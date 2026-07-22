"""Unified LLM client wrapping MatrixLLM gateway.

Claude models go through the Anthropic SDK (Messages API).
GPT + Gemini models go through the OpenAI SDK (OpenAI-compatible endpoint).
Both share one API key.
"""

from __future__ import annotations

import base64
import os
import random
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()


# Transient error classes that should be retried with exponential backoff.
# These cover the three failure modes observed in production runs:
#   - Anthropic APIConnectionError (Claude trajectory ~70% kill rate)
#   - OpenAI RateLimitError 429 (Gemini ~30%)
#   - OpenAI InternalServerError 5xx (GPT-4o ~12%)
#   - APITimeoutError (both)
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)

_MAX_RETRIES = 5
_BASE_BACKOFF = 1.5  # seconds; doubled each attempt with ±20% jitter


def _retry(fn, *, what: str = "llm call"):
    """Call `fn` with exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except _RETRYABLE as e:
            last_exc = e
            if attempt == _MAX_RETRIES:
                break
            sleep_s = _BASE_BACKOFF * (2 ** attempt)
            sleep_s *= 1.0 + random.uniform(-0.2, 0.2)
            time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]

_KEEPALIVE_SOCKET_OPTIONS = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30),
]
if hasattr(socket, "TCP_KEEPALIVE"):
    _KEEPALIVE_SOCKET_OPTIONS.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))
if hasattr(socket, "TCP_KEEPIDLE"):
    _KEEPALIVE_SOCKET_OPTIONS.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60))


def _build_http_client() -> httpx.Client:
    transport = httpx.HTTPTransport(socket_options=_KEEPALIVE_SOCKET_OPTIONS)
    return httpx.Client(transport=transport, timeout=httpx.Timeout(300.0, connect=15.0))


def _api_key() -> str:
    key = os.environ.get("MATRIXLLM_API_KEY") or os.environ.get("ak")
    if not key:
        raise RuntimeError("MATRIXLLM_API_KEY not set. Copy .env.example to .env and fill it in.")
    return key


def _anthropic_base() -> str:
    return os.environ.get("MATRIXLLM_ANTHROPIC_BASE_URL", "https://matrixllm.alipay.com/anthropic")


def _openai_base() -> str:
    return os.environ.get("MATRIXLLM_OPENAI_BASE_URL", "https://matrixllm.alipay.com/v1")


CLAUDE_PREFIX = ("claude",)
OPENAI_PREFIX = ("gpt", "gemini")


def _provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith(CLAUDE_PREFIX):
        return "anthropic"
    if m.startswith(OPENAI_PREFIX):
        return "openai"
    raise ValueError(f"Unknown provider for model {model!r}")


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: dict = field(default_factory=dict)


_MAX_IMAGE_DIM = 1280
_MAX_IMAGE_BYTES = 1_500_000


def _resize_if_needed(data: bytes) -> tuple[bytes, str]:
    """If image is too big, downscale + re-encode as JPEG at q=85."""
    from io import BytesIO

    from PIL import Image
    if len(data) <= _MAX_IMAGE_BYTES:
        try:
            img = Image.open(BytesIO(data))
            if max(img.size) <= _MAX_IMAGE_DIM:
                fmt = (img.format or "JPEG").lower()
                return data, f"image/{'jpeg' if fmt == 'jpg' else fmt}"
        except Exception:
            return data, "image/jpeg"
    img = Image.open(BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_IMAGE_DIM:
        scale = _MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _encode_image(path_or_bytes, media_type: str | None = None) -> tuple[str, str]:
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = path_or_bytes
    data, mt = _resize_if_needed(data)
    return base64.b64encode(data).decode("ascii"), mt


def _to_anthropic_content(parts: Iterable[dict]) -> list[dict]:
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            b64, mt = _encode_image(p["source"], p.get("media_type"))
            out.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
        else:
            raise ValueError(f"Unknown part type {p['type']}")
    return out


def _to_openai_content(parts: Iterable[dict]) -> list[dict]:
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            b64, mt = _encode_image(p["source"], p.get("media_type"))
            out.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}})
        else:
            raise ValueError(f"Unknown part type {p['type']}")
    return out


class LLMClient:
    def __init__(self):
        self._http = _build_http_client()
        self._anthropic = Anthropic(
            base_url=_anthropic_base(),
            api_key=_api_key(),
            http_client=self._http,
            max_retries=0,
        )
        self._openai = OpenAI(
            base_url=_openai_base(),
            api_key=_api_key(),
            http_client=self._http,
            max_retries=0,
        )

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        provider = _provider_for(model)
        if provider == "anthropic":
            return self._chat_anthropic(model, messages, system=system,
                                        max_tokens=max_tokens, temperature=temperature)
        return self._chat_openai(model, messages, system=system,
                                 max_tokens=max_tokens, temperature=temperature)

    def _chat_anthropic(self, model, messages, *, system, max_tokens, temperature) -> LLMResponse:
        ant_msgs = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                ant_msgs.append({"role": m["role"], "content": content})
            else:
                ant_msgs.append({"role": m["role"], "content": _to_anthropic_content(content)})
        kwargs = dict(model=model, max_tokens=max_tokens, messages=ant_msgs, temperature=temperature)
        if system:
            kwargs["system"] = system
        def _call():
            try:
                return self._anthropic.messages.create(**kwargs)
            except Exception as e:
                # Some upstream Bedrock backends reject `temperature` for newer models.
                # Drop it and try again immediately (this branch is non-transient).
                if "temperature" in str(e).lower() and "deprecated" in str(e).lower():
                    kwargs.pop("temperature", None)
                    return self._anthropic.messages.create(**kwargs)
                raise
        resp = _retry(_call, what=f"anthropic chat({model})")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return LLMResponse(text=text, raw=resp, usage=usage)

    def _chat_openai(self, model, messages, *, system, max_tokens, temperature) -> LLMResponse:
        oa_msgs = []
        if system:
            oa_msgs.append({"role": "system", "content": system})
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                oa_msgs.append({"role": m["role"], "content": content})
            else:
                oa_msgs.append({"role": m["role"], "content": _to_openai_content(content)})
        effective_max = max_tokens
        if model.lower().startswith("gemini-2.5-pro") and effective_max < 3072:
            effective_max = 3072
        resp = _retry(
            lambda: self._openai.chat.completions.create(
                model=model, messages=oa_msgs, max_tokens=effective_max,
                temperature=temperature, stream=False,
            ),
            what=f"openai chat({model})",
        )
        choice = resp.choices[0]
        msg = choice.message
        text = ""
        if msg is not None and msg.content:
            text = msg.content
        finish = getattr(choice, "finish_reason", None)
        usage = {}
        if resp.usage:
            ct = resp.usage.completion_tokens or 0
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": ct,
                     "finish_reason": finish}
            details = getattr(resp.usage, "completion_tokens_details", None)
            if details and getattr(details, "reasoning_tokens", None):
                usage["reasoning_tokens"] = details.reasoning_tokens
        if not text and finish == "length":
            text = ""
        return LLMResponse(text=text, raw=resp, usage=usage)


_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
