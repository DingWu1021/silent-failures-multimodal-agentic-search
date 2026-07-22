"""Tool implementations for the ReAct agent.

All tools return a dict with a `summary` string (shown to the model) and optional
`raw` payload kept only in the trajectory log.

Search backends are selected by env var: SERPAPI_KEY > TAVILY_API_KEY > GOOGLE_API_KEY/GOOGLE_CSE_ID.
If none is set, tools fall back to a stub that returns an error message; callers
must check `is_stub`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


# Image-host upload cache. Litterbox URLs expire after their TTL window
# (default 1h), so we record upload_time and re-upload if stale. Cache lives
# under data/ so it is reproducible across runs of the same campaign.
_IMG_UPLOAD_CACHE = Path(__file__).resolve().parents[2] / "data" / "image_uploads.json"
_LITTERBOX_TTL_S = 60 * 60  # 1 hour, matches default `time=1h` upload


def _http() -> httpx.Client:
    return httpx.Client(timeout=30.0, follow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; SilentFailureBot/0.1)"})


def _serpapi_key() -> str | None:
    return os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY")


def _tavily_key() -> str | None:
    return os.environ.get("TAVILY_API_KEY")


def _backend() -> str:
    if _serpapi_key():
        return "serpapi"
    if _tavily_key():
        return "tavily"
    if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
        return "google_cse"
    return "stub"


def web_search(query: str, *, num: int = 5) -> dict[str, Any]:
    backend = _backend()
    if backend == "stub":
        return {"summary": f"[web_search unavailable: no search API key configured]",
                "is_stub": True, "results": []}
    with _http() as cli:
        if backend == "serpapi":
            r = cli.get("https://serpapi.com/search.json",
                        params={"q": query, "engine": "google", "num": num,
                                "api_key": _serpapi_key()})
            r.raise_for_status()
            data = r.json()
            results = [{"title": x.get("title"), "url": x.get("link"),
                        "snippet": x.get("snippet")} for x in data.get("organic_results", [])[:num]]
        elif backend == "tavily":
            r = cli.post("https://api.tavily.com/search",
                         json={"api_key": _tavily_key(), "query": query,
                               "max_results": num, "include_answer": False})
            r.raise_for_status()
            data = r.json()
            results = [{"title": x.get("title"), "url": x.get("url"),
                        "snippet": x.get("content")} for x in data.get("results", [])[:num]]
        else:  # google_cse
            r = cli.get("https://www.googleapis.com/customsearch/v1",
                        params={"q": query, "num": num,
                                "key": os.environ["GOOGLE_API_KEY"],
                                "cx": os.environ["GOOGLE_CSE_ID"]})
            r.raise_for_status()
            data = r.json()
            results = [{"title": x.get("title"), "url": x.get("link"),
                        "snippet": x.get("snippet")} for x in data.get("items", [])[:num]]
    lines = [f"[{i+1}] {x['title']}\n    URL: {x['url']}\n    {x['snippet']}"
             for i, x in enumerate(results)]
    return {"summary": "\n".join(lines) if lines else "(no results)",
            "is_stub": False, "results": results, "backend": backend}


def image_search(query: str, *, num: int = 5) -> dict[str, Any]:
    if not _serpapi_key():
        return {"summary": f"[image_search needs SERPAPI_API_KEY]",
                "is_stub": True, "results": []}
    with _http() as cli:
        r = cli.get("https://serpapi.com/search.json",
                    params={"q": query, "engine": "google_images", "num": num,
                            "api_key": _serpapi_key()})
        r.raise_for_status()
        data = r.json()
    results = [{"title": x.get("title"), "image_url": x.get("original"),
                "source_url": x.get("link"), "snippet": x.get("snippet")}
               for x in data.get("images_results", [])[:num]]
    lines = [f"[{i+1}] {x['title']}\n    image: {x['image_url']}\n    page: {x['source_url']}"
             for i, x in enumerate(results)]
    return {"summary": "\n".join(lines) if lines else "(no results)",
            "is_stub": False, "results": results, "backend": "serpapi"}


def web_fetch(url: str, *, max_chars: int = 4000) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"summary": f"[web_fetch: invalid url {url}]", "is_stub": True}
    try:
        with _http() as cli:
            r = cli.get(url)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "html" in ctype or ctype == "":
                soup = BeautifulSoup(r.text, "lxml")
                for s in soup(["script", "style", "noscript"]):
                    s.extract()
                text = soup.get_text(separator="\n", strip=True)
            else:
                text = r.text
    except Exception as e:
        return {"summary": f"[web_fetch error: {type(e).__name__}: {e}]", "is_stub": True}
    truncated = text[:max_chars]
    return {"summary": truncated, "is_stub": False,
            "url": url, "truncated": len(text) > max_chars, "full_length": len(text)}


def _load_upload_cache() -> dict[str, dict]:
    if _IMG_UPLOAD_CACHE.exists():
        try:
            return json.loads(_IMG_UPLOAD_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_upload_cache(cache: dict[str, dict]) -> None:
    _IMG_UPLOAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _IMG_UPLOAD_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _upload_image_for_lens(image_path: str | Path) -> tuple[str | None, str | None]:
    """Upload local image to a public host and return (url, error).

    Tries litterbox.catbox.moe (anonymous, 1-hour TTL by default).
    Returns (None, error_message) on failure.

    Uploads are cached by file hash to avoid re-uploading the same image
    across multiple trajectories.
    """
    p = Path(image_path)
    if not p.exists():
        return None, f"image not found: {image_path}"
    digest = _hash_file(p)
    cache = _load_upload_cache()
    entry = cache.get(digest)
    if entry and (time.time() - entry.get("uploaded_at", 0)) < _LITTERBOX_TTL_S - 60:
        return entry["url"], None

    try:
        with p.open("rb") as f:
            files = {"fileToUpload": (p.name, f, "application/octet-stream")}
            data = {"reqtype": "fileupload", "time": "1h"}
            r = httpx.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                files=files, data=data, timeout=60.0,
            )
        text = (r.text or "").strip()
        if r.status_code != 200 or not text.startswith("http"):
            return None, f"upload failed: status={r.status_code} body={text[:120]}"
    except Exception as e:
        return None, f"upload exception: {type(e).__name__}: {e}"

    cache[digest] = {"url": text, "uploaded_at": time.time(), "src": str(p)}
    _save_upload_cache(cache)
    return text, None


def reverse_image_search(image_path: str, *, num: int = 8) -> dict[str, Any]:
    """Identify the source/context of an input image via Google Lens.

    Pipeline: upload local file to a temporary public host, then call
    SerpAPI's google_lens engine with the resulting URL. Returns the top
    `num` visual-match titles + source URLs in the same format as other
    search tools, plus a knowledge-graph summary if Google Lens identifies
    a recognised entity.
    """
    if not _serpapi_key():
        return {"summary": "[reverse_image_search needs SERPAPI_API_KEY]",
                "is_stub": True, "results": []}
    url, err = _upload_image_for_lens(image_path)
    if err:
        return {"summary": f"[reverse_image_search: {err}]",
                "is_stub": True, "results": []}
    try:
        with _http() as cli:
            r = cli.get("https://serpapi.com/search.json",
                        params={"engine": "google_lens", "url": url,
                                "api_key": _serpapi_key()})
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return {"summary": f"[reverse_image_search SerpAPI error: {type(e).__name__}: {e}]",
                "is_stub": True, "results": []}

    matches = data.get("visual_matches") or []
    results = []
    for m in matches[:num]:
        results.append({
            "title": m.get("title") or "",
            "url": m.get("link") or "",
            "source": m.get("source") or "",
            "snippet": m.get("snippet") or "",
        })
    kg = data.get("knowledge_graph") or {}
    kg_summary = ""
    if kg:
        bits = [kg.get("title")] + [v for k, v in kg.items()
                                     if isinstance(v, str) and k not in ("title", "image")]
        kg_summary = " | ".join([b for b in bits if b])[:300]

    lines = []
    if kg_summary:
        lines.append(f"[knowledge_graph] {kg_summary}")
    for i, x in enumerate(results):
        lines.append(f"[{i+1}] {x['title']}\n    URL: {x['url']}\n    source: {x['source']}")
    summary = "\n".join(lines) if lines else "(no visual matches)"
    return {"summary": summary, "is_stub": False, "results": results,
            "knowledge_graph": kg, "uploaded_url": url, "backend": "serpapi_lens"}


def crop(image_path: str, box: list[int]) -> dict[str, Any]:
    """Crop image to (left, top, right, bottom). Saves to same dir with suffix."""
    p = Path(image_path)
    if not p.exists():
        return {"summary": f"[crop: file not found {image_path}]", "is_stub": True}
    if len(box) != 4:
        return {"summary": "[crop: box must be [left, top, right, bottom]]", "is_stub": True}
    img = Image.open(p)
    left, top, right, bottom = box
    left = max(0, min(left, img.width))
    right = max(0, min(right, img.width))
    top = max(0, min(top, img.height))
    bottom = max(0, min(bottom, img.height))
    if right <= left or bottom <= top:
        return {"summary": "[crop: degenerate box]", "is_stub": True}
    cropped = img.crop((left, top, right, bottom))
    out = p.with_name(f"{p.stem}_crop_{left}_{top}_{right}_{bottom}{p.suffix}")
    cropped.save(out)
    return {"summary": f"Cropped to {cropped.size}, saved to {out.name}",
            "is_stub": False, "path": str(out), "size": cropped.size}


TOOLS = {
    "web_search": web_search,
    "image_search": image_search,
    "reverse_image_search": reverse_image_search,
    "web_fetch": web_fetch,
    "crop": crop,
}


def tool_catalog_text() -> str:
    return (
        "- reverse_image_search(image_path: str) -> for an INPUT image, find pages where it appears, plus visually similar images and any recognised entity (Google Lens). Use this FIRST whenever the question depends on identifying what is shown.\n"
        "- web_search(query: str) -> text snippets from top web results\n"
        "- image_search(query: str) -> image URLs from Google Images by text query (needs SERPAPI)\n"
        "- web_fetch(url: str) -> main text content of a web page (truncated)\n"
        "- crop(image_path: str, box: [l,t,r,b]) -> cropped region of an input image; useful before a second reverse_image_search on a sub-region"
    )
