"""Minimal ReAct-style multimodal search agent.

Produces a structured trajectory JSON so downstream judges can inspect each step.
The prompt format is text-only parsing (THOUGHT/TOOL/INPUT/FINAL_ANSWER) to keep
behavior identical across providers — we do NOT use provider-native tool-calling
because that introduces cross-provider variability.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import get_client
from .tools import TOOLS, tool_catalog_text


SYSTEM_PROMPT = """You are a multimodal search agent. You have access to these tools:
{tool_catalog}

The input image paths are listed below the question. Use the tools iteratively to reach a grounded answer.

At each step, output EXACTLY ONE of these two forms and NOTHING else (no preamble, no numbered lists, no markdown):

THOUGHT: <one or two sentences, concise>
TOOL: <tool_name>
INPUT: <compact JSON, e.g. {{"query": "..."}} or {{"url": "..."}} or {{"image_path": "data/raw/images/mmsp_XXXX/img_1.png"}} or {{"image_path": "...", "box": [0,0,100,100]}}>

-- OR --

THOUGHT: <one or two sentences, concise>
FINAL_ANSWER: <your concise answer>

Strategy:
- If identifying what is shown matters for the answer, START with reverse_image_search on the supplied image path. This is far more reliable than describing the image in words and using web_search.
- If reverse_image_search returns generic/noisy matches, try crop on a salient region first, then reverse_image_search on the crop.
- Use web_search / web_fetch to verify specific facts on a candidate source page found by reverse_image_search.
- Only use image_search (a TEXT-to-image lookup) when you already know what to search for.

Rules:
- THOUGHT must be at most 2 sentences. Do not plan multiple steps; plan only this one step.
- Do not exceed {max_steps} tool calls.
- For image-related tools, use the EXACT image_path from the question. Do not invent paths like 'image.png'.
- Cite any URL you relied on inside the FINAL_ANSWER when relevant.
- If a tool returns an error or no useful info, try an alternative approach; do not invent sources.
"""


@dataclass
class Step:
    idx: int
    raw_model_output: str
    thought: str = ""
    tool: str | None = None
    tool_input: dict | None = None
    observation: str = ""
    observation_raw: Any = None
    final_answer: str | None = None
    error: str | None = None


@dataclass
class Trajectory:
    task_id: str
    model: str
    question: str
    image_paths: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    stopped_reason: str = ""
    token_cost: dict = field(default_factory=dict)
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


_SEP = r"[:\s]"
_STEP_RE = re.compile(
    r"THOUGHT" + _SEP + r"\s*(?P<thought>.*?)"
    r"(?:\n\s*(?:"
    r"TOOL" + _SEP + r"\s*(?P<tool>\S+)\s*\n\s*INPUT" + _SEP + r"\s*(?P<input>.*)"
    r"|FINAL_ANSWER" + _SEP + r"\s*(?P<final>.*)"
    r"))",
    re.DOTALL,
)


def _parse_step(text: str) -> dict:
    m = _STEP_RE.search(text)
    if not m:
        return {"thought": text.strip(), "tool": None, "input": None, "final": None,
                "parse_error": "no THOUGHT/TOOL or THOUGHT/FINAL_ANSWER match"}
    thought = m.group("thought").strip()
    if m.group("final") is not None:
        return {"thought": thought, "tool": None, "input": None,
                "final": m.group("final").strip(), "parse_error": None}
    tool = m.group("tool").strip()
    inp_raw = m.group("input").strip()
    inp_raw = re.split(r"\n\s*(?:THOUGHT|FINAL_ANSWER|TOOL)[:\s]", inp_raw)[0].strip()
    if inp_raw.startswith("```"):
        inp_raw = re.sub(r"^```(?:json)?\n?|\n?```$", "", inp_raw, flags=re.MULTILINE).strip()
    try:
        inp = json.loads(inp_raw)
    except json.JSONDecodeError:
        try:
            decoder = json.JSONDecoder()
            inp, _end = decoder.raw_decode(inp_raw)
        except Exception as e:
            return {"thought": thought, "tool": tool, "input": None, "final": None,
                    "parse_error": f"input not valid JSON: {e}; raw={inp_raw[:200]}"}
    return {"thought": thought, "tool": tool, "input": inp, "final": None, "parse_error": None}


def _obs_to_str(obs: dict) -> str:
    if "summary" in obs:
        return obs["summary"]
    return json.dumps(obs, ensure_ascii=False)[:2000]


def run_agent(
    *,
    task_id: str,
    model: str,
    question: str,
    image_paths: list[str] | None = None,
    max_steps: int = 10,
    max_tokens_per_call: int = 1024,
    temperature: float = 0.2,
) -> Trajectory:
    image_paths = image_paths or []
    client = get_client()
    system = SYSTEM_PROMPT.format(tool_catalog=tool_catalog_text(), max_steps=max_steps)

    user_parts: list[dict] = [{"type": "text", "text": f"Question: {question}"}]
    for i, ip in enumerate(image_paths):
        user_parts.append({"type": "text", "text": f"image_path: {ip}"})
        user_parts.append({"type": "image", "source": ip})

    traj = Trajectory(task_id=task_id, model=model, question=question, image_paths=list(image_paths))
    messages: list[dict] = [{"role": "user", "content": user_parts}]

    total_in = 0
    total_out = 0
    start = time.time()

    for step_idx in range(max_steps):
        try:
            resp = client.chat(model=model, messages=messages, system=system,
                               max_tokens=max_tokens_per_call, temperature=temperature)
        except Exception as e:
            step = Step(idx=step_idx, raw_model_output="", error=f"LLM call failed: {type(e).__name__}: {e}")
            traj.steps.append(step)
            traj.stopped_reason = "llm_error"
            break

        total_in += resp.usage.get("input_tokens", 0) or 0
        total_out += resp.usage.get("output_tokens", 0) or 0
        parsed = _parse_step(resp.text)
        step = Step(idx=step_idx, raw_model_output=resp.text, thought=parsed["thought"])

        if parsed.get("parse_error"):
            step.error = parsed["parse_error"]
            step.observation = "Parse error. Please output exactly: THOUGHT/TOOL/INPUT or THOUGHT/FINAL_ANSWER."
            traj.steps.append(step)
            messages.append({"role": "assistant", "content": resp.text})
            messages.append({"role": "user", "content": step.observation})
            continue

        if parsed["final"] is not None:
            step.final_answer = parsed["final"]
            traj.steps.append(step)
            traj.final_answer = parsed["final"]
            traj.stopped_reason = "final_answer"
            break

        tool_name = parsed["tool"]
        tool_input = parsed["input"] or {}
        step.tool = tool_name
        step.tool_input = tool_input

        if tool_name not in TOOLS:
            step.error = f"Unknown tool {tool_name}"
            step.observation = f"Error: unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
        else:
            try:
                raw_obs = TOOLS[tool_name](**tool_input)
            except TypeError as e:
                raw_obs = {"summary": f"[tool arg error: {e}]", "is_stub": True}
            except Exception as e:
                raw_obs = {"summary": f"[tool error: {type(e).__name__}: {e}]", "is_stub": True}
            step.observation_raw = raw_obs
            step.observation = _obs_to_str(raw_obs)

        traj.steps.append(step)
        messages.append({"role": "assistant", "content": resp.text})
        messages.append({"role": "user", "content": f"OBSERVATION: {step.observation}"})
    else:
        traj.stopped_reason = "max_steps"

    traj.wall_time_s = round(time.time() - start, 2)
    traj.token_cost = {"input_tokens": total_in, "output_tokens": total_out}
    return traj


def save_trajectory(traj: Trajectory, root: str | Path) -> Path:
    root = Path(root)
    out_dir = root / traj.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{traj.task_id}.json"
    out.write_text(json.dumps(traj.to_dict(), ensure_ascii=False, indent=2))
    return out
