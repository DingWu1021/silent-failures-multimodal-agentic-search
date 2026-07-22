"""LLM-judge runner: applies the silent-failure rubric to a trajectory.

Input: trajectory JSON (from react_agent) + task record (question, ground truth, image paths).
Output: JSON with {answer_correct, failures{...}, overall_rationale}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agents.llm_client import get_client


RUBRIC_PATH = Path(__file__).parent / "rubric.md"

FAILURE_KEYS = [
    "modality_shortcut",
    "phantom_grounding",
    "wrong_evidence_right_answer",
    "over_retrieval_laundering",
    "cross_modal_contradiction",
    "provenance_hallucination",
]


JUDGE_SYSTEM = """You are a strict meta-evaluator for multimodal search agents.
Follow the rubric exactly. Always return valid JSON matching the schema in the rubric."""


def _trajectory_to_text(traj: dict) -> str:
    lines = [f"Final answer: {traj.get('final_answer', '')}"]
    lines.append(f"Stopped reason: {traj.get('stopped_reason', '')}")
    lines.append(f"Steps ({len(traj.get('steps', []))}):")
    for s in traj.get("steps", []):
        idx = s.get("idx")
        thought = (s.get("thought") or "").strip()
        tool = s.get("tool")
        tinput = s.get("tool_input")
        obs = (s.get("observation") or "").strip()
        if len(obs) > 1500:
            obs = obs[:1500] + " …[truncated]"
        if tool:
            lines.append(f"[step {idx}] THOUGHT: {thought}\n  TOOL: {tool} INPUT: {json.dumps(tinput, ensure_ascii=False)}\n  OBSERVATION: {obs}")
        elif s.get("final_answer"):
            lines.append(f"[step {idx}] THOUGHT: {thought}\n  FINAL_ANSWER: {s['final_answer']}")
        else:
            lines.append(f"[step {idx}] THOUGHT: {thought}\n  (no tool, no final — parse error: {s.get('error')})")
    return "\n".join(lines)


def build_judge_prompt(*, rubric: str, question: str, ground_truth: str,
                       trajectory: dict, image_paths: list[str]) -> list[dict]:
    traj_text = _trajectory_to_text(trajectory)
    parts: list[dict] = [{
        "type": "text",
        "text": (
            f"RUBRIC:\n{rubric}\n\n"
            f"---\nTASK QUESTION: {question}\n"
            f"GROUND-TRUTH ANSWER: {ground_truth}\n"
            f"---\nAGENT TRAJECTORY:\n{traj_text}\n"
            f"---\nNow output the JSON verdict. Only the JSON, no prose outside it."
        ),
    }]
    for ip in image_paths:
        parts.append({"type": "image", "source": ip})
    return [{"role": "user", "content": parts}]


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON found in judge output: {text[:300]}")
    return json.loads(m.group(0))


def _sanity_check(verdict: dict) -> list[str]:
    errs: list[str] = []
    if verdict.get("answer_correct") not in ("True", "False", "Partial"):
        errs.append(f"answer_correct invalid: {verdict.get('answer_correct')!r}")
    failures = verdict.get("failures", {})
    for k in FAILURE_KEYS:
        if k not in failures:
            errs.append(f"missing failure key {k}")
            continue
        v = failures[k]
        if not isinstance(v, dict) or "present" not in v:
            errs.append(f"failure {k} malformed")
            continue
        if not isinstance(v["present"], bool):
            errs.append(f"failure {k}.present not bool")
    return errs


@dataclass
class JudgeVerdict:
    task_id: str
    model: str
    judge_model: str
    answer_correct: str
    failures: dict
    overall_rationale: str
    sanity_errors: list[str]
    raw_text: str


def judge_trajectory(
    *,
    trajectory: dict,
    question: str,
    ground_truth: str,
    image_paths: list[str],
    judge_model: str,
    max_tokens: int = 1024,
) -> JudgeVerdict:
    rubric = RUBRIC_PATH.read_text()
    messages = build_judge_prompt(rubric=rubric, question=question, ground_truth=ground_truth,
                                  trajectory=trajectory, image_paths=image_paths)
    client = get_client()
    resp = client.chat(model=judge_model, messages=messages, system=JUDGE_SYSTEM,
                       max_tokens=max_tokens, temperature=0.0)
    try:
        verdict = _extract_json(resp.text)
    except Exception as e:
        return JudgeVerdict(task_id=trajectory.get("task_id", ""), model=trajectory.get("model", ""),
                            judge_model=judge_model, answer_correct="False", failures={},
                            overall_rationale="", sanity_errors=[f"json_parse_failed: {e}"],
                            raw_text=resp.text)
    errs = _sanity_check(verdict)
    return JudgeVerdict(
        task_id=trajectory.get("task_id", ""),
        model=trajectory.get("model", ""),
        judge_model=judge_model,
        answer_correct=verdict.get("answer_correct", "False"),
        failures=verdict.get("failures", {}),
        overall_rationale=verdict.get("overall_rationale", ""),
        sanity_errors=errs,
        raw_text=resp.text,
    )


def save_verdict(verdict: JudgeVerdict, root: str | Path) -> Path:
    root = Path(root)
    out_dir = root / verdict.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{verdict.task_id}.json"
    out.write_text(json.dumps({
        "task_id": verdict.task_id,
        "model": verdict.model,
        "judge_model": verdict.judge_model,
        "answer_correct": verdict.answer_correct,
        "failures": verdict.failures,
        "overall_rationale": verdict.overall_rationale,
        "sanity_errors": verdict.sanity_errors,
        "raw_text": verdict.raw_text,
    }, ensure_ascii=False, indent=2))
    return out
