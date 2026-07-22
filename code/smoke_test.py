"""End-to-end smoke test for the three MatrixLLM channels.

Verifies:
  1. Claude (Anthropic SDK)      — claude-sonnet-4-6
  2. GPT-4o (OpenAI SDK)         — gpt-4o-2024-08-06
  3. Gemini (OpenAI SDK)         — gemini-3-pro-preview

Also runs a one-step text-only ReAct agent call against Claude to exercise the
trajectory pipeline end to end (no external tools needed).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.agents.llm_client import get_client
from code.agents.react_agent import run_agent


TRIALS = [
    ("claude-sonnet-4-6",   "Anthropic SDK → MatrixLLM /anthropic"),
    ("gpt-4o-2024-11-20",   "OpenAI SDK → MatrixLLM /v1"),
    ("gemini-2.5-pro",      "OpenAI SDK → MatrixLLM /v1"),
]


def ping(model: str, label: str) -> dict:
    client = get_client()
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
            max_tokens=16,
            temperature=0.0,
        )
        text = (resp.text or "").strip()
        ok = "PONG" in text.upper()
        return {"model": model, "label": label, "ok": ok, "text": text, "usage": resp.usage}
    except Exception as e:
        return {"model": model, "label": label, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3)}


def main() -> int:
    if not (os.environ.get("MATRIXLLM_API_KEY") or os.environ.get("ak")):
        print("ERROR: MATRIXLLM_API_KEY not set (or ak). Copy .env.example to .env.")
        return 2

    print("=" * 60)
    print("Smoke test: MatrixLLM channels")
    print("=" * 60)
    any_fail = False
    for model, label in TRIALS:
        r = ping(model, label)
        status = "OK " if r["ok"] else "FAIL"
        print(f"[{status}] {model:30s} {label}")
        if r["ok"]:
            print(f"       reply: {r['text']!r}  usage: {r.get('usage')}")
        else:
            any_fail = True
            print(f"       {r.get('error')}")
        print()

    print("=" * 60)
    print("Smoke test: one-step ReAct on claude-sonnet-4-6 (no external tools)")
    print("=" * 60)
    try:
        traj = run_agent(
            task_id="smoke_0",
            model="claude-sonnet-4-6",
            question="What is 17 * 23? Answer concisely.",
            image_paths=[],
            max_steps=2,
            max_tokens_per_call=256,
        )
        print(f"stopped_reason: {traj.stopped_reason}")
        print(f"final_answer: {traj.final_answer!r}")
        print(f"steps: {len(traj.steps)}  tokens: {traj.token_cost}  wall: {traj.wall_time_s}s")
        for s in traj.steps:
            print(f"  step {s.idx}: tool={s.tool!r} final={s.final_answer!r} err={s.error!r}")
    except Exception as e:
        any_fail = True
        print(f"ReAct run failed: {type(e).__name__}: {e}")

    print()
    print("DONE.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
