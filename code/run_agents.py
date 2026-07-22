"""Phase 3 — batch-run the ReAct agent across (model × task) combinations.

Resumable: skips any trajectory already present on disk.
Caches per-run progress; prints cost summary at the end.

Usage:
    python code/run_agents.py --models claude-sonnet-4-6 gpt-4o-2024-08-06 gemini-2.5-pro \
                              --max-steps 10 --limit 200
    python code/run_agents.py --models claude-sonnet-4-6 --limit 10   # small pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.agents.react_agent import run_agent, save_trajectory

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
TRAJ_ROOT = ROOT / "data" / "trajectories"


def load_sample() -> list[dict]:
    recs = [json.loads(l) for l in SAMPLE_PATH.read_text().splitlines() if l.strip()]
    return [r for r in recs if "task_id" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="run only first N tasks")
    ap.add_argument("--start", type=int, default=0, help="skip first N tasks")
    ap.add_argument("--overwrite", action="store_true", help="ignore cached trajectories")
    ap.add_argument("--retry-errors", action="store_true",
                    help="redo trajectories whose stopped_reason is llm_error")
    args = ap.parse_args()

    sample = load_sample()
    if args.limit is not None:
        sample = sample[args.start: args.start + args.limit]
    else:
        sample = sample[args.start:]
    print(f"Loaded {len(sample)} tasks.")

    totals: dict[str, dict] = {}
    t0 = time.time()

    for model in args.models:
        print(f"\n{'='*60}\nMODEL: {model}\n{'='*60}")
        m_totals = {"done": 0, "skipped": 0, "errors": 0,
                    "input_tokens": 0, "output_tokens": 0, "wall_s": 0.0}
        out_dir = TRAJ_ROOT / model.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(sample):
            tid = task["task_id"]
            out_p = out_dir / f"{tid}.json"
            if out_p.exists() and not args.overwrite:
                if args.retry_errors:
                    try:
                        cached = json.loads(out_p.read_text())
                        if cached.get("stopped_reason") != "llm_error":
                            m_totals["skipped"] += 1
                            continue
                    except Exception:
                        pass
                else:
                    m_totals["skipped"] += 1
                    continue
            t1 = time.time()
            try:
                traj = run_agent(
                    task_id=tid,
                    model=model,
                    question=task["question"],
                    image_paths=task.get("image_paths") or [],
                    max_steps=args.max_steps,
                    max_tokens_per_call=args.max_tokens,
                )
                save_trajectory(traj, TRAJ_ROOT)
                m_totals["done"] += 1
                m_totals["input_tokens"] += traj.token_cost.get("input_tokens") or 0
                m_totals["output_tokens"] += traj.token_cost.get("output_tokens") or 0
                m_totals["wall_s"] += traj.wall_time_s
                print(f"  [{i+1:4d}/{len(sample)}] {tid} steps={len(traj.steps)} "
                      f"stop={traj.stopped_reason} {traj.wall_time_s:.1f}s "
                      f"in={traj.token_cost.get('input_tokens')} "
                      f"out={traj.token_cost.get('output_tokens')} "
                      f"ans={(traj.final_answer or '')[:60]!r}")
            except KeyboardInterrupt:
                print("\ninterrupted.")
                raise
            except Exception as e:
                m_totals["errors"] += 1
                elapsed = time.time() - t1
                err_p = out_dir / f"{tid}.error.json"
                err_p.write_text(json.dumps({"task_id": tid, "model": model,
                                             "error": f"{type(e).__name__}: {e}",
                                             "traceback": traceback.format_exc(limit=4),
                                             "elapsed_s": elapsed},
                                            ensure_ascii=False, indent=2))
                print(f"  [{i+1:4d}/{len(sample)}] {tid} ERROR {type(e).__name__}: {str(e)[:160]}")

        totals[model] = m_totals
        print(f"\nModel {model} summary: {m_totals}")

    print(f"\n{'='*60}\nGrand total wall: {time.time() - t0:.1f}s")
    for m, t in totals.items():
        print(f"  {m}: done={t['done']} skipped={t['skipped']} errors={t['errors']} "
              f"in={t['input_tokens']} out={t['output_tokens']} agent_wall={t['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
