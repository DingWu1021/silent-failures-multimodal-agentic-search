"""Phase 5 — run the LLM judge on every (model, task) trajectory.

Resumable: skips verdicts already on disk.
Also records judge cost totals.

Usage:
    python code/run_judge_all.py --judge-model claude-opus-4-7
    python code/run_judge_all.py --judge-model gpt-4o-2024-08-06 --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.agents.llm_client import get_client  # ensure .env loaded
from code.judges.run_judge import judge_trajectory, save_verdict

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
TRAJ_ROOT = ROOT / "data" / "trajectories"
VERDICT_ROOT = ROOT / "data" / "annotations" / "judge"


def load_tasks_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in SAMPLE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "task_id" in r:
            out[r["task_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "claude-opus-4-7"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these agent-model subdirs; default: all under data/trajectories/")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()

    tasks = load_tasks_by_id()
    if not TRAJ_ROOT.exists():
        print(f"No trajectories at {TRAJ_ROOT}. Run Phase 3 first.")
        return

    model_dirs = [p for p in TRAJ_ROOT.iterdir() if p.is_dir()]
    if args.models:
        model_dirs = [p for p in model_dirs if p.name in args.models]

    grand = {"done": 0, "skipped": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0, "wall_s": 0.0}
    t_start = time.time()

    for mdir in model_dirs:
        print(f"\n=== Judging trajectories in {mdir.name} ===")
        traj_files = sorted(mdir.glob("mmsp_*.json"))
        traj_files = [p for p in traj_files if not p.name.endswith(".error.json")]
        if args.limit:
            traj_files = traj_files[: args.limit]
        out_dir = VERDICT_ROOT / mdir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, tp in enumerate(traj_files):
            out_p = out_dir / tp.name
            if out_p.exists() and not args.overwrite:
                grand["skipped"] += 1
                continue
            try:
                traj = json.loads(tp.read_text())
                task = tasks.get(traj["task_id"])
                if task is None:
                    print(f"  [{i+1}] SKIP {traj['task_id']} (not in sample)")
                    continue
                t0 = time.time()
                verdict = judge_trajectory(
                    trajectory=traj,
                    question=task["question"],
                    ground_truth=", ".join(task.get("answer") or []),
                    image_paths=task.get("image_paths") or [],
                    judge_model=args.judge_model,
                    max_tokens=args.max_tokens,
                )
                save_verdict(verdict, VERDICT_ROOT)
                dt = time.time() - t0
                grand["done"] += 1
                grand["wall_s"] += dt
                flags = [k for k, v in (verdict.failures or {}).items()
                         if isinstance(v, dict) and v.get("present")]
                print(f"  [{i+1:3d}/{len(traj_files)}] {traj['task_id']} {dt:.1f}s "
                      f"ans={verdict.answer_correct} flags={flags} "
                      f"sanity_errs={len(verdict.sanity_errors)}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                grand["errors"] += 1
                err_p = out_dir / f"{tp.stem}.error.json"
                err_p.write_text(json.dumps({"task_id": tp.stem, "model": mdir.name,
                                             "error": f"{type(e).__name__}: {e}",
                                             "traceback": traceback.format_exc(limit=4)},
                                            ensure_ascii=False, indent=2))
                print(f"  [{i+1:3d}/{len(traj_files)}] {tp.stem} ERROR {type(e).__name__}: {e}")

    print(f"\nGrand total: {grand}  wall={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
