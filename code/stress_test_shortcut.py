"""Phase 7 (optional) — modality-shortcut stress test.

For N tasks where the judge labeled "no failure AND answer correct", re-run the
agent with each input image replaced by a **blank white image of the same
dimensions**. Measure Δaccuracy.

If the model still answers correctly at a similar rate, it implies the model
was not actually relying on the image — a latent shortcut the original run did
not surface as a silent failure.

Outputs:
  data/trajectories_blank/<model>/<task_id>.json
  results/tables/modality_shortcut.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.agents.react_agent import run_agent, save_trajectory

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
JUDGE_ROOT = ROOT / "data" / "annotations" / "judge"
BLANK_TRAJ_ROOT = ROOT / "data" / "trajectories_blank"
BLANK_IMG_ROOT = ROOT / "data" / "raw" / "images_blank"
TAB_DIR = ROOT / "results" / "tables"

FAILURE_KEYS = [
    "modality_shortcut", "phantom_grounding", "wrong_evidence_right_answer",
    "over_retrieval_laundering", "cross_modal_contradiction", "provenance_hallucination",
]


def load_tasks_by_id() -> dict[str, dict]:
    out = {}
    for line in SAMPLE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "task_id" in r:
            out[r["task_id"]] = r
    return out


def make_blank_for(image_path: Path) -> Path:
    img = Image.open(image_path)
    blank = Image.new("RGB", img.size, "white")
    out = BLANK_IMG_ROOT / image_path.parent.name / image_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    blank.save(out)
    return out


def pick_candidates(model: str, n: int, seed: int, *, require_no_failure: bool = False) -> list[str]:
    """Pick task_ids the model previously got correct (optionally also flag-clean)."""
    verdicts = list((JUDGE_ROOT / model).glob("mmsp_*.json"))
    rng = random.Random(seed)
    rng.shuffle(verdicts)
    picks = []
    for vp in verdicts:
        if vp.name.endswith(".error.json"):
            continue
        v = json.loads(vp.read_text())
        if str(v.get("answer_correct", "")).lower() != "true":
            continue
        if require_no_failure:
            any_fail = any(
                isinstance(v.get("failures", {}).get(k, {}), dict)
                and v["failures"][k].get("present")
                for k in FAILURE_KEYS
            )
            if any_fail:
                continue
        picks.append(v["task_id"])
        if len(picks) >= n:
            break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260419)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--require-no-failure", action="store_true",
                    help="restrict to tasks judged 'no failure' (strict, often very small pool)")
    args = ap.parse_args()

    tasks = load_tasks_by_id()
    picks = pick_candidates(args.model, args.n, args.seed,
                            require_no_failure=args.require_no_failure)
    print(f"Picked {len(picks)} no-failure + answer-correct tasks for model {args.model}.")

    out_dir = BLANK_TRAJ_ROOT / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = 0
    for i, tid in enumerate(picks):
        if (out_dir / f"{tid}.json").exists():
            continue
        task = tasks[tid]
        blank_paths = [str(make_blank_for(ROOT / p).relative_to(ROOT))
                       for p in task.get("image_paths", [])]
        t0 = time.time()
        traj = run_agent(
            task_id=tid,
            model=args.model,
            question=task["question"],
            image_paths=blank_paths,
            max_steps=args.max_steps,
            max_tokens_per_call=args.max_tokens,
        )
        save_trajectory(traj, BLANK_TRAJ_ROOT)
        started += 1
        print(f"  [{i+1:3d}/{len(picks)}] {tid} {time.time()-t0:.1f}s stop={traj.stopped_reason} "
              f"ans={(traj.final_answer or '')[:60]!r}")

    print(f"\nRe-ran {started} tasks with blank images. Trajectories at {out_dir}.")


if __name__ == "__main__":
    main()
