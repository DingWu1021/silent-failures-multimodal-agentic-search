"""Phase 6 (substitute) — cross-validation by a second LLM judge.

We cannot do human spot-checks, so we use claude-opus-4-7 as an independent
second judge on a stratified 50-trajectory sample. Cohen's κ between the
primary judge (claude-opus-4-6) and the cross-validator (claude-opus-4-7)
serves as a proxy for judge–human agreement.

Outputs:
  data/annotations/validation/cross_judge_opus47/<agent_model>/<task_id>.json
  data/annotations/validation/validation_labels.jsonl
      (in the schema compute_metrics.compute_kappa expects, with the cross-
       validator's labels as `human_*` and the primary judge's as `judge_*`)
  data/annotations/validation/sample_manifest.json

Usage:
    python code/cross_validate.py                # sample + judge + emit jsonl
    python code/cross_validate.py --judge-only   # only re-emit the jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.agents.llm_client import get_client  # noqa: F401  (loads .env)
from code.judges.run_judge import judge_trajectory, save_verdict, FAILURE_KEYS

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
TRAJ_ROOT = ROOT / "data" / "trajectories"
PRIMARY_VERDICT_ROOT = ROOT / "data" / "annotations" / "judge"
VAL_ROOT = ROOT / "data" / "annotations" / "validation"

DEFAULT_CROSS_MODEL = "claude-opus-4-7"
SAMPLE_SIZE = 50
SAMPLE_SEED = 20260425


def _cross_dir(cross_model: str) -> Path:
    safe = cross_model.replace("/", "_").replace(".", "")
    return VAL_ROOT / f"cross_judge_{safe}"


def _manifest_path(cross_model: str) -> Path:
    return VAL_ROOT / f"sample_manifest_{cross_model.replace('/', '_').replace('.', '')}.json"


def _labels_path(cross_model: str) -> Path:
    return VAL_ROOT / f"validation_labels_{cross_model.replace('/', '_').replace('.', '')}.jsonl"


def _load_tasks_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in SAMPLE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "task_id" in r:
            out[r["task_id"]] = r
    return out


def _has_any_failure(verdict: dict) -> bool:
    for k in FAILURE_KEYS:
        f = verdict.get("failures", {}).get(k, {})
        if isinstance(f, dict) and f.get("present"):
            return True
    return False


def _collect_primary_verdicts() -> list[dict]:
    """Returns list of {task_id, agent_model, verdict_path, verdict, flagged, correct}."""
    items: list[dict] = []
    if not PRIMARY_VERDICT_ROOT.exists():
        return items
    for mdir in sorted(p for p in PRIMARY_VERDICT_ROOT.iterdir() if p.is_dir()):
        for vp in sorted(mdir.glob("mmsp_*.json")):
            if vp.name.endswith(".error.json"):
                continue
            try:
                v = json.loads(vp.read_text())
            except Exception:
                continue
            if v.get("sanity_errors"):
                continue
            items.append({
                "task_id": v["task_id"],
                "agent_model": mdir.name,
                "verdict_path": str(vp),
                "verdict": v,
                "flagged": _has_any_failure(v),
                "correct": str(v.get("answer_correct", "")).lower() == "true",
            })
    return items


def _stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by (agent_model, flagged), then by correctness within strata if possible."""
    rng = random.Random(seed)
    by_model: dict[str, list[dict]] = {}
    for it in items:
        by_model.setdefault(it["agent_model"], []).append(it)

    models = sorted(by_model.keys())
    base_per_model = n // len(models)
    extras = n - base_per_model * len(models)
    quotas = {m: base_per_model + (1 if i < extras else 0) for i, m in enumerate(models)}

    chosen: list[dict] = []
    for m, quota in quotas.items():
        pool = by_model[m]
        flagged = [x for x in pool if x["flagged"]]
        clean = [x for x in pool if not x["flagged"]]
        rng.shuffle(flagged)
        rng.shuffle(clean)
        # Aim for ~50/50 flagged/clean inside each model's quota,
        # falling back to whichever side has supply.
        want_flag = quota // 2
        want_clean = quota - want_flag
        take_flag = flagged[: min(want_flag, len(flagged))]
        take_clean = clean[: min(want_clean, len(clean))]
        deficit = quota - len(take_flag) - len(take_clean)
        if deficit > 0:
            leftover = flagged[len(take_flag):] + clean[len(take_clean):]
            rng.shuffle(leftover)
            take_flag.extend(leftover[:deficit])
        chosen.extend(take_flag + take_clean)
    rng.shuffle(chosen)
    return chosen


def _run_cross_judge(sample: list[dict], *, max_tokens: int, overwrite: bool,
                     cross_model: str) -> int:
    tasks = _load_tasks_by_id()
    cross_root = _cross_dir(cross_model)
    cross_root.mkdir(parents=True, exist_ok=True)
    done = skipped = errors = 0
    for i, it in enumerate(sample, 1):
        agent_model = it["agent_model"]
        task_id = it["task_id"]
        out_dir = cross_root / agent_model
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / f"{task_id}.json"
        if out_p.exists() and not overwrite:
            skipped += 1
            continue

        traj_path = TRAJ_ROOT / agent_model / f"{task_id}.json"
        if not traj_path.exists():
            print(f"  [{i:2d}] missing trajectory {traj_path}")
            errors += 1
            continue
        traj = json.loads(traj_path.read_text())
        task = tasks.get(task_id)
        if task is None:
            print(f"  [{i:2d}] {task_id}: no task record")
            errors += 1
            continue
        try:
            t0 = time.time()
            verdict = judge_trajectory(
                trajectory=traj,
                question=task["question"],
                ground_truth=", ".join(task.get("answer") or []),
                image_paths=task.get("image_paths") or [],
                judge_model=cross_model,
                max_tokens=max_tokens,
            )
            save_verdict(verdict, cross_root)
            dt = time.time() - t0
            flags = [k for k, v in (verdict.failures or {}).items()
                     if isinstance(v, dict) and v.get("present")]
            print(f"  [{i:2d}/{len(sample)}] {agent_model}/{task_id} {dt:.1f}s "
                  f"ans={verdict.answer_correct} flags={flags} "
                  f"sanity_errs={len(verdict.sanity_errors)}")
            done += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors += 1
            err_p = out_dir / f"{task_id}.error.json"
            err_p.write_text(json.dumps({
                "task_id": task_id, "agent_model": agent_model,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4),
            }, ensure_ascii=False, indent=2))
            print(f"  [{i:2d}] {agent_model}/{task_id} ERROR {type(e).__name__}: {e}")
    print(f"\nCross-judge: done={done} skipped={skipped} errors={errors}")
    return done + skipped


def _emit_validation_labels(sample: list[dict], cross_model: str) -> int:
    """Pair primary judge with cross judge into the schema that
    compute_metrics.compute_kappa expects.

    Cross-validator labels go into `human_*` fields, primary judge into `judge_*`.
    """
    cross_root = _cross_dir(cross_model)
    labels_path = _labels_path(cross_model)
    rows: list[dict] = []
    for it in sample:
        agent_model = it["agent_model"]
        task_id = it["task_id"]
        cross_p = cross_root / agent_model / f"{task_id}.json"
        if not cross_p.exists():
            continue
        try:
            cross_v = json.loads(cross_p.read_text())
        except Exception:
            continue
        if cross_v.get("sanity_errors"):
            continue
        primary_v = it["verdict"]
        rows.append({
            "task_id": task_id,
            "model": agent_model,
            "human_answer_correct": cross_v.get("answer_correct"),
            "human_failures": cross_v.get("failures", {}),
            "human_judge_model": cross_v.get("judge_model"),
            "judge_answer_correct": primary_v.get("answer_correct"),
            "judge_failures": primary_v.get("failures", {}),
            "judge_model": primary_v.get("judge_model"),
        })
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Also expose the *default* cross-validator as the legacy
    # validation_labels.jsonl so compute_metrics.py needs no flags.
    if cross_model == DEFAULT_CROSS_MODEL:
        legacy = VAL_ROOT / "validation_labels.jsonl"
        with legacy.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} paired labels → {labels_path}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cross-model", default=DEFAULT_CROSS_MODEL,
                    help="judge model to act as the cross-validator (e.g. claude-opus-4-7, gpt-4o-2024-11-20)")
    ap.add_argument("--labels-only", action="store_true",
                    help="skip judging, just rebuild validation labels jsonl from existing files")
    args = ap.parse_args()

    VAL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(args.cross_model)

    if args.labels_only and manifest_path.exists():
        sample = json.loads(manifest_path.read_text())["sample"]
        # rehydrate primary verdict text
        for it in sample:
            it["verdict"] = json.loads(Path(it["verdict_path"]).read_text())
    else:
        items = _collect_primary_verdicts()
        if not items:
            print("No primary verdicts found.")
            return
        print(f"Pool: {len(items)} primary verdicts")
        sample = _stratified_sample(items, args.n, args.seed)
        # persist a slim manifest (without bulky verdict body) for resumability
        manifest_sample = [
            {k: v for k, v in it.items() if k != "verdict"} for it in sample
        ]
        manifest_path.write_text(json.dumps({
            "n": len(sample),
            "seed": args.seed,
            "cross_model": args.cross_model,
            "primary_judge": "claude-opus-4-6",
            "sample": manifest_sample,
        }, ensure_ascii=False, indent=2))
        print(f"Wrote sample manifest → {manifest_path}")

    if not args.labels_only:
        _run_cross_judge(sample, max_tokens=args.max_tokens, overwrite=args.overwrite,
                         cross_model=args.cross_model)
    _emit_validation_labels(sample, args.cross_model)


if __name__ == "__main__":
    main()
