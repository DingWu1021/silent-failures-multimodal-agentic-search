"""Modality-shortcut analysis (Phase 7 post-processing).

Inputs:
  data/trajectories_blank/<model>/<task_id>.json    (rerun w/ blank images)
  data/annotations/judge/<model>/<task_id>.json     (original verdicts)

Process:
  For every blank trajectory, run the primary judge (claude-opus-4-6)
  against the same task ground-truth and save the verdict to
  data/annotations/judge_blank/<model>/<task_id>.json. Resumable: skips
  already-judged trajectories.

Outputs:
  data/annotations/judge_blank/<model>/<task_id>.json
  results/tables/modality_shortcut.md
  results/figures/modality_shortcut.pdf
  results/shortcut_summary.json

Interpretation:
  Survival rate = fraction of tasks the agent answered correctly with the
  real image that it still answers correctly with a blank image. A high
  survival rate is direct evidence that the agent never needed the image,
  i.e. a modality shortcut at the benchmark or model level.

Usage:
  python code/analysis/analyze_shortcut.py
  python code/analysis/analyze_shortcut.py --judge-only      # skip rejudging
  python code/analysis/analyze_shortcut.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.agents.llm_client import get_client  # noqa: F401
from code.judges.run_judge import judge_trajectory, save_verdict

SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
BLANK_TRAJ_ROOT = ROOT / "data" / "trajectories_blank"
ORIG_JUDGE_ROOT = ROOT / "data" / "annotations" / "judge"
BLANK_JUDGE_ROOT = ROOT / "data" / "annotations" / "judge_blank"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
SUMMARY_PATH = ROOT / "results" / "shortcut_summary.json"

JUDGE_MODEL = "claude-opus-4-6"


def _load_tasks_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in SAMPLE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "task_id" in r:
            out[r["task_id"]] = r
    return out


def _is_correct(v: dict | None) -> bool:
    if not v:
        return False
    return str(v.get("answer_correct", "")).lower() == "true"


def rejudge_blanks(*, max_tokens: int = 1500) -> None:
    if not BLANK_TRAJ_ROOT.exists():
        print(f"No blank trajectories at {BLANK_TRAJ_ROOT}; run stress_test_shortcut.py first.")
        return
    tasks = _load_tasks_by_id()
    BLANK_JUDGE_ROOT.mkdir(parents=True, exist_ok=True)
    for mdir in sorted(p for p in BLANK_TRAJ_ROOT.iterdir() if p.is_dir()):
        out_dir = BLANK_JUDGE_ROOT / mdir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_files = sorted(mdir.glob("mmsp_*.json"))
        traj_files = [p for p in traj_files if not p.name.endswith(".error.json")]
        print(f"\n=== Re-judging {mdir.name}: {len(traj_files)} blank trajectories ===")
        for i, tp in enumerate(traj_files, 1):
            out_p = out_dir / tp.name
            if out_p.exists():
                continue
            traj = json.loads(tp.read_text())
            task = tasks.get(traj["task_id"])
            if task is None:
                print(f"  [{i}] no task record for {traj['task_id']}")
                continue
            # The blank trajectory's image_paths point to data/raw/images_blank/...
            # but for judging, we want the *real* image so the judge can spot
            # answer correctness; image content does not matter for the judge's
            # answer-correctness check (it only needs ground truth + trajectory),
            # so we pass the original image paths.
            try:
                t0 = time.time()
                verdict = judge_trajectory(
                    trajectory=traj,
                    question=task["question"],
                    ground_truth=", ".join(task.get("answer") or []),
                    image_paths=task.get("image_paths") or [],
                    judge_model=JUDGE_MODEL,
                    max_tokens=max_tokens,
                )
                save_verdict(verdict, BLANK_JUDGE_ROOT)
                dt = time.time() - t0
                print(f"  [{i:2d}/{len(traj_files)}] {traj['task_id']} {dt:.1f}s "
                      f"ans={verdict.answer_correct}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  [{i:2d}] {traj['task_id']} ERROR {type(e).__name__}: {e}")


def compute_summary() -> dict:
    summary: dict[str, dict] = {}
    for mdir in sorted(p for p in BLANK_JUDGE_ROOT.iterdir() if p.is_dir()):
        rows = []
        for vp in sorted(mdir.glob("mmsp_*.json")):
            if vp.name.endswith(".error.json"):
                continue
            blank_v = json.loads(vp.read_text())
            tid = blank_v["task_id"]
            orig_p = ORIG_JUDGE_ROOT / mdir.name / f"{tid}.json"
            if not orig_p.exists():
                continue
            orig_v = json.loads(orig_p.read_text())
            rows.append({
                "task_id": tid,
                "orig_correct": _is_correct(orig_v),
                "blank_correct": _is_correct(blank_v),
            })
        n = len(rows)
        n_orig_correct = sum(1 for r in rows if r["orig_correct"])
        n_blank_correct = sum(1 for r in rows if r["blank_correct"])
        survived = sum(1 for r in rows if r["orig_correct"] and r["blank_correct"])
        summary[mdir.name] = {
            "n": n,
            "orig_correct": n_orig_correct,
            "blank_correct": n_blank_correct,
            "survived_correct": survived,
            "survival_rate": survived / n_orig_correct if n_orig_correct else None,
            "delta_accuracy": (n_blank_correct - n_orig_correct) / n if n else None,
            "rows": rows,
        }
    return summary


def render_table(summary: dict) -> str:
    lines = [
        "| Model | N | Orig correct | Blank correct | Survived | Survival rate | Δ accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, s in summary.items():
        sr = f"{s['survival_rate']:.2f}" if s["survival_rate"] is not None else "—"
        da = f"{s['delta_accuracy']:+.2f}" if s["delta_accuracy"] is not None else "—"
        lines.append(
            f"| {m} | {s['n']} | {s['orig_correct']} | {s['blank_correct']} | "
            f"{s['survived_correct']} | {sr} | {da} |"
        )
    lines.append("")
    lines.append("Survival rate = fraction of originally-correct tasks the agent still "
                 "answered correctly when the input image was replaced by a same-size "
                 "blank white image. High survival rate is evidence of a modality shortcut.")
    return "\n".join(lines)


def plot_survival(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    models = list(summary.keys())
    if not models:
        return
    sr = [summary[m]["survival_rate"] or 0.0 for m in models]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(models))
    bars = ax.bar(x, [v * 100 for v in sr], color="#cf8a3d", width=0.55,
                  edgecolor="#1a1a1a", linewidth=0.6)
    for i, m in enumerate(models):
        label = f"{summary[m]['survived_correct']}/{summary[m]['orig_correct']}"
        ax.text(i, sr[i] * 100 + 1.5, label, ha="center", fontsize=8)
    ax.set_ylim(0, max(105, max([v * 100 for v in sr] + [0]) + 10))
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Survival rate under blank image (%)")
    ax.set_title("Modality-shortcut stress test")
    ax.axhline(0, color="grey", lw=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-only", action="store_true",
                    help="skip rejudging, just regenerate the table from existing blank verdicts")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not args.judge_only:
        rejudge_blanks(max_tokens=args.max_tokens)

    if not BLANK_JUDGE_ROOT.exists():
        print("No blank verdicts; nothing to summarize.")
        return

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    summary = compute_summary()
    table = render_table(summary)
    (TAB_DIR / "modality_shortcut.md").write_text(table)
    print("\n" + table)

    if not args.no_plot:
        plot_survival(summary, FIG_DIR / "modality_shortcut.pdf")

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {TAB_DIR / 'modality_shortcut.md'}, "
          f"{FIG_DIR / 'modality_shortcut.pdf'}, {SUMMARY_PATH}.")


if __name__ == "__main__":
    main()
