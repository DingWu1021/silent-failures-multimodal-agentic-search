"""Correlate per-trajectory tool-call budget with OR-LD flag rate.

Hypothesis: Gemini's elevated OR-LD rate may simply reflect its longer
trajectories rather than a behavioural tendency. We test this by
binning trajectories on tool-call count and computing OR-LD rate per
bin, per model.

Outputs:
  results/tables/or_ld_vs_budget.md
  results/figures/or_ld_vs_budget.pdf
  results/or_ld_vs_budget.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAJ_ROOT = ROOT / "data" / "trajectories"
JUDGE_ROOT = ROOT / "data" / "annotations" / "judge"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
OUT_JSON = ROOT / "results" / "or_ld_vs_budget.json"

BUCKETS = [(1, 2), (3, 4), (5, 6), (7, 10)]  # tool-call counts


def _tool_calls(traj: dict) -> int:
    return sum(1 for s in traj.get("steps") or [] if s.get("tool"))


def _has_or_ld(verdict: dict) -> bool:
    f = (verdict.get("failures") or {}).get("over_retrieval_laundering") or {}
    return bool(f.get("present"))


def main():
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for mdir in sorted(p for p in TRAJ_ROOT.iterdir() if p.is_dir()):
        bucket_counts = {b: {"n": 0, "or_ld": 0} for b in BUCKETS}
        total_n = 0
        total_or_ld = 0
        for fp in mdir.glob("mmsp_*.json"):
            if fp.name.endswith(".error.json"):
                continue
            t = json.loads(fp.read_text())
            vp = JUDGE_ROOT / mdir.name / fp.name
            if not vp.exists():
                continue
            v = json.loads(vp.read_text())
            if v.get("sanity_errors"):
                continue
            n_tools = _tool_calls(t)
            ord_flag = _has_or_ld(v)
            total_n += 1
            total_or_ld += int(ord_flag)
            rows.append({"model": mdir.name, "task_id": t.get("task_id"),
                         "n_tools": n_tools, "or_ld": ord_flag})
            for lo, hi in BUCKETS:
                if lo <= n_tools <= hi:
                    bucket_counts[(lo, hi)]["n"] += 1
                    bucket_counts[(lo, hi)]["or_ld"] += int(ord_flag)
                    break
        summary[mdir.name] = {
            "total_n": total_n,
            "total_or_ld_rate": total_or_ld / total_n if total_n else 0.0,
            "buckets": {f"{lo}-{hi}":
                        {"n": v["n"],
                         "or_ld_rate": v["or_ld"] / v["n"] if v["n"] else None}
                        for (lo, hi), v in bucket_counts.items()},
        }

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows},
                                    ensure_ascii=False, indent=2))

    headers = ["Model", "Total"] + [f"{lo}-{hi} steps" for lo, hi in BUCKETS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for m, s in summary.items():
        row = [m, f"{s['total_or_ld_rate']*100:.1f}% (n={s['total_n']})"]
        for lo, hi in BUCKETS:
            b = s["buckets"][f"{lo}-{hi}"]
            row.append("—" if b["or_ld_rate"] is None
                       else f"{b['or_ld_rate']*100:.1f}% (n={b['n']})")
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    (TAB_DIR / "or_ld_vs_budget.md").write_text(table)
    print(table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    bucket_labels = [f"{lo}–{hi}" for lo, hi in BUCKETS]
    width = 0.27
    x = np.arange(len(bucket_labels))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    colors = {"claude-sonnet-4-6": "#4a6fa5", "gemini-2.5-pro": "#cf8a3d",
              "gpt-4o-2024-11-20": "#4a8a5e"}
    for i, (m, s) in enumerate(summary.items()):
        rates = [s["buckets"][f"{lo}-{hi}"]["or_ld_rate"] or 0.0 for lo, hi in BUCKETS]
        ax.bar(x + i * width - width, [r * 100 for r in rates], width,
               label=m, color=colors.get(m, f"C{i}"), edgecolor="#1a1a1a", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Tool calls in trajectory")
    ax.set_ylabel("Over-retrieval laundering rate (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "or_ld_vs_budget.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
