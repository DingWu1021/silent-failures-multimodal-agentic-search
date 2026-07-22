"""Figure 1 — diagnostic pipeline diagram.

Boxes for: (Task + Image) → Unified ReAct agent → Trajectory → Primary judge
+ Cross-validator → Verdict & κ. Saves results/figures/pipeline.pdf.

Usage:
    python code/analysis/make_pipeline_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures" / "pipeline.pdf"


def _box(ax, x, y, w, h, text, *, fc="#eef4fb", ec="#4a6fa5", lw=1.2, fontsize=8.5):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=lw, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1a1a1a")


def _arrow(ax, x1, y1, x2, y2, *, color="#4a6fa5"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                        shrinkA=2, shrinkB=2,
                        mutation_scale=12),
    )


def main():
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    # Inputs
    _box(ax, 0.2, 3.1, 2.3, 1.2,
         "Task\n(question + image)",
         fc="#fff5e6", ec="#cf8a3d")

    # Agent box
    _box(ax, 3.1, 2.6, 2.7, 2.0,
         "Unified ReAct agent\n"
         r"$\{$web\_search, image\_search,$\}$" + "\n" +
         r"$\{$web\_fetch, crop$\}$" + "\n"
         "(identical scaffold across\nall agent models)",
         fc="#eef4fb", ec="#4a6fa5", fontsize=7.4)

    # Trajectory
    _box(ax, 6.4, 3.1, 2.4, 1.2,
         "Trajectory JSON\n(thoughts, tools, obs)",
         fc="#eef4fb", ec="#4a6fa5")

    # Primary judge
    _box(ax, 9.3, 3.7, 2.5, 0.95,
         "Primary judge\n(Claude Opus 4.6)",
         fc="#e8f4ec", ec="#4a8a5e", fontsize=8.5)

    # Cross validator
    _box(ax, 9.3, 2.3, 2.5, 0.95,
         "Cross-validator\n(Claude Opus 4.7)",
         fc="#fbeef4", ec="#a54a78", fontsize=8.5)

    # Output
    _box(ax, 12.2, 2.95, 1.7, 1.4,
         "Verdict\n+ Cohen's $\\kappa$",
         fc="#f3eefb", ec="#6f4aa5", fontsize=8.5)

    # Rubric (as a separate annotation feeding into both judges)
    _box(ax, 9.55, 0.55, 2.0, 1.0,
         "Rubric (6 categories)\n+ JSON schema",
         fc="#f7f7f7", ec="#777777", fontsize=7.6)

    # Arrows
    _arrow(ax, 2.5, 3.7, 3.1, 3.6)                  # task -> agent
    _arrow(ax, 5.8, 3.6, 6.4, 3.7)                  # agent -> trajectory
    _arrow(ax, 8.8, 3.85, 9.3, 4.15)                # traj -> primary
    _arrow(ax, 8.8, 3.55, 9.3, 2.78)                # traj -> cross
    _arrow(ax, 11.8, 4.18, 12.2, 3.95)              # primary -> verdict
    _arrow(ax, 11.8, 2.78, 12.2, 3.35)              # cross -> verdict
    _arrow(ax, 10.55, 1.55, 10.55, 2.30, color="#777777")  # rubric -> cross
    _arrow(ax, 10.55, 1.55, 10.55, 3.70, color="#777777")  # rubric -> primary

    # Legend / footnote
    fig.text(0.01, 0.02,
             "Solid arrows: data flow.  Grey arrows: rubric is shared by both judges.",
             fontsize=6.8, color="#555555")

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
