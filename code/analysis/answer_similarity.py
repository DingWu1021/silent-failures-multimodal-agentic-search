"""Compute TF-IDF cosine similarity between each (original, blank) answer pair.

Binary survival rate (Figure 3) is a coarse signal. Cosine similarity gives
a denser distribution: are the blank answers totally unrelated to the
original (cos ≈ 0), partially related (cos ≈ 0.3), or near-duplicates
(cos ≈ 0.9 — would imply a modality shortcut)?

Outputs:
  results/tables/answer_similarity.md
  results/figures/answer_similarity.pdf
  results/answer_similarity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
ORIG_TRAJ = ROOT / "data" / "trajectories"
BLANK_TRAJ = ROOT / "data" / "trajectories_blank"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
OUT_JSON = ROOT / "results" / "answer_similarity.json"


def _final(traj_path: Path) -> str:
    if not traj_path.exists():
        return ""
    try:
        return (json.loads(traj_path.read_text()).get("final_answer") or "").strip()
    except Exception:
        return ""


def main():
    rows: list[dict] = []
    for mdir in sorted(p for p in BLANK_TRAJ.iterdir() if p.is_dir()):
        for fp in sorted(mdir.glob("mmsp_*.json")):
            if fp.name.endswith(".error.json"):
                continue
            tid = fp.stem
            blank = _final(fp)
            orig = _final(ORIG_TRAJ / mdir.name / fp.name)
            if not orig or not blank:
                continue
            rows.append({"model": mdir.name, "task_id": tid,
                         "orig": orig, "blank": blank})
    if not rows:
        print("No paired trajectories found.")
        return

    corpus = []
    for r in rows:
        corpus.extend([r["orig"], r["blank"]])
    vec = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2))
    X = vec.fit_transform(corpus).toarray()
    sims = []
    for i, r in enumerate(rows):
        a = X[2 * i]
        b = X[2 * i + 1]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        sim = float(a @ b / denom) if denom else 0.0
        r["cosine"] = sim
        sims.append((r["model"], sim))

    OUT_JSON.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))

    # Per-model summary
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[float]] = {}
    for m, s in sims:
        by_model.setdefault(m, []).append(s)
    lines = ["| Model | n | mean cos | median | p90 |",
             "|---|---:|---:|---:|---:|"]
    for m, vs in sorted(by_model.items()):
        arr = np.array(vs)
        lines.append(f"| {m} | {len(arr)} | {arr.mean():.3f} | "
                     f"{np.median(arr):.3f} | {np.quantile(arr, 0.9):.3f} |")
    table = "\n".join(lines)
    (TAB_DIR / "answer_similarity.md").write_text(table)
    print(table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    colors = {"claude-sonnet-4-6": "#4a6fa5", "gemini-2.5-pro": "#cf8a3d",
              "gpt-4o-2024-11-20": "#4a8a5e"}
    for i, (m, vs) in enumerate(sorted(by_model.items())):
        ax.scatter([i] * len(vs), vs, color=colors.get(m, f"C{i}"),
                   alpha=0.7, s=28, edgecolor="#1a1a1a", linewidth=0.4)
        ax.scatter([i], [np.mean(vs)], color="black", marker="_", s=120, linewidth=2)
    ax.set_xticks(range(len(by_model)))
    ax.set_xticklabels(sorted(by_model.keys()), rotation=15, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("TF-IDF cosine(orig, blank)")
    ax.set_title("Answer similarity under blank-image substitution")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "answer_similarity.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
