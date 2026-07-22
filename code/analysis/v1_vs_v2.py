"""Controlled comparison: v1 (no reverse_image_search) vs v2 (with).

Same 200 tasks, same 3 models, same primary judge (opus-4-6). The only
difference is the agent's tool surface. So Δ in silent-failure rates
quantifies the *causal* effect of providing a faithful image-grounding tool.

Inputs:
  data/annotations/judge_v1/<model>/*.json      (v1 verdicts, backup)
  data/annotations/judge/<model>/*.json         (v2 verdicts)

Output:
  results/tables/v1_vs_v2.md
  results/v1_vs_v2.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = ROOT / "data" / "annotations" / "judge_v1"
V2_ROOT = ROOT / "data" / "annotations" / "judge"
TRAJ_V1 = ROOT / "data" / "trajectories_v1_pre_lens"
TRAJ_V2 = ROOT / "data" / "trajectories"
TAB_DIR = ROOT / "results" / "tables"
OUT_JSON = ROOT / "results" / "v1_vs_v2.json"

FAILURE_KEYS = [
    "modality_shortcut", "phantom_grounding", "wrong_evidence_right_answer",
    "over_retrieval_laundering", "cross_modal_contradiction", "provenance_hallucination",
]
SHORT = {
    "modality_shortcut": "MOD-SC", "phantom_grounding": "PHT-GR",
    "wrong_evidence_right_answer": "WE-RA", "over_retrieval_laundering": "OR-LD",
    "cross_modal_contradiction": "CM-CT", "provenance_hallucination": "PRV-HL",
}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def _committed_task_ids(traj_root: Path, model: str) -> set[str]:
    out = set()
    mdir = traj_root / model
    if not mdir.exists():
        return out
    for fp in mdir.glob("mmsp_*.json"):
        if fp.name.endswith(".error.json"):
            continue
        try:
            t = json.loads(fp.read_text())
        except Exception:
            continue
        if t.get("stopped_reason") == "final_answer" and (t.get("final_answer") or "").strip():
            out.add(t["task_id"])
    return out


def _load_verdicts(root: Path, model: str) -> dict[str, dict]:
    out = {}
    mdir = root / model
    if not mdir.exists():
        return out
    for fp in mdir.glob("mmsp_*.json"):
        if fp.name.endswith(".error.json"):
            continue
        try:
            v = json.loads(fp.read_text())
        except Exception:
            continue
        if v.get("sanity_errors"):
            continue
        out[v.get("task_id") or fp.stem] = v
    return out


def _is_correct(v: dict | None) -> bool:
    return bool(v) and str(v.get("answer_correct", "")).lower() == "true"


def _has(v: dict | None, k: str) -> bool:
    if not v:
        return False
    f = (v.get("failures") or {}).get(k) or {}
    return bool(f.get("present"))


def _stats_per_committed(verdicts: dict, committed: set, key_fn) -> tuple[int, int]:
    """Return (k, n) counts on the committed subset for a given predicate."""
    n = len(committed)
    k = sum(1 for tid in committed if key_fn(verdicts.get(tid)))
    return k, n


def main():
    models = sorted({p.name for p in V1_ROOT.iterdir() if p.is_dir()} &
                    {p.name for p in V2_ROOT.iterdir() if p.is_dir()})
    summary = {}
    for m in models:
        v1 = _load_verdicts(V1_ROOT, m)
        v2 = _load_verdicts(V2_ROOT, m)
        c1 = _committed_task_ids(TRAJ_V1, m)
        c2 = _committed_task_ids(TRAJ_V2, m)
        # Restrict to committed and judge-present.
        c1 &= set(v1.keys())
        c2 &= set(v2.keys())
        n1, n2 = len(c1), len(c2)

        row = {"n_v1": n1, "n_v2": n2}
        for label, predicate in [
            ("accuracy", _is_correct),
            *[(SHORT[k], (lambda key: lambda v: _has(v, key))(k)) for k in FAILURE_KEYS],
        ]:
            k1, _ = _stats_per_committed(v1, c1, predicate)
            k2, _ = _stats_per_committed(v2, c2, predicate)
            r1 = k1 / n1 if n1 else 0.0
            r2 = k2 / n2 if n2 else 0.0
            ci1 = _wilson(k1, n1)
            ci2 = _wilson(k2, n2)
            row[label] = {"v1": {"k": k1, "n": n1, "rate": r1, "ci": ci1},
                          "v2": {"k": k2, "n": n2, "rate": r2, "ci": ci2},
                          "delta": r2 - r1}
        summary[m] = row

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    headers = ["Metric"] + [f"{m} v1" for m in models] + [f"{m} v2" for m in models] + [f"{m} Δ" for m in models]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    metric_keys = ["accuracy"] + [SHORT[k] for k in FAILURE_KEYS]
    for mk in metric_keys:
        cells = [mk]
        for m in models:
            d = summary[m][mk]
            cells.append(f"{d['v1']['rate']*100:.1f}% (n={d['v1']['n']})")
        for m in models:
            d = summary[m][mk]
            cells.append(f"{d['v2']['rate']*100:.1f}% (n={d['v2']['n']})")
        for m in models:
            d = summary[m][mk]
            cells.append(f"{d['delta']*100:+.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (TAB_DIR / "v1_vs_v2.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
