"""Phase 8 — compute all metrics from task.md §3.5.

Inputs:
  data/annotations/judge/<model>/<task_id>.json  — judge verdicts
  data/annotations/validation/validation_labels.jsonl (optional) — human labels

Outputs:
  results/tables/accuracy_vs_tcr.md           — Table 2 (per-model accuracy vs TCR)
  results/tables/failure_distribution.md      — Per-model per-category rate
  results/tables/agreement.md                 — κ agreement (if validation present)
  results/figures/failure_distribution.pdf    — Figure 2 stacked-bar
  results/summary.json                        — machine-readable everything
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.analysis.trajectory_status import classify_trajectory, TerminalState  # noqa: E402

VERDICT_ROOT = ROOT / "data" / "annotations" / "judge"
TRAJ_ROOT = ROOT / "data" / "trajectories"
SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
VALIDATION_PATH = ROOT / "data" / "annotations" / "validation" / "validation_labels.jsonl"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
SUMMARY_PATH = ROOT / "results" / "summary.json"

FAILURE_KEYS = [
    "modality_shortcut",
    "phantom_grounding",
    "wrong_evidence_right_answer",
    "over_retrieval_laundering",
    "cross_modal_contradiction",
    "provenance_hallucination",
]

SHORT_LABEL = {
    "modality_shortcut": "MOD-SC",
    "phantom_grounding": "PHT-GR",
    "wrong_evidence_right_answer": "WE-RA",
    "over_retrieval_laundering": "OR-LD",
    "cross_modal_contradiction": "CM-CT",
    "provenance_hallucination": "PRV-HL",
}


def load_verdicts() -> dict[str, list[dict]]:
    """model -> list of verdict dicts."""
    out: dict[str, list[dict]] = {}
    if not VERDICT_ROOT.exists():
        return out
    for mdir in sorted(p for p in VERDICT_ROOT.iterdir() if p.is_dir()):
        verdicts = []
        for vp in sorted(mdir.glob("mmsp_*.json")):
            if vp.name.endswith(".error.json"):
                continue
            try:
                verdicts.append(json.loads(vp.read_text()))
            except Exception:
                pass
        out[mdir.name] = verdicts
    return out


def _is_correct(v: dict) -> bool:
    return str(v.get("answer_correct", "")).lower() == "true"


def _has_any_failure(v: dict) -> bool:
    for k in FAILURE_KEYS:
        f = v.get("failures", {}).get(k, {})
        if isinstance(f, dict) and f.get("present"):
            return True
    return False


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a Bernoulli proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_accuracy_and_tcr(verdicts: list[dict]) -> dict:
    n = len(verdicts)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "tcr": 0.0,
                "accuracy_ci": (0.0, 0.0), "tcr_ci": (0.0, 0.0),
                "sfr_among_correct": 0.0}
    correct = sum(1 for v in verdicts if _is_correct(v))
    silent_among_correct = sum(1 for v in verdicts if _is_correct(v) and _has_any_failure(v))
    tcr_count = correct - silent_among_correct
    return {
        "n": n,
        "correct": correct,
        "tcr_count": tcr_count,
        "accuracy": correct / n,
        "tcr": tcr_count / n,
        "accuracy_ci": _wilson_ci(correct, n),
        "tcr_ci": _wilson_ci(tcr_count, n),
        "silent_among_correct": silent_among_correct,
        "sfr_among_correct": silent_among_correct / correct if correct else 0.0,
    }


def compute_category_rates(verdicts: list[dict]) -> dict[str, float]:
    if not verdicts:
        return {k: 0.0 for k in FAILURE_KEYS}
    rates = {}
    for k in FAILURE_KEYS:
        cnt = sum(1 for v in verdicts
                  if isinstance(v.get("failures", {}).get(k), dict)
                  and v["failures"][k].get("present"))
        rates[k] = cnt / len(verdicts)
    return rates


def _load_trajectories_with_state() -> dict[str, dict[str, dict]]:
    """model -> {task_id -> {trajectory dict, terminal_state, difficulty, category}}."""
    # First collect difficulty/category from sample.
    meta = {}
    if SAMPLE_PATH.exists():
        for line in SAMPLE_PATH.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "task_id" in r:
                meta[r["task_id"]] = {
                    "difficulty": r.get("difficulty"),
                    "category": r.get("category"),
                }
    out: dict[str, dict[str, dict]] = {}
    if not TRAJ_ROOT.exists():
        return out
    for mdir in sorted(p for p in TRAJ_ROOT.iterdir() if p.is_dir()):
        out[mdir.name] = {}
        for fp in mdir.glob("mmsp_*.json"):
            if fp.name.endswith(".error.json"):
                continue
            try:
                t = json.loads(fp.read_text())
            except Exception:
                continue
            tid = t.get("task_id") or fp.stem
            out[mdir.name][tid] = {
                "trajectory": t,
                "state": classify_trajectory(t).value,
                "difficulty": meta.get(tid, {}).get("difficulty"),
                "category": meta.get(tid, {}).get("category"),
            }
    return out


def _terminal_state_breakdown(model_traj: dict[str, dict]) -> dict[str, int]:
    states = Counter(t["state"] for t in model_traj.values())
    return {s.value: states.get(s.value, 0) for s in TerminalState}


def compute_committed_only(verdicts: list[dict], model_traj: dict[str, dict]) -> dict:
    """Subset metrics to trajectories whose terminal state is `committed`."""
    sub = [v for v in verdicts
           if model_traj.get(v["task_id"], {}).get("state") == TerminalState.COMMITTED.value]
    return compute_accuracy_and_tcr(sub)


def compute_by_difficulty(verdicts: list[dict], model_traj: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for diff in ("easy", "difficult"):
        sub = [v for v in verdicts
               if model_traj.get(v["task_id"], {}).get("difficulty") == diff]
        out[diff] = compute_accuracy_and_tcr(sub)
    return out


def render_table_accuracy(per_model: dict) -> str:
    lines = [
        "| Model | N | Accuracy [95% CI] | TCR [95% CI] | SFR\\|correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, s in per_model.items():
        ac = s.get("accuracy_ci") or (s["accuracy"], s["accuracy"])
        tc = s.get("tcr_ci") or (s["tcr"], s["tcr"])
        lines.append(
            f"| {m} | {s['n']} | "
            f"{s['accuracy']*100:.1f} [{ac[0]*100:.1f}, {ac[1]*100:.1f}] | "
            f"{s['tcr']*100:.1f} [{tc[0]*100:.1f}, {tc[1]*100:.1f}] | "
            f"{s['sfr_among_correct']:.3f} |"
        )
    return "\n".join(lines)


def render_table_state_breakdown(state_breakdown: dict[str, dict[str, int]]) -> str:
    states = [s.value for s in TerminalState]
    header = ["Model"] + states
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for m, br in state_breakdown.items():
        n = sum(br.values())
        row = [m] + [f"{br.get(s, 0)} ({br.get(s, 0)/n*100:.0f}%)" if n else "0"
                     for s in states]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_table_committed(per_model: dict, committed: dict) -> str:
    lines = [
        "| Model | Raw N | Acc (raw) | Committed N | Acc (committed) [95% CI] | TCR (committed) [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, s in per_model.items():
        c = committed[m]
        ac = c.get("accuracy_ci") or (c["accuracy"], c["accuracy"])
        tc = c.get("tcr_ci") or (c["tcr"], c["tcr"])
        lines.append(
            f"| {m} | {s['n']} | {s['accuracy']*100:.1f}% | {c['n']} | "
            f"{c['accuracy']*100:.1f} [{ac[0]*100:.1f}, {ac[1]*100:.1f}] | "
            f"{c['tcr']*100:.1f} [{tc[0]*100:.1f}, {tc[1]*100:.1f}] |"
        )
    return "\n".join(lines)


def render_table_difficulty(by_diff: dict[str, dict[str, dict]]) -> str:
    lines = [
        "| Model | Easy N | Easy Acc | Easy TCR | Diff N | Diff Acc | Diff TCR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, dd in by_diff.items():
        e, d = dd["easy"], dd["difficult"]
        lines.append(
            f"| {m} | {e['n']} | {e['accuracy']*100:.1f}% | {e['tcr']*100:.1f}% | "
            f"{d['n']} | {d['accuracy']*100:.1f}% | {d['tcr']*100:.1f}% |"
        )
    return "\n".join(lines)


def render_table_failures(per_model: dict[str, dict]) -> str:
    header = ["Model"] + [SHORT_LABEL[k] for k in FAILURE_KEYS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for m, rates in per_model.items():
        row = [m] + [f"{rates[k]*100:.1f}%" for k in FAILURE_KEYS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_failure_distribution(per_model_rates: dict[str, dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    models = list(per_model_rates.keys())
    cats = FAILURE_KEYS
    data = np.array([[per_model_rates[m][k] * 100 for k in cats] for m in models])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, k in enumerate(cats):
        ax.bar(x, data[:, i], bottom=bottom, label=SHORT_LABEL[k], color=colors[i])
        bottom += data[:, i]
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Silent-failure rate (%)")
    ax.set_title("Silent-failure category distribution per model")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_kappa(validation: list[dict]) -> dict | None:
    """validation entries have {task_id, model, human_answer_correct, human_failures, judge_*}.

    Returns kappa + per-category confusion counts + base rates. The fields named
    `human_*` may be filled by a second LLM judge (cross-validation) when
    human labels aren't feasible.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return None
    if not validation:
        return None
    kappas: dict[str, float] = {}
    confusions: dict[str, dict] = {}
    labels = ["True", "False", "Partial"]
    h_ans = [v.get("human_answer_correct") for v in validation]
    j_ans = [v.get("judge_answer_correct") for v in validation]
    try:
        kappas["answer_correct"] = cohen_kappa_score(h_ans, j_ans, labels=labels)
    except Exception:
        pass
    confusions["answer_correct"] = {
        "agree": sum(1 for a, b in zip(h_ans, j_ans) if a == b),
        "disagree": sum(1 for a, b in zip(h_ans, j_ans) if a != b),
    }
    for k in FAILURE_KEYS:
        h = [bool(v.get("human_failures", {}).get(k, {}).get("present")) for v in validation]
        j = [bool(v.get("judge_failures", {}).get(k, {}).get("present")) for v in validation]
        try:
            kappas[k] = cohen_kappa_score(h, j)
        except Exception:
            pass
        tt = sum(1 for a, b in zip(h, j) if a and b)
        ff = sum(1 for a, b in zip(h, j) if not a and not b)
        tf = sum(1 for a, b in zip(h, j) if a and not b)
        ft = sum(1 for a, b in zip(h, j) if not a and b)
        confusions[k] = {"tt": tt, "tf": tf, "ft": ft, "ff": ff,
                         "cross_pos": sum(h), "primary_pos": sum(j)}
    primary_judge = next((v.get("judge_model") for v in validation if v.get("judge_model")), None)
    cross_judge = next((v.get("human_judge_model") for v in validation if v.get("human_judge_model")), None)
    return {
        "n": len(validation),
        "kappa": kappas,
        "confusion": confusions,
        "primary_judge": primary_judge,
        "cross_judge": cross_judge,
    }


_KAPPA_INTERPRETATION = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.0, "slight"),
    (-1.0, "below chance"),
]


def _interpret_kappa(k: float) -> str:
    for thresh, label in _KAPPA_INTERPRETATION:
        if k >= thresh:
            return label
    return "below chance"


def render_agreement_md(kappa: dict) -> str:
    cross = kappa.get("cross_judge") or "cross"
    primary = kappa.get("primary_judge") or "primary"
    is_cross = "claude-opus-4-7" in (cross or "") or cross != primary
    title = "Cross-judge agreement" if is_cross else "Judge–human agreement"
    lines = [f"# {title} (n={kappa['n']})", ""]
    lines.append(f"- Primary judge: `{primary}`")
    lines.append(f"- {'Cross-validator' if is_cross else 'Human annotator'}: `{cross}`")
    lines.append("")
    lines.append("## Cohen's κ")
    lines.append("")
    lines.append("| Metric | κ | Interpretation | Cross+ | Primary+ | Both+ | Disagree |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    confusion = kappa.get("confusion", {})
    lines.append(
        f"| answer_correct | {kappa['kappa'].get('answer_correct', float('nan')):.3f} | "
        f"{_interpret_kappa(kappa['kappa'].get('answer_correct', 0))} | "
        f"— | — | "
        f"{confusion.get('answer_correct', {}).get('agree', 0)} | "
        f"{confusion.get('answer_correct', {}).get('disagree', 0)} |"
    )
    for k in FAILURE_KEYS:
        c = confusion.get(k, {})
        kv = kappa['kappa'].get(k, float('nan'))
        lines.append(
            f"| {k} | {kv:.3f} | {_interpret_kappa(kv)} | "
            f"{c.get('cross_pos', 0)} | {c.get('primary_pos', 0)} | "
            f"{c.get('tt', 0)} | {c.get('tf', 0) + c.get('ft', 0)} |"
        )
    lines.append("")
    lines.append("Cross+/Primary+ are positive-flag counts; Both+ is agreements on positive; "
                 "Disagree counts mismatched positives in either direction.")
    return "\n".join(lines)


def main():
    verdicts_by_model = load_verdicts()
    if not verdicts_by_model:
        print(f"No verdicts under {VERDICT_ROOT}; run Phase 5 first.")
        return

    traj_by_model = _load_trajectories_with_state()

    per_model = {m: compute_accuracy_and_tcr(vs) for m, vs in verdicts_by_model.items()}
    per_model_rates = {m: compute_category_rates(vs) for m, vs in verdicts_by_model.items()}
    # Failure-category rates restricted to trajectories that committed to a final
    # answer. This is fairer across models with very different infra-failure rates.
    per_model_rates_committed = {}
    for m, vs in verdicts_by_model.items():
        traj = traj_by_model.get(m, {})
        committed_v = [v for v in vs
                       if traj.get(v["task_id"], {}).get("state") == TerminalState.COMMITTED.value]
        per_model_rates_committed[m] = compute_category_rates(committed_v)
    state_breakdown = {m: _terminal_state_breakdown(traj_by_model.get(m, {}))
                       for m in verdicts_by_model}
    committed = {m: compute_committed_only(vs, traj_by_model.get(m, {}))
                 for m, vs in verdicts_by_model.items()}
    by_difficulty = {m: compute_by_difficulty(vs, traj_by_model.get(m, {}))
                     for m, vs in verdicts_by_model.items()}

    # Counts of dropped (sanity-error) verdicts.
    drop_counts: dict[str, int] = {}
    if VERDICT_ROOT.exists():
        for mdir in sorted(p for p in VERDICT_ROOT.iterdir() if p.is_dir()):
            n_drop = 0
            for vp in mdir.glob("mmsp_*.json"):
                if vp.name.endswith(".error.json"):
                    continue
                try:
                    v = json.loads(vp.read_text())
                except Exception:
                    n_drop += 1
                    continue
                if v.get("sanity_errors"):
                    n_drop += 1
            drop_counts[mdir.name] = n_drop

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / "accuracy_vs_tcr.md").write_text(render_table_accuracy(per_model))
    (TAB_DIR / "failure_distribution.md").write_text(render_table_failures(per_model_rates))
    (TAB_DIR / "failure_distribution_committed.md").write_text(
        render_table_failures(per_model_rates_committed))
    (TAB_DIR / "state_breakdown.md").write_text(render_table_state_breakdown(state_breakdown))
    (TAB_DIR / "committed_only.md").write_text(render_table_committed(per_model, committed))
    (TAB_DIR / "by_difficulty.md").write_text(render_table_difficulty(by_difficulty))
    plot_failure_distribution(per_model_rates_committed,
                              FIG_DIR / "failure_distribution.pdf")

    kappa = None
    if VALIDATION_PATH.exists():
        validation = [json.loads(l) for l in VALIDATION_PATH.read_text().splitlines() if l.strip()]
        kappa = compute_kappa(validation)
        if kappa:
            (TAB_DIR / "agreement.md").write_text(render_agreement_md(kappa))

    summary = {
        "per_model": per_model,
        "per_model_rates": per_model_rates,
        "per_model_rates_committed": per_model_rates_committed,
        "state_breakdown": state_breakdown,
        "committed_only": committed,
        "by_difficulty": by_difficulty,
        "verdict_drop_counts": drop_counts,
        "kappa": kappa,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("Accuracy vs TCR (raw N):\n" + render_table_accuracy(per_model) + "\n")
    print("Trajectory state breakdown:\n" + render_table_state_breakdown(state_breakdown) + "\n")
    print("Committed-only:\n" + render_table_committed(per_model, committed) + "\n")
    print("By difficulty:\n" + render_table_difficulty(by_difficulty) + "\n")
    print("Failure distribution:\n" + render_table_failures(per_model_rates) + "\n")
    if kappa:
        print(f"Kappa (n={kappa['n']}): {kappa['kappa']}")
    print(f"\nWrote {TAB_DIR}, {FIG_DIR}, {SUMMARY_PATH}.")


if __name__ == "__main__":
    main()
