"""Lightweight structural detectors for silent-failure categories.

These rules use *only* the trajectory + final answer (no LLM call). The
goal is a per-category precision/recall versus the LLM-judge gold labels.
A high-recall cheap detector reduces the LLM-judge cost in practice.

Detectors implemented:

- PRV-HL  : URL membership. Final answer cites URL X but X never appears
            in any observation.
- MOD-SC  : No image-conditioned tool call. Trajectory contains no
            reverse_image_search / image_search / crop call AND no thought
            string mentions image-specific descriptors.
- OR-LD   : Three-or-more web_search calls AND ≤1 fetch+useful observation
            AND final answer is long (>40 words = "narrative").
- PHT-GR  : Final answer cites a URL/DOI string that *does* appear in
            observations, but the cited assertion does not appear in any
            observation snippet (BM25-like keyword overlap < 0.1).
- CM-CT   : Skipped - structurally hard to detect without semantic reasoning.
- WE-RA   : Skipped - extremely rare (<1%) and structurally invisible.

Outputs:
  results/tables/cheap_detector.md
  results/cheap_detector.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAJ_ROOT = ROOT / "data" / "trajectories"
JUDGE_ROOT = ROOT / "data" / "annotations" / "judge"
TAB_DIR = ROOT / "results" / "tables"
OUT_JSON = ROOT / "results" / "cheap_detector.json"

# Image-conditioned tools (anything that operates on the input image directly).
IMAGE_TOOLS = {"reverse_image_search", "image_search", "crop"}

URL_RE = re.compile(r"https?://[^\s)\"'\]]+")


def _trajectory_text(traj: dict) -> tuple[str, str, str]:
    """Return concatenated thoughts, observations, and the final answer."""
    thoughts = []
    obs = []
    for s in traj.get("steps") or []:
        if s.get("thought"):
            thoughts.append(s["thought"])
        if s.get("observation"):
            obs.append(s["observation"])
    return "\n".join(thoughts), "\n".join(obs), (traj.get("final_answer") or "").strip()


def detect_prv_hl(traj: dict) -> bool:
    """Final answer cites a URL not present in any observation."""
    _thoughts, observations, final = _trajectory_text(traj)
    final_urls = URL_RE.findall(final)
    if not final_urls:
        return False
    obs_blob = observations.lower()
    for u in final_urls:
        # Tolerate trailing punctuation, trailing slash differences.
        u_norm = u.rstrip(".,);").lower()
        # Strip protocol for membership check (some obs strip http).
        host_path = re.sub(r"^https?://", "", u_norm)
        if host_path not in obs_blob and u_norm not in obs_blob:
            return True
    return False


def detect_mod_sc(traj: dict) -> bool:
    """No image-conditioned tool was used in the entire trajectory."""
    for s in traj.get("steps") or []:
        if s.get("tool") in IMAGE_TOOLS:
            return False
    # Even if no image tool, an attentive trajectory may quote image content
    # in thoughts; if it does, do not flag (heuristic).
    thoughts = " ".join(s.get("thought") or "" for s in traj.get("steps") or [])
    image_descriptor_hits = sum(1 for kw in (
        "image shows", "image depicts", "in the image", "on the image",
        "from the image", "the image is", "image contains"
    ) if kw in thoughts.lower())
    return image_descriptor_hits == 0


def detect_or_ld(traj: dict) -> bool:
    """Many redundant web_search calls + sparse useful observations + long
    narrative final answer."""
    n_web_search = 0
    n_useful_obs = 0
    for s in traj.get("steps") or []:
        if s.get("tool") == "web_search":
            n_web_search += 1
        obs_raw = s.get("observation_raw") or {}
        if isinstance(obs_raw, dict) and not obs_raw.get("is_stub", False):
            n_useful_obs += 1
    final = (traj.get("final_answer") or "").strip()
    final_words = len(final.split())
    return n_web_search >= 3 and n_useful_obs <= 1 and final_words > 40


def detect_pht_gr(traj: dict) -> bool:
    """A URL in the final answer DOES appear in observations, but the
    sentence around the citation has near-zero keyword overlap with any
    observation snippet that mentions the same URL.

    Cheap proxy: if final answer contains a numeric or proper-noun token
    that is absent from observations, flag it.
    """
    _thoughts, observations, final = _trajectory_text(traj)
    obs_blob = observations.lower()
    # If the final answer makes a quantitative or named-entity claim
    # whose token never appears in observations, flag.
    candidates = re.findall(r"\b\d{4}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", final)
    if not candidates:
        return False
    seen_in_obs = sum(1 for c in candidates if c.lower() in obs_blob)
    # If less than half of named/numeric claims appear, treat as phantom.
    return seen_in_obs < len(candidates) / 2


DETECTORS = {
    "provenance_hallucination": detect_prv_hl,
    "modality_shortcut": detect_mod_sc,
    "over_retrieval_laundering": detect_or_ld,
    "phantom_grounding": detect_pht_gr,
}


def _judge_label(verdict: dict, key: str) -> bool:
    f = (verdict.get("failures") or {}).get(key) or {}
    return bool(f.get("present"))


def main():
    rows = []
    counts: dict[str, dict[str, int]] = {k: Counter() for k in DETECTORS}
    n_total = 0
    for mdir in sorted(p for p in TRAJ_ROOT.iterdir() if p.is_dir()):
        for fp in mdir.glob("mmsp_*.json"):
            if fp.name.endswith(".error.json"):
                continue
            t = json.loads(fp.read_text())
            vp = JUDGE_ROOT / mdir.name / fp.name
            if not vp.exists():
                continue
            try:
                v = json.loads(vp.read_text())
            except Exception:
                continue
            if v.get("sanity_errors"):
                continue
            n_total += 1
            row = {"model": mdir.name, "task_id": t.get("task_id")}
            for cat, fn in DETECTORS.items():
                rule = bool(fn(t))
                gold = _judge_label(v, cat)
                row[f"{cat}_rule"] = rule
                row[f"{cat}_gold"] = gold
                if rule and gold:
                    counts[cat]["tp"] += 1
                elif rule and not gold:
                    counts[cat]["fp"] += 1
                elif (not rule) and gold:
                    counts[cat]["fn"] += 1
                else:
                    counts[cat]["tn"] += 1
            rows.append(row)

    print(f"n verdicts: {n_total}")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"counts": {k: dict(v) for k, v in counts.items()},
                                     "rows": rows},
                                    ensure_ascii=False, indent=2))

    lines = [
        "| Category | TP | FP | FN | TN | Precision | Recall | F1 | Gold rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, c in counts.items():
        tp = c.get("tp", 0); fp = c.get("fp", 0)
        fn = c.get("fn", 0); tn = c.get("tn", 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        gold = (tp + fn) / max(1, tp + fp + fn + tn)
        lines.append(
            f"| {cat} | {tp} | {fp} | {fn} | {tn} | "
            f"{prec:.3f} | {rec:.3f} | {f1:.3f} | {gold*100:.1f}% |"
        )
    table = "\n".join(lines)
    (TAB_DIR / "cheap_detector.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
