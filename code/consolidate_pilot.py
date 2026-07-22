"""Phase 4 consolidate — parse YAML front matter from pilot markdown files
and emit data/annotations/pilot_labels.jsonl.

Also parses data/annotations/validation/*.md the same way for Phase 6 validation
labels (writes validation_labels.jsonl alongside the judge verdicts for κ compute).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data" / "annotations" / "pilot"
VALIDATION_DIR = ROOT / "data" / "annotations" / "validation"
JUDGE_ROOT = ROOT / "data" / "annotations" / "judge"


FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_front_matter(text: str) -> dict | None:
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except Exception as e:
        print(f"YAML parse error: {e}")
        return None
    return data if isinstance(data, dict) else None


def _norm_present(v) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return None


def _norm_answer(v) -> str | None:
    if not v:
        return None
    s = str(v).strip().lower()
    if s == "true":
        return "True"
    if s == "false":
        return "False"
    if s in ("partial", "part"):
        return "Partial"
    return None


def consolidate(source_dir: Path, out_path: Path, *, kind: str):
    records = []
    skipped = 0
    for p in sorted(source_dir.glob("*.md")):
        text = p.read_text()
        fm = parse_front_matter(text)
        if fm is None:
            skipped += 1
            continue
        ans = _norm_answer(fm.get("answer_correct"))
        if ans is None:
            skipped += 1
            continue
        rec = {
            "task_id": fm.get("task_id"),
            "agent_model": fm.get("agent_model"),
            "human_answer_correct": ans,
            "human_failures": {},
            "notes": fm.get("notes", ""),
        }
        failures = fm.get("failures", {}) or {}
        for k, v in failures.items():
            if isinstance(v, dict):
                rec["human_failures"][k] = {
                    "present": _norm_present(v.get("present")),
                    "justification": v.get("justification", ""),
                }
        records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{kind}] wrote {len(records)} labels -> {out_path} (skipped unlabeled: {skipped})")

    if kind == "validation" and records:
        merged_path = out_path.parent / "validation_labels.jsonl"
        merged = []
        for rec in records:
            agent_m = rec["agent_model"]
            tid = rec["task_id"]
            verdict_p = JUDGE_ROOT / agent_m / f"{tid}.json"
            if not verdict_p.exists():
                continue
            verdict = json.loads(verdict_p.read_text())
            merged.append({
                **rec,
                "judge_answer_correct": verdict.get("answer_correct"),
                "judge_failures": verdict.get("failures", {}),
            })
        with merged_path.open("w") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{kind}] merged w/ judge verdicts: {len(merged)} -> {merged_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("pilot", "validation"), default="pilot")
    args = ap.parse_args()
    if args.kind == "pilot":
        consolidate(PILOT_DIR, PILOT_DIR.parent / "pilot_labels.jsonl", kind="pilot")
    else:
        consolidate(VALIDATION_DIR, VALIDATION_DIR / "validation_raw.jsonl", kind="validation")


if __name__ == "__main__":
    main()
