"""Classify each trajectory's terminal state into one of:

  committed  — final_answer present, content looks like an actual answer
  refused    — final_answer present, content is a refusal / "I don't know"
  exhausted  — stopped_reason == max_steps and no final_answer
  crashed    — stopped_reason == llm_error
  parse_err  — stopped_reason == max_steps because every step failed parsing

This separation lets us report accuracy on the *committed* subset (the only
subset where the model actually attempted an answer), so that infra
failures and refusals don't mask model capability.

Usage as a library:
    from code.analysis.trajectory_status import classify_trajectory, TerminalState
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path


class TerminalState(str, Enum):
    COMMITTED = "committed"
    REFUSED = "refused"
    EXHAUSTED = "exhausted"
    CRASHED = "crashed"
    PARSE_ERR = "parse_err"


# Curated refusal-language patterns. Conservative: we want false-negatives
# (counting a refusal as committed) to be rare; false-positives (counting a
# committed answer as refusal) to be even rarer because that would shrink
# the accuracy denominator artificially.
_REFUSAL_PATTERNS = [
    # generic
    re.compile(r"\bi (?:cannot|can't|am unable to|am not able to)\b", re.I),
    re.compile(r"\b(?:i (?:do not|don't) (?:know|have))\b", re.I),
    re.compile(r"\bi (?:do not|don't) have (?:enough|sufficient) information\b", re.I),
    re.compile(r"\bi'?m sorry,?\b.*\b(?:cannot|can't|unable|not able)\b", re.I),
    re.compile(r"\bunable to (?:determine|identify|answer|provide)\b", re.I),
    # blank-image acknowledgements (caused by the stress test rerun)
    re.compile(r"\b(?:image|images|picture)s?\s+(?:is|are)\s+(?:blank|completely blank|white|empty)\b", re.I),
    re.compile(r"\b(?:blank|empty|white)\s+(?:image|images|picture)s?\b", re.I),
    # tool-failure surrender
    re.compile(r"\bsearch (?:returned|gave) no useful (?:results|information)\b", re.I),
    re.compile(r"\bI (?:do not|don't) have access\b", re.I),
    # explicit refusal
    re.compile(r"\b(?:cannot|can't) (?:provide|determine|identify|verify) (?:a|the|an) answer\b", re.I),
]


def _looks_like_refusal(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    if len(s) <= 2:
        return False  # too short to confidently call a refusal
    for p in _REFUSAL_PATTERNS:
        if p.search(s):
            return True
    return False


def classify_trajectory(traj: dict) -> TerminalState:
    """Return the canonical terminal state of a trajectory dict."""
    stopped = traj.get("stopped_reason")
    final = (traj.get("final_answer") or "").strip()
    if stopped == "llm_error":
        return TerminalState.CRASHED
    if stopped == "max_steps":
        # If steps all fall back to parse errors, treat as parse_err.
        if traj.get("steps") and all(
            (s.get("error") and "parse" in s["error"].lower()) for s in traj["steps"]
        ):
            return TerminalState.PARSE_ERR
        return TerminalState.EXHAUSTED
    if stopped == "final_answer":
        if not final or _looks_like_refusal(final):
            return TerminalState.REFUSED
        return TerminalState.COMMITTED
    # Unknown / legacy stops: treat empty as exhausted, otherwise committed.
    if not final:
        return TerminalState.EXHAUSTED
    return TerminalState.REFUSED if _looks_like_refusal(final) else TerminalState.COMMITTED


def main():
    """CLI: print state distribution per agent model."""
    from collections import Counter
    root = Path(__file__).resolve().parents[2] / "data" / "trajectories"
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        states = Counter()
        for fp in mdir.glob("mmsp_*.json"):
            if fp.name.endswith(".error.json"):
                continue
            t = json.loads(fp.read_text())
            states[classify_trajectory(t)] += 1
        n = sum(states.values())
        print(f"=== {mdir.name} (n={n}) ===")
        for s in TerminalState:
            c = states.get(s, 0)
            print(f"  {s.value:11s} {c:4d}  ({c/n*100:5.1f}%)" if n else f"  {s.value:11s} 0")


if __name__ == "__main__":
    main()
