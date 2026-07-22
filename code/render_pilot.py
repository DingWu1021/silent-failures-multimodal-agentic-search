"""Phase 4 — render a stratified pilot of trajectories as markdown for human labeling.

Selects N trajectories (default 30) across all agent models present under
data/trajectories/, balanced per-model. Output goes to data/annotations/pilot/.

Each rendered file is ready for the user to fill in a small YAML header with
labels; `consolidate_pilot.py` (to be written in Phase 5) will parse them back.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "mmsearch_plus_sample.jsonl"
TRAJ_ROOT = ROOT / "data" / "trajectories"
PILOT_DIR = ROOT / "data" / "annotations" / "pilot"

LABEL_HEADER = """---
# HUMAN LABELING — fill in:
#   answer_correct: True | False | Partial
#   Each failure: present true/false + one-sentence justification.
task_id: {task_id}
agent_model: {model}
answer_correct: ""
failures:
  modality_shortcut:
    present:
    justification: ""
  phantom_grounding:
    present:
    justification: ""
  wrong_evidence_right_answer:
    present:
    justification: ""
  over_retrieval_laundering:
    present:
    justification: ""
  cross_modal_contradiction:
    present:
    justification: ""
  provenance_hallucination:
    present:
    justification: ""
notes: ""
---
"""


def load_tasks() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in SAMPLE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "task_id" in r:
            out[r["task_id"]] = r
    return out


def render_trajectory(task: dict, traj: dict) -> str:
    imgs = task.get("image_paths") or []
    img_md = "\n".join(f"![{ip}](../../../{ip})" for ip in imgs)

    out = [
        LABEL_HEADER.format(task_id=task["task_id"], model=traj.get("model", "?")),
        "",
        f"# {task['task_id']} ({traj.get('model')})",
        "",
        f"**Category**: {task.get('category')} / {task.get('difficulty')} / {task.get('subtask')}",
        "",
        f"**Question**: {task['question']}",
        "",
        f"**Ground-truth answer**: {task.get('answer')}",
        "",
        f"**Agent final answer**: {traj.get('final_answer', '')}",
        "",
        "## Images",
        img_md,
        "",
        f"## Trajectory ({len(traj.get('steps', []))} steps, stopped: {traj.get('stopped_reason')})",
        "",
    ]

    for s in traj.get("steps", []):
        idx = s.get("idx")
        thought = (s.get("thought") or "").strip()
        tool = s.get("tool")
        tinput = s.get("tool_input")
        obs = (s.get("observation") or "").strip()
        final = s.get("final_answer")
        err = s.get("error")
        out.append(f"### Step {idx}")
        if thought:
            out.append(f"**Thought**: {thought}")
        if tool:
            out.append(f"**Tool**: `{tool}`  **Input**: `{json.dumps(tinput, ensure_ascii=False)}`")
            if len(obs) > 1200:
                obs = obs[:1200] + " …_[truncated]_"
            out.append(f"**Observation**:\n\n```\n{obs}\n```")
        if final:
            out.append(f"**FINAL_ANSWER**: {final}")
        if err:
            out.append(f"_(step error: {err})_")
        out.append("")

    token = traj.get("token_cost", {})
    out.append(f"_tokens: in={token.get('input_tokens')} out={token.get('output_tokens')} wall={traj.get('wall_time_s')}s_")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260418)
    args = ap.parse_args()

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()

    candidates: list[tuple[str, Path]] = []
    for mdir in sorted(p for p in TRAJ_ROOT.iterdir() if p.is_dir()):
        for tp in sorted(mdir.glob("mmsp_*.json")):
            if tp.name.endswith(".error.json"):
                continue
            candidates.append((mdir.name, tp))
    print(f"Found {len(candidates)} trajectories across {len({c[0] for c in candidates})} models.")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    per_model: dict[str, list] = {}
    for m, p in candidates:
        per_model.setdefault(m, []).append(p)

    model_names = list(per_model.keys())
    if not model_names:
        print("No trajectories found. Run Phase 3 first.")
        return
    per_quota = args.n // len(model_names)
    picks: list[tuple[str, Path]] = []
    for m in model_names:
        picks.extend([(m, p) for p in per_model[m][:per_quota]])
    # top up with remainders
    leftover = [(m, p) for m in model_names for p in per_model[m][per_quota:]]
    rng.shuffle(leftover)
    picks.extend(leftover[: args.n - len(picks)])

    for model, tp in picks:
        traj = json.loads(tp.read_text())
        task = tasks.get(traj["task_id"])
        if task is None:
            continue
        md = render_trajectory(task, traj)
        out = PILOT_DIR / f"{traj['task_id']}__{model}.md"
        out.write_text(md)
    print(f"Wrote {len(picks)} pilot files -> {PILOT_DIR}")


if __name__ == "__main__":
    main()
