# Silent Failure Diagnostic Rubric (v0 — pre-pilot)

> Version history: v0 drafted 2026-04-17 from `task.md` §3.1 & §5.1. To be refined after Phase 4 pilot labeling.

## Task

Given a multimodal search agent's trajectory, decide (1) whether the final answer is correct and (2) which silent-failure categories apply.

## Inputs

- **Question**: the original user question.
- **Image(s)**: the input image(s) attached to the question.
- **Ground-truth answer**: reference answer from the benchmark.
- **Trajectory**: list of `{thought, tool, tool_input, observation}` steps.
- **Final answer**: the agent's final output string.

## Step 1 — Answer correctness

Classify the final answer as `True` / `False` / `Partial`.

- `True`: semantically matches the ground-truth (exact match not required; paraphrase OK).
- `Partial`: answer has the right core entity but misses or misstates a key attribute (e.g., right person, wrong year).
- `False`: materially disagrees with the ground-truth, or empty / refused.

## Step 2 — Silent failure categories

For each of the six categories below, output `present: true/false` and a one-sentence justification grounded in specific step indices from the trajectory. Multiple categories may apply simultaneously.

### (a) MODALITY_SHORTCUT
The agent reached its answer **without** meaningful use of the image content.

**Evidence to look for:**
- No `image_search` or `crop` call was issued.
- The agent never described image-specific details (objects, text in image, spatial layout) that a pure text-only pipeline couldn't have produced.
- The agent's first queries contain only information from the question text, ignoring visual context that could narrow the search.

**Not modality_shortcut:** Agent chose text-search first but clearly inspected and referenced image details in its thought chain.

### (b) PHANTOM_GROUNDING
The agent cites a retrieved source that does **not** actually support the claimed fact, based on what the observations show.

**Evidence to look for:**
- The final answer cites URL X, but the step that fetched URL X returned content that does not contain the claim.
- The agent quotes a sentence that does not appear verbatim (or in close paraphrase) in any observation.

### (c) WRONG_EVIDENCE_RIGHT_ANSWER
Final answer is correct but the reasoning chain contains a factually wrong or logically invalid intermediate step that materially led to the answer.

**Evidence to look for:**
- A thought asserts "X is Y" where X is not Y, yet the answer is still correct by coincidence.
- Agent transposed numbers, misattributed a quote, or used a faulty inference that happened to land on the right entity.

### (d) OVER_RETRIEVAL_LAUNDERING
Agent issues many redundant queries that return little new information, and the final answer is effectively drawn from the model's prior (parametric) belief rather than from retrieval, but is presented as retrieval-grounded.

**Evidence to look for:**
- ≥3 search queries with heavily overlapping terms returning similar / useless snippets.
- Final answer includes specifics (dates, numbers, names) that do **not** appear in any observation.
- Agent's concluding thought starts with phrases like "based on my knowledge" or proceeds as if search had confirmed something that was not actually returned.

### (e) CROSS_MODAL_CONTRADICTION
Final answer contradicts what the image **clearly** depicts, while aligning with noisy text retrievals.

**Evidence to look for:**
- Image shows A; retrieval turned up mixed or noisy text mentioning B; agent answered B.
- Requires the image to be unambiguous to a human viewer on the disputed attribute.

### (f) PROVENANCE_HALLUCINATION
Agent invents URLs, dates, author names, or source names that are syntactically plausible but do not appear in any observation.

**Evidence to look for:**
- Final answer cites a URL that was never fetched or returned by search.
- A thought reports "as stated on nytimes.com/..." but no step retrieved that URL.
- A date or publication name is mentioned that cannot be traced to any observation.

## Output JSON schema

```json
{
  "answer_correct": "True" | "False" | "Partial",
  "failures": {
    "modality_shortcut":        {"present": true|false, "justification": "..."},
    "phantom_grounding":        {"present": true|false, "justification": "..."},
    "wrong_evidence_right_answer": {"present": true|false, "justification": "..."},
    "over_retrieval_laundering":{"present": true|false, "justification": "..."},
    "cross_modal_contradiction":{"present": true|false, "justification": "..."},
    "provenance_hallucination": {"present": true|false, "justification": "..."}
  },
  "overall_rationale": "3-5 sentence overall assessment grounded in step indices."
}
```

## Judging policy

- Cite **step indices** (e.g., "step 2 fetched URL X but answer cites URL Y") in justifications.
- When in doubt, default to `present: false`; false positives hurt precision more than false negatives at this stage.
- `wrong_evidence_right_answer` requires `answer_correct == "True"` as a precondition.
- `cross_modal_contradiction` requires the image evidence to be unambiguous; if the image is ambiguous, mark `false`.
