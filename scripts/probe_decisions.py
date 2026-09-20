"""Measure candidate-order stability and independent-question consistency."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from qev.data import load_split
from qev.inference import Agent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data", default="data/v1")
parser.add_argument("--backend", default="auto")
parser.add_argument("--device")
parser.add_argument("--per-source", type=int, default=4)
parser.add_argument("--output", required=True)
args = parser.parse_args()
agent = Agent(args.checkpoint, backend=args.backend, device=args.device)
selected, counts = [], Counter()
for record in load_split(args.data, "development"):
    source = record.get("_meta", {}).get("source", "unknown")
    for qid, question in record["questions"].items():
        if question["type"] == "choice" and counts[source] < args.per_source:
            selected.append((record, qid, question, source))
            counts[source] += 1
permutations = []
for record, qid, question, source in selected:
    # Data records may include labels/metadata beside the request. Only send
    # fields declared by the typed public API.
    question = {k: v for k, v in question.items() if k in {"type", "instructions", "criteria"}}
    reversed_question = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
    original = agent.predict(record["state"], {qid: question})["answers"][qid]
    reverse = agent.predict(record["state"], {qid: reversed_question})["answers"][qid]
    permutations.append({"source": source, "question_id": qid,
                         "options": len(question["criteria"]),
                         "original_choice": original["choice"], "reversed_choice": reverse["choice"],
                         "same_choice": original["choice"] == reverse["choice"],
                         "max_probability_difference": max(abs(original["probabilities"][key] - reverse["probabilities"][key])
                                                           for key in question["criteria"])})
state = "A customer was charged twice and requests a refund of the extra payment."
q = {"type": "choice", "instructions": "Select the appropriate department.",
     "criteria": {"billing": "Payments and refunds", "technical": "Software problems", "sales": "New purchases"}}
before = agent.predict(state, {"route": q})["answers"]["route"]["probabilities"]
combined = agent.predict(state, {"first": q, "second": {"type": "noul", "instructions": "Is a refund requested?"},
                                "last": q})["answers"]
after = agent.predict(state, {"route": q})["answers"]["route"]["probabilities"]
errors = [max(abs(before[k] - values[k]) for k in before) for values in
          (combined["first"]["probabilities"], combined["last"]["probabilities"], after)]
report = {"backend": agent.backend, "checkpoint": str(Path(args.checkpoint).resolve()),
          "permutation": {"questions": len(permutations), "sources": dict(counts),
                          "same_choice_fraction": float(np.mean([p["same_choice"] for p in permutations])) if permutations else None,
                          "rows": permutations},
          "independent_questions": {"maximum_probability_difference": max(errors), "passed": max(errors) <= 0.002},
          "scope": "Deterministic development probes. Candidate reversal is measured, not promised invariant. These are not new held-out accuracy or Jev comparisons."}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False))
if not report["independent_questions"]["passed"]:
    raise SystemExit("Independent-question probability check failed")
