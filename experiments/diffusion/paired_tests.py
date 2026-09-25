"""Exact McNemar tests between arms on identical development questions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

SNAKE = "synthetic_snake_observable_teacher"
PAIRS = [("a2d-block", "qwen3-causal"), ("qwen3-block", "qwen3-causal"), ("a2d-block", "qwen3-block"),
         ("lfm2-causal", "qwen35-causal"), ("lfm2-block", "lfm2-causal"), ("qwen3-causal", "qwen35-causal")]


def exact_mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    rows = {}
    for path in Path(args.runs).glob("*/development_rows.json"):
        data = json.loads(path.read_text())
        rows[path.parent.name] = {r["index"]: (int(np.argmax(r["logits"]) == r["label"]), r["source"],
                                               r["record_id"], r["question_id"]) for r in data}
    reference = next(iter(rows.values()))
    for arm, values in rows.items():
        if any(values[k][2:] != reference[k][2:] for k in reference):
            raise ValueError(f"{arm} rows do not align with the other arms")
    result = []
    for a, b in PAIRS:
        for subset, keep in (("general", lambda s: s != SNAKE), ("snake", lambda s: s == SNAKE)):
            keys = [k for k in reference if keep(reference[k][1])]
            x = np.array([rows[a][k][0] for k in keys])
            y = np.array([rows[b][k][0] for k in keys])
            only_a, only_b = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
            result.append({"a": a, "b": b, "subset": subset, "n": len(keys), "accuracy_a": float(x.mean()),
                           "accuracy_b": float(y.mean()), "only_a_correct": only_a, "only_b_correct": only_b,
                           "exact_mcnemar_p": exact_mcnemar(only_a, only_b)})
            print(f"{subset:7s} {a} vs {b}: {100 * x.mean():.1f}% vs {100 * y.mean():.1f}% "
                  f"(p={result[-1]['exact_mcnemar_p']:.3g})")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
