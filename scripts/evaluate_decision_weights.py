r"""Compare MLX decision-weight modes on the development split and Snake latency.

Loads one MLX checkpoint and scores every development question with LoRA
computed separately (``adapter``, the validated path), then with merged float32
and merged bfloat16 decision weights. Reports accuracy, calibrated NLL,
agreement with ``adapter``, Snake-request latency and MLX memory::

    uv run python scripts/evaluate_decision_weights.py --model models/qev-snake-0.8b-mlx \
        --data data/snake-v1 --output runs/decision-weights.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qev.data import load_split, materialize
from qev.evaluation import metrics
from qev.mlx_runtime import MLXRuntime
from qev.tokenization import encode_question

SNAKE = "synthetic_snake_observable_teacher"


def encode(runtime, data, limit):
    items = []
    for record in load_split(data, "development"):
        rec = materialize(record)
        for question in rec["questions"]:
            item = encode_question(rec["state"], question, runtime.tokenizer,
                                   runtime.config.get("max_length", 1024), runtime.config.get("max_state", 384))
            item.update(label=int(question["label"]), source=question.get("src", "unknown"))
            items.append(item)
    return items[:limit] if limit else items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/qev-snake-0.8b-mlx")
    ap.add_argument("--data", default="data/snake-v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--latency-requests", type=int, default=40)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    import mlx.core as mx

    runtime = MLXRuntime.from_checkpoint(args.model)
    temperature = float(runtime.config.get("temperature", 1.0))
    items = encode(runtime, args.data, args.limit)
    snake = [item for item in items if item["source"] == SNAKE][:args.latency_requests]
    result = {"model": str(Path(args.model).resolve()), "questions": len(items), "temperature": temperature,
              "mlx": mx.__version__, "modes": {}}
    reference = None
    for mode in MLXRuntime.DECISION_WEIGHTS:
        runtime.merge_decision_weights(mode)
        started = time.perf_counter()
        logits = []
        for index, item in enumerate(items, 1):
            logits.append(runtime.predict_logits([item])[0])
            if index % 300 == 0:
                print(f"{mode}: {index}/{len(items)}", flush=True)
        seconds = time.perf_counter() - started
        rows = [{"logits": row.tolist(), "label": item["label"]} for row, item in zip(logits, items)]
        for item in snake[:5]:
            runtime.predict_logits([item])
        times = []
        for item in snake:
            t0 = time.perf_counter()
            runtime.predict_logits([item])
            times.append((time.perf_counter() - t0) * 1000)
        entry = {"development": metrics(rows, temperature),
                 "snake": metrics([r for r, i in zip(rows, items) if i["source"] == SNAKE], temperature),
                 "general": metrics([r for r, i in zip(rows, items) if i["source"] != SNAKE], temperature),
                 "seconds": seconds, "snake_request_ms_p50": statistics.median(times),
                 "active_memory_gib": mx.get_active_memory() / 2**30, "peak_memory_gib": mx.get_peak_memory() / 2**30}
        if reference is None:
            reference = logits
        else:
            def probs(z):
                z = np.asarray(z, dtype=np.float64) / temperature
                e = np.exp(z - z.max())
                return e / e.sum()
            entry["argmax_changes_vs_adapter"] = sum(int(np.argmax(a) != np.argmax(b)) for a, b in zip(reference, logits))
            entry["max_probability_difference_vs_adapter"] = max(
                float(np.abs(probs(a) - probs(b)).max()) for a, b in zip(reference, logits))
        result["modes"][mode] = entry
        print(mode, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()
                                if not isinstance(v, dict)} | {"accuracy": round(entry["development"]["accuracy"], 4)}),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
