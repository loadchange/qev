"""Compare an exported LFM2 MLX model with its PyTorch development rows.

    uv run python scripts/validate_lfm2_mlx.py --model models/qev-450m-mlx \
        --rows runs/lfm2-v2/development_rows.json --output runs/lfm2-mlx-validation.json
"""
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
from qev.lfm2_runtime import LFM2Runtime
from qev.tokenization import encode_question

SNAKE = "synthetic_snake_observable_teacher"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True, help="development_rows.json from the PyTorch run")
    ap.add_argument("--data", default="data/snake-v1")
    ap.add_argument("--modes", nargs="+", default=["adapter", "bf16"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    reference = sorted(json.loads(Path(args.rows).read_text()), key=lambda row: row["index"])
    runtime = LFM2Runtime.from_checkpoint(args.model)
    temperature = float(runtime.config["temperature"])
    items = []
    for record in load_split(args.data, "development"):
        rec = materialize(record)
        for question in rec["questions"]:
            item = encode_question(rec["state"], question, runtime.tokenizer, runtime.config["max_length"],
                                   runtime.config["max_state"], family="lfm2")
            item.update(label=int(question["label"]), source=question.get("src"))
            items.append(item)
    assert len(items) == len(reference) and all(i["label"] == r["label"] for i, r in zip(items, reference))

    def probs(z):
        z = np.asarray(z, dtype=np.float64) / temperature
        e = np.exp(z - z.max())
        return e / e.sum()

    result = {"model": str(Path(args.model).resolve()), "questions": len(items), "torch": {
        "development": metrics(reference, temperature)}, "modes": {}}
    import mlx.core as mx

    for mode in args.modes:
        runtime.merge_decision_weights(mode)
        mx.reset_peak_memory()
        started = time.perf_counter()
        logits = [runtime.predict_logits([item])[0] for item in items]
        seconds = time.perf_counter() - started
        rows = [{"logits": z.tolist(), "label": i["label"]} for z, i in zip(logits, items)]
        snake = [i for i in items if i["source"] == SNAKE][:40]
        for item in snake[:5]:
            runtime.predict_logits([item])
        times = []
        for item in snake:
            t0 = time.perf_counter()
            runtime.predict_logits([item])
            times.append((time.perf_counter() - t0) * 1000)
        result["modes"][mode] = {
            "development": metrics(rows, temperature),
            "snake": metrics([r for r, i in zip(rows, items) if i["source"] == SNAKE], temperature),
            "general": metrics([r for r, i in zip(rows, items) if i["source"] != SNAKE], temperature),
            "argmax_changes_vs_torch": sum(int(np.argmax(z) != np.argmax(r["logits"])) for z, r in zip(logits, reference)),
            "max_probability_difference_vs_torch": max(float(np.abs(probs(z) - probs(r["logits"])).max())
                                                      for z, r in zip(logits, reference)),
            "by_source": {source: metrics([r for r, i in zip(rows, items) if i["source"] == source], temperature)
                          for source in sorted({i["source"] for i in items})},
            "seconds": seconds, "snake_request_ms_p50": statistics.median(times),
            "active_memory_gib": mx.get_active_memory() / 2**30, "peak_memory_gib": mx.get_peak_memory() / 2**30}
        print(mode, json.dumps({k: v for k, v in result["modes"][mode].items() if not isinstance(v, dict)}
                               | {"accuracy": result["modes"][mode]["development"]["accuracy"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
