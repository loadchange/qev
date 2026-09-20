"""Evaluate the shipped MLX checkpoint on every development question."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qev.data import load_split
from qev.evaluation import encode_records, probabilities, report, write_json
from qev.mlx_runtime import MLXRuntime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data", default="data/v1")
parser.add_argument("--output", required=True)
parser.add_argument("--torch-rows", help="Optional saved CUDA development logits, with matching record identities")
args = parser.parse_args()
runtime = MLXRuntime.from_checkpoint(args.checkpoint)
config = runtime.config
temperature = float(config.get("temperature", 1.0))
items = encode_records(load_split(args.data, "development"), runtime.tokenizer,
                       config.get("max_length", 1024), config.get("max_state", 384))
rows = []
started = time.perf_counter()
for index, item in enumerate(items):
    logits = runtime.predict_logits([item])[0]
    if not np.isfinite(logits).all():
        raise FloatingPointError(f"Non-finite MLX logits at development question {index}")
    row = {key: item[key] for key in ("label", "source", "language", "qtype", "record_id", "question_id", "state_truncated")}
    row.update(index=index, logits=logits.tolist(), precision=f"mlx:{config['mlx_dtype']}:fp32_pointer")
    rows.append(row)
    if (index + 1) % 100 == 0:
        print(json.dumps({"questions": index + 1, "total": len(items), "seconds": time.perf_counter() - started}), flush=True)
result = {"development": report(rows, temperature), "seconds": time.perf_counter() - started,
          "backend": "mlx_vlm", "mlx_dtype": config["mlx_dtype"],
          "temperature_fit": "Original CUDA calibration partition; not refitted on development or MLX predictions",
          "checkpoint": str(Path(args.checkpoint).resolve())}
if args.torch_rows:
    reference = json.loads(Path(args.torch_rows).read_text())
    if len(reference) != len(rows):
        raise ValueError("CUDA/MLX question counts differ")
    errors, flips = [], 0
    for expected, observed in zip(reference, rows, strict=True):
        for key in ("record_id", "question_id", "label", "source"):
            if expected[key] != observed[key]:
                raise ValueError(f"CUDA/MLX question identity mismatch: {key}")
        p = probabilities(expected["logits"], temperature)
        q = probabilities(observed["logits"], temperature)
        errors.append(float(np.max(np.abs(p - q))))
        flips += int(p.argmax() != q.argmax())
    result["cuda_bf16_comparison"] = {
        "questions": len(rows), "argmax_flips": flips, "max_probability_difference": max(errors),
        "median_probability_difference": float(np.median(errors)),
        "scope": "Different inference arithmetic (CUDA BF16 vs MLX FP32); diagnostic, not the CPU FP32 export-parity gate."}
output = Path(args.output)
write_json(output, result)
write_json(output.with_name(output.stem + "_rows.json"), rows)
print("QEV_MLX_EVALUATION_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)
