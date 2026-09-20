"""Measure warm local decision latency; does not call an external model API."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from qev.inference import Agent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--backend", default="auto")
parser.add_argument("--device")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("repeats must be positive")
hardware = {"processor": platform.processor()}
if platform.system() == "Darwin":
    hardware = {
        "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
    }
versions = {"python": platform.python_version()}
for package in ("mlx", "mlx-vlm", "torch", "transformers"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
started = time.perf_counter()
agent = Agent(args.checkpoint, backend=args.backend, device=args.device)
load_seconds = time.perf_counter() - started
state = "A customer reports a duplicate credit card charge and requests a refund."
question = {"type": "choice", "instructions": "Select the appropriate department.",
            "criteria": {"billing": "charges and refunds", "technical": "software problems", "sales": "new orders"}}
results = []
for count in (1, 4, 16):
    questions = {f"route_{i}": question for i in range(count)}
    agent.predict(state, questions)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        response = agent.predict(state, questions)
        samples.append((time.perf_counter() - started) * 1000)
    results.append({"questions": count, "milliseconds": samples,
                    "median_ms": statistics.median(samples), "max_ms": max(samples),
                    "questions_per_second": count * 1000 / statistics.median(samples),
                    "input_tokens": response["usage"]["input_tokens"]})
report = {"checkpoint": str(Path(args.checkpoint).resolve()), "backend": agent.backend,
          "machine": platform.machine(), "platform": platform.platform(),
          "hardware": hardware, "packages": versions, "dtype": agent.config.get("mlx_dtype"),
          "load_seconds": load_seconds, "repeats": args.repeats, "results": results,
          "scope": "Warm sequential wall-clock latency for a short fixed request; independent question rows, no shared-prefix cache. Not a general speed benchmark."}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
