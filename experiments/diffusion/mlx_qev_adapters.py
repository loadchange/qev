"""Qev production MLX runtime with decision adapters on vs off (fresh process).

Adapters off is the cost floor of a merged-LoRA decision path, the fair
comparison for candidates timed with merged LoRA. Run alone: switching dtypes
inside a longer benchmark process on a 16 GB Mac inflated bf16 timings.
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx

from experiments.diffusion.mlx_latency import snake_requests, time_calls
from qev.data import materialize
from qev.inference import Agent
from qev.tokenization import encode_question

agent = Agent(sys.argv[1] if len(sys.argv) > 1 else "models/qev-snake-0.8b-mlx", backend="mlx")
encodings = []
for record in snake_requests(40):
    rec = materialize(record)
    encodings.append(encode_question(rec["state"], rec["questions"][0], agent.tokenizer, 1024, 384))
runtime = agent.model
result = {"adapters_on_fp32_ms": time_calls(lambda e: runtime.predict_logits([e]), encodings)}
original = runtime._decision_mode
runtime._decision_mode = types.MethodType(lambda self, enabled: original(False), runtime)
result["adapters_off_fp32_ms"] = time_calls(lambda e: runtime.predict_logits([e]), encodings)
runtime.backbone.set_dtype(mx.bfloat16)
result["adapters_off_bf16_ms"] = time_calls(lambda e: runtime.predict_logits([e]), encodings)
print(json.dumps({key: round(value["p50"], 1) for key, value in result.items()}))
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
