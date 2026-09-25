"""Apple Silicon (MLX) per-decision latency on real Snake development requests.

``qev``: the production Qev MLX runtime (Qwen3.5-0.8B, switchable LoRA,
float32), timed end to end through ``Agent.predict`` and as pure inference.
Candidates: the decision forward of each backbone (hidden states plus the
pointer head). LoRA is treated as merged, which leaves the compute unchanged,
so untrained weights give the same timing as trained ones.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diffusion import qevd
from qev.data import load_split, materialize

SNAKE_SOURCE = "synthetic_snake_observable_teacher"


def summary(values):
    ordered = sorted(values)
    return {"n": len(values), "p50": statistics.median(values),
            "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "mean": statistics.mean(values)}


def snake_requests(count):
    records = [r for r in load_split(ROOT / "data/snake-v1", "development")
               if next(iter(r["questions"].values())).get("src") == SNAKE_SOURCE]
    return records[:count]


def time_calls(fn, items, warmup=5):
    for item in items[:warmup]:
        fn(item)
    times = []
    for item in items:
        started = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - started) * 1000)
    return summary(times)


def qev_production(path, records):
    from qev.inference import Agent

    agent = Agent(path, backend="mlx")
    requests = []
    for record in records:
        questions = {qid: {k: v for k, v in q.items() if k not in ("label", "src")}
                     for qid, q in record["questions"].items()}
        requests.append((record["state"], questions))
    end_to_end = time_calls(lambda r: agent.predict(r[0], r[1]), requests)
    encodings = []
    for record in records:
        rec = materialize(record)
        from qev.tokenization import encode_question
        encodings.append(encode_question(rec["state"], rec["questions"][0], agent.tokenizer, 1024, 384))
    result = {"dtype": agent.config.get("mlx_dtype"), "agent_predict_ms": end_to_end,
              "predict_logits_ms": time_calls(lambda e: agent.model.predict_logits([e]), encodings),
              "tokens": summary([len(e["ids"]) for e in encodings])}
    # Same runtime with a cheaper backbone precision: separates the effect of
    # precision from the effect of a smaller model. Latency only, no accuracy audit.
    # Adapters-off (merged-LoRA floor) timing lives in mlx_qev_adapters.py: it
    # must run in a fresh process to avoid memory-pressure artifacts.
    reference = [agent.model.predict_logits([e])[0] for e in encodings]
    for precision in ("bf16", "q8"):
        try:
            if precision == "bf16":
                agent.model.backbone.set_dtype(mx.bfloat16)
            else:
                nn.quantize(agent.model.backbone.language_model, group_size=64, bits=8)
            timing = time_calls(lambda e: agent.model.predict_logits([e]), encodings)
            changed = sum(int(np.argmax(agent.model.predict_logits([e])[0]) != np.argmax(r))
                          for e, r in zip(encodings, reference))
            result[f"predict_logits_{precision}_ms"] = {**timing, "argmax_changes_vs_fp32": changed}
        except Exception as error:  # noqa: BLE001 -- report and keep the fp32 measurement
            result[f"predict_logits_{precision}_ms"] = {"error": f"{type(error).__name__}: {error}"[:300]}
    return result


class Candidate:
    """Backbone decision forward with an explicit attention mask when needed."""

    def __init__(self, name):
        self.name = name
        spec = qevd.BACKBONES[name]
        self.family = spec["family"]
        self.tokenizer = qevd.load_tokenizer(name)
        if spec["kind"] == "lfm2_vl":
            from huggingface_hub import snapshot_download
            from mlx_vlm import load

            model, _ = load(snapshot_download(spec["repo"], revision=spec["revision"]))
            self.model, self.text = model, model.language_model.model
            self.norm = self.text.embedding_norm
        else:
            from huggingface_hub import snapshot_download
            from mlx_lm import load

            model, _ = load(snapshot_download(spec["repo"], revision=spec["revision"]))
            self.model, self.text = model, model.model
            self.norm = self.text.norm
        hidden = self.text.embed_tokens.weight.shape[-1]
        rng = np.random.default_rng(0)
        self.q = mx.array(rng.standard_normal((256, hidden), dtype=np.float32) / math.sqrt(hidden))
        self.k = mx.array(rng.standard_normal((256, hidden), dtype=np.float32) / math.sqrt(hidden))
        self.ids = qevd.special_ids(self.tokenizer, self.family)
        self.prefix = qevd.prefix_ids(self.tokenizer, self.family)

    def set_precision(self, precision):
        if precision == "q8":
            nn.quantize(self.model, group_size=64, bits=8)
        else:
            self.model.set_dtype({"fp32": mx.float32, "bf16": mx.bfloat16}[precision])

    def encode(self, record):
        rec = materialize(record)
        return qevd.encode_question(rec["state"], rec["questions"][0], self.tokenizer, self.ids,
                                    family=self.family, prefix=self.prefix)

    def decide(self, encoding, attention):
        ids = mx.array([encoding["ids"]])
        if attention == "causal":
            hidden = self.text(ids)
        else:
            segments = mx.array(encoding["segments"])
            mask = (segments[None, :] == 0) | (segments[:, None] == segments[None, :])
            h = self.text.embed_tokens(ids)
            for layer in self.text.layers:
                is_attention = getattr(layer, "is_attention_layer", True)
                h = layer(h, mask if is_attention else None, cache=None)
            hidden = self.norm(h)
        hidden = hidden[0].astype(mx.float32)
        query = hidden[encoding["decision_position"]] @ self.q.T
        keys = hidden[mx.array(encoding["option_positions"])] @ self.k.T
        logits = keys @ query
        mx.eval(logits)
        return logits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--qev", default="models/qev-snake-0.8b-mlx")
    ap.add_argument("--candidates", nargs="*", default=["lfm2-vl", "qwen3"])
    ap.add_argument("--precisions", nargs="*", default=["fp32", "bf16", "q8"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    records = snake_requests(args.requests)
    result = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                          "mlx": mx.__version__}, "requests": len(records),
              "scope": "Warm single-request wall time on Snake development states; one decision per call."}
    if args.qev and Path(args.qev).exists():
        result["qev_production"] = qev_production(args.qev, records)
        print("QEV", json.dumps(result["qev_production"]), flush=True)
    for name in args.candidates:
        entry = {}
        for precision in args.precisions:
            candidate = Candidate(name)  # fresh weights per precision
            candidate.set_precision(precision)
            encodings = [candidate.encode(r) for r in records]
            entry[precision] = {"tokens": summary([len(e["ids"]) for e in encodings])}
            for attention in ("causal", "block"):
                entry[precision][f"{attention}_ms"] = time_calls(lambda e, c=candidate, a=attention: c.decide(e, a), encodings)
            print(name, precision, json.dumps(entry[precision]), flush=True)
            del candidate
            mx.clear_cache()
        result[name] = entry
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
