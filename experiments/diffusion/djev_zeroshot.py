"""Zero-shot DiffusionGemma structured reads (the djev method) on Qev's splits.

Reproduces djev-run's server.py: the questions go in the system message with
single-token labels, the state is the user message, the reply template
``id: <label>`` is pinned on the diffusion canvas except the label slot, and a
single read-only denoise step returns the slot distribution. Probabilities are
a softmax over the labels' logprobs (labels outside the returned top-k get a
floor), exactly as djev-run does. No training and no Qev weights are involved.

Requires a running ``vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4`` with the
diffusion structured-read extensions (vLLM nightly).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qev.data import load_split, materialize
from qev.evaluation import fit_temperature, metrics

LABELS = [chr(97 + i) for i in range(26)] + [chr(65 + i) for i in range(26)] + [str(i) for i in range(10)]
SNAKE_SOURCE = "synthetic_snake_observable_teacher"


def post(url, payload, timeout=60):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())


class Reader:
    def __init__(self, base, model, canvas, top_logprobs):
        self.base, self.model, self.canvas, self.top = base.rstrip("/"), model, canvas, top_logprobs
        self.cache = {}

    def tokenize(self, text):
        if text not in self.cache:
            self.cache[text] = post(f"{self.base}/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]
        return self.cache[text]

    def read(self, state, question):
        options = question["options"]
        labels = LABELS[:len(options)]
        system = ("Answer a fixed set of questions about the state the user provides. Each question lists its "
                  "allowed answers; reply with exactly one label per question.\n")
        system += f"\nQuestion answer: {question['instr'].strip()}\n"
        system += "".join(f"  {label}: {option}\n" for label, option in zip(labels, options))
        system += '\nReply with one line per question, in this order, formatted as "id: label".'
        base_toks = self.tokenize(f"answer: {labels[0]}")
        alt_toks = self.tokenize(f"answer: {labels[min(1, len(labels) - 1)]}")
        slots = [i for i, (a, b) in enumerate(zip(base_toks, alt_toks)) if a != b]
        if len(base_toks) != len(alt_toks) or len(slots) != 1:
            raise ValueError("labels must occupy one aligned token")
        slot = slots[0]
        rng = random.Random(42)
        seed = list(base_toks)
        seed[slot] = rng.randrange(262144)
        pinned = [i for i in range(len(base_toks)) if i != slot]
        seed += [1] * (self.canvas - len(seed))
        payload = {"model": self.model, "max_tokens": len(base_toks), "logprobs": True, "top_logprobs": self.top,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": state}],
                   "vllm_xargs": {"diffusion_seed_canvas": seed, "diffusion_pinned": pinned,
                                  "diffusion_max_steps": 1, "diffusion_read_only": True}}
        started = time.perf_counter()
        out = post(f"{self.base}/v1/chat/completions", payload)
        elapsed = (time.perf_counter() - started) * 1000
        content = out["choices"][0]["logprobs"]["content"]
        table = {}
        if slot < len(content):
            for top in content[slot]["top_logprobs"]:
                token = str(top["token"]).strip()
                table[token] = max(table.get(token, -math.inf), float(top["logprob"]))
        floor = (min(table.values()) if table else -20.0) - 5.0
        logits = [table.get(label, floor) for label in labels]
        return logits, elapsed, sum(label in table for label in labels)


def items(split, limit):
    rows = []
    for record in load_split(ROOT / "data/snake-v1", split):
        rec, meta = materialize(record), record.get("_meta", {})
        for question in rec["questions"]:
            rows.append({"state": rec["state"], "question": question, "label": int(question["label"]),
                         "source": question.get("src", meta.get("source", "unknown")),
                         "qtype": question.get("qtype", "choice"), "language": meta.get("language", "en")})
    return rows[:limit] if limit else rows


def run(reader, rows, workers):
    def one(row):
        if len(row["question"]["options"]) > len(LABELS):
            return None
        logits, ms, covered = reader.read(row["state"], row["question"])
        return {**{k: row[k] for k in ("label", "source", "qtype", "language")},
                "logits": logits, "ms": ms, "labels_in_topk": covered, "options": len(logits)}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="djev-dgemma")
    ap.add_argument("--canvas", type=int, default=128)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    reader = Reader(args.base, args.model, args.canvas, args.top_logprobs)
    result = {"model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4", "method": "djev-run server.py structured read, 1 step",
              "top_logprobs": args.top_logprobs}
    rows = {}
    for split in ("calibration", "development"):
        split_rows = items(split, args.limit)
        started = time.time()
        read = run(reader, split_rows, args.workers)
        rows[split] = [r for r in read if r is not None]
        result[f"{split}_skipped_over_{len(LABELS)}_options"] = sum(r is None for r in read)
        result[f"{split}_seconds"] = time.time() - started
        print(split, len(rows[split]), "read", flush=True)
    temperature = fit_temperature(rows["calibration"])
    dev = rows["development"]
    result.update(temperature=temperature,
                  development={"raw": metrics(dev), "calibrated": metrics(dev, temperature),
                               "snake": metrics([r for r in dev if r["source"] == SNAKE_SOURCE], temperature),
                               "general": metrics([r for r in dev if r["source"] != SNAKE_SOURCE], temperature),
                               "by_source": {s: metrics([r for r in dev if r["source"] == s], temperature)
                                             for s in sorted({r["source"] for r in dev})}},
                  labels_in_topk_fraction=statistics.mean(r["labels_in_topk"] / r["options"] for r in dev))
    serial = [reader.read(r["state"], r["question"])[1] for r in items("development", 0)[:40]]
    result["serial_request_ms"] = {"p50": statistics.median(serial), "max": max(serial), "n": len(serial)}
    Path(args.output).write_text(json.dumps({**result, "development_rows": dev}, indent=2) + "\n")
    print("DJEV", json.dumps({k: v for k, v in result.items() if k != "development"}), flush=True)
    print("DJEV_DEV", json.dumps({k: result["development"][k] for k in ("raw", "calibrated", "snake", "general")}), flush=True)


if __name__ == "__main__":
    main()
