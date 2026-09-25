"""Train and evaluate one arm of the Qev diffusion experiment.

Every arm uses the same data, LoRA rank, pointer head, optimizer and schedule;
only the backbone and the decision attention pattern change. ``--backbone
qwen3_5`` trains the unchanged Qev architecture (Qwen3.5-0.8B, causal) with the
same one-stage recipe as the other arms.

Outputs under ``--out``: checkpoint files, provenance.json, training.jsonl,
evaluation.json (calibrated development metrics, Snake/general split),
latency.json (GPU single/multi-question timing) and snake.json (closed-loop
games with the published 8x8 protocol, argmax actions, no guardrail).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diffusion import qevd
from qev.api import to_answers, to_record
from qev.data import load_split, materialize
from qev.evaluation import fit_temperature, metrics, probabilities, report, write_json

SNAKE_SOURCE = "synthetic_snake_observable_teacher"
QWEN35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def encode_split(records, tokenizer, ids, family, prefix, max_length, max_state):
    items = []
    for index, record in enumerate(records):
        rec, meta = materialize(record), record.get("_meta", {})
        for question in rec["questions"]:
            item = qevd.encode_question(rec["state"], question, tokenizer, ids, family=family, prefix=prefix,
                                        max_length=max_length, max_state=max_state)
            item.update(label=int(question["label"]), source=question.get("src", meta.get("source", "unknown")),
                        language=meta.get("language", "en"), qtype=question.get("qtype", "choice"),
                        record_id=meta.get("id", str(index)), question_id=question.get("qid", ""))
            items.append(item)
    return items


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


class Runner:
    """Uniform calls for QevD arms and the unchanged QevModel baseline."""

    def __init__(self, args, device):
        self.device, self.backbone = device, args.backbone
        if args.backbone == "qwen3_5":
            from transformers import AutoTokenizer

            from qev.model import QevModel

            self.model = QevModel.from_pretrained(
                "Qwen/Qwen3.5-0.8B", revision=QWEN35_REVISION, lora_rank=args.lora_rank,
                device=device, max_length=args.max_length, max_state=args.max_state)
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=QWEN35_REVISION)
            self.family, self.segments = "qwen", False
        else:
            self.model = qevd.QevDModel.from_pretrained(
                args.backbone, attention=args.attention, lora_rank=args.lora_rank, device=device,
                max_length=args.max_length, max_state=args.max_state)
            self.tokenizer = qevd.load_tokenizer(args.backbone)
            self.family, self.segments = qevd.BACKBONES[args.backbone]["family"], True
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ids = qevd.special_ids(self.tokenizer, self.family)
        self.prefix = qevd.prefix_ids(self.tokenizer, self.family)

    def batch(self, items):
        batch = qevd.collate(items, self.tokenizer.pad_token_id, self.device)
        if not self.segments:
            batch.pop("segments")
        return batch

    def encode(self, state, question, max_length, max_state):
        return qevd.encode_question(state, question, self.tokenizer, self.ids, family=self.family,
                                    prefix=self.prefix, max_length=max_length, max_state=max_state)

    @torch.inference_mode()
    def logits(self, items, bf16=True):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=bf16 and self.device.type == "cuda"):
            return self.model(**self.batch(items)).float().cpu().numpy()


def predict(runner, items, batch_size):
    runner.model.eval()
    rows = []
    ordered = sorted(enumerate(items), key=lambda pair: len(pair[1]["ids"]))
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        logits = runner.logits([item for _, item in chunk])
        if not np.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        for (index, item), row in zip(chunk, logits):
            rows.append({"index": index, "logits": row[:len(item["option_positions"])].tolist(),
                         **{k: item[k] for k in ("label", "source", "language", "qtype", "record_id", "question_id")}})
    return sorted(rows, key=lambda row: row["index"])


def split_report(rows, temperature):
    result = report(rows, temperature)
    snake = [row for row in rows if row["source"] == SNAKE_SOURCE]
    general = [row for row in rows if row["source"] != SNAKE_SOURCE]
    result["snake"] = metrics(snake, temperature)
    result["general"] = metrics(general, temperature)
    return result


def train(runner, items, args, log_path):
    model, device = runner.model, runner.device
    params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    updates_per_epoch = math.ceil(math.ceil(len(items) / args.batch) / args.accum)
    total = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1., (step + 1) / max(1, total * .1)) * .5 *
        (1 + math.cos(math.pi * max(0, step - total * .1) / max(1, total * .9))))
    step, tokens, started = 0, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(items)))
        random.shuffle(order)
        buckets = [sorted(order[i:i + args.batch * 32], key=lambda j: len(items[j]["ids"]))
                   for i in range(0, len(order), args.batch * 32)]
        order = [j for bucket in buckets for j in bucket]
        chunks = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        losses = []
        for group_start in range(0, len(chunks), args.accum):
            group = chunks[group_start:group_start + args.accum]
            examples = sum(map(len, group))
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                chunk = [items[i] for i in indices]
                target = torch.tensor([item["label"] for item in chunk], device=device)
                with autocast(device):
                    loss = F.cross_entropy(model(**runner.batch(chunk)), target, reduction="sum") / examples
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                losses.append((float(loss.detach()) * examples, len(chunk)))
                tokens += sum(len(item["ids"]) for item in chunk)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradients")
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 10 == 0 or step == total:
                recent = losses[-args.accum * 10:]
                record = {"epoch": epoch + 1, "step": step, "total_steps": total,
                          "loss": sum(v for v, _ in recent) / sum(n for _, n in recent),
                          "elapsed_seconds": time.time() - started, "tokens": tokens,
                          "lr": scheduler.get_last_lr()[0]}
                print(json.dumps(record), flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
    return {"training_seconds": time.time() - started, "optimizer_steps": step, "forward_tokens": tokens}


def _timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000


def summary(values):
    ordered = sorted(values)
    return {"n": len(values), "p50": statistics.median(values),
            "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "mean": statistics.mean(values)}


@contextmanager
def merged_adapter(model):
    """Fold LoRA into the base weights while timing; unmerge afterwards."""
    adapter = model.text_model if isinstance(model, qevd.QevDModel) else model.backbone
    adapter.merge_adapter()
    try:
        yield
    finally:
        adapter.unmerge_adapter()


def multi_question_encodings(runner, args, count, rng):
    record = next(r for r in load_split(args.data, "development")
                  if next(iter(r["questions"].values())).get("src") == SNAKE_SOURCE)
    rec = materialize(record)
    questions = []
    for _ in range(count):
        options = list(rec["questions"][0]["options"])
        rng.shuffle(options)
        questions.append({**rec["questions"][0], "options": options})
    return [runner.encode(rec["state"], q, args.max_length, args.max_state) for q in questions]


def packable(model):
    return isinstance(model, qevd.QevDModel) and model.config["attention"] != "full" \
        and model.config["kind"] not in ("lfm2", "lfm2_vl")


def latency(runner, dev_items, args):
    """GPU wall-clock after warmup, as served (separate adapter) and with LoRA merged.

    Small models on a fast GPU are launch-overhead bound, so batched and packed
    timings matter as much as single-question ones. Packed-vs-independent
    agreement is checked in fp32 (exactness) and bf16 (serving numerics).
    """
    device = runner.device
    result = {"device": str(device), "precision": "bf16_autocast" if device.type == "cuda" else "native",
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    snake = [item for item in dev_items if item["source"] == SNAKE_SOURCE][:args.latency_items]
    general = [item for item in dev_items if item["source"] != SNAKE_SOURCE][:args.latency_items]
    multi = {count: multi_question_encodings(runner, args, count, random.Random(count))
             for count in args.latency_questions}
    model = runner.model

    def measure():
        out = {}
        for _ in range(5):
            runner.logits(snake[:1])
        for name, items in (("snake", snake), ("general", general)):
            out[f"single_{name}_ms"] = summary([_timed(lambda item=item: runner.logits([item]), device)[1]
                                                for item in items])
        out["multi_question"] = {}
        for count, encodings in multi.items():
            for _ in range(3):
                runner.logits(encodings)
            entry = {"sequential_ms": summary([_timed(lambda es=encodings: [runner.logits([e]) for e in es], device)[1]
                                               for _ in range(args.latency_repeats)]),
                     "batched_ms": summary([_timed(lambda es=encodings: runner.logits(es), device)[1]
                                            for _ in range(args.latency_repeats)])}
            if packable(model):
                packed = qevd.pack_questions(encodings)
                with autocast(device):
                    for _ in range(3):
                        model.forward_packed(packed, device)
                    entry["packed_ms"] = summary([_timed(lambda p=packed: model.forward_packed(p, device), device)[1]
                                                  for _ in range(args.latency_repeats)])
            out["multi_question"][str(count)] = entry
        return out

    result["adapter"] = measure()
    with merged_adapter(model):
        result["merged"] = measure()
    for name, items in (("snake", snake), ("general", general)):
        result[f"single_{name}_tokens"] = summary([len(item["ids"]) for item in items])
    result["tokens"] = {str(c): {"independent": sum(len(e["ids"]) for e in encs),
                                 "packed": len(qevd.pack_questions(encs)["ids"])} for c, encs in multi.items()}
    if packable(model):
        agreement = {}
        for bf16 in (False, True):
            worst = 0.0
            for encodings in multi.values():
                packed = qevd.pack_questions(encodings)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=bf16 and device.type == "cuda"):
                    rows = [row.float().cpu().numpy() for row in model.forward_packed(packed, device)]
                single = [runner.logits([e], bf16=bf16)[0][:len(e["option_positions"])] for e in encodings]
                worst = max([worst] + [float(np.abs(probabilities(p) - probabilities(s)).max())
                                       for p, s in zip(rows, single)])
            agreement["bf16" if bf16 else "fp32"] = worst
        result["packed_vs_independent_max_probability_difference"] = agreement
    return result


class Predictor:
    """scripts/benchmark_snake.py predictor interface over one arm."""

    def __init__(self, runner, temperature, args, label):
        self.runner, self.temperature, self.args = runner, temperature, args
        self.model_name = label
        self.source = {"kind": "qevd-experiment", "label": label, "backbone": args.backbone,
                       "attention": args.attention, "temperature": temperature,
                       "precision": "cuda bf16 autocast, fp32 pointer" if runner.device.type == "cuda" else "native"}

    def predict(self, requests):
        encodings, metadata = [], []
        for request in requests:
            record, meta = to_record(request)
            encodings.append(self.runner.encode(record["state"], record["questions"][0],
                                                self.args.max_length, self.args.max_state))
            metadata.append(meta)
        logits = self.runner.logits(encodings)
        return [{"model": request["model"],
                 "answers": to_answers([probabilities(row[:len(e["option_positions"])], self.temperature).tolist()], meta),
                 "usage": {"input_tokens": len(e["ids"]), "output_tokens": 0},
                 "qev": {"truncated_questions": ["move"] if e["state_truncated"] else []}}
                for request, e, meta, row in zip(requests, encodings, metadata, logits)]


def snake_benchmark(runner, temperature, args, out):
    spec = importlib.util.spec_from_file_location("benchmark_snake", ROOT / "scripts/benchmark_snake.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bench = Namespace(seeds=list(range(10000, 10000 + args.snake_seeds)), size=8,
                      max_steps=args.snake_max_steps, observation="spatial", batch=8, warmup=1)
    predictor = Predictor(runner, temperature, args, args.label)
    with (out / "snake.steps.jsonl").open("w") as trace:
        result = module.run_benchmark(bench, predictor, trace)
    write_json(out / "snake.json", result)
    return {"complete": result["complete"], **{k: result["aggregate"][k] for k in ("food", "collisions", "terminal_reasons")},
            "prediction_call_ms": result["timing"]["prediction_call_ms"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=[*qevd.BACKBONES, "qwen3_5"])
    ap.add_argument("--attention", default="causal", choices=qevd.ATTENTION_MODES)
    ap.add_argument("--label")
    ap.add_argument("--data", default="data/snake-v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-state", type=int, default=384)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--latency-items", type=int, default=30)
    ap.add_argument("--latency-repeats", type=int, default=10)
    ap.add_argument("--latency-questions", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--skip-snake", action="store_true")
    ap.add_argument("--snake-seeds", type=int, default=20, help="Held-out seeds 10000.. (published protocol: 20)")
    ap.add_argument("--snake-max-steps", type=int, default=500)
    ap.add_argument("--checkpointing", action="store_true")
    args = ap.parse_args(argv)
    if args.backbone == "qwen3_5" and args.attention != "causal":
        ap.error("Qwen3.5 Gated DeltaNet layers are recurrent; only causal attention is supported")
    args.label = args.label or f"{args.backbone}-{args.attention}"
    out = Path(args.out)
    if (out / "evaluation.json").exists():
        raise FileExistsError(f"Arm already evaluated: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    load_started = time.time()
    runner = Runner(args, device)
    load_seconds = time.time() - load_started
    splits = {split: load_split(args.data, split) for split in ("train", "calibration", "development")}
    if args.limit_train:
        splits["train"] = splits["train"][:args.limit_train]
    if args.eval_limit:
        for split in ("calibration", "development"):
            # Keep both Snake and general questions in smoke runs.
            records = splits[split]
            splits[split] = records[:args.eval_limit // 2] + records[-(args.eval_limit // 2):]
    encoded = {split: encode_split(records, runner.tokenizer, runner.ids, runner.family, runner.prefix,
                                   args.max_length, args.max_state) for split, records in splits.items()}
    params = runner.model.trainable_parameters()
    provenance = {"arguments": vars(args), "load_seconds": load_seconds,
                  "dataset_manifest_sha256": hashlib.sha256((Path(args.data) / "manifest.json").read_bytes()).hexdigest(),
                  "questions": {k: len(v) for k, v in encoded.items()},
                  "state_truncations": {k: sum(i["state_truncated"] for i in v) for k, v in encoded.items()},
                  "tokens_per_question": {k: summary([len(i["ids"]) for i in v]) for k, v in encoded.items()},
                  "trainable_parameters": sum(p.numel() for p in params),
                  "total_parameters": sum(p.numel() for p in runner.model.parameters()),
                  "backbone": qevd.BACKBONES.get(args.backbone, {"repo": "Qwen/Qwen3.5-0.8B", "revision": QWEN35_REVISION}),
                  "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else args.device,
                  "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "objective": "supervised cross-entropy over candidate options (same recipe for every arm)"}
    write_json(out / "provenance.json", provenance)
    print(json.dumps(provenance), flush=True)
    if args.checkpointing:
        runner.model.enable_gradient_checkpointing()
    training = train(runner, encoded["train"], args, out / "training.jsonl")
    calibration = predict(runner, encoded["calibration"], args.eval_batch)
    temperature = fit_temperature(calibration)
    development = predict(runner, encoded["development"], args.eval_batch)
    evaluation = {"label": args.label, "backbone": args.backbone, "attention": args.attention,
                  "calibration": split_report(calibration, temperature),
                  "development": split_report(development, temperature), **training}
    write_json(out / "development_rows.json", development)
    if isinstance(runner.model, qevd.QevDModel):
        runner.model.save_pretrained(out, {"temperature": temperature, "label": args.label})
    else:
        runner.model.save_pretrained(out, tokenizer=runner.tokenizer,
                                     extra_config={"model_name": args.label, "temperature": temperature})
    print("QEVD_EVALUATION", json.dumps({"development": evaluation["development"]["calibrated"],
                                         "snake": evaluation["development"]["snake"],
                                         "general": evaluation["development"]["general"]}), flush=True)
    if not args.skip_snake:
        evaluation["snake_closed_loop"] = snake_benchmark(runner, temperature, args, out)
        print("QEVD_SNAKE", json.dumps(evaluation["snake_closed_loop"]), flush=True)
    timing = latency(runner, encoded["development"], args)
    write_json(out / "latency.json", timing)
    print("QEVD_LATENCY", json.dumps(timing), flush=True)
    write_json(out / "evaluation.json", evaluation)
    print("QEVD_ARM_COMPLETE", args.label, flush=True)


if __name__ == "__main__":
    main()
