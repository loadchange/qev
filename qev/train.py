"""Train the Qev adapter and readout using labelled decisions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .data import load_split
from .evaluation import (
    encode_records,
    fit_temperature,
    predict_items,
    prediction_precision,
    report,
    write_json,
)
from .model import QevModel
from .tokenization import collate_encodings

BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def frozen_digest(model):
    """Hash every frozen tensor in bounded chunks, before and after optimization."""
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 8_000_000):
            data = flat[start:start + 8_000_000].cpu().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(data))
        count += parameter.numel()
    return {"sha256": digest.hexdigest(), "parameters": count}


def accumulation_loss(logits, target, examples_in_group):
    """Equal question weight across uneven microbatches, including the last group."""
    if examples_in_group < target.numel() or target.numel() == 0:
        raise ValueError("The accumulation denominator must cover this nonempty microbatch")
    return F.cross_entropy(logits, target, reduction="sum") / examples_in_group


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-state", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit-train", type=int, default=0, help="Smoke-test subset; recorded in training metadata")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                    help="Measure the random pointer + zero-initialized adapters on development before training")
    args = ap.parse_args(argv)
    if min(args.epochs, args.batch, args.accum) < 1:
        ap.error("epochs, batch, and accum must be positive")
    if args.limit_train < 0 or args.eval_limit < 0:
        ap.error("limit-train and eval-limit must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error("lr must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                ap.error("CUDA training requires bf16 support; select a Colab L4/A100 or newer GPU")
    out = Path(args.out)
    if (out / "qev_config.json").exists():
        raise FileExistsError(f"Checkpoint already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    partitions = {split: load_split(args.data, split) for split in ("train", "calibration", "development")}
    if args.limit_train:
        partitions["train"] = partitions["train"][:args.limit_train]
    if args.eval_limit:
        for split in ("calibration", "development"):
            partitions[split] = partitions[split][:args.eval_limit]
    encoded = {split: encode_records(records, tokenizer, args.max_length, args.max_state)
               for split, records in partitions.items()}
    if any(not items for items in encoded.values()):
        raise ValueError("Train, calibration and development must each contain labelled questions")
    dataset_hash = hashlib.sha256((Path(args.data) / "manifest.json").read_bytes()).hexdigest()
    dtype = torch.float32
    model = QevModel.from_pretrained(args.base, revision=args.revision, dtype=dtype,
        device=args.device, max_length=args.max_length, max_state=args.max_state)
    frozen_before = frozen_digest(model)
    if args.baseline:
        baseline = predict_items(model, encoded["development"], tokenizer, args.batch)
        write_json(out / "untrained_development.json", report(baseline, 1.))
        print("QEV_BASELINE_COMPLETE", json.dumps(report(baseline, 1.)["raw"]), flush=True)
    if args.checkpointing:
        model.enable_gradient_checkpointing()
    params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    batches = math.ceil(len(encoded["train"]) / args.batch)
    updates_per_epoch = math.ceil(batches / args.accum)
    total_steps = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(1., (step + 1) / max(1, total_steps * .1)) * .5 *
        (1 + math.cos(math.pi * max(0, step - total_steps * .1) / max(1, total_steps * .9))))
    provenance = {"arguments": vars(args), "dataset_manifest_sha256": dataset_hash,
                  "questions": {k: len(v) for k, v in encoded.items()},
                  "state_truncations": {k: sum(i["state_truncated"] for i in v) for k, v in encoded.items()},
                  "trainable_parameters": sum(p.numel() for p in params),
                  "torch": torch.__version__, "training_dtype": "fp32 master weights; bf16 autocast; fp32 pointer" if device.type == "cuda" else str(dtype),
                  "evaluation_precision": prediction_precision(model),
                  "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else args.device,
                  "frozen_before": frozen_before,
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "peft", "safetensors", "numpy")},
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "objective": "supervised cross-entropy over candidate options; no Jev labels or RL"}
    write_json(out / "provenance.json", provenance)
    print(json.dumps(provenance), flush=True)
    start_time = time.time(); step = 0; tokens = 0
    # Augment before encoding in data creation; shuffle batches every epoch.
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(encoded["train"]))); random.shuffle(order)
        # Local buckets avoid wasting memory on a 77-option outlier in every batch.
        buckets = [sorted(order[i:i + args.batch * 32], key=lambda j: len(encoded['train'][j]['ids']))
                   for i in range(0, len(order), args.batch * 32)]
        order = [j for bucket in buckets for j in bucket]
        chunks = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        losses = []
        for group_start in range(0, len(chunks), args.accum):
            group = chunks[group_start:group_start + args.accum]
            examples_in_group = sum(map(len, group))
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                items = [encoded["train"][i] for i in indices]
                batch = collate_encodings(items, tokenizer.pad_token_id, args.device)
                target = torch.tensor([i["label"] for i in items], device=args.device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(**batch)
                    loss = accumulation_loss(logits, target, examples_in_group)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                losses.append((float(loss.detach()) * examples_in_group, len(items)))
                tokens += sum(len(i["ids"]) for i in items)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradients")
            optimizer.step(); scheduler.step(); step += 1
            if step % 10 == 0 or step == total_steps:
                recent = losses[-args.accum * 10:]
                record = {"epoch": epoch + 1, "step": step, "total_steps": total_steps,
                          "loss": sum(value for value, _ in recent) / sum(n for _, n in recent),
                          "elapsed_seconds": time.time() - start_time, "tokens": tokens,
                          "lr": scheduler.get_last_lr()[0]}
                print(json.dumps(record), flush=True)
                with (out / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
        model.save_pretrained(out, tokenizer=tokenizer,
            extra_config={"model_name": "qev-0.8b", "completed_epochs": epoch + 1,
                          "dataset_manifest_sha256": dataset_hash})
    training_seconds = time.time() - start_time
    frozen_after = frozen_digest(model)
    integrity = {"before": frozen_before, "after": frozen_after,
                 "unchanged": frozen_before == frozen_after,
                 "generation": "Native generation disables the decision LoRA adapter; original vision and language weights are frozen."}
    write_json(out / "base_integrity.json", integrity)
    if not integrity["unchanged"]:
        raise RuntimeError("Frozen base weights changed during training")
    calibration = predict_items(model, encoded["calibration"], tokenizer, args.batch)
    temperature = fit_temperature(calibration)
    development = predict_items(model, encoded["development"], tokenizer, args.batch)
    write_json(out / "calibration_rows.json", calibration)
    write_json(out / "development_rows.json", development)
    evaluation = {"calibration": report(calibration, temperature),
                  "development": report(development, temperature),
                  "evaluation_precision": prediction_precision(model),
                  "calibration_fit_partition": "calibration",
                  "training_seconds": training_seconds, "optimizer_steps": step,
                  "forward_tokens": tokens,
                  "scope": "Development results; public task families also occur in training. Chinese data are programmatic rules, not a broad Chinese benchmark."}
    write_json(out / "evaluation.json", evaluation)
    model.save_pretrained(out, tokenizer=tokenizer,
        extra_config={"model_name": "qev-0.8b", "completed_epochs": args.epochs,
                      "temperature": temperature, "calibration_split": "calibration",
                      "calibration_precision": prediction_precision(model),
                      "dataset_manifest_sha256": dataset_hash})
    print("QEV_TRAINING_COMPLETE", json.dumps(evaluation), flush=True)


if __name__ == "__main__":
    main()
