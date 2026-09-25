"""Candidate-order robustness: score every development question with its options reversed.

Reports accuracy in both orders, how often the chosen option (by identity, not
position) changes, and how far the calibrated probability of each option moves.
Causal readouts see earlier candidates only; block attention lets every
candidate see all others, which is where a diffusion-style read could help.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diffusion import qevd
from qev.data import load_split, materialize
from qev.evaluation import probabilities

SNAKE_SOURCE = "synthetic_snake_observable_teacher"


def load(path, device):
    path = Path(path)
    if (path / "qevd_config.json").exists():
        model = qevd.QevDModel.from_checkpoint(path, device=device)
        tokenizer = qevd.load_tokenizer(model.config["backbone"])
        family, segments = model.config["family"], True
    else:
        from transformers import AutoTokenizer

        from qev.model import QevModel
        model = QevModel.from_checkpoint(path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        family, segments = "qwen", False
    return model, tokenizer, family, segments, float(model.config.get("temperature", 1.0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, help="label=checkpoint_dir")
    ap.add_argument("--data", default="data/snake-v1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="Records per half (smoke tests)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    results = {}
    for spec in args.arm:
        label, path = spec.split("=", 1)
        model, tokenizer, family, segments, temperature = load(path, device)
        ids, prefix = qevd.special_ids(tokenizer, family), qevd.prefix_ids(tokenizer, family)
        rows = []
        records = load_split(ROOT / args.data, "development")
        if args.limit:
            records = records[:args.limit] + records[-args.limit:]
        for record in records:
            rec = materialize(record)
            for question in rec["questions"]:
                if len(question["options"]) < 2:
                    continue
                flipped = {**question, "options": question["options"][::-1]}
                pair = [qevd.encode_question(rec["state"], q, tokenizer, ids, family=family, prefix=prefix)
                        for q in (question, flipped)]
                rows.append({"pair": pair, "label": int(question["label"]),
                             "source": question.get("src", "unknown")})
        flat = [(i, j, row["pair"][j]) for i, row in enumerate(rows) for j in (0, 1)]
        flat.sort(key=lambda item: len(item[2]["ids"]))
        logits = {}
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                    enabled=device.type == "cuda"):
            for start in range(0, len(flat), args.batch):
                chunk = flat[start:start + args.batch]
                batch = qevd.collate([item[2] for item in chunk], tokenizer.pad_token_id, device)
                if not segments:
                    batch.pop("segments")
                out = model(**batch).float().cpu().numpy()
                for (i, j, encoding), row in zip(chunk, out):
                    logits[i, j] = row[:len(encoding["option_positions"])]
        per_source, flips, shifts, correct = {}, [], [], [[], []]
        for i, row in enumerate(rows):
            p = probabilities(logits[i, 0], temperature)
            q = probabilities(logits[i, 1], temperature)[::-1]  # back to original option order
            flip = int(p.argmax() != q.argmax())
            shift = float(np.abs(p - q).max())
            flips.append(flip)
            shifts.append(shift)
            correct[0].append(int(p.argmax() == row["label"]))
            correct[1].append(int(q.argmax() == row["label"]))
            group = "snake" if row["source"] == SNAKE_SOURCE else row["source"]
            per_source.setdefault(group, []).append(flip)
        results[label] = {"questions": len(rows), "accuracy_original": float(np.mean(correct[0])),
                          "accuracy_reversed": float(np.mean(correct[1])), "choice_flip_rate": float(np.mean(flips)),
                          "mean_max_probability_shift": float(np.mean(shifts)),
                          "p95_max_probability_shift": float(np.percentile(shifts, 95)),
                          "flip_rate_by_source": {k: float(np.mean(v)) for k, v in sorted(per_source.items())}}
        print("ORDER", label, json.dumps({k: v for k, v in results[label].items() if k != "flip_rate_by_source"}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
