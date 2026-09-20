"""Outcome-based evaluation and calibration; no generated judge labels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def probabilities(logits, temperature=1.0):
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Expected a nonempty vector of finite candidate logits")
    z = z / temperature
    if not np.isfinite(z).all():
        raise ValueError("Temperature-scaled logits overflowed")
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def metrics(rows, temperature=1.0):
    if not rows:
        return {"n": 0}
    correct, confidence, nll, brier, score_error = [], [], [], [], []
    for row in rows:
        p = probabilities(row["logits"], temperature)
        y = int(row["label"])
        if y != row["label"] or not 0 <= y < len(p):
            raise ValueError("Evaluation label is not an in-range candidate index")
        correct.append(int(p.argmax() == y))
        confidence.append(float(p.max()))
        # log-sum-exp preserves the true loss even when p[y] underflows to 0.
        # Clipping p[y] would cap a catastrophically confident error at ~27.6.
        shifted = np.asarray(row["logits"], dtype=np.float64) / temperature
        shifted -= shifted.max()
        nll.append(float(np.log(np.exp(shifted).sum()) - shifted[y]))
        target = np.zeros(len(p)); target[y] = 1
        brier.append(float(np.square(p - target).sum()))
        if row.get("qtype") == "score":
            score_error.append(abs(float(p @ np.arange(len(p))) - y))
    confidence, correct = np.array(confidence), np.array(correct)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence > lo) & (confidence <= hi)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    out = {"n": len(rows), "accuracy": float(correct.mean()), "nll": float(np.mean(nll)),
           "brier": float(np.mean(brier)), "ece": float(ece),
           "confident_error_rate": float(np.mean((confidence >= .9) & (correct == 0)))}
    if score_error:
        out["score_mae"] = float(np.mean(score_error))
    return out


def fit_temperature(rows):
    if not rows:
        raise ValueError("Cannot fit temperature without calibration questions")
    candidates = np.unique(np.r_[1.0, np.exp(np.linspace(np.log(.25), np.log(8), 101))])
    return float(min(candidates, key=lambda t: metrics(rows, t)["nll"]))


def report(rows, temperature=1.0):
    result = {"raw": metrics(rows), "calibrated": metrics(rows, temperature),
              "temperature": temperature}
    for field in ("source", "language", "qtype"):
        result["by_" + field] = {
            key: metrics([r for r in rows if r.get(field, "unknown") == key], temperature)
            for key in sorted({r.get(field, "unknown") for r in rows})
        }
    precision = sorted({r["precision"] for r in rows if "precision" in r})
    if precision:
        result["prediction_precision"] = precision
    return result


def encode_records(records, tokenizer, max_length=1024, max_state=384):
    from .data import materialize
    from .tokenization import encode_question
    items = []
    for index, record in enumerate(records):
        rec = materialize(record)
        meta = record.get("_meta", {})
        for q in rec["questions"]:
            enc = encode_question(rec["state"], q, tokenizer, max_length, max_state)
            enc.update(label=int(q["label"]), source=q.get("src", meta.get("source", "unknown")),
                       language=meta.get("language", "en"), qtype=q.get("qtype", "choice"),
                       record_id=meta.get("id", str(index)), question_id=q.get("qid", ""))
            items.append(enc)
    return items


def prediction_precision(model):
    """Describe the actual evaluation policy, not just weight storage dtype.

    CUDA uses bf16 autocast with the fp32 readout. Transformers dispatches to
    installed FLA kernels without automatically substituting its fp32 reference;
    evaluating fp32 master weights without autocast is not the training path.
    CPU/MPS remain native dtype and do not require the optional CUDA kernels.
    """
    parameter = next(model.parameters())
    device = parameter.device
    return {
        "device": str(device),
        "weight_dtype": str(parameter.dtype).removeprefix("torch."),
        "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
        "pointer_dtype": "float32",
        "arithmetic": "bf16_autocast" if device.type == "cuda" else "native_parameter_dtype",
    }


def inference_autocast(model):
    import torch

    device_type = next(model.parameters()).device.type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda")


def predict_items(model, items, tokenizer, batch_size=4):
    import torch

    from .tokenization import collate_encodings
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    device = next(model.parameters()).device
    rows = []
    policy = prediction_precision(model)
    precision = f"{device.type}:{policy['autocast_dtype'] or policy['weight_dtype']}:fp32_pointer"
    # Length sorting reduces padding while retaining full record identity.
    ordered = sorted(enumerate(items), key=lambda x: len(x[1]["ids"]))
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            batch = collate_encodings([item for _, item in chunk], tokenizer.pad_token_id, device)
            with inference_autocast(model):
                output = model(**batch)
            logits = output.float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits; no partial report was produced")
            for (index, item), z in zip(chunk, logits):
                row = {k: item[k] for k in ("label", "source", "language", "qtype", "record_id", "question_id", "state_truncated")}
                row.update(index=index, logits=z[:len(item["option_positions"])].tolist(), precision=precision)
                rows.append(row)
    return sorted(rows, key=lambda r: r["index"])


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
