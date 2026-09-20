"""Validate post-training native vision preservation and media decision plumbing.

Run after training on the same Colab GPU/runtime used by colab_smoke.py::

    python scripts/validate_native.py --checkpoint models/qev-0.8b \
        --baseline runs/native_baseline.json --output runs/native_validation.json

The image fixture and decoding settings exactly match colab_smoke.py. Video
generation and media-pointer outputs are functional probes, not accuracy tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from qev.evaluation import inference_autocast, prediction_precision, write_json
from qev.model import QevModel
from qev.tokenization import encode_multimodal_question

IMAGE_PROMPT = "What color is the shape in the image? Answer in one word."


def image_fixture():
    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).rectangle((48, 48, 176, 176), fill="red")
    return image


def video_fixture():
    frames = []
    for x in (48, 88, 128, 168):
        frame = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(frame).ellipse((x - 28, 84, x + 28, 140), fill="blue")
        frames.append(np.asarray(frame))
    return np.stack(frames)


def move_inputs(inputs, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()}


def generate(model, processor, content, *, max_new_tokens, video_metadata=None):
    kwargs = {}
    if video_metadata:
        kwargs.update(video_metadata=video_metadata, do_sample_frames=False, cap_pixels_per_frame=False)
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt", processor_kwargs=kwargs,
    )
    device = next(model.parameters()).device
    inputs = move_inputs(inputs, device)
    started = time.perf_counter()
    with torch.inference_mode(), inference_autocast(model):
        output = model.generate_native(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    continuation = output[0, inputs["input_ids"].shape[-1]:].tolist()
    return {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "token_ids": continuation,
        "text": processor.tokenizer.decode(continuation, skip_special_tokens=True),
        "max_new_tokens": max_new_tokens,
        "seconds": time.perf_counter() - started,
        "decision_adapter_enabled": False,
        "processor_fields": sorted(inputs),
    }


def pointer_probe(model, processor, state, question, **media):
    encoding = encode_multimodal_question(
        state, question, processor, max_length=8192, **media,
    )
    inputs = move_inputs(encoding["inputs"], next(model.parameters()).device)
    with torch.inference_mode(), inference_autocast(model):
        logits = model(**inputs)[0].float()
        probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError("Multimodal pointer produced non-finite outputs")
    values = probabilities.cpu().tolist()
    return {
        "input_tokens": len(encoding["ids"]),
        "options": question["options"],
        "raw_probabilities": values,
        "choice": question["options"][int(np.argmax(values))],
        "probability_sum": float(sum(values)),
        "temperature": 1.0,
        "processor_fields": sorted(encoding["inputs"]),
        "accuracy_asserted": False,
        "training_modalities": ["text"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="runs/native_baseline.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    baseline = json.loads(Path(args.baseline).read_text())
    if not isinstance(baseline.get("token_ids"), list) or "input_tokens" not in baseline:
        raise ValueError("Baseline must be the native_baseline.json written by colab_smoke.py")
    model = QevModel.from_checkpoint(args.checkpoint, device=args.device, dtype=torch.float32)
    processor = model.get_processor()
    image, video = image_fixture(), video_fixture()
    precision = prediction_precision(model)
    precision_label = "cuda bf16 autocast" if torch.device(args.device).type == "cuda" else f"{args.device} float32"
    image_result = generate(model, processor, [
        {"type": "image", "image": image}, {"type": "text", "text": IMAGE_PROMPT},
    ], max_new_tokens=16)
    same_tokens = image_result["token_ids"] == baseline["token_ids"]
    comparable = (image_result["input_tokens"] == baseline["input_tokens"]
                  and precision_label == baseline.get("precision"))
    image_result.update(
        baseline_token_ids=baseline["token_ids"], baseline_text=baseline.get("text"),
        baseline_precision=baseline.get("precision"), same_token_ids=same_tokens,
        same_input_length=image_result["input_tokens"] == baseline["input_tokens"],
        comparable_precision=comparable,
        preservation_passed=same_tokens and comparable,
    )
    fps = 2.0
    video_result = generate(model, processor, [
        {"type": "video", "video": video},
        {"type": "text", "text": "Describe the motion of the blue circle in this video in one short sentence."},
    ], max_new_tokens=48, video_metadata=[{
        "fps": fps, "total_num_frames": len(video), "frames_indices": list(range(len(video))),
    }])
    video_result.update(frames=len(video), fps=fps, accuracy_asserted=False)
    image_pointer = pointer_probe(model, processor, "Inspect the supplied image.", {
        "instr": "What is the color of the shape?", "options": ["red", "blue", "green"],
    }, images=[image])
    video_pointer = pointer_probe(model, processor, "Inspect the supplied video frames in order.", {
        "instr": "In which direction does the blue circle move?",
        "options": ["left to right", "right to left", "stationary"],
    }, videos=[video], video_fps=[fps])
    integrity_path = Path(args.checkpoint) / "base_integrity.json"
    integrity = json.loads(integrity_path.read_text()) if integrity_path.is_file() else None
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_model": model.config["base_model"], "base_revision": model.config["base_revision"],
        "torch": torch.__version__, "precision": precision,
        "vision_and_lm_head_frozen": (
            not any(p.requires_grad for p in model.foundation.model.visual.parameters())
            and not any(p.requires_grad for p in model.foundation.lm_head.parameters())
        ),
        "base_integrity": integrity,
        "image_native": image_result, "video_native": video_result,
        "image_decision": image_pointer, "video_decision": video_pointer,
        "scope": "Original foundation generation with decision adapters disabled. Image tokens compared with the same pre-training fixture; video and media decisions are functional smoke checks. The pointer was trained on text outcomes, not supervised image/video decisions.",
    }
    result["passed"] = (image_result["preservation_passed"]
                        and result["vision_and_lm_head_frozen"]
                        and (integrity is None or integrity.get("unchanged") is True))
    write_json(args.output, result)
    print("QEV_NATIVE_VALIDATION", json.dumps(result, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit("Native preservation check failed; inspect the saved validation report")


if __name__ == "__main__":
    main()
