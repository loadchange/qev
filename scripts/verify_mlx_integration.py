"""Export and audit a trained full multimodal MLX checkpoint against Torch CPU.

The fixed 20 probes include 14 development questions (all 10 public data sources,
English and Chinese executable rules), 3 manual instruction probes, and 3 media
probes. Four native-generation probes check unchanged text/image/video/mixed
behavior with decision adapters disabled. This is a numerical integration test,
not a multimodal accuracy benchmark or the full development evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from qev.evaluation import encode_records, probabilities
from qev.export import export_checkpoint, verify_export
from qev.mlx_runtime import MLXRuntime
from qev.processing import load_processor
from qev.tokenization import encode_multimodal_question, encode_question


def build_probes(data: Path, processor, config):
    records = [json.loads(line) for line in (data / "development.jsonl").read_text().splitlines() if line]
    grouped = {}
    for record in records:
        grouped.setdefault(record.get("_meta", {}).get("source", "unknown"), []).append(record)
    required = {"agnews", "amazon", "banking77", "boolq", "dbpedia14", "imdb", "mnli",
                "sst5", "trec", "yelp", "synthetic_rules_en", "synthetic_rules_zh"}
    if not required <= set(grouped):
        raise ValueError(f"Missing audit development sources: {sorted(required - set(grouped))}")
    max_length, max_state = config.get("max_length", 1024), config.get("max_state", 384)
    cases = []
    for source in sorted(required):
        pool = grouped[source]
        if source == "banking77":
            # Exercise all 77 candidates and the longest available state after
            # the same explicit text budget used by training.
            candidates = encode_records(pool, processor.tokenizer, max_length, max_state)
            cases.append(max(candidates, key=lambda encoded: len(encoded["ids"])))
        elif source.startswith("synthetic_rules_"):
            families = set()
            for record in pool:
                family = record.get("_meta", {}).get("family")
                if family in families:
                    continue
                cases.append(encode_records([record], processor.tokenizer, max_length, max_state)[0])
                families.add(family)
                if len(families) == 2:
                    break
            if len(families) != 2:
                raise ValueError(f"Need two executable rule families in {source}")
        else:
            cases.append(encode_records([pool[0]], processor.tokenizer, max_length, max_state)[0])
    manual = [
        ("The customer loved the product, but delivery arrived two days late.",
         {"instr": "Does the customer express a positive opinion about the product itself?", "options": ["false", "true"]}),
        ("The customer loved the product, but delivery arrived two days late.",
         {"instr": "Did the delivery arrive on time?", "options": ["false", "true"]}),
        ("用户说：昨天重复扣费了两次，请退还多扣的钱。",
         {"instr": "请选择最适合处理这条请求的部门。", "options": ["账单与退款", "技术支持", "销售咨询"]}),
    ]
    manual_start = len(cases)
    for index, (state, question) in enumerate(manual):
        encoded = encode_question(state, question, processor.tokenizer, max_length, max_state)
        encoded["probe_id"] = f"manual_instruction_{index}"
        cases.append(encoded)
    image = Image.new("RGB", (128, 128), (255, 0, 0))
    # Four frames create two temporal patch groups, exercising timestamped
    # MRoPE beyond a single image-like video group.
    video = np.stack([np.asarray(image)] * 4)
    question = {"instr": "Choose the dominant color.", "options": ["red", "blue", "green"]}
    media_options = [
        ("image", {"images": [image]}),
        ("video", {"videos": [video], "video_fps": [1.0]}),
        ("mixed_image_video", {"images": [image], "videos": [video], "video_fps": [1.0]}),
    ]
    for kind, kwargs in media_options:
        encoded = encode_multimodal_question(
            "Media is attached.", question, processor, **kwargs,
            max_length=config.get("multimodal_max_length", 8192), max_state=None,
        )
        encoded["probe_id"] = f"media_{kind}"
        cases.append(encoded)
    native = {}
    for kind in ("text", "image", "video", "mixed_image_video"):
        content = []
        if kind in {"image", "mixed_image_video"}:
            content.append({"type": "image", "image": image})
        if kind in {"video", "mixed_image_video"}:
            content.append({"type": "video", "video": video})
        instruction = "Write only the word ready." if kind == "text" else "What color is the media? Answer with one word."
        content.append({"type": "text", "text": instruction})
        video_kwargs = {}
        if kind in {"video", "mixed_image_video"}:
            video_kwargs = {"video_metadata": [{"fps": 1.0, "total_num_frames": len(video), "frames_indices": list(range(len(video)))}],
                            "do_sample_frames": False, "cap_pixels_per_frame": False}
        native[kind] = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True, enable_thinking=False,
            processor_kwargs=video_kwargs,
        )
    if len(cases) != 20:
        raise RuntimeError(f"Expected exactly 20 audit cases, got {len(cases)}")
    return cases, manual_start, manual, native


def run(args):
    import torch

    from qev.model import QevModel

    start = time.perf_counter()
    source, target = Path(args.checkpoint).resolve(), Path(args.output_model).resolve()
    config = json.loads((source / "qev_config.json").read_text())
    if not target.exists():
        export_checkpoint(source, target, dtype=args.dtype, backbone=args.backbone)
    processor = load_processor(source / "processor", local_files_only=True)
    cases, manual_start, manual, native_inputs = build_probes(Path(args.data), processor, config)
    report = verify_export(source, target, cases, atol=args.atol, torch_device="cpu")
    # Compare each native sequence with the original full foundation exposed by
    # the Torch checkpoint's disabled-adapter mode, including its actual EOS.
    reference = QevModel.from_checkpoint(source, device="cpu", dtype=torch.float32)
    expected_native = {}
    for kind, inputs in native_inputs.items():
        output = reference.generate_native(**inputs, max_new_tokens=24, do_sample=False)
        expected_native[kind] = output[0, inputs["input_ids"].shape[1]:].tolist()
    del reference, output
    gc.collect()
    runtime = MLXRuntime.from_checkpoint(target)
    logits = runtime.predict_logits(cases)
    temperature = float(config.get("temperature", 1.0))
    manual_outputs = []
    for index, (state, question) in enumerate(manual):
        scores = logits[manual_start + index]
        manual_outputs.append({"state": state, **question,
                               "raw_probabilities": probabilities(scores).tolist(),
                               "calibrated_probabilities": probabilities(scores, temperature).tolist()})
    native = {}
    for kind, inputs in native_inputs.items():
        output = runtime.generate_native(**inputs, max_new_tokens=24)
        actual = output[0, inputs["input_ids"].shape[1]:].tolist()
        native[kind] = {"torch_tokens": expected_native[kind], "mlx_tokens": actual,
                        "tokens_equal": actual == expected_native[kind],
                        "torch_text": processor.tokenizer.decode(expected_native[kind], skip_special_tokens=True),
                        "mlx_text": processor.tokenizer.decode(actual, skip_special_tokens=True)}
    native_passed = all(probe["tokens_equal"] for probe in native.values())
    report.update(
        checkpoint=str(source), output_model=str(target), completed_epochs=config.get("completed_epochs"),
        validation_scope="Numerical integration and four native-generation smoke cases; not a statistical accuracy benchmark",
        text_cases=17, media_cases=3, dataset_cases=14,
        media_validation_scope="No media supervision was used; numerical parity and input plumbing only, no multimodal decision accuracy claim",
        manual_instruction_probabilities=manual_outputs,
        native_generation=native, native_tokens_equal=native_passed,
        passed=bool(report["passed"] and native_passed), elapsed_seconds=time.perf_counter() - start,
        cases=[{"source": enc.get("source", enc.get("probe_id")), "language": enc.get("language"),
                "record_id": enc.get("record_id"), "question_id": enc.get("question_id"),
                "tokens": len(enc["ids"]), "options": len(enc["option_positions"])} for enc in cases],
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(report_path), "passed": report["passed"],
                      "max_probability_error": report["max_probability_error"],
                      "calibrated_max_probability_error": report["calibrated"]["max_probability_error"],
                      "native_tokens_equal": native_passed}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--backbone", default="runs/base-mlx")
    parser.add_argument("--data", default="data/v1")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--atol", type=float, default=0.002)
    args = parser.parse_args()
    if not run(args)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
