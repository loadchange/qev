"""Compare native text/image/video generation before and after decision continuation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from qev.evaluation import prediction_precision, write_json
from qev.model import QevModel
from scripts.validate_native import generate, image_fixture, video_fixture


def native_cases(checkpoint, device):
    model = QevModel.from_checkpoint(checkpoint, device=device)
    processor = model.get_processor()
    video = video_fixture()
    cases = {
        "text": generate(model, processor, [{"type": "text", "text": "What is 2 + 3? Answer with one number."}], max_new_tokens=12),
        "image": generate(model, processor, [
            {"type": "image", "image": image_fixture()},
            {"type": "text", "text": "What color is the shape in the image? Answer in one word."}], max_new_tokens=16),
        "video": generate(model, processor, [
            {"type": "video", "video": video},
            {"type": "text", "text": "Describe the motion of the blue circle in this video in one short sentence."}],
            max_new_tokens=48, video_metadata=[{"fps": 2., "total_num_frames": len(video), "frames_indices": list(range(len(video)))}]),
    }
    result = {"checkpoint": str(checkpoint), "precision": prediction_precision(model), "cases": cases,
              "base_revision": model.config["base_revision"],
              "native_generation": model.config["native_generation"],
              "vision_and_lm_head_frozen": not any(p.requires_grad for module in
                  (model.foundation.model.visual, model.foundation.lm_head) for p in module.parameters())}
    del model, processor
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    parent = native_cases(Path(args.parent), args.device)
    trained = native_cases(Path(args.checkpoint), args.device)
    integrity = json.loads((Path(args.checkpoint) / "base_integrity.json").read_text())
    matching = {name: all(parent["cases"][name][key] == trained["cases"][name][key]
                         for key in ("token_ids", "input_tokens", "processor_fields")) for name in parent["cases"]}
    digests = {label: {str(file): hashlib.sha256((Path(folder) / file).read_bytes()).hexdigest()
        for file in ("adapter/adapter_model.safetensors", "pointer.safetensors")}
        for label, folder in (("parent", args.parent), ("trained", args.checkpoint))}
    passed = (all(matching.values()) and integrity["unchanged"] and
              parent["precision"] == trained["precision"] and
              parent["base_revision"] == trained["base_revision"] and
              trained["native_generation"] == "adapter_disabled" and trained["vision_and_lm_head_frozen"] and
              digests["parent"] != digests["trained"])
    result = {"passed": passed, "native_tokens_identical": matching, "base_integrity": integrity,
              "checkpoint_hashes": digests, "parent": parent, "trained": trained,
              "scope": "Same-device deterministic native generation with decision adapters disabled. Three finite probes plus frozen-weight integrity; not a general multimodal accuracy benchmark."}
    write_json(args.output, result)
    print("QEV_CONTINUATION_NATIVE", json.dumps({"passed": passed, "native_tokens_identical": matching}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
