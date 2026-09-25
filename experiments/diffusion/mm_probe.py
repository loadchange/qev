"""Zero-shot image decision probe on A-OKVQA validation (4-way multiple choice).

Every Qev-style checkpoint in this experiment was trained on text decisions
only, so this measures how far the decision adapter and pointer head transfer
to images, not tuned visual accuracy. Each question is scored three ways:

``decision``         the checkpoint's pointer head with the image in the state
``decision_no_image`` the same question with no image (how much the image helps)
``letter``           the frozen foundation (adapter disabled) reading the next-token
                     probability of A/B/C/D after its own chat template
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diffusion import qevd

DATASET, REVISION = "HuggingFaceM4/A-OKVQA", "d1b0efa3a436e9101dfbde3752db7607da696c35"
FILE = "data/validation-00000-of-00001-b2bd0de231b6326a.parquet"
LETTERS = "ABCD"


def load_rows(limit):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    table = pq.read_table(hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION))
    rows = []
    for row in table.slice(0, limit).to_pylist():
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        rows.append({"id": row["question_id"], "image": image, "question": row["question"],
                     "choices": list(row["choices"]), "label": int(row["correct_choice_idx"])})
    return rows


class QevArm:
    """The unchanged Qwen3.5 Qev checkpoint (Torch)."""

    def __init__(self, path, device):
        from qev.model import QevModel

        self.device = torch.device(device)
        self.model = QevModel.from_checkpoint(path, device=self.device)
        self.processor = self.model.get_processor()

    def decision(self, state, question, images):
        from qev.tokenization import encode_multimodal_question

        encoding = encode_multimodal_question(state, question, self.processor, images=images or None)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in encoding["inputs"].items()}
        return self.model(**inputs)[0].float().cpu().numpy()[:len(question["options"])]

    def letter_inputs(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                                  add_generation_prompt=True, enable_thinking=False)

    def foundation_logits(self, inputs):
        with self.model.backbone.disable_adapter():
            return self.model.foundation(**inputs).logits


class QevDArm:
    """A QevD checkpoint on LFM2.5-VL."""

    def __init__(self, path, device):
        self.device = torch.device(device)
        self.model = qevd.QevDModel.from_checkpoint(path, device=self.device)
        if self.model.config["kind"] != "lfm2_vl":
            raise ValueError("Only the multimodal LFM2-VL arm can read images")
        self.processor = qevd.load_processor(self.model.config["backbone"])
        self.ids = qevd.special_ids(self.processor.tokenizer, "lfm2")

    def decision(self, state, question, images):
        inputs = qevd.encode_multimodal_question(state, question, self.processor, self.ids, images=images)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)[0].float().cpu().numpy()[:len(question["options"])]

    def letter_inputs(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                                  add_generation_prompt=True)

    def foundation_logits(self, inputs):
        with self.model.text_model.disable_adapter():
            return self.model.foundation(**inputs).logits


def evaluate(arm, rows, autocast):
    tokenizer = arm.processor.tokenizer
    letter_ids = [tokenizer.convert_tokens_to_ids(letter) for letter in LETTERS]
    if len(set(letter_ids)) != 4 or tokenizer.unk_token_id in letter_ids:
        raise ValueError("Letters A-D must be distinct single tokens")
    counts = {"decision": 0, "decision_no_image": 0, "letter": 0}
    started, per_question = time.perf_counter(), []
    for row in rows:
        question = {"instr": row["question"], "options": row["choices"]}
        with torch.inference_mode(), autocast:
            t0 = time.perf_counter()
            with_image = arm.decision("[Attached image 1]", question, [row["image"]])
            per_question.append((time.perf_counter() - t0) * 1000)
            without = arm.decision("", question, [])
            prompt = row["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, row["choices"]))
            messages = [{"role": "user", "content": [{"type": "image", "image": row["image"]},
                                                     {"type": "text", "text": prompt + "\nAnswer with the option's letter."}]}]
            inputs = {k: v.to(arm.device) if hasattr(v, "to") else v for k, v in arm.letter_inputs(messages).items()}
            letters = arm.foundation_logits(inputs)[0, -1, letter_ids].float().cpu().numpy()
        counts["decision"] += int(np.argmax(with_image) == row["label"])
        counts["decision_no_image"] += int(np.argmax(without) == row["label"])
        counts["letter"] += int(np.argmax(letters) == row["label"])
    n = len(rows)
    return {"n": n, **{f"{k}_accuracy": v / n for k, v in counts.items()},
            "chance": 0.25, "decision_ms_p50": float(np.median(per_question)),
            "seconds": time.perf_counter() - started}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, help="label=kind:path, kind is qev or qevd")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    rows = load_rows(args.limit)
    results = {"dataset": DATASET, "revision": REVISION, "split": "validation", "rows": len(rows),
               "scope": "Zero-shot transfer of text-trained decision heads; letter = frozen foundation via chat template.",
               "arms": {}}
    for spec in args.arm:
        label, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        arm = (QevArm if kind == "qev" else QevDArm)(path, args.device)
        autocast = torch.autocast(device_type=arm.device.type, dtype=torch.bfloat16, enabled=arm.device.type == "cuda")
        results["arms"][label] = {"kind": kind, "path": path, **evaluate(arm, rows, autocast)}
        print("MM_PROBE", label, json.dumps(results["arms"][label]), flush=True)
        del arm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
