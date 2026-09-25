"""Encode independent candidate questions without changing Qwen's vocabulary."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

SPECIAL = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "option": "<|box_start|>",
    "end_option": "<|box_end|>",
    "decision": "<|fim_suffix|>",
}
_SPECIAL_RE = re.compile(r"<\|([^<>\r\n]*?)\|>")


def user_tokens(tokenizer, text: str) -> list[int]:
    """Prevent user text from creating structural special tokens."""
    text = _SPECIAL_RE.sub(r"<¦\1¦>", str(text))
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def special_token_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for name, token in SPECIAL.items():
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id or encoded != [token_id]:
            raise ValueError(f"Tokenizer lacks the existing Qwen delimiter {token!r}")
        ids[name] = int(token_id)
    if len(set(ids.values())) != len(ids):
        raise ValueError("Structural delimiter IDs must be distinct")
    return ids


def encode_question(
    state: str,
    question: dict[str, Any],
    tok,
    max_length: int = 512,
    max_state: int = 320,
) -> dict[str, Any]:
    """Encode state + one question, preserving every option's boundary marker.

    State is the only automatically truncated content. Instructions and options
    are kept verbatim: if all candidates cannot fit, fail rather than silently
    changing a criterion (for example truncating its negation or numeric limit).
    ``max_state`` includes the leading state delimiter.
    """
    if max_length < 8 or not 1 <= max_state <= max_length:
        raise ValueError("Expected max_length >= 8 and 1 <= max_state <= max_length")
    options = question.get("options")
    if not isinstance(options, list) or not options or not all(isinstance(v, str) for v in options):
        raise ValueError("question.options must be a nonempty list of strings")
    if len(options) > 255:
        raise ValueError("At most 255 options are supported")
    instruction = question.get("instr")
    if not isinstance(instruction, str):
        raise ValueError("question.instr must be a string")  # noqa: TRY004 -- request validation uses ValueError
    special = special_token_ids(tok)
    instruction_ids = user_tokens(tok, instruction)
    spans = [user_tokens(tok, option) for option in options]
    # One state marker, question marker, two markers per option, decision marker.
    fixed = 3 + len(instruction_ids) + sum(len(span) + 2 for span in spans)
    if fixed > max_length:
        raise ValueError(
            f"Question and all {len(options)} options require {fixed} tokens before state; "
            f"increase max_length={max_length} or shorten the criteria"
        )
    state_ids = user_tokens(tok, state)
    state_budget = min(max_state - 1, max_length - fixed)
    ids = [special["state"], *state_ids[:state_budget], special["question"], *instruction_ids]
    positions = []
    for span in spans:
        ids.extend([special["option"], *span, special["end_option"]])
        positions.append(len(ids) - 1)
    ids.append(special["decision"])
    return {
        "ids": ids,
        "option_positions": positions,
        "decision_position": len(ids) - 1,
        "state_truncated": len(state_ids) > state_budget,
        "state_tokens": min(len(state_ids), state_budget),
    }


def collate_encodings(encodings, pad_token_id: int, device=None):
    """Right-pad independent rows; labels remain outside the model input dict."""
    import torch

    if not encodings:
        raise ValueError("Cannot collate an empty batch")
    if pad_token_id is None:
        raise ValueError("A pad_token_id is required")
    batch = len(encodings)
    length = max(len(enc["ids"]) for enc in encodings)
    count = max(len(enc["option_positions"]) for enc in encodings)
    if not length or not count:
        raise ValueError("Each encoding needs tokens and at least one option")
    inputs = {
        "input_ids": torch.full((batch, length), pad_token_id, dtype=torch.long, device=device),
        "attention_mask": torch.zeros((batch, length), dtype=torch.long, device=device),
        "option_positions": torch.zeros((batch, count), dtype=torch.long, device=device),
        "option_mask": torch.zeros((batch, count), dtype=torch.bool, device=device),
        "decision_positions": torch.empty(batch, dtype=torch.long, device=device),
    }
    for row, enc in enumerate(encodings):
        size, n = len(enc["ids"]), len(enc["option_positions"])
        if not size or not n:
            raise ValueError("Each encoding needs tokens and at least one option")
        inputs["input_ids"][row, :size] = torch.tensor(enc["ids"], dtype=torch.long, device=device)
        inputs["attention_mask"][row, :size] = 1
        inputs["option_positions"][row, :n] = torch.tensor(enc["option_positions"], dtype=torch.long, device=device)
        inputs["option_mask"][row, :n] = True
        inputs["decision_positions"][row] = enc["decision_position"]
    return inputs


def encode_multimodal_question(
    state: str,
    question: dict[str, Any],
    processor,
    *,
    images=None,
    videos=None,
    video_fps=None,
    option_images=None,
    max_length: int = 8192,
    max_state: int | None = None,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    """Process one independent image/video decision with the original processor.

    Inputs are decoded in-memory images and ``[frames, height, width, channels]``
    videos. No local paths or remote URLs are opened here. Submitted video frames
    are preserved, with explicit frame rates to produce correct timestamps.
    Media placeholders are expanded before locating option boundaries. Unlike
    text training, this path never silently truncates state or visual content.
    ``option_images`` contains one image list per candidate. Their placeholders
    stay inside that candidate's boundaries, after its text and before readout.

    The returned ``inputs`` dict can be passed directly to ``QevModel.forward``;
    the remaining fields are metadata, not model keyword arguments. Run each
    question independently; the text-only collator cannot combine pixel grids.
    ``return_tensors="np"`` keeps the MLX runtime free of torch.
    """
    import numpy as np

    if return_tensors not in ("pt", "np"):
        raise ValueError("return_tensors must be 'pt' or 'np'")
    tok = processor.tokenizer
    special = special_token_ids(tok)
    options, instruction = question.get("options"), question.get("instr")
    if not isinstance(instruction, str) or not isinstance(options, list) or not 1 <= len(options) <= 255:
        raise ValueError("Expected an instruction and 1..255 candidate options")
    if not all(isinstance(option, str) for option in options):
        raise ValueError("Candidate options must be strings")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    state_tokens = user_tokens(tok, state)
    if max_state is not None and (max_state < 1 or len(state_tokens) + 1 > max_state):
        raise ValueError("Multimodal state exceeds max_state; content was not truncated")
    images, videos = list(images or []), list(videos or [])
    if option_images is not None and len(option_images) != len(options):
        raise ValueError("Provide one option_images list per candidate")
    candidate_images = [list(group) for group in option_images] if option_images is not None else [[] for _ in options]
    all_images = [*images, *(image for group in candidate_images for image in group)]
    if any(isinstance(media, (str, bytes, Path)) for media in [*all_images, *videos]):
        raise ValueError("Media must be decoded in-memory objects, not paths or URLs")
    if videos and (video_fps is None or len(video_fps) != len(videos)):
        raise ValueError("Provide one explicit video_fps value per submitted video")
    metadata = []
    for video, fps in zip(videos, video_fps or []):
        if not hasattr(video, "shape") or len(video.shape) != 4 or len(video) == 0:
            raise ValueError("Each video must contain a nonempty [T,H,W,C] frame array")
        if not math.isfinite(float(fps)) or fps <= 0:
            raise ValueError("Video frame rates must be finite and positive")
        metadata.append({"fps": float(fps), "total_num_frames": len(video),
                         "frames_indices": list(range(len(video)))})
    clean = lambda text: _SPECIAL_RE.sub(r"<¦\1¦>", str(text))
    prompt = SPECIAL["state"] + clean(state)
    for _ in images:
        prompt += processor.vision_start_token + processor.image_token + processor.vision_end_token
    # Qwen3VLProcessor expands each video placeholder into timestamped frame
    # groups, each already wrapped in vision_start/vision_end tokens.
    prompt += processor.video_token * len(videos)
    prompt += SPECIAL["question"] + clean(instruction)
    for option, attached in zip(options, candidate_images, strict=True):
        prompt += SPECIAL["option"] + clean(option)
        for _ in attached:
            prompt += processor.vision_start_token + processor.image_token + processor.vision_end_token
        prompt += SPECIAL["end_option"]
    prompt += SPECIAL["decision"]
    kwargs = {"text": [prompt], "return_tensors": return_tensors, "add_special_tokens": False}
    if all_images:
        kwargs["images"] = all_images
    if videos:
        kwargs.update(videos=videos, video_metadata=metadata, do_sample_frames=False)
    processed = processor(**kwargs)
    inputs = dict(processed)
    ids = inputs["input_ids"][0].tolist()
    if len(ids) > max_length:
        raise ValueError(
            f"Expanded multimodal question needs {len(ids)} tokens, exceeding max_length={max_length}; "
            "no images, video frames, state text, or options were truncated"
        )
    positions = [i for i, token_id in enumerate(ids) if token_id == special["end_option"]]
    decisions = [i for i, token_id in enumerate(ids) if token_id == special["decision"]]
    if len(positions) != len(options) or decisions != [len(ids) - 1]:
        raise ValueError("Processor did not preserve the complete candidate structure")
    markers = {"option_positions": np.asarray([positions], dtype=np.int64),
               "option_mask": np.ones((1, len(positions)), dtype=bool),
               "decision_positions": np.asarray(decisions, dtype=np.int64)}
    if "attention_mask" not in inputs:
        markers["attention_mask"] = np.ones(np.asarray(inputs["input_ids"]).shape, dtype=np.int64)
    if return_tensors == "pt":
        import torch

        markers = {key: torch.from_numpy(value) for key, value in markers.items()}
    inputs.update(markers)
    return {"inputs": inputs, "ids": ids, "option_positions": positions,
            "decision_position": decisions[0], "state_truncated": False,
            "state_tokens": len(state_tokens)}
