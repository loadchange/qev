"""TypeSafe-shaped typed decisions, with probabilities supplied by a local model.

Rendering and confidence conventions are adapted from jaredpalmer/kev (Apache-2.0).
The score confidence statistic is an approximation, not TypeSafe's unpublished
formula. Neither a confidence value nor an uncalibrated softmax implies accuracy.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_OPTIONS = 255
JSONContent = JsonValue


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instructions: JSONContent = None


class Noul(QuestionBase):
    type: Literal["noul"]
    criteria: dict[str, JSONContent] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        if self.criteria is not None and set(self.criteria) - {"false", "true"}:
            raise ValueError("noul criteria may contain only 'false' and 'true'")
        return self


class Choice(QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


class Score(QuestionBase):
    type: Literal["score"]
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    state: JSONContent
    model: str = "qev-latest"
    questions: dict[str, Question] = Field(min_length=1)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str | list[JSONContent]


class ChatCompletionRequest(BaseModel):
    """OpenAI-shaped non-streaming native generation, with inline media only."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = "qev-native"
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: Literal[False] = False
    enable_thinking: bool = False

    @model_validator(mode="after")
    def validate_budget(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("set max_tokens or max_completion_tokens, not both")
        return self

    @property
    def generation_budget(self):
        return self.max_completion_tokens or self.max_tokens or 256


@dataclass
class MediaBudget:
    """Budgets apply across every image and video frame in the complete request."""
    max_image_bytes: int = 2 * 1024 * 1024
    max_total_bytes: int = 8 * 1024 * 1024
    max_frames: int = 8
    max_image_pixels: int = 4_000_000
    max_total_pixels: int = 8_000_000
    total_bytes: int = 0
    total_pixels: int = 0
    frames: int = 0


@dataclass
class DecodedContent:
    text: str
    content: list[dict]
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    video_fps: list[float] = field(default_factory=list)

    @property
    def has_media(self):
        return bool(self.images or self.videos)


def state_content(state: JSONContent) -> list | None:
    """Recognize an explicit content wrapper or an OpenAI text/image content array.

    Other JSON objects/arrays retain the historical TypeSafe rendering behavior.
    Video is Qev's explicit sampled-frame extension, not a URL fetch facility.
    """
    if isinstance(state, dict) and set(state) == {"content"} and isinstance(state["content"], list):
        return state["content"]
    if (isinstance(state, list) and state and all(isinstance(item, dict) for item in state)
        and any(isinstance(item.get("type"), str)
                and item["type"] in {"text", "image_url", "video", "video_url", "audio_url", "input_audio"}
                for item in state)):
        return state
    return None


def _content_text(content: list) -> str:
    parts, image_index, video_index = [], 0, 0
    for item in content:
        if not isinstance(item, dict):
            # Protocol validation consistently maps to HTTP 422, like schema errors.
            raise ValueError("content items must be objects")  # noqa: TRY004
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif item.get("type") == "image_url":
            image_index += 1
            parts.append(f"[Attached image {image_index}]")
        elif item.get("type") == "video":
            video_index += 1
            parts.append(f"[Attached video {video_index}]")
        else:
            raise ValueError("supported content types are text, image_url (inline data URL), and video (inline frames)")
    return "\n".join(parts)


def render_state(state: JSONContent) -> str:
    content = state_content(state)
    return render(state) if content is None else _content_text(content)


def decode_image_data_url(value: str, budget: MediaBudget):
    """Decode bounded PNG/JPEG/WebP bytes without ever opening a path or URL."""
    from PIL import Image, UnidentifiedImageError
    if not isinstance(value, str) or len(value) > 4 * ((budget.max_image_bytes + 2) // 3) + 64:
        raise ValueError("image data URL exceeds the per-image byte limit")
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", value)
    if not match:
        raise ValueError("images must be inline base64 PNG/JPEG/WebP data URLs; paths and remote URLs are not supported")
    try:
        body = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid image base64") from exc
    if len(body) > budget.max_image_bytes or budget.total_bytes + len(body) > budget.max_total_bytes:
        raise ValueError("decoded media exceeds the request byte budget")
    if budget.frames + 1 > budget.max_frames:
        raise ValueError("too many images/video frames in one request")
    try:
        with Image.open(io.BytesIO(body)) as image:
            expected = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[match[1]]
            if image.format != expected:
                raise ValueError("image MIME type does not match its bytes")
            pixels = image.width * image.height
            if pixels > budget.max_image_pixels or budget.total_pixels + pixels > budget.max_total_pixels:
                raise ValueError("decoded image dimensions exceed the pixel budget")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated images must be supplied as explicit video frames")
            decoded = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid or oversized image") from exc
    budget.frames += 1
    budget.total_bytes += len(body)
    budget.total_pixels += pixels
    return decoded


def decode_content(content: list, budget: MediaBudget | None = None) -> DecodedContent:
    import numpy as np
    budget = budget or MediaBudget()
    text = _content_text(content)
    decoded = DecodedContent(text=text, content=[])
    for item in content:
        kind = item["type"]
        if kind == "text":
            decoded.content.append({"type": "text", "text": item["text"]})
        elif kind == "image_url":
            spec = item.get("image_url")
            if not isinstance(spec, dict) or not isinstance(spec.get("url"), str):
                raise ValueError("image_url must contain a string url field")
            image = decode_image_data_url(spec["url"], budget)
            decoded.images.append(image)
            decoded.content.append({"type": "image", "image": image})
        else:
            frames, fps = item.get("frames"), item.get("fps", 1.0)
            if not isinstance(frames, list) or not 1 <= len(frames) <= budget.max_frames:
                raise ValueError("video must contain 1..8 inline image data URL frames")
            if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 60:
                raise ValueError("video fps must be finite and between zero and 60")
            images = [decode_image_data_url(frame, budget) for frame in frames]
            if any(image.size != images[0].size for image in images):
                raise ValueError("video frames must all have the same dimensions")
            video = np.stack([np.asarray(image) for image in images])
            decoded.videos.append(video)
            decoded.video_fps.append(float(fps))
            decoded.content.append({"type": "video", "video": video})
    return decoded


def decode_state(state: JSONContent, budget: MediaBudget | None = None) -> DecodedContent:
    content = state_content(state)
    if content is None:
        text = render(state)
        return DecodedContent(text=text, content=[{"type": "text", "text": text}])
    return decode_content(content, budget)


def decode_question_options(question: Question, budget: MediaBudget) -> list[DecodedContent]:
    """Decode candidate images in answer-key order using the whole request budget.

    Explicit content wrappers/arrays use the same inline-image protocol as state.
    Ordinary strings and JSON descriptions keep their existing text rendering.
    Candidate video is not part of this protocol; state video remains supported.
    """
    if question.type == "noul":
        criteria = question.criteria or {}
        values = [criteria.get("false"), criteria.get("true")]
    elif question.type == "choice":
        values = list(question.criteria.values())
    else:
        values = question.criteria
    decoded = []
    for value in values:
        content = state_content(value)
        if content is not None and any(isinstance(item, dict) and item.get("type") == "video" for item in content):
            raise ValueError("candidate content supports text and inline images; video belongs in state")
        decoded.append(decode_state(value, budget))
    return decoded


def decode_chat_messages(messages: list[ChatMessage]) -> tuple[list[dict], MediaBudget, list[float]]:
    budget, result, video_fps = MediaBudget(), [], []
    for message in messages:
        decoded = (DecodedContent(text=message.content, content=[{"type": "text", "text": message.content}])
                   if isinstance(message.content, str) else decode_content(message.content, budget))
        result.append({"role": message.role, "content": decoded.content})
        video_fps.extend(decoded.video_fps)
    return result, budget, video_fps


def render(value: JSONContent, indent: int = 0) -> str:
    """Deterministically render JSON, preserving field names and array order."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite numbers are not JSON content")
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    if isinstance(value, dict):
        return "\n".join(
            f"{pad}{key}:\n{render(item, indent + 1)}"
            if isinstance(item, (dict, list))
            else f"{pad}{key}: {render(item)}"
            for key, item in value.items()
        )
    raise TypeError(f"unsupported JSON content: {type(value).__name__}")


def option_text(name: str, description: JSONContent) -> str:
    return name if description is None or description == "" else f"{name}: {render_state(description)}"


def to_record(request: SystemOneRequest | dict) -> tuple[dict, list[dict]]:
    """Render a request for the encoder, plus metadata for mapping results back."""
    if not isinstance(request, SystemOneRequest):
        request = SystemOneRequest.model_validate(request)
    questions, metadata = [], []
    for qid, question in request.questions.items():
        meta = {"id": qid, "type": question.type}
        if question.type == "noul":
            criteria = question.criteria or {}
            keys = ["false", "true"]
            options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
        elif question.type == "choice":
            keys = list(question.criteria)
            options = [option_text(key, value) for key, value in question.criteria.items()]
        else:
            options = [render_state(value) for value in question.criteria]
            keys = [str(i) for i in range(len(options))]
            meta["legend"] = dict(zip(keys, options))
        meta["keys"] = keys
        metadata.append(meta)
        questions.append({"instr": render(question.instructions), "options": options,
                          "qtype": question.type, "qid": qid, "keys": keys})
    return {"state": render_state(request.state), "questions": questions}, metadata


def choice_confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    return 1.0 if count == 1 else max(0.0, min(1.0, (max(probabilities) - 1 / count) / (1 - 1 / count)))


def score_confidence(probabilities: list[float]) -> float:
    """Approximation: 1 minus normalized expected distance from the modal level."""
    mode = max(range(len(probabilities)), key=probabilities.__getitem__)
    return max(0.0, min(1.0, 1 - sum(p * abs(i - mode) for i, p in enumerate(probabilities)) / (len(probabilities) - 1)))


def to_answers(probabilities: list[list[float]], metadata: list[dict]) -> dict[str, Any]:
    """Map softmax values to typed answers; do not round or claim calibration.

    Tiny floating point summation errors are normalized. Invalid lengths, NaNs,
    negative values, or distributions materially different from one are rejected.
    """
    if len(probabilities) != len(metadata):
        raise ValueError("one probability vector is required per question")
    answers = {}
    for values, meta in zip(probabilities, metadata):
        p = [float(value) for value in values]
        expected = len(meta["keys"])
        if len(p) != expected or not p:
            raise ValueError(f"wrong number of probabilities for {meta['id']}")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p):
            raise ValueError("probabilities must be finite and between zero and one")
        total = math.fsum(p)
        if not math.isclose(total, 1.0, abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError("probabilities must sum to one")
        p = [value / total for value in p]
        if meta["id"] in answers:
            raise ValueError("duplicate question metadata")
        if meta["type"] == "noul":
            answers[meta["id"]] = {"type": "noul", "noul": p[1]}
        elif meta["type"] == "choice":
            chosen = max(range(len(p)), key=p.__getitem__)
            answers[meta["id"]] = {"type": "choice", "choice": meta["keys"][chosen],
                "confidence": choice_confidence(p), "probabilities": dict(zip(meta["keys"], p))}
        elif meta["type"] == "score":
            answers[meta["id"]] = {"type": "score", "score": sum(i * value for i, value in enumerate(p)),
                "confidence": score_confidence(p), "legend": meta["legend"],
                "probabilities": dict(zip(meta["keys"], p))}
        else:
            raise ValueError(f"unsupported answer type {meta['type']}")
    return answers
