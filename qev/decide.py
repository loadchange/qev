"""Build decision requests from command-line arguments and render the answers."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

# Mirrors qev.api.MediaBudget so CLI requests are accepted as sent.
MAX_ITEMS, MAX_IMAGE_PIXELS, MAX_TOTAL_PIXELS = 8, 4_000_000, 8_000_000
MAX_IMAGE_BYTES, MAX_TOTAL_BYTES = 2 * 1024 * 1024, 8 * 1024 * 1024
_BARS = " ▏▎▍▌▋▊▉█"


def _encode(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return buffer.getvalue()


def _data_url(raw, fmt):
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(raw).decode('ascii')}"


def media_urls(paths, *, frames=False, notes=None):
    """Inline data URLs within the API's per-image and per-request budgets.

    Photos are rotated by their EXIF orientation and downscaled or re-encoded
    as JPEG only when needed; video frames are all resized to the first one.
    """
    from PIL import Image, ImageOps

    if not paths:
        return []
    pixel_cap = min(MAX_IMAGE_PIXELS, MAX_TOTAL_PIXELS // len(paths))
    byte_cap = min(MAX_IMAGE_BYTES, MAX_TOTAL_BYTES // len(paths))
    urls, first_size = [], None
    for path in paths:
        raw = Path(path).expanduser().read_bytes()
        with Image.open(io.BytesIO(raw)) as source:
            fmt, size = source.format, source.size
            orientation = source.getexif().get(0x0112, 1)
            animated = getattr(source, "n_frames", 1) > 1
            target = first_size if frames and first_size else size
            if (fmt in ("PNG", "JPEG", "WEBP") and not animated and orientation == 1 and target == size
                    and size[0] * size[1] <= pixel_cap and len(raw) <= byte_cap):
                urls.append(_data_url(raw, fmt))
                first_size = first_size or size
                continue
            image = ImageOps.exif_transpose(source).convert("RGB")
        if frames and first_size:
            image = image.resize(first_size, Image.Resampling.LANCZOS)
        elif image.width * image.height > pixel_cap:
            scale = math.sqrt(pixel_cap / (image.width * image.height))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                                 Image.Resampling.LANCZOS)
        encoded = _encode(image, "JPEG")
        while len(encoded) > byte_cap and min(image.size) > 64:
            image = image.resize((image.width * 3 // 4, image.height * 3 // 4), Image.Resampling.LANCZOS)
            encoded = _encode(image, "JPEG")
        first_size = first_size or image.size
        if notes is not None:
            notes.append(f"{path}: sent as JPEG {image.width}x{image.height} (was {fmt} {size[0]}x{size[1]})")
        urls.append(_data_url(encoded, "JPEG"))
    return urls


def build_state(text, *, images=(), frames=(), fps=1.0, state_json=None, notes=None):
    if len(images) + len(frames) > MAX_ITEMS:
        raise ValueError(f"At most {MAX_ITEMS} images and video frames per request")
    if state_json is not None:
        if images or frames:
            raise ValueError("--state-json cannot be combined with --image or --frame")
        return state_json
    if not images and not frames:
        return text
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "image_url", "image_url": {"url": url}}
                for url in media_urls(list(images), notes=notes)]
    if frames:
        content.append({"type": "video", "frames": media_urls(list(frames), frames=True, notes=notes), "fps": fps})
    return {"content": content}


def build_question(kind, values=(), *, instructions=None, true=None, false=None):
    if kind == "choice":
        criteria = {}
        for value in values:
            key, _, description = value.partition("=")
            key = key.strip()
            if not key or key in criteria:
                raise ValueError(f"Choice keys must be unique and nonempty: {value!r}")
            criteria[key] = description.strip() or None
        return {"type": "choice", "instructions": instructions, "criteria": criteria}
    if kind == "noul":
        question = {"type": "noul", "instructions": instructions}
        criteria = {key: value for key, value in (("true", true), ("false", false)) if value}
        if criteria:
            question["criteria"] = criteria
        return question
    if kind == "score":
        if len(values) < 2:
            raise ValueError("--score needs at least two ordered levels")
        return {"type": "score", "instructions": instructions, "criteria": list(values)}
    raise ValueError("Choose a question type: --choice, --noul or --score")


def answer_value(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return "yes" if answer["noul"] >= 0.5 else "no"
    return f"{answer['score']:.3f}"


def _bar(fraction, width=24):
    cells = max(0.0, min(1.0, fraction)) * width
    full = int(cells)
    rest = _BARS[int((cells - full) * 8)] if full < width else ""
    return ("█" * full + rest).rstrip()


def render_text(response, *, color=False, footer=None, limit=12):
    bold = (lambda s: f"\033[1m{s}\033[0m") if color else (lambda s: s)
    dim = (lambda s: f"\033[2m{s}\033[0m") if color else (lambda s: s)
    lines = []
    for qid, answer in response["answers"].items():
        if answer["type"] == "noul":
            p = answer["noul"]
            lines.append(f"{qid}: {bold(answer_value(answer))}  {dim(f'yes/no · p(yes) = {p:.3f}')}")
            rows = [("yes", p), ("no", 1 - p)]
        elif answer["type"] == "choice":
            lines.append(f"{qid}: {bold(answer['choice'])}  {dim(f'choice · confidence {answer['confidence']:.2f}')}")
            rows = sorted(answer["probabilities"].items(), key=lambda item: -item[1])
        else:
            lines.append(f"{qid}: {bold(answer_value(answer))}  "
                         f"{dim(f'score 0..{len(answer['legend']) - 1} · confidence {answer['confidence']:.2f}')}")
            rows = [(f"{key} {answer['legend'][key]}", p) for key, p in answer["probabilities"].items()]
        width = max(len(label) for label, _ in rows[:limit])
        for label, p in rows[:limit]:
            lines.append(f"  {label.ljust(width)}  {p:.3f}  {_bar(p)}")
        if len(rows) > limit:
            lines.append(dim(f"  … {len(rows) - limit} more, {sum(p for _, p in rows[limit:]):.3f} total"))
    if footer:
        lines.append(dim(footer))
    return "\n".join(lines)


def render_quiet(response):
    answers = response["answers"]
    if len(answers) == 1:
        return answer_value(next(iter(answers.values())))
    return "\n".join(f"{qid}={answer_value(answer)}" for qid, answer in answers.items())


def render_json(response):
    return json.dumps(response, ensure_ascii=False, indent=2)
