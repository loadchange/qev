import base64
import io

import pytest
from PIL import Image
from pydantic import ValidationError

from qev.api import (
    ChatCompletionRequest,
    MediaBudget,
    decode_chat_messages,
    decode_content,
    decode_image_data_url,
    decode_state,
    to_record,
)


def image_url(color="red", size=(8, 8)):
    stream = io.BytesIO()
    Image.new("RGB", size, color=color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def test_string_and_unstructured_json_are_backward_compatible():
    assert decode_state("hello").text == "hello"
    assert decode_state({"customer": "Mira", "count": 2}).text == "customer: Mira\ncount: 2"
    assert not decode_state([{"role": "user", "content": "hello"}]).has_media


@pytest.mark.parametrize("value", [[], {}, ["text"], {"name": "image_url"}])
def test_unstructured_json_type_values_remain_ordinary_data(value):
    state = [{"type": value}]
    decoded = decode_state(state)
    assert not decoded.has_media
    record, _ = to_record({"state": state, "questions": {"q": {"type": "noul"}}})
    assert record["state"] == decoded.text


def test_image_decode_and_decision_render_do_not_tokenize_base64():
    state = [{"type": "text", "text": "What color?"}, {"type": "image_url", "image_url": {"url": image_url()}}]
    decoded = decode_state(state)
    assert decoded.has_media and decoded.images[0].size == (8, 8)
    assert decoded.images[0].getpixel((0, 0)) == (255, 0, 0)
    record, _ = to_record({"state": state, "questions": {"q": {"type": "noul"}}})
    assert "base64" not in record["state"]
    assert "[Attached image 1]" in record["state"]
    assert record["questions"][0]["instr"] == ""


@pytest.mark.parametrize("url", ["https://example.com/pic.png", "/etc/passwd", "file:///etc/passwd", "data:image/png;base64,###", "data:image/jpeg;base64,AAAA"])
def test_remote_paths_and_invalid_images_are_rejected(url):
    with pytest.raises(ValueError):
        decode_image_data_url(url, MediaBudget())


def test_media_budgets_cover_every_frame_and_message():
    url = image_url()
    with pytest.raises(ValueError, match="pixel budget"):
        decode_image_data_url(url, MediaBudget(max_image_pixels=32))
    with pytest.raises(ValueError, match="byte limit"):
        decode_image_data_url(url, MediaBudget(max_image_bytes=1))
    with pytest.raises(ValueError, match="too many"):
        decode_content([{"type": "image_url", "image_url": {"url": url}}] * 9)
    req = ChatCompletionRequest(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}] * 5},
                                          {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}] * 4}])
    with pytest.raises(ValueError, match="too many"):
        decode_chat_messages(req.messages)


def test_video_frames_preserve_order_fps_and_pixels():
    decoded = decode_state({"content": [{"type": "video", "frames": [image_url("red"), image_url("blue")], "fps": 2.0}]})
    assert decoded.video_fps == [2.0]
    assert decoded.videos[0].shape == (2, 8, 8, 3)
    assert decoded.videos[0][0, 0, 0].tolist() == [255, 0, 0]
    assert decoded.videos[0][1, 0, 0].tolist() == [0, 0, 255]
    with pytest.raises(ValueError, match="same dimensions"):
        decode_content([{"type": "video", "frames": [image_url(), image_url(size=(4, 4))]}])
    with pytest.raises(ValueError, match="fps"):
        decode_content([{"type": "video", "frames": [image_url()], "fps": float("nan")}])


def test_unsupported_content_or_streaming_fails_explicitly():
    with pytest.raises(ValueError, match="supported content"):
        decode_content([{"type": "video_url", "video_url": {"url": "https://example.com/x.mp4"}}])
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=8, max_completion_tokens=8)
