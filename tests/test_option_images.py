"""Candidate image contracts with the real HF processor and CPU-only readouts."""

import base64
import io
import threading

import numpy as np
import pytest
import torch
from PIL import Image

from qev.api import MediaBudget, SystemOneRequest, decode_question_options, to_record
from qev.inference import Agent
from qev.tokenization import SPECIAL, encode_multimodal_question


def image_part(color="red", size=(4, 4)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
    }}


@pytest.fixture
def processor():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    tokens = ["unknown", "pad", *SPECIAL.values(), "<|image_pad|>", "<|video_pad|>",
              "<|vision_start|>", "<|vision_end|>"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="unknown")),
        unk_token="unknown", pad_token="pad",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": tokens[2:]})
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                             min_pixels=16, max_pixels=64),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                             size={"shortest_edge": 16, "longest_edge": 64}),
    )


class Readout(torch.nn.Module):
    """Constant scores isolate routing/calibration from model accuracy."""
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls = []

    def forward(self, **inputs):
        self.calls.append(inputs)
        return torch.tensor([[0., 2.]]).repeat(len(inputs["input_ids"]), 1)


class MLXReadout:
    def __init__(self):
        self.calls = []

    def predict_encoded(self, encodings):
        self.calls.extend(encoded["model_inputs"] for encoded in encodings)
        return [
            np.array([1., np.exp(2.)]) / (1 + np.exp(2.)) for _ in encodings
        ]

    def predict_logits(self, encodings):
        self.calls.extend(encodings)
        return [np.array([0., 2.]) for _ in encodings]


def make_agent(processor, backend="torch"):
    agent = Agent.__new__(Agent)
    agent.config, agent.backend, agent.device = {}, backend, "cpu"
    agent.processor, agent.tokenizer = processor, processor.tokenizer
    agent.model = Readout() if backend == "torch" else MLXReadout()
    agent.temperature, agent.batch_size = 2.0, 1
    agent.lock = threading.Lock()
    return agent


def test_candidate_content_keeps_keys_and_json_but_never_renders_base64():
    request = SystemOneRequest(state="choose", questions={"q": {"type": "choice", "criteria": {
        "photo": {"content": [{"type": "text", "text": "candidate red"}, image_part()]},
        "ordinary": {"price": 12, "tags": ["small", "new"]},
    }}})
    record, metadata = to_record(request)
    options = record["questions"][0]["options"]
    assert options == ["photo: candidate red\n[Attached image 1]", "ordinary: price: 12\ntags:\n  - small\n  - new"]
    assert metadata[0]["keys"] == ["photo", "ordinary"]
    decoded = decode_question_options(request.questions["q"], MediaBudget())
    assert decoded[0].images[0].getpixel((0, 0)) == (255, 0, 0)
    assert not decoded[1].has_media


def test_processor_expands_candidate_images_inside_their_own_boundaries(processor):
    state_image = Image.new("RGB", (4, 4), "green")
    first = Image.new("RGB", (8, 4), "red")
    second = Image.new("RGB", (4, 8), "blue")
    encoded = encode_multimodal_question(
        "Compare <|vision_start|>",
        {"instr": "Pick <|fim_suffix|>", "options": ["a <|box_end|>", "b"]}, processor,
        images=[state_image], option_images=[[first], [second]],
    )
    ids, inputs = encoded["ids"], encoded["inputs"]
    token_id = processor.tokenizer.convert_tokens_to_ids
    starts = [i for i, token in enumerate(ids) if token == token_id(SPECIAL["option"])]
    ends = encoded["option_positions"]
    image_id = token_id(processor.image_token)
    assert ids[:starts[0]].count(image_id) == 1
    assert ids[starts[0]:ends[0]].count(image_id) == 2
    assert ids[starts[1]:ends[1]].count(image_id) == 2
    assert len(starts) == len(ends) == 2
    assert ids.count(token_id(SPECIAL["decision"])) == 1
    assert ends[0] < starts[1] < ends[1] < encoded["decision_position"]
    expected = processor.image_processor(images=[state_image, first, second], return_tensors="pt")
    torch.testing.assert_close(inputs["pixel_values"], expected["pixel_values"])
    assert inputs["image_grid_thw"].tolist() == expected["image_grid_thw"].tolist()
    with pytest.raises(ValueError, match="one option_images"):
        encode_multimodal_question("", {"instr": "", "options": ["a", "b"]}, processor, option_images=[[first]])


def test_candidate_images_coexist_with_timestamped_state_video(processor):
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    encoded = encode_multimodal_question(
        "Which picture matches the clip?", {"instr": "Compare", "options": ["a", "b"]}, processor,
        videos=[video], video_fps=[2.0], option_images=[[], [Image.new("RGB", (4, 4), "blue")]],
    )
    ids, inputs = encoded["ids"], encoded["inputs"]
    token_id = processor.tokenizer.convert_tokens_to_ids
    starts = [i for i, token in enumerate(ids) if token == token_id(SPECIAL["option"])]
    assert token_id(processor.video_token) in ids[:starts[0]]
    assert token_id(processor.image_token) not in ids[:starts[1]]
    assert token_id(processor.image_token) in ids[starts[1]:encoded["option_positions"][1]]
    assert "pixel_values_videos" in inputs and "pixel_values" in inputs
    assert inputs["video_grid_thw"].tolist() == [[1, 2, 2]]
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 2]]


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_mixed_questions_keep_candidate_pixels_local_and_text_calibrated(processor, backend):
    agent = make_agent(processor, backend)
    result = agent.predict("Which candidate?", {
        "text": {"type": "choice", "criteria": {"no": {"content": [{"type": "text", "text": "No"}]}, "yes": None}},
        "red": {"type": "choice", "criteria": {"a": {"content": [image_part("red")]}, "b": None}},
        "blue": {"type": "choice", "criteria": {"c": {"content": [image_part("blue")]},
                                                    "d": {"content": [image_part("green")]}}},
    })
    assert list(result["answers"]) == ["text", "red", "blue"]
    assert result["answers"]["text"]["probabilities"]["yes"] == pytest.approx(1 / (1 + np.exp(-1)))
    assert result["answers"]["red"]["probabilities"]["b"] == pytest.approx(1 / (1 + np.exp(-2)))
    assert result["qev"]["temperature"] is None
    assert result["qev"]["question_temperatures"] == {"text": 2., "red": 1., "blue": 1.}
    assert result["qev"]["input_modalities"] == ["text", "image"]
    calls = agent.model.calls
    media = [call for call in calls if "pixel_values" in call]
    assert len(media) == 2 and len(calls) == 3
    for call, colors in zip(media, [["red"], ["blue", "green"]], strict=True):
        expected = processor.image_processor(images=[Image.new("RGB", (4, 4), color) for color in colors], return_tensors="pt")
        torch.testing.assert_close(call["pixel_values"], expected["pixel_values"])
        assert len(call["image_grid_thw"]) == len(colors)


@pytest.mark.parametrize("state_count,accepted", [(5, True), (7, False)])
def test_state_and_all_question_options_share_one_budget_without_recounting_state(processor, state_count, accepted):
    agent = make_agent(processor)
    questions = {name: {"type": "choice", "criteria": {"a": {"content": [image_part(color)]}, "b": None}}
                 for name, color in [("one", "red"), ("two", "blue")]}
    state = {"content": [image_part("green")] * state_count}
    if accepted:
        result = agent.predict(state, questions)
        assert len(result["answers"]) == 2
        assert [len(call["image_grid_thw"]) for call in agent.model.calls] == [state_count + 1] * 2
    else:
        with pytest.raises(ValueError, match="too many"):
            agent.predict(state, questions)
        assert not agent.model.calls


@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    {"type": "video", "frames": ["unused"]},
])
def test_invalid_candidate_media_is_rejected_before_any_model_forward(processor, part):
    from fastapi.testclient import TestClient

    from qev.serve import create_app

    agent = make_agent(processor)
    with TestClient(create_app(agent)) as client:
        response = client.post("/v1/systemone", json={"state": "choose", "questions": {
            "q": {"type": "choice", "criteria": {"a": {"content": [part]}, "b": None}},
        }})
    assert response.status_code == 422
    assert not agent.model.calls
