import base64
import io
import json
import threading
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from qev.inference import Agent


def png_url():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def batch_decode(self, rows, **kwargs):
        self.decoded = rows
        return ["native answer"]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.processor = FakeProcessor()

    def get_processor(self):
        return self.processor

    def generate_native(self, **kwargs):
        self.native_kwargs = kwargs
        return torch.tensor([[1, 2, 77, 78]])

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.tensor([[0., 1.]])


def fake_agent():
    agent = Agent.__new__(Agent)
    agent.config, agent.backend, agent.device = {}, "torch", "cpu"
    agent.model, agent.processor = FakeModel(), None
    agent.temperature, agent.batch_size = 2.0, 1
    agent.lock = threading.Lock()
    return agent


@pytest.mark.parametrize("batch_options,expected_calls", [({}, [1, 1, 1]), ({"batch_size": 4}, [3])])
def test_torch_default_runs_questions_separately_and_explicit_batching_remains_available(
    tmp_path, monkeypatch, batch_options, expected_calls,
):
    from qev.model import QevModel

    class ShapeSensitiveModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.batch_calls = []

        def get_tokenizer(self):
            return SimpleNamespace(pad_token_id=0)

        def forward(self, input_ids, **kwargs):
            self.batch_calls.append(len(input_ids))
            logits = torch.zeros((len(input_ids), 2))
            # Represent padding-dependent low-precision arithmetic so the test
            # detects batching changes without requiring a CUDA checkpoint.
            logits[:, 1] = input_ids.shape[1] * 0.1
            return logits

    def encode(state, question, tokenizer, max_length, max_state):
        length = 8 if question["instr"] == "long" else 4
        return {"ids": [1] * length, "option_positions": [1, 2],
                "decision_position": length - 1, "state_truncated": False}

    model = ShapeSensitiveModel()
    monkeypatch.setattr(QevModel, "from_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr("qev.inference.encode_question", encode)
    (tmp_path / "qev_config.json").write_text(json.dumps({"runtime": "torch"}))
    agent = Agent(tmp_path, device="cpu", **batch_options)
    question = {"type": "noul", "instructions": "short"}
    before = agent.predict("state", {"q": question})["answers"]["q"]
    model.batch_calls.clear()
    combined = agent.predict("state", {
        "first": question, "middle": {"type": "noul", "instructions": "long"}, "last": question,
    })["answers"]
    assert model.batch_calls == expected_calls
    after = agent.predict("state", {"q": question})["answers"]["q"]
    assert before == after
    if not batch_options:
        assert combined["first"] == combined["last"] == before


def test_native_chat_uses_original_generation_path_and_decodes_only_completion():
    agent = fake_agent()
    result = agent.chat_completions({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": png_url()}},
    ]}], "max_tokens": 8})
    assert result["choices"][0]["message"]["content"] == "native answer"
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert result["qev"]["decision_adapter_enabled"] is False
    assert agent.model.processor.decoded == [[77, 78]]
    assert agent.model.native_kwargs["do_sample"] is False
    assert "temperature" not in agent.model.native_kwargs
    assert agent.model.processor.messages[0]["content"][1]["image"].size == (8, 8)


def test_native_video_passes_explicit_timestamps_without_resampling():
    agent = fake_agent()
    agent.chat_completions({"messages": [{"role": "user", "content": [
        {"type": "video", "frames": [png_url(), png_url()], "fps": 2},
    ]}], "temperature": 0.6, "top_p": 0.8})
    assert agent.model.processor.kwargs["processor_kwargs"]["do_sample_frames"] is False
    assert agent.model.processor.kwargs["processor_kwargs"]["video_metadata"] == [{"fps": 2.0, "total_num_frames": 2, "frames_indices": [0, 1]}]
    assert agent.model.native_kwargs["temperature"] == 0.6
    assert agent.model.native_kwargs["top_p"] == 0.8


def test_native_context_budget_is_checked_before_generation():
    agent = fake_agent()
    agent.config["native_max_length"] = 4
    with pytest.raises(ValueError, match="context limit"):
        agent.chat_completions({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4})
    assert not hasattr(agent.model, "native_kwargs")


def test_native_eos_at_token_budget_reports_stop_not_length():
    agent = fake_agent()
    agent.model.foundation = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[78]))
    result = agent.chat_completions({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2})
    assert result["choices"][0]["finish_reason"] == "stop"


def test_multimodal_decisions_keep_mrope_inputs_and_do_not_apply_text_temperature(monkeypatch):
    def encode(state, question, processor, **kwargs):
        assert len(kwargs["images"]) == 1
        return {"inputs": {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones((1, 4)),
                           "option_positions": torch.tensor([[1, 2]]), "option_mask": torch.tensor([[True, True]]),
                           "decision_positions": torch.tensor([3]), "mm_token_type_ids": torch.tensor([[1, 0, 0, 0]]),
                           "pixel_values": torch.zeros((1, 3, 8, 8)), "image_grid_thw": torch.tensor([[1, 1, 1]])},
                "ids": [1, 2, 3, 4], "option_positions": [1, 2], "decision_position": 3, "state_truncated": False}
    monkeypatch.setattr("qev.tokenization.encode_multimodal_question", encode)
    agent = fake_agent()
    answer = agent.predict([{"type": "image_url", "image_url": {"url": png_url()}}], {"q": {"type": "noul"}})
    assert answer["answers"]["q"]["noul"] == pytest.approx(0.7310586)
    assert answer["qev"]["temperature"] == 1.0
    assert answer["qev"]["multimodal_decision_accuracy_validated"] is False
    assert "mm_token_type_ids" in agent.model.forward_kwargs
    assert answer["usage"]["input_tokens"] == 4


def test_real_processor_agent_and_mlx_runtime_multimodal_contract(tmp_path):
    """End-to-end CPU tiny VLM: real PNG/frames -> processor -> Agent -> MLX.

    Specifically covers pointer metadata stripping and a media sequence longer
    than the trained text context, which mocked runtime tests cannot detect.
    """
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from qev.export import export_checkpoint
    from qev.model import QevModel
    from qev.tokenization import SPECIAL
    original_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    try:
        torch.manual_seed(73)
        config = Qwen3_5Config(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 16, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                "layer_types": ["linear_attention", "full_attention"], "tie_word_embeddings": True,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                    "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 48,
                "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2},
            tie_word_embeddings=True, image_token_id=60, video_token_id=61,
            vision_start_token_id=62, vision_end_token_id=63)
        base = tmp_path / "base"
        Qwen3_5ForConditionalGeneration(config).eval().save_pretrained(base)
        vocab = {f"t{i}": i for i in range(60)}
        for index, token in enumerate(SPECIAL.values(), 10):
            del vocab[f"t{index}"]
            vocab[token] = index
        vision_tokens = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
        vocab.update({token: index for index, token in enumerate(vision_tokens, 60)})
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="t0")),
                                            unk_token="t0", pad_token="t1", eos_token="t2")
        tokenizer.add_special_tokens({"additional_special_tokens": [*SPECIAL.values(), *vision_tokens]})
        processor = Qwen3VLProcessor(tokenizer=tokenizer,
            image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=16),
            video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=16))
        processor.save_pretrained(base)
        model = QevModel.from_pretrained(str(base), lora_rank=2, pointer_dim=8,
                                         max_length=8, max_state=4, local_files_only=True).eval()
        source = tmp_path / "checkpoint"
        model.save_pretrained(source, tokenizer=tokenizer, processor=processor)
        target = export_checkpoint(source, tmp_path / "mlx", dtype="float32")
        agent = Agent(target, backend="mlx")
        assert type(agent._processor()).__module__.startswith("transformers.")
        assert agent._processor().image_processor.patch_size == 2
        assert agent._processor().video_processor.patch_size == 2
        question = {"color": {"type": "choice", "criteria": {"red": None, "blue": None}}}
        for media in ({"type": "image_url", "image_url": {"url": png_url()}},
                      {"type": "video", "frames": [png_url(), png_url()], "fps": 2.0}):
            answer = agent.predict([{"type": "text", "text": "What color?"}, media], question)
            assert answer["usage"]["input_tokens"] > 8
            assert sum(answer["answers"]["color"]["probabilities"].values()) == pytest.approx(1)
            assert answer["qev"]["temperature"] == 1.0
            assert all(not adapter.enabled for adapter in agent.model._adapters)
    finally:
        mx.set_default_device(original_device)
        mx.enable_compile()
