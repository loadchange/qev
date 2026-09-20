"""Tiny real hybrid Qwen3.5 tests; no pretrained weights or network needed."""

import json
from pathlib import Path

import pytest
import torch

from qev.model import QevModel, _load_foundation
from qev.tokenization import SPECIAL, collate_encodings, encode_question, user_tokens


@pytest.fixture
def tiny_base(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(7)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": True,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 48,
            "num_heads": 2, "out_hidden_size": 32,
            "num_position_embeddings": 16, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
        tie_word_embeddings=True,
        image_token_id=60, video_token_id=61,
        vision_start_token_id=62, vision_end_token_id=63,
    )
    full = Qwen3_5ForConditionalGeneration(config)
    path = tmp_path / "base"
    full.save_pretrained(path)
    expected = {key: value.detach().clone() for key, value in full.model.language_model.state_dict().items()}
    return path, expected


def _enc(ids, positions):
    return {"ids": ids, "option_positions": positions, "decision_position": len(ids) - 1}


def test_official_multimodal_checkpoint_maps_all_text_weights(tiny_base):
    path, expected = tiny_base
    text = _load_foundation(str(path), None, torch.float32, "eager", True).model.language_model
    assert set(text.state_dict()) == set(expected)
    assert all(torch.equal(value, text.state_dict()[key]) for key, value in expected.items())


def test_hybrid_rows_isolate_questions_and_mask_candidate_padding(tiny_base):
    path, _ = tiny_base
    model = QevModel.from_pretrained(str(path), lora_rank=2, pointer_dim=8, local_files_only=True).eval()
    a = _enc([1, 2, 3, 4, 5, 6], [2, 4])
    b = _enc([7, 8, 9, 10, 11, 12, 13, 14], [2, 4, 6])
    with torch.no_grad():
        solo = model(**collate_encodings([a], 0))[0]
        batch = model(**collate_encodings([a, b], 0))
        changed = model(**collate_encodings([a, _enc([21, 22, 23, 24, 25, 26, 27, 28], [2, 4, 6])], 0))
    torch.testing.assert_close(batch[0, :2], solo, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(batch[0], changed[0], atol=0, rtol=0)
    assert torch.softmax(batch, -1)[0, 2].item() == 0
    assert batch.dtype == torch.float32


def test_lora_receives_gradients_and_checkpoint_roundtrip(tiny_base, tmp_path):
    path, _ = tiny_base
    model = QevModel.from_pretrained(str(path), lora_rank=2, pointer_dim=8, local_files_only=True)
    model.enable_gradient_checkpointing()
    inputs = collate_encodings([_enc([1, 2, 3, 4, 5, 6], [2, 4])], 0)
    loss = torch.nn.functional.cross_entropy(model(**inputs), torch.tensor([1]))
    loss.backward()
    grads = [p.grad for name, p in model.named_parameters() if "lora_B" in name]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert model.head.q.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad for name, p in model.backbone.named_parameters() if "lora_" not in name)
    assert all(not p.requires_grad for p in model.foundation.model.visual.parameters())
    assert all(not p.requires_grad for p in model.foundation.lm_head.parameters())
    # Nonzero adapter weights ensure the round trip tests more than the base.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.005)
    model.eval()
    with torch.no_grad():
        expected = model(**inputs)
    checkpoint = tmp_path / "checkpoint"
    class ProcessorStub:
        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True)
            (Path(directory) / "processor_config.json").write_text('{}')

    model.save_pretrained(checkpoint, processor=ProcessorStub(), extra_config={"max_length": 128})
    assert json.loads((checkpoint / "qev_config.json").read_text())["max_length"] == 128
    loaded = QevModel.from_checkpoint(checkpoint, local_files_only=True)
    with torch.no_grad():
        actual = loaded(**inputs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError, match="native base"):
        QevModel.from_checkpoint(checkpoint, merge=True, local_files_only=True)


@pytest.mark.parametrize("modality", ["text", "image", "video"])
def test_native_generation_matches_original_after_nonzero_lora(tiny_base, modality):
    from transformers import Qwen3_5ForConditionalGeneration

    path, _ = tiny_base
    base = Qwen3_5ForConditionalGeneration.from_pretrained(path).eval()
    model = QevModel.from_pretrained(str(path), lora_rank=2, pointer_dim=8, local_files_only=True).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * .1)
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    if modality in ("image", "video"):
        image = modality == "image"
        inputs.update(
            input_ids=torch.tensor([[1, 62, 60 if image else 61, 63, 4, 5]]),
            mm_token_type_ids=torch.tensor([[0, 0, 1 if image else 2, 0, 0, 0]]),
        )
        inputs["pixel_values" if image else "pixel_values_videos"] = torch.randn(4, 24)
        inputs["image_grid_thw" if image else "video_grid_thw"] = torch.tensor([[1, 2, 2]])
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    with torch.inference_mode():
        original_logits = base(**inputs, use_cache=False).logits
        adapted_logits = model.foundation(**inputs, use_cache=False).logits
        pointer_logits = model(
            **inputs, option_positions=torch.tensor([[0, 1]]),
            option_mask=torch.ones((1, 2), dtype=torch.bool),
            decision_positions=torch.tensor([inputs["input_ids"].shape[1] - 1]),
        )
        assert pointer_logits.shape == (1, 2) and torch.isfinite(pointer_logits).all()
        with model.backbone.disable_adapter():
            restored_logits = model.foundation(**inputs, use_cache=False).logits
        assert not torch.equal(adapted_logits, original_logits)
        torch.testing.assert_close(restored_logits, original_logits, atol=0, rtol=0)
        expected = base.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=0)
        actual = model.generate_native(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=0)
    assert torch.equal(actual, expected)
    assert not model.training
    # Native generation must restore the enabled adapter for subsequent decisions.
    with torch.inference_mode():
        torch.testing.assert_close(model.foundation(**inputs, use_cache=False).logits, adapted_logits, atol=0, rtol=0)


class TinyTokenizer:
    unk_token_id = None

    def __init__(self):
        self.special = {token: i + 1 for i, token in enumerate(SPECIAL.values())}

    def convert_tokens_to_ids(self, token):
        return self.special.get(token)

    def __call__(self, text, add_special_tokens=False):
        if text in self.special:
            return {"input_ids": [self.special[text]]}
        return {"input_ids": [10 + ord(char) % 40 for char in text]}


def test_token_budget_preserves_every_candidate_and_sanitizes_control_tokens():
    tok = TinyTokenizer()
    question = {"instr": "pick", "options": ["a", "longer", "c"]}
    enc = encode_question("very long state" * 8, question, tok, max_length=32, max_state=20)
    assert len(enc["ids"]) == 32 and enc["state_truncated"]
    assert len(enc["option_positions"]) == 3
    assert all(enc["ids"][p] == tok.special[SPECIAL["end_option"]] for p in enc["option_positions"])
    assert not set(tok.special.values()) & set(user_tokens(tok, "<|fim_suffix|>"))
    with pytest.raises(ValueError, match="all 3 options"):
        encode_question("state", question, tok, max_length=12, max_state=5)


def test_markers_cannot_point_into_padding(tiny_base):
    path, _ = tiny_base
    model = QevModel.from_pretrained(str(path), lora_rank=2, pointer_dim=8, local_files_only=True)
    inputs = collate_encodings([_enc([1, 2, 3], [1]), _enc([1, 2, 3, 4], [2])], 0)
    inputs["decision_positions"][0] = 3
    with pytest.raises(ValueError, match="padding"):
        model(**inputs)


@pytest.mark.parametrize("nested", [False, True])
def test_original_processor_is_independent_of_auto_registrations(tiny_base, tmp_path, monkeypatch, nested):
    """Loading MLX in the same process must not replace HF video patch settings."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoImageProcessor,
        AutoProcessor,
        AutoVideoProcessor,
        PreTrainedTokenizerFast,
    )
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    vocab = {"unknown": 0, "<|image_pad|>": 1, "<|video_pad|>": 2,
             "<|vision_start|>": 3, "<|vision_end|>": 4}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="unknown")),
                                        unk_token="unknown")
    template = "{% for message in messages %}{{ message['role'] }}{% endfor %}"
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer, chat_template=template,
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                              min_pixels=16, max_pixels=64),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                              min_pixels=16, max_pixels=64),
    )
    source = tmp_path / "processor"
    processor.save_pretrained(source)
    if not nested:
        processor.image_processor.save_pretrained(source)
        processor.video_processor.save_pretrained(source)
        (source / "processor_config.json").unlink()

    def reject_auto(*args, **kwargs):
        raise AssertionError("Official processor loading must not use global Auto processor dispatch")
    for auto_class in (AutoProcessor, AutoImageProcessor, AutoVideoProcessor):
        monkeypatch.setattr(auto_class, "from_pretrained", reject_auto)
    model = QevModel.from_pretrained(str(tiny_base[0]), lora_rank=2, pointer_dim=8, local_files_only=True)
    model._processor_source = str(source)
    restored = model.get_processor(local_files_only=True)
    assert type(restored) is Qwen3VLProcessor
    assert type(restored.image_processor) is Qwen2VLImageProcessor
    assert type(restored.video_processor) is Qwen3VLVideoProcessor
    assert restored.image_processor.patch_size == restored.video_processor.patch_size == 2
    assert restored.video_processor.temporal_patch_size == 2
    assert restored.image_processor.size == processor.image_processor.size
    assert restored.video_processor.size == processor.video_processor.size
    assert restored.chat_template == template
