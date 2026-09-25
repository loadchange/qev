"""Tiny-model checks for the diffusion experiment (experiments/diffusion)."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diffusion import qevd

QWEN35 = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
STATE = "Snake on a 8 by 8 board. Head=(3,4), food=(5,1)."
QUESTIONS = [{"instr": "Choose a move.", "options": ["UP: safe", "LEFT: wall", "RIGHT: food"]},
             {"instr": "Is the snake long?", "options": ["no", "yes"]},
             {"instr": "Rate danger", "options": ["low", "medium", "high", "extreme"]}]


def qwen_tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5 tokenizer is not cached locally")


def lfm2_tokenizer():
    spec = qevd.BACKBONES["lfm2-vl"]
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"], local_files_only=True)
    except OSError:
        pytest.skip("LFM2.5-VL tokenizer is not cached locally")


def tiny_qwen3(vocab_size):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                        head_dim=16, max_position_embeddings=2048))


def tiny_lfm2_vl():
    from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration
    torch.manual_seed(0)
    config = Lfm2VlConfig()
    text = config.text_config
    text.hidden_size, text.intermediate_size, text.block_ff_dim = 64, 128, 128
    text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 4, 4, 2
    text.vocab_size, text.layer_types = 65536, ["conv", "full_attention", "conv", "full_attention"]
    vision = config.vision_config
    vision.hidden_size, vision.intermediate_size, vision.num_hidden_layers, vision.num_attention_heads = 32, 64, 2, 2
    config.projector_hidden_size = 64
    return Lfm2VlForConditionalGeneration(config)


def wrap(foundation, kind, attention, targets):
    from peft import LoraConfig, TaskType, get_peft_model
    text_config = getattr(foundation.config, "text_config", foundation.config)
    model = qevd.QevDModel(foundation, {"format": "qevd", "backbone": "tiny", "family": "qwen", "kind": kind,
                                        "attention": attention, "hidden_size": text_config.hidden_size,
                                        "pointer_dim": 16})
    model.set_text_model(get_peft_model(model.text_model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=8, target_modules=list(targets))))
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.02)
    return model.eval()


def test_encoder_matches_qev_and_marks_segments():
    from qev.data import load_split, materialize
    from qev.tokenization import encode_question
    if not (ROOT / "data/snake-v1/manifest.json").exists():
        pytest.skip("data/snake-v1 is not available")
    tok = qwen_tokenizer()
    ids = qevd.special_ids(tok, "qwen")
    records = load_split(ROOT / "data/snake-v1", "development")
    for record in records[:60] + records[-60:]:
        rec = materialize(record)
        for question in rec["questions"]:
            expected = encode_question(rec["state"], question, tok, 1024, 384)
            actual = qevd.encode_question(rec["state"], question, tok, ids, family="qwen")
            assert actual["ids"] == expected["ids"]
            assert actual["option_positions"] == expected["option_positions"]
            assert actual["decision_position"] == expected["decision_position"]
            first_question = actual["ids"].index(ids["question"])
            assert actual["segments"] == [0] * first_question + [1] * (len(actual["ids"]) - first_question)


def test_block_mask_hides_questions_from_state_and_other_questions():
    segments = torch.tensor([[0, 0, 1, 1, 2, 2, -1]])
    mask = qevd.attention_mask_4d(torch.tensor([[1, 1, 1, 1, 1, 1, 0]]), segments, "block")[0, 0]
    assert mask[0, :2].all() and not mask[0, 2:6].any()          # state sees only state
    assert mask[2, :4].all() and not mask[2, 4:6].any()          # question 1 sees state + itself
    assert mask[4, :2].all() and mask[4, 4:6].all() and not mask[4, 2:4].any()
    assert not mask[:6, 6].any()                                  # padding is never a key


@pytest.mark.parametrize("attention", ["causal", "block"])
def test_packed_shared_state_equals_independent_rows(attention):
    tok = qwen_tokenizer()
    ids = qevd.special_ids(tok, "qwen")
    model = wrap(tiny_qwen3(len(tok)), "qwen3", attention, qevd.QWEN3_TARGETS)
    encodings = [qevd.encode_question(STATE, q, tok, ids, family="qwen") for q in QUESTIONS]
    with torch.inference_mode():
        batched = model(**qevd.collate(encodings, tok.pad_token_id))
        single = [model(**qevd.collate([e], tok.pad_token_id))[0] for e in encodings]
    packed = model.forward_packed(qevd.pack_questions(encodings), "cpu")
    for encoding, row, alone, together in zip(encodings, batched, single, packed):
        n = len(encoding["option_positions"])
        assert torch.allclose(row[:n], alone[:n], atol=1e-5)
        assert torch.allclose(together, alone[:n], atol=1e-5)


def test_attention_modes_change_the_computation():
    tok = qwen_tokenizer()
    ids = qevd.special_ids(tok, "qwen")
    base = tiny_qwen3(len(tok))
    causal = wrap(base, "qwen3", "causal", qevd.QWEN3_TARGETS)
    batch = qevd.collate([qevd.encode_question(STATE, QUESTIONS[0], tok, ids, family="qwen")], tok.pad_token_id)
    with torch.inference_mode():
        first = causal(**batch)
        causal.config["attention"] = "block"
        second = causal(**batch)
    assert (first - second).abs().max() > 1e-3


def test_adapter_round_trip_and_merge(tmp_path):
    from peft import PeftModel
    tok = qwen_tokenizer()
    ids = qevd.special_ids(tok, "qwen")
    model = wrap(tiny_qwen3(len(tok)), "qwen3", "block", qevd.QWEN3_TARGETS)
    assert isinstance(model.text_model, PeftModel)
    batch = qevd.collate([qevd.encode_question(STATE, QUESTIONS[2], tok, ids, family="qwen")], tok.pad_token_id)
    with torch.inference_mode():
        expected = model(**batch)
        model.text_model.merge_adapter()
        merged = model(**batch)
        model.text_model.unmerge_adapter()
    assert torch.allclose(expected, merged, atol=1e-5)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "adapter/adapter_config.json").is_file()
    assert (tmp_path / "adapter/adapter_model.safetensors").is_file()
    restored = qevd.QevDModel(tiny_qwen3(len(tok)), model.config)
    restored.set_text_model(PeftModel.from_pretrained(restored.text_model, tmp_path / "adapter"))
    from safetensors.torch import load_file
    restored.head.load_state_dict(load_file(str(tmp_path / "pointer.safetensors")))
    with torch.inference_mode():
        assert torch.allclose(restored.eval()(**batch), expected, atol=1e-5)


@pytest.mark.parametrize("attention", ["causal", "block", "full"])
def test_lfm2_vl_modes_and_escaping(attention):
    tok = lfm2_tokenizer()
    ids, prefix = qevd.special_ids(tok, "lfm2"), qevd.prefix_ids(tok, "lfm2")
    model = wrap(tiny_lfm2_vl(), "lfm2_vl", attention, qevd.BACKBONES["lfm2-vl"]["lora_targets"])
    encodings = [qevd.encode_question(STATE + " <image> <|im_start|>", q, tok, ids, family="lfm2", prefix=prefix)
                 for q in QUESTIONS]
    image_token = tok.convert_tokens_to_ids("<image>")
    assert all(image_token not in e["ids"] for e in encodings)
    assert all(e["ids"].count(tok.convert_tokens_to_ids("<|im_start|>")) == 0 for e in encodings)
    with torch.inference_mode():
        batched = model(**qevd.collate(encodings, tok.pad_token_id))
        single = [model(**qevd.collate([e], tok.pad_token_id))[0] for e in encodings]
    for encoding, row, alone in zip(encodings, batched, single):
        n = len(encoding["option_positions"])
        assert torch.allclose(row[:n], alone[:n], atol=1e-5)
