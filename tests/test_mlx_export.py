"""Real tiny multimodal conversion and native parity, without downloads/GPU."""

import json

import numpy as np
import pytest

from qev.export import convert_adapter_weights, export_checkpoint, multimodal_config, verify_export
from qev.mlx_runtime import MLXRuntime, compare_probabilities, validate_encoding


def test_export_requires_complete_foundation_and_preserves_vision_and_vocab():
    config = {"model_type": "qwen3_5", "vision_config": {"depth": 1},
              "text_config": {"tie_word_embeddings": False,
                              "layer_types": ["linear_attention", "full_attention"]},
              "tie_word_embeddings": False}
    actual = multimodal_config(config)
    assert actual["vision_config"] == config["vision_config"]
    assert actual["text_config"]["tie_word_embeddings"] is False
    assert actual["text_config"]["full_attention_interval"] == 2
    assert "full_attention_interval" not in config["text_config"]
    with pytest.raises(ValueError, match="complete"):
        multimodal_config({"model_type": "qwen3_5_text"})
    config["text_config"]["layer_types"] += ["full_attention"]
    with pytest.raises(ValueError, match="periodically"):
        multimodal_config(config)


def test_adapter_conversion_preserves_separate_unmerged_pairs():
    prefix = "base_model.model.layers.0.self_attn.q_proj"
    a, b = np.ones((2, 4)), np.ones((4, 2))
    cfg = {"peft_type": "LORA", "bias": "none"}
    actual = convert_adapter_weights({prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b}, cfg)
    assert actual["language_model.model.layers.0.self_attn.q_proj.lora_a"] is a
    assert actual["language_model.model.layers.0.self_attn.q_proj.lora_b"] is b
    with pytest.raises(ValueError, match="plain"):
        convert_adapter_weights({}, {**cfg, "use_dora": True})
    with pytest.raises(ValueError, match="pair"):
        convert_adapter_weights({prefix + ".lora_A.weight": a}, cfg)


def test_parity_gate_rejects_invalid_or_changed_decisions():
    assert compare_probabilities([[0.7, 0.3]], [[0.7001, 0.2999]])["passed"]
    report = compare_probabilities([[0.5001, 0.4999]], [[0.4999, 0.5001]])
    assert report["max_probability_error"] < 0.002
    assert report["argmax_flips"] == 1 and not report["passed"]
    for values in ([[np.nan, 0.3]], [[0.6, 0.3]], [[-0.1, 1.1]]):
        with pytest.raises(ValueError):
            compare_probabilities([[0.7, 0.3]], values)
    with pytest.raises(ValueError):
        compare_probabilities([], [])


def test_encoding_rejects_cross_branch_or_padding_positions():
    encoded = {"ids": [1, 2, 3, 4], "option_positions": [0, 2], "decision_position": 3}
    validate_encoding(encoded, 4)
    for change in ({"option_positions": [2, 2]}, {"decision_position": 1},
                   {"option_positions": [0, 4]}, {"ids": [True, 2, 3, 4]}):
        with pytest.raises(ValueError):
            validate_encoding({**encoded, **change}, 4)


@pytest.fixture
def cpu_mlx():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    original_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    try:
        yield mx
    finally:
        mx.set_default_device(original_device)
        mx.enable_compile()


def test_real_multimodal_export_decisions_and_unchanged_native_generation(tmp_path, cpu_mlx):
    import torch
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

    from qev.model import QevModel

    torch.manual_seed(31)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        },
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 48,
                       "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                       "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2},
        tie_word_embeddings=True, image_token_id=60, video_token_id=61,
        vision_start_token_id=62, vision_end_token_id=63,
    )
    base = tmp_path / "base"
    original = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).eval()
    with torch.no_grad():
        original.model.language_model.layers[0].input_layernorm.weight.fill_(0.001953125)
    original.save_pretrained(base)
    original.float()
    vocab = {f"t{i}": i for i in range(60)}
    vocab.update({"<|image_pad|>": 60, "<|video_pad|>": 61,
                  "<|vision_start|>": 62, "<|vision_end|>": 63})
    tok = Tokenizer(WordLevel(vocab, unk_token="t0"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t0", pad_token="t1", eos_token="t2")
    tokenizer.add_special_tokens({"additional_special_tokens": list(vocab)[60:]})
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
    )
    processor.save_pretrained(base)
    model = QevModel.from_pretrained(str(base), lora_rank=2, pointer_dim=8, local_files_only=True).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.randn_like(parameter) * 0.1)
    source = tmp_path / "checkpoint"
    model.save_pretrained(source, tokenizer=tokenizer, processor=processor)
    target = export_checkpoint(source, tmp_path / "native", dtype="float32")
    exported = json.loads((target / "backbone" / "config.json").read_text())
    assert exported["vision_config"]["hidden_size"] == 32
    assert exported["text_config"]["vocab_size"] == 64
    assert (target / "decision_adapters.safetensors").is_file()
    from safetensors import safe_open

    with safe_open(target / "backbone" / "model.safetensors", framework="pt") as weights:
        shifted = weights.get_tensor("language_model.model.layers.0.input_layernorm.weight")
    torch.testing.assert_close(shifted, original.model.language_model.layers[0].input_layernorm.weight + 1,
                               atol=0, rtol=0)
    cases = []
    native_inputs = []
    for modality in ("text", "image", "video", "both"):
        ids = [3, 4, 5, 6, 7, 8]
        types = [0] * len(ids)
        media = {}
        if modality != "text":
            ids = [3, 62, 60 if modality != "video" else 61, 63, 7, 8]
            types = [0, 0, 1 if modality != "video" else 2, 0, 0, 0]
            kind = "video" if modality == "video" else "image"
            media["pixel_values" if kind == "image" else "pixel_values_videos"] = torch.randn(4, 24)
            media["image_grid_thw" if kind == "image" else "video_grid_thw"] = torch.tensor([[1, 2, 2]])
            if modality == "both":
                ids = [*ids[:4], 62, 61, 63, *ids[4:]]
                types = [*types[:4], 0, 2, 0, *types[4:]]
                media["pixel_values_videos"] = torch.randn(4, 24)
                media["video_grid_thw"] = torch.tensor([[1, 2, 2]])
        inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                  "mm_token_type_ids": torch.tensor([types]), **media}
        cases.append({"ids": ids, "option_positions": [0, len(ids) - 2],
                      "decision_position": len(ids) - 1, "inputs": inputs})
        native_inputs.append(inputs)
    report = verify_export(source, target, cases, atol=5e-5)
    assert report["passed"], report
    runtime = MLXRuntime.from_checkpoint(target)
    assert runtime.processor is not None and runtime.backbone.vision_tower is not None
    first, _, repeated = runtime.predict_encoded([cases[0], cases[1], cases[0]])
    np.testing.assert_array_equal(first, repeated)
    assert all(not adapter.enabled for adapter in runtime._adapters)
    for inputs in native_inputs:
        expected = original.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=1)
        actual = runtime.generate_native(**inputs, max_new_tokens=3)
        np.testing.assert_array_equal(actual, expected.cpu().numpy())
    # Generation must not leave state or adapter mode behind.
    np.testing.assert_array_equal(runtime.predict_encoded([cases[0]])[0], first)
    with pytest.raises(FileExistsError):
        export_checkpoint(source, target)
    with pytest.raises(ValueError, match="Unsupported processor"):
        runtime.generate_native(input_ids=[[3, 4]], audio_values=[[0.0]])
    # Failure during scoring restores native adapter mode.
    with pytest.raises(ValueError, match="Missing pixel_values"):
        runtime.predict_encoded([{"ids": [3, 60, 4], "option_positions": [0], "decision_position": 2}])
    assert all(not adapter.enabled for adapter in runtime._adapters)


def test_native_eos_uses_original_text_config_not_extra_tokenizer_stops(cpu_mlx):
    from types import SimpleNamespace

    mx = cpu_mlx

    class Language:
        def __init__(self):
            self.model = SimpleNamespace(layers=[])
            self.step = 0

        def make_cache(self):
            self.step = 0
            return []

        def __call__(self, *args, **kwargs):
            token = [3, 4, 2][self.step]
            self.step += 1
            logits = np.full((1, 1, 8), -20, dtype=np.float32)
            logits[0, 0, token] = 20
            return SimpleNamespace(logits=mx.array(logits))

    backbone = SimpleNamespace(language_model=Language(), eval=lambda: None,
                               config=SimpleNamespace(eos_token_id=[2, 3], text_config=SimpleNamespace(eos_token_id=2)))
    runtime = MLXRuntime(backbone, {"q.weight": mx.zeros((1, 2)), "k.weight": mx.zeros((1, 2))},
                         {"question_isolation": "independent_rows", "pointer_dim": 1, "hidden_size": 2},
                         tokenizer=SimpleNamespace(eos_token_id=3))
    runtime._embeddings = lambda _: {"rope_deltas": mx.zeros((1, 1))}
    np.testing.assert_array_equal(runtime.generate_native(input_ids=[[7]], max_new_tokens=6), [[7, 3, 4, 2]])
    # A one-token sampling support is deterministic and still respects the
    # foundation EOS instead of the tokenizer's additional stop token.
    np.testing.assert_array_equal(
        runtime.generate_native(input_ids=[[7]], max_new_tokens=6, do_sample=True, top_k=1, top_p=0.9),
        [[7, 3, 4, 2]],
    )


@pytest.mark.parametrize("device_name", ["cpu", "gpu"])
def test_runtime_materializes_loading_graphs_and_runs_across_threads(monkeypatch, device_name):
    """Reproduce main-thread lazy loads plus alternating FastAPI-like workers.

    Only tiny arrays/modules are used. Include a private cache created just
    before an exception, then consumed by the other thread on its next call.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from types import SimpleNamespace

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    if device_name == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    import qev.mlx_runtime as module

    old_device = mx.default_device()
    device = mx.cpu if device_name == "cpu" else mx.gpu
    mx.set_default_device(device)
    if device_name == "cpu":
        mx.disable_compile()

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []
            self.embed_tokens = nn.Embedding(64, 2)
            # Keep genuinely lazy CPU source and compute-device cast graphs.
            self.embed_tokens.weight = mx.arange(128, stream=mx.cpu).reshape(64, 2).astype(mx.float32) / 100
            self._private_gain = (mx.arange(2, stream=mx.cpu) + 1).astype(mx.float32)
            self._after_failure = None
            self.fail_once = True

        def __call__(self, ids, inputs_embeds=None, **kwargs):
            if self.fail_once:
                self.fail_once = False
                self._after_failure = mx.ones((1,), stream=mx.cpu) * 2
                raise ValueError("Injected failure after constructing a lazy private cache")
            hidden = self.embed_tokens(ids) if inputs_embeds is None else inputs_embeds
            return mx.cumsum(hidden, axis=1) * self._private_gain * self._after_failure

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Text()
            self.lm_head = nn.Linear(2, 64, bias=False)
            weight = np.zeros((64, 2), dtype=np.float32)
            weight[2] = 1
            self.lm_head.weight = mx.array(weight) + 0
            self._position_ids = None
            self._rope_deltas = None

        def get_rope_index(self, ids, *args):
            return mx.arange(ids.shape[1])[None], mx.zeros((1, 1), dtype=mx.int32)

        def make_cache(self):
            return []

        def __call__(self, ids, **kwargs):
            kwargs.pop("rope_deltas", None)
            return SimpleNamespace(logits=self.lm_head(self.model(ids, **kwargs)))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Language()
            self.config = SimpleNamespace(image_token_id=60, video_token_id=61,
                                          text_config=SimpleNamespace(eos_token_id=2))

    try:
        runtime = MLXRuntime(
            Backbone(), {"q.weight": mx.array([[0.2, 0.1]]) + 0,
                         "k.weight": mx.array([[0.1, 0.3]]) + 0},
            {"question_isolation": "independent_rows", "pointer_dim": 1, "hidden_size": 2},
        )
        runtime._adapters = [SimpleNamespace(enabled=False)]
        # Leave the global RNG's next key lazy on the constructing thread.
        # Worker sampling must not consume this graph.
        mx.random.normal((2,))
        original_processed = module._processed_inputs

        def checked_processed(values):
            # This also verifies native preprocessing is inside the context.
            assert mx.default_device().type == device
            assert mx.default_stream(mx.cpu) == runtime._thread_streams.cpu
            assert mx.default_stream(device) == runtime._thread_streams.compute
            return original_processed(values)

        monkeypatch.setattr(module, "_processed_inputs", checked_processed)
        encoded = {"ids": [3, 4, 5, 6], "option_positions": [0, 2], "decision_position": 3}
        barrier = threading.Barrier(2)
        failure_done = threading.Event()

        @contextmanager
        def caller_scope():
            # Default streams are thread-local, but MLX's default device is
            # process-global. Include this caller context and its restoration
            # assertions in the runtime lock to avoid observing another
            # worker's temporary device change between a call and its checks.
            other_device = mx.gpu if device_name == "cpu" and mx.metal.is_available() else mx.cpu
            with runtime._lock, mx.stream(other_device):
                previous_device = mx.default_device()
                previous_cpu = mx.default_stream(mx.cpu)
                previous_compute = mx.default_stream(device)
                try:
                    yield
                finally:
                    assert mx.default_device() == previous_device
                    assert mx.default_stream(mx.cpu) == previous_cpu
                    assert mx.default_stream(device) == previous_compute
                    assert not runtime._adapters[0].enabled

        def worker(index):
            barrier.wait(timeout=10)
            if index == 0:
                try:
                    with caller_scope(), pytest.raises(ValueError, match="lazy private cache"):
                        runtime.predict_logits([encoded])
                finally:
                    failure_done.set()
            else:
                assert failure_done.wait(timeout=10)
            barrier.wait(timeout=10)
            with caller_scope():
                first = runtime.predict_logits([encoded])[0]
            barrier.wait(timeout=10)
            with caller_scope():
                generated = runtime.generate_native(input_ids=[encoded["ids"]], max_new_tokens=2)
                sampled = runtime.generate_native(input_ids=[encoded["ids"]], max_new_tokens=2,
                                                  do_sample=True, temperature=0.7, top_p=0.9, top_k=1)
                np.testing.assert_array_equal(sampled, generated)
                second = runtime.predict_logits([encoded])[0]
                np.testing.assert_array_equal(first, second)
            return threading.get_ident(), first, generated

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, index) for index in range(2)]
            results = [future.result(timeout=30) for future in futures]
        assert results[0][0] != results[1][0]
        np.testing.assert_array_equal(results[0][1], results[1][1])
        np.testing.assert_array_equal(results[0][2], [[3, 4, 5, 6, 2]])
        np.testing.assert_array_equal(results[0][2], results[1][2])
        np.testing.assert_array_equal(runtime.predict_logits([encoded])[0], results[0][1])
        assert not runtime._adapters[0].enabled
    finally:
        mx.set_default_device(old_device)
        if device_name == "cpu":
            mx.enable_compile()
