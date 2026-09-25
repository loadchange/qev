"""Native multimodal MLX Qev: complete foundation plus switchable decision LoRA.

Every decision row gets a fresh recurrent/attention state. Native generation
uses the unchanged foundation with decision adapters disabled. No global patch
or merged foundation is used.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


def load_original_processor(path: str | Path):
    """Load exact Transformers processors despite mlx-vlm's global registry."""
    from qev.processing import load_processor

    return load_processor(path, local_files_only=True)


def _state_arrays(value):
    """Visit parameters and private cached arrays in an MLX module's state."""
    import mlx.core as mx

    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _state_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_arrays(item)


def _correct_delta_normalization(model):
    """Match HF/FLA's sum-of-squares epsilon (mlx-vlm 0.7.1 uses mean)."""
    import mlx.core as mx

    def normalize(self, q, k):
        inv_scale = k.shape[-1] ** -0.5
        rms_eps = 1e-6 * inv_scale**2
        return (inv_scale**2 * mx.fast.rms_norm(q, None, rms_eps),
                inv_scale * mx.fast.rms_norm(k, None, rms_eps))

    for layer in model.language_model.model.layers:
        if layer.is_linear:
            # Bind only this model instance; do not change another application's
            # mlx-vlm model or the installed library.
            object.__setattr__(layer.linear_attn, "_normalize_qk", MethodType(normalize, layer.linear_attn))


def _install_adapters(model, weights, scale):
    import mlx.core as mx
    from mlx import nn

    class DecisionLinear(nn.Module):
        def __init__(self, linear, a, b):
            super().__init__()
            self.linear = linear
            self.lora_a = a.astype(mx.float32)
            self.lora_b = b.astype(mx.float32)
            self.enabled = False

        @property
        def weight(self):
            return self.linear.weight

        def merge(self, dtype):
            """Keep a separate decision weight W + scale*B@A; W itself is untouched."""
            merged = self.linear.weight.astype(mx.float32) + scale * (self.lora_b @ self.lora_a)
            self.merged = merged.astype(dtype)

        def __call__(self, inputs):
            if self.enabled and "merged" in self:
                output = inputs.astype(self.merged.dtype) @ self.merged.T
                if "bias" in self.linear:
                    output = output + self.linear.bias.astype(output.dtype)
                return output.astype(inputs.dtype)
            base = self.linear(inputs)
            if not self.enabled:
                return base
            update = (inputs.astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
            return base + (update * scale).astype(base.dtype)

    groups = {}
    for key, value in weights.items():
        path, side = key.rsplit(".", 1)
        if side not in {"lora_a", "lora_b"} or not path.startswith("language_model.model.layers."):
            raise ValueError(f"Unsupported decision adapter key: {key}")
        groups.setdefault(path, {})[side] = value
    if not groups:
        raise ValueError("Decision adapter contains no LoRA weights")
    replacements, wrappers = [], []
    modules = dict(model.named_modules())
    for path, pair in groups.items():
        original = modules.get(path)
        if not isinstance(original, nn.Linear) or set(pair) != {"lora_a", "lora_b"}:
            raise ValueError(f"Missing LoRA pair or unsupported linear target: {path}")
        a, b = pair["lora_a"], pair["lora_b"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != original.weight.shape:
            raise ValueError(f"Incompatible LoRA shapes for {path}")
        wrapped = DecisionLinear(original, a, b)
        replacements.append((path, wrapped))
        wrappers.append(wrapped)
    from mlx.utils import tree_unflatten

    model.update_modules(tree_unflatten(replacements))
    return wrappers


def _array(value):
    import mlx.core as mx

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no native bfloat16 representation.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return value if isinstance(value, mx.array) else mx.array(value)


_INPUT_FIELDS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                 "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"}
_MEDIA_FIELDS = _INPUT_FIELDS - {"input_ids", "attention_mask"}


def _processed_inputs(values):
    """Single, unpadded processor result. Never silently discard media tensors."""
    unknown = set(values) - _INPUT_FIELDS
    if unknown:
        raise ValueError(f"Unsupported processor fields: {sorted(unknown)}")
    converted = {key: _array(value) for key, value in values.items() if value is not None}
    ids = converted.get("input_ids")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("MLX expects one unpadded input_ids row")
    mask = converted.pop("attention_mask", None)
    if mask is not None and (mask.shape != ids.shape or not bool((mask == 1).all().item())):
        raise ValueError("MLX accepts one unpadded processor row with attention_mask=1")
    # These token types are processor metadata. Qwen3.5 reconstructs MRoPE
    # positions from the actual image/video token IDs and grid shapes.
    converted.pop("mm_token_type_ids", None)
    for pixels, grid in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
        if (pixels in converted) != (grid in converted):
            raise ValueError(f"{pixels} and {grid} must be supplied together")
    return converted


def validate_encoding(encoded: Mapping[str, Any], max_length: int) -> None:
    """Reject malformed marker positions before touching a device."""
    ids = encoded.get("ids")
    options = encoded.get("option_positions")
    decision = encoded.get("decision_position")
    if not isinstance(ids, (list, tuple)) or not ids or len(ids) > max_length:
        raise ValueError(f"ids must contain 1..{max_length} token IDs")
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in ids):
        raise ValueError("Token IDs must be nonnegative integers")
    if not isinstance(options, (list, tuple)) or not 1 <= len(options) <= 255:
        raise ValueError("Each question needs 1..255 option positions")
    positions = [*options, decision]
    if any(isinstance(p, bool) or not isinstance(p, (int, np.integer)) for p in positions):
        raise ValueError("Marker positions must be integers")
    if any(p < 0 or p >= len(ids) for p in positions):
        raise ValueError("Marker position outside the encoded sequence")
    if len(set(options)) != len(options) or any(p >= decision for p in options):
        raise ValueError("Distinct option markers must precede the decision marker")


def compare_probabilities(
    reference: Sequence[Any],
    actual: Sequence[Any],
    *,
    atol: float = 0.002,
    require_same_argmax: bool = True,
) -> dict[str, Any]:
    """Numerical acceptance check shared by export validation and benchmarks.

    Tolerance is a declared deployment choice, not a promise that low-precision
    conversion preserves all decisions. Empty or malformed probes never pass.
    """
    if not reference or len(reference) != len(actual):
        raise ValueError("Parity requires equally sized, nonempty prediction lists")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")
    max_error = 0.0
    flips = 0
    for expected, observed in zip(reference, actual, strict=True):
        expected = np.asarray(expected, dtype=np.float64)
        observed = np.asarray(observed, dtype=np.float64)
        if expected.ndim != 1 or expected.size < 1 or expected.shape != observed.shape:
            raise ValueError("Parity vectors must have matching one-dimensional shapes")
        for values in (expected, observed):
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError("Parity vectors contain invalid probabilities")
            if not np.isclose(values.sum(), 1.0, atol=1e-5, rtol=0):
                raise ValueError("Parity vectors must sum to one")
        max_error = max(max_error, float(np.max(np.abs(expected - observed))))
        flips += int(np.argmax(expected) != np.argmax(observed))
    return {
        "questions": len(reference),
        "max_probability_error": max_error,
        "argmax_flips": flips,
        "atol": atol,
        "require_same_argmax": require_same_argmax,
        "passed": max_error <= atol and (not require_same_argmax or flips == 0),
    }



class MLXRuntime:
    """Full Qwen3.5 mlx-vlm foundation and separate Qev decision adapters."""

    def __init__(self, backbone: Any, head_weights: Mapping[str, Any], config: dict,
                 tokenizer: Any = None, *, processor: Any = None, adapter_weights=None, generation_config=None):
        import mlx.core as mx

        if config.get("question_isolation") != "independent_rows":
            raise ValueError("Qwen3.5 MLX inference requires independent_rows isolation")
        if config.get("pointer_bias", False):
            raise ValueError("Only bias-free q/k pointer heads are supported")
        if set(head_weights) != {"q.weight", "k.weight"}:
            raise ValueError("Pointer weights must contain exactly q.weight and k.weight")
        shape = (int(config["pointer_dim"]), int(config["hidden_size"]))
        if min(shape) <= 0:
            raise ValueError("Pointer dimensions must be positive")
        for name, value in head_weights.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        self._device = mx.default_device()
        self._thread_streams = threading.local()
        self.backbone = backbone
        self.q_weight = head_weights["q.weight"].astype(mx.float32)
        self.k_weight = head_weights["k.weight"].astype(mx.float32)
        self.config = dict(config)
        self.generation_config = dict(generation_config or {})
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self._lock = threading.RLock()
        self._adapters = []
        if adapter_weights is not None:
            self._adapters = _install_adapters(
                backbone, adapter_weights, config["lora_alpha"] / config["lora_rank"]
            )
        self._prepare_backbone(backbone)
        backbone.eval()
        # Safetensors loads and dtype casts are lazy. MLX 0.32 primitive streams
        # belong to their creating thread; evaluating a loading-thread graph in
        # a FastAPI worker fails even if that worker creates its own streams.
        # Materialize all persistent arrays here, including private constants,
        # before publishing this runtime to any request thread.
        self._materialize_state()

    def _materialize_state(self):
        import mlx.core as mx

        arrays = list(_state_arrays(self.backbone))
        mx.eval(arrays, self.q_weight, self.k_weight)

    @contextmanager
    def _execution_streams(self):
        """Use CPU/compute streams owned by the current request thread."""
        import mlx.core as mx

        previous_device = mx.default_device()
        previous_cpu = mx.default_stream(mx.cpu)
        previous_compute = mx.default_stream(self._device)
        streams = self._thread_streams
        try:
            if not hasattr(streams, "cpu"):
                cpu = mx.new_stream(mx.cpu)
                compute = cpu if self._device.type == mx.cpu else mx.new_stream(self._device)
                streams.cpu, streams.compute = cpu, compute
            # CPU defaults also matter for host-side primitives in otherwise
            # GPU graphs. Never reuse streams created by a different thread.
            with mx.stream(streams.cpu), mx.stream(streams.compute):
                try:
                    yield
                finally:
                    mx.synchronize(streams.cpu)
                    if streams.compute != streams.cpu:
                        mx.synchronize(streams.compute)
        finally:
            # mx.stream restores the entry device's stream, but can leave a
            # different device's default changed. Restore both explicitly,
            # including on the first call when thread streams were created.
            mx.set_default_stream(previous_cpu)
            mx.set_default_stream(previous_compute)
            mx.set_default_device(previous_device)

    DECISION_WEIGHTS = ("adapter", "merged", "merged-bf16", "bf16")

    def merge_decision_weights(self, mode: str = "merged") -> None:
        """Serve decisions from merged LoRA weights; native generation is unchanged.

        ``adapter`` computes LoRA separately (the validated default). ``merged``
        stores float32 W + scale*B@A per adapted layer (same math, fewer kernel
        launches, more memory). ``merged-bf16`` stores that copy in bfloat16.
        ``bf16`` also casts the whole foundation, vision tower included, to
        bfloat16: the smallest and fastest mode, with native generation in
        bfloat16 as well. Merges use the float32 LoRA factors before any cast.
        """
        import mlx.core as mx

        if mode not in self.DECISION_WEIGHTS:
            raise ValueError(f"decision weights must be one of {self.DECISION_WEIGHTS}")
        if mode == "adapter":
            return
        with self._lock:
            for adapter in self._adapters:
                adapter.merge(mx.float32 if mode == "merged" else mx.bfloat16)
            if mode == "bf16":
                self.backbone.set_dtype(mx.bfloat16)
            self.config = {**self.config, "decision_weights": mode}
            self._materialize_state()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, decision_weights: str = "adapter") -> MLXRuntime:
        """Load local version-2 exports including the complete vision tower."""
        import mlx.core as mx
        from mlx_vlm.utils import load_model

        path = Path(path).resolve()
        config = json.loads((path / "qev_config.json").read_text())
        if config.get("runtime") != "mlx" or config.get("format") != "qev":
            raise ValueError("Not a native Qev MLX export; run python -m qev.export first")
        if config.get("format_version") != 2 or config.get("backend") != "mlx_vlm":
            raise ValueError("A version-2 multimodal mlx-vlm export is required; text-only exports are unsupported")
        if config.get("native_generation") != "adapter_disabled":
            raise ValueError("Native generation requires unmerged decision adapters")
        backbone_path = path / "backbone"
        model_config = json.loads((backbone_path / "config.json").read_text())
        if model_config.get("model_type") != "qwen3_5" or not model_config.get("vision_config"):
            raise ValueError("The complete Qwen3.5 multimodal foundation is required")
        backbone = load_model(backbone_path, lazy=True, strict=True)
        processor = load_original_processor(backbone_path)
        runtime = cls(backbone, mx.load(str(path / "pointer.safetensors")), config,
                      processor=processor,
                      generation_config=json.loads((backbone_path / "generation_config.json").read_text())
                      if (backbone_path / "generation_config.json").is_file() else None,
                      adapter_weights=mx.load(str(path / "decision_adapters.safetensors")))
        runtime.merge_decision_weights(decision_weights)
        return runtime

    @contextmanager
    def _decision_mode(self, enabled):
        with self._lock, self._execution_streams():
            previous = [adapter.enabled for adapter in self._adapters]
            try:
                for adapter in self._adapters:
                    adapter.enabled = enabled
                self.backbone.language_model._position_ids = None
                self.backbone.language_model._rope_deltas = None
                yield
            finally:
                for adapter, state in zip(self._adapters, previous, strict=True):
                    adapter.enabled = state
                self.backbone.language_model._position_ids = None
                self.backbone.language_model._rope_deltas = None
                # Layers can add private lazy caches even before raising an
                # input error. Finish them in their owning thread on both the
                # success and failure paths before another worker can reuse it.
                self._materialize_state()

    # Model-specific hooks; LFM2Runtime overrides them for LFM2-VL.
    def _prepare_backbone(self, backbone):
        _correct_delta_normalization(backbone)

    def _processed_inputs(self, values):
        return _processed_inputs(values)

    def _hidden_states(self, inputs):
        features = self._embeddings(inputs)
        return self.backbone.language_model.model(
            inputs["input_ids"], inputs_embeds=features["inputs_embeds"],
            position_ids=features["position_ids"], cache=None)

    def _prefill(self, language, ids, cache, features):
        return language(ids, cache=cache, **features)

    def _step(self, language, token, cache, features):
        import mlx.core as mx

        return language(mx.array([[token]], dtype=mx.int32), cache=cache, rope_deltas=features["rope_deltas"])

    def _embeddings(self, inputs):
        """Official vision and MRoPE; also handle images and videos together.

        mlx-vlm 0.7.1's convenience method selects only one pixel tensor when
        both kinds are present, so scatter each modality independently here.
        """
        import mlx.core as mx
        from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter

        ids = inputs["input_ids"]
        model = self.backbone
        hidden = model.language_model.model.embed_tokens(ids)
        for pixel_key, grid_key, token_id in (
            ("pixel_values", "image_grid_thw", model.config.image_token_id),
            ("pixel_values_videos", "video_grid_thw", model.config.video_token_id),
        ):
            token_mask = ids == token_id
            n_tokens = int(token_mask.sum().item())
            if pixel_key not in inputs:
                if n_tokens:
                    raise ValueError(f"Missing {pixel_key} for {n_tokens} media tokens")
                continue
            dtype = model.vision_tower.patch_embed.proj.weight.dtype
            features, _ = model.vision_tower(inputs[pixel_key].astype(dtype), inputs[grid_key])
            if n_tokens != features.shape[0]:
                raise ValueError(f"{pixel_key}: {n_tokens} tokens != {features.shape[0]} visual features")
            mask = mx.broadcast_to(token_mask[..., None], hidden.shape)
            hidden = masked_scatter(hidden, mask, features.astype(hidden.dtype))
        positions, deltas = model.language_model.get_rope_index(
            ids, inputs.get("image_grid_thw"), inputs.get("video_grid_thw")
        )
        return {"inputs_embeds": hidden, "position_ids": positions, "rope_deltas": deltas}

    def predict_logits(self, encodings: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        """Raw float32 option scores, retaining precision for later calibration."""
        import mlx.core as mx

        result = []
        with self._decision_mode(True):
            for encoded in encodings:
                values = dict(encoded.get("inputs", {}))
                values.update(encoded.get("model_inputs", {}))
                for marker_field in ("option_positions", "option_mask", "decision_positions"):
                    values.pop(marker_field, None)
                values.update({key: encoded[key] for key in _INPUT_FIELDS if key in encoded})
                has_media = any(values.get(key) is not None for key in ("pixel_values", "pixel_values_videos"))
                limit = self.config.get("multimodal_max_length", 8192) if has_media else self.config.get("max_length", 512)
                validate_encoding(encoded, int(limit))
                values["input_ids"] = [encoded["ids"]]
                inputs = self._processed_inputs(values)
                hidden = self._hidden_states(inputs).astype(mx.float32)
                query = hidden[0, encoded["decision_position"]] @ self.q_weight.T
                keys = hidden[0, mx.array(encoded["option_positions"])] @ self.k_weight.T
                logits = (keys @ query) / math.sqrt(self.config["pointer_dim"])
                # Materialize while adapters are still enabled.
                mx.eval(logits)
                result.append(np.asarray(logits).copy())
        return result

    def predict_encoded(self, encodings: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        """Uncalibrated probabilities; use predict_logits for temperature fitting."""
        result = []
        for logits in self.predict_logits(encodings):
            probabilities = np.exp(logits - logits.max())
            result.append(probabilities / probabilities.sum())
        return result

    def generate_native(self, *, max_new_tokens=256, do_sample=False,
                        temperature=1.0, top_p=1.0, top_k=None, **processed_inputs):
        """Return [1, prompt + continuation] IDs with decision adapters disabled.

        Supports greedy decoding or temperature/nucleus sampling on original
        text, image, and video processor inputs. Every call starts a new cache.
        """
        import mlx.core as mx

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or (do_sample and temperature <= 0):
            raise ValueError("Sampling temperature must be finite and positive")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        top_k = self.generation_config.get("top_k", 50) if top_k is None else top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        with self._decision_mode(False):
            # Tensor conversion and tolist() can themselves schedule MLX work,
            # so they also belong inside the lock and thread-local streams.
            inputs = self._processed_inputs(processed_inputs)
            ids = inputs["input_ids"]
            generated = ids.tolist()[0]
            if max_new_tokens == 0:
                return np.asarray([generated], dtype=np.int64)
            features = self._embeddings(inputs)
            language = self.backbone.language_model
            cache = language.make_cache()
            output = self._prefill(language, ids, cache, features)
            # MLX's process-global RNG may retain a lazy key graph from the
            # loading thread (including unused random model initialization).
            # Keep sampling state local to this call and pass explicit keys.
            sampling_key = mx.random.key(secrets.randbits(32)) if do_sample else None
            # Match HF GenerationConfig: explicit generation settings, then the
            # original text config. mlx-vlm's top-level config adds im_end to
            # EOS, which would stop earlier than the original Qwen3.5 model.
            text_config = getattr(self.backbone.config, "text_config", self.backbone.config)
            eos = self.generation_config.get("eos_token_id", text_config.eos_token_id)
            eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
            for step in range(max_new_tokens):
                logits = output.logits[0, -1].astype(mx.float32)
                if do_sample:
                    sampling_key, token_key = mx.random.split(sampling_key)
                    logits = logits / temperature
                    if top_k > 0 or top_p < 1:
                        order = mx.argsort(-logits)
                        if top_k > 0:
                            order = order[:min(top_k, logits.shape[-1])]
                        sorted_logits = logits[order]
                        probabilities = mx.softmax(sorted_logits)
                        retained = mx.cumsum(probabilities) - probabilities < top_p
                        selected = mx.random.categorical(mx.where(retained, sorted_logits, -float("inf")), key=token_key)
                        token = int(order[selected].item())
                    else:
                        token = int(mx.random.categorical(logits, key=token_key).item())
                else:
                    token = int(mx.argmax(logits).item())
                generated.append(token)
                if token in eos or step + 1 == max_new_tokens:
                    break
                output = self._step(language, token, cache, features)
        return np.asarray([generated], dtype=np.int64)


# A descriptive alias for callers that use the Torch class name QevModel.
QevMLX = MLXRuntime
