"""LFM2.5 decisions and native generation on MLX (qev-450m and the text-only qev-230m).

Reuses MLXRuntime's thread-owned streams, switchable decision LoRA, merged
decision weights and sampling loop; only embeddings, the hidden-state forward
and generation calls differ. Images use mlx-vlm's numpy LFM2-VL processor, so
no torch is needed. Text-only checkpoints (LFM2.5-230M) load with mlx-vlm's
lfm2 text model, which uses the same language_model.model module paths, and
chat through the tokenizer's template.
"""

from __future__ import annotations

import json
from pathlib import Path

from .mlx_runtime import MLXRuntime, _array

_FIELDS = {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask", "spatial_shapes"}


def load_lfm2_processor(path, *, images=True):
    if not images:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(path))
    import mlx_vlm.models.lfm2_vl.processing_lfm2_vl  # noqa: F401 -- registers the numpy image processor
    from transformers import Lfm2VlProcessor

    processor = Lfm2VlProcessor.from_pretrained(str(path))
    template = Path(path) / "chat_template.jinja"
    if getattr(processor, "chat_template", None) is None and template.is_file():
        # mlx-vlm's loader attaches the template to the tokenizer only.
        processor.chat_template = template.read_text()
    return processor


class LFM2Runtime(MLXRuntime):
    def _prepare_backbone(self, backbone):
        """LFM2 needs no numerical patch."""

    def _processed_inputs(self, values):
        unknown = set(values) - _FIELDS
        if unknown:
            raise ValueError(f"Unsupported processor fields: {sorted(unknown)}")
        converted = {key: _array(value) for key, value in values.items() if value is not None}
        for key in ("input_ids", "attention_mask"):
            # Text-only chat templates return one unbatched row.
            if key in converted and converted[key].ndim == 1:
                converted[key] = converted[key][None]
        ids = converted.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("MLX expects one unpadded input_ids row")
        mask = converted.pop("attention_mask", None)
        if mask is not None and (mask.shape != ids.shape or not bool((mask == 1).all().item())):
            raise ValueError("MLX accepts one unpadded processor row with attention_mask=1")
        if ("pixel_values" in converted) != ("spatial_shapes" in converted):
            raise ValueError("pixel_values and spatial_shapes must be supplied together")
        return converted

    def _embeddings(self, inputs):
        ids = inputs["input_ids"]
        if "pixel_values" not in inputs:
            return {"inputs_embeds": self.backbone.language_model.model.embed_tokens(ids)}
        features = self.backbone.get_input_embeddings(
            ids, inputs["pixel_values"], spatial_shapes=inputs["spatial_shapes"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"))
        return {"inputs_embeds": features.inputs_embeds}

    def _hidden_states(self, inputs):
        embeddings = self._embeddings(inputs)["inputs_embeds"]
        return self.backbone.language_model.model(inputs["input_ids"], input_embeddings=embeddings)

    def _prefill(self, language, ids, cache, features):
        return language(ids, cache=cache, inputs_embeds=features["inputs_embeds"])

    def _step(self, language, token, cache, features):
        import mlx.core as mx

        return language(mx.array([[token]], dtype=mx.int32), cache=cache)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, decision_weights: str = "adapter") -> LFM2Runtime:
        import mlx.core as mx
        from mlx_vlm.utils import load_model

        path = Path(path).resolve()
        config = json.loads((path / "qev_config.json").read_text())
        if config.get("format") != "qev" or config.get("family") != "lfm2" or config.get("runtime") != "mlx":
            raise ValueError("Not a Qev LFM2 MLX checkpoint")
        backbone_path = path / "backbone"
        backbone = load_model(backbone_path, lazy=True)
        generation = backbone_path / "generation_config.json"
        images = "image" in config.get("modalities", [])
        processor = load_lfm2_processor(backbone_path, images=images)
        runtime = cls(backbone, mx.load(str(path / "pointer.safetensors")), config,
                      None if images else processor, processor=processor,
                      generation_config=json.loads(generation.read_text()) if generation.is_file() else None,
                      adapter_weights=mx.load(str(path / "decision_adapters.safetensors")))
        runtime.merge_decision_weights(decision_weights)
        return runtime
