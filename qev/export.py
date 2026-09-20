"""Export full Qwen3.5 vision/language weights and unmerged Qev LoRA to MLX.

The official mlx-vlm converter handles convolution layouts and zero-centred
RMSNorm conventions. Decision adapters remain separate, so native generation
can use the original foundation without the decision fine-tuning applied.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def multimodal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the complete foundation and align explicit hybrid layer layout."""
    result = json.loads(json.dumps(config))
    if result.get("model_type") != "qwen3_5" or not result.get("vision_config"):
        raise ValueError("Export requires the complete dense multimodal Qwen3.5 foundation")
    text = result.get("text_config", {})
    if not text or text.get("num_experts", 0):
        raise ValueError("Export requires a dense Qwen3.5 text_config")
    layer_types = text.get("layer_types")
    if layer_types:
        if "full_attention" not in layer_types or "linear_attention" not in layer_types:
            raise ValueError("Expected Qwen3.5 hybrid full/linear attention layers")
        interval = layer_types.index("full_attention") + 1
        expected = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(len(layer_types))]
        if layer_types != expected:
            raise ValueError("mlx-vlm requires periodically spaced full-attention layers")
        text["full_attention_interval"] = interval
    return result


def convert_adapter_weights(weights: Mapping[str, Any], adapter_config: Mapping[str, Any]) -> dict[str, Any]:
    """Map PEFT's unmerged linear LoRA pairs without changing their arithmetic."""
    if (adapter_config.get("peft_type") != "LORA" or adapter_config.get("bias", "none") != "none"
            or any(adapter_config.get(key) for key in ("use_dora", "use_rslora", "fan_in_fan_out",
                                                        "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias"))):
        raise ValueError("Export supports plain bias-free linear LoRA only")
    result = {}
    pattern = re.compile(r"^base_model\.model\.(layers\..+)\.lora_([AB])(?:\.default)?\.weight$")
    for key, value in weights.items():
        match = pattern.fullmatch(key)
        if not match:
            raise ValueError(f"Unsupported PEFT adapter tensor: {key}")
        mapped = f"language_model.model.{match[1]}.lora_{match[2].lower()}"
        if mapped in result:
            raise ValueError(f"Duplicate LoRA adapter tensor: {mapped}")
        result[mapped] = value
    if not result:
        raise ValueError("Adapter contains no tensors")
    for key, value in result.items():
        partner = key[:-1] + ("b" if key.endswith("a") else "a")
        if partner not in result or value.ndim != 2:
            raise ValueError(f"Missing or malformed LoRA pair: {key}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PROVENANCE = {
    "qev_config.json": "source_config_sha256",
    "pointer.safetensors": "source_pointer_sha256",
    "adapter/adapter_model.safetensors": "source_adapter_sha256",
    "adapter/adapter_config.json": "source_adapter_config_sha256",
}


def _preserve_foundation_documents(source: Path, output: Path) -> None:
    for pattern in ("LICENSE*", "NOTICE*"):
        for file in source.glob(pattern):
            if file.is_file():
                shutil.copy2(file, output / file.name)
    if (source / "README.md").is_file():
        shutil.copy2(source / "README.md", output / "BASE_MODEL_README.md")


def _stage_foundation(source: Path, staged: Path, config: dict) -> None:
    """Leave large weights in place; promote only zero-centred norms before +1.

    mlx-vlm sanitizes BF16 norms before applying its requested output dtype.
    BF16(x + 1) followed by float32 differs from float32(x) + 1. The original
    Torch fp32 foundation uses the latter, so provide fp32 source norm tensors.
    """
    from mlx_vlm.models.qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES
    from safetensors import safe_open
    from safetensors.torch import save_file

    staged.mkdir()
    for file in source.iterdir():
        if file.is_file() and file.name != "config.json":
            target = staged / file.name
            if file.suffix == ".safetensors":
                target.symlink_to(file.resolve())
            else:
                shutil.copy2(file, target)
                target.chmod(target.stat().st_mode | 0o200)
    (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"weight_map": {}}
    if not index["weight_map"]:
        for file in sorted(source.glob("*.safetensors")):
            with safe_open(file, framework="pt") as weights:
                index["weight_map"].update({key: file.name for key in weights.keys()})  # noqa: SIM118
    norms = {}
    for file in sorted(set(index["weight_map"].values())):
        with safe_open(source / file, framework="pt") as weights:
            for key in weights.keys():  # noqa: SIM118
                if key.startswith("model.language_model.") and any(key.endswith(suffix) for suffix in NORM_WEIGHT_SUFFIXES):
                    norms[key] = weights.get_tensor(key).float().contiguous()
    if not norms:
        raise ValueError("Original Qwen3.5 foundation is missing zero-centred RMSNorm weights")
    # Loader processes shards in sorted order; this override must follow the
    # source shard, whose file also contains original copies of these tensors.
    promoted_name = "zz-qev-fp32-source-norms.safetensors"
    if any(name >= promoted_name for name in index["weight_map"].values()):
        raise ValueError("Unsupported source shard naming for FP32 norm promotion")
    save_file(norms, str(staged / promoted_name))
    index["weight_map"].update({key: promoted_name for key in norms})
    (staged / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def export_foundation(base_model: str, output: str | Path, *, revision: str | None = None,
                      dtype: str = "float32") -> Path:
    """Preconvert an unchanged foundation while an independent adapter trains."""
    from mlx_vlm.convert import convert
    from mlx_vlm.utils import get_model_path

    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16, or float16")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Foundation destination already exists: {output}")
    source = get_model_path(base_model, revision=revision)
    config = multimodal_config(json.loads((source / "config.json").read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as work:
        work = Path(work)
        hf = work / "foundation"
        _stage_foundation(source, hf, config)
        staged = work / "export"
        convert(hf_path=str(hf), mlx_path=str(staged), dtype=dtype, trust_remote_code=False)
        from qev.mlx_runtime import load_original_processor

        load_original_processor(hf).save_pretrained(staged)
        _preserve_foundation_documents(source, staged)
        manifest = {"format": "qev_foundation", "norm_promotion": "float32_before_offset",
                    "processor": "original_transformers", "base_model": base_model,
                    "base_revision": revision, "dtype": dtype,
                    "source_config_sha256": _sha256(source / "config.json"),
                    "converter": "mlx_vlm.convert.convert",
                    "mlx_vlm_version": importlib.metadata.version("mlx-vlm")}
        (staged / "foundation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return output


def export_checkpoint(checkpoint: str | Path, output: str | Path, *, dtype: str = "float32",
                      backbone: str | Path | None = None) -> Path:
    """Atomically export a complete foundation plus separate adapter and head.

    No LoRA is merged and no vision/language output weights are removed.
    Existing outputs are never overwritten; low-precision exports still need
    measured parity before they can be treated as numerically interchangeable.
    """
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16, or float16")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    source_config = json.loads((checkpoint / "qev_config.json").read_text())
    if source_config.get("runtime") == "mlx":
        raise ValueError("Input is already an MLX export; provide a trained Torch checkpoint")
    if (source_config.get("format_version") != 2 or source_config.get("native_generation") != "adapter_disabled"
            or source_config.get("question_isolation") != "independent_rows"):
        raise ValueError("Export requires a version-2 unmerged multimodal Qev checkpoint")

    import mlx.core as mx
    from mlx_vlm.convert import convert
    from mlx_vlm.utils import get_model_path

    adapter_config = json.loads((checkpoint / "adapter" / "adapter_config.json").read_text())
    if (adapter_config.get("r") != source_config["lora_rank"]
            or adapter_config.get("lora_alpha") != source_config["lora_alpha"]):
        raise ValueError("LoRA scaling metadata does not match the saved PEFT adapter")
    adapter = convert_adapter_weights(mx.load(str(checkpoint / "adapter" / "adapter_model.safetensors")), adapter_config)
    source = get_model_path(source_config["base_model"], revision=source_config.get("base_revision"))
    foundation_config = multimodal_config(json.loads((source / "config.json").read_text()))
    reused = Path(backbone).resolve() if backbone is not None else None
    if reused is not None:
        manifest = json.loads((reused / "foundation_manifest.json").read_text())
        expected = {"format": "qev_foundation", "norm_promotion": "float32_before_offset",
                    "processor": "original_transformers", "base_model": source_config["base_model"],
                    "base_revision": source_config.get("base_revision"), "dtype": dtype,
                    "source_config_sha256": _sha256(source / "config.json")}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Preconverted foundation provenance does not match this checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as work:
        work = Path(work)
        # A local view keeps large original weight files in place. Only config
        # needs adjustment because HF dispatches layer_types, MLX an interval.
        hf = work / "foundation"
        _stage_foundation(source, hf, foundation_config)
        # Preserve the checkpoint's complete processor assets when available.
        processor_path = checkpoint / "processor"
        if processor_path.is_dir():
            for file in processor_path.iterdir():
                if file.is_file() and file.name != "config.json":
                    target = hf / file.name
                    if target.is_symlink() or target.exists():
                        target.unlink()
                    shutil.copy2(file, target)
        staged = work / "export"
        staged.mkdir()
        mx.save_safetensors(str(staged / "decision_adapters.safetensors"), adapter)
        shutil.copy2(checkpoint / "pointer.safetensors", staged / "pointer.safetensors")
        if reused is None:
            convert(hf_path=str(hf), mlx_path=str(staged / "backbone"), dtype=dtype, trust_remote_code=False)
        else:
            shutil.copytree(reused, staged / "backbone")
            for file in processor_path.glob("*"):
                if file.is_file() and file.name != "config.json":
                    shutil.copy2(file, staged / "backbone" / file.name)
        from qev.mlx_runtime import load_original_processor

        load_original_processor(hf).save_pretrained(staged / "backbone")
        _preserve_foundation_documents(source, staged / "backbone")
        exported = {
            **source_config,
            "format": "qev", "format_version": 2,
            "runtime": "mlx", "backend": "mlx_vlm", "mlx_dtype": dtype,
            "native_generation": "adapter_disabled",
            "modalities": ["text", "image", "video"],
            "export": {
                "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
                "mlx_version": importlib.metadata.version("mlx"),
                **{field: _sha256(checkpoint / filename) for filename, field in _PROVENANCE.items()},
                "converter": "mlx_vlm.convert.convert",
                "parity_status": "not_run",
                "foundation": "original_unmerged_full_multimodal",
                "norm_promotion": "float32_before_offset",
                "processor": "original_transformers",
                "vocabulary_projection": "preserved_for_native_generation",
                "gated_delta_normalization": "HF/FLA sum-of-squares epsilon=1e-6",
            },
        }
        (staged / "qev_config.json").write_text(json.dumps(exported, indent=2) + "\n")
        # Load validates that all adapter targets exist and that full processor
        # and vision/language weights survived conversion. Materialize weights
        # for thread-safe loading; this does not run model inference.
        from qev.mlx_runtime import MLXRuntime

        checked = MLXRuntime.from_checkpoint(staged)
        del checked
        staged.rename(output)
    return output


def _json_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot encode parity input type {type(value).__name__}")


def verify_export(checkpoint: str | Path, output: str | Path, encodings: list[dict[str, Any]],
                  *, atol: float = 0.002, torch_device: str = "cpu") -> dict[str, Any]:
    """Compare Torch and native MLX decisions on identical text/media inputs."""
    if not encodings:
        raise ValueError("At least one encoded parity case is required")
    import torch

    from qev.evaluation import inference_autocast, probabilities
    from qev.mlx_runtime import _MEDIA_FIELDS, MLXRuntime, compare_probabilities, validate_encoding
    from qev.model import QevModel
    from qev.tokenization import collate_encodings

    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    config = json.loads((output / "qev_config.json").read_text())
    provenance = config.get("export", {})
    for filename, field in _PROVENANCE.items():
        if provenance.get(field) != _sha256(checkpoint / filename):
            raise ValueError(f"Parity reference does not match exported source {filename}")
    for encoded in encodings:
        has_media = any(key in values for values in (encoded, encoded.get("inputs", {}), encoded.get("model_inputs", {}))
                        for key in ("pixel_values", "pixel_values_videos"))
        validate_encoding(encoded, int(config.get("multimodal_max_length", 8192) if has_media else config.get("max_length", 512)))
    model = QevModel.from_checkpoint(checkpoint, device=torch_device, dtype=torch.float32)
    tokenizer = model.get_tokenizer()
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    expected, expected_logits = [], []
    with torch.inference_mode():
        for encoded in encodings:
            batch = collate_encodings([encoded], pad_token, device=torch_device)
            media = dict(encoded.get("inputs", {}))
            media.update(encoded.get("model_inputs", {}))
            media.update({key: encoded[key] for key in _MEDIA_FIELDS if key in encoded})
            for key in _MEDIA_FIELDS:
                if key in media and media[key] is not None:
                    tensor = media[key]
                    batch[key] = (tensor.to(torch_device) if isinstance(tensor, torch.Tensor)
                                  else torch.as_tensor(tensor, device=torch_device))
            with inference_autocast(model):
                logits = model(**batch)[0].float()
                expected_logits.append(logits.cpu().numpy())
                expected.append(logits.softmax(dim=-1).cpu().numpy())
    del model, batch
    gc.collect()
    if torch_device.startswith("mps"):
        torch.mps.empty_cache()
    elif torch_device.startswith("cuda"):
        torch.cuda.empty_cache()
    runtime = MLXRuntime.from_checkpoint(output)
    actual_logits = runtime.predict_logits(encodings)
    report = compare_probabilities(expected, [probabilities(row, 1.0) for row in actual_logits], atol=atol)
    temperature = float(config.get("temperature", 1.0))
    calibrated = compare_probabilities(
        [probabilities(row, temperature) for row in expected_logits],
        [probabilities(row, temperature) for row in actual_logits], atol=atol,
    )
    report["calibrated"] = {"temperature": temperature, **calibrated}
    report["passed"] = report["passed"] and calibrated["passed"]
    report.update(
        reference=("Torch bf16 autocast full foundation + unmerged LoRA, independent rows" if torch_device.startswith("cuda")
                   else "Torch fp32 full foundation + unmerged LoRA, independent rows"),
        candidate="Native mlx-vlm full foundation + switchable unmerged LoRA + float32 pointer",
        mlx_dtype=config.get("mlx_dtype"),
        encoded_cases_sha256=hashlib.sha256(json.dumps(
            encodings, sort_keys=True, separators=(",", ":"), default=_json_value
        ).encode()).hexdigest(),
    )
    (output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    config["export"]["parity_status"] = "passed" if report["passed"] else "failed"
    (output / "qev_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--run", required=True)
    parser.add_argument("--output", "--out", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--backbone", help="Previously exported complete foundation with matching manifest")
    parser.add_argument("--verify-cases", help="JSON list of encoded question dictionaries (IDs, markers, optional media arrays)")
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--torch-device", default="cpu")
    args = parser.parse_args()
    path = export_checkpoint(args.checkpoint, args.output, dtype=args.dtype, backbone=args.backbone)
    result = {"output": str(path), "dtype": args.dtype, "parity_status": "not_run"}
    if args.verify_cases:
        report = verify_export(args.checkpoint, path, json.loads(Path(args.verify_cases).read_text()),
                               atol=args.atol, torch_device=args.torch_device)
        result["parity"] = report
        result["parity_status"] = "passed" if report["passed"] else "failed"
    print(json.dumps(result))
    if result["parity_status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
