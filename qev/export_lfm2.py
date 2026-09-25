"""Export a trained LFM2.5 decision checkpoint to a Qev MLX model directory.

    python -m qev.export_lfm2 --checkpoint runs/lfm2-v2 --output models/qev-450m-mlx
    python -m qev.export_lfm2 --checkpoint runs/t230-a --output models/qev-230m-mlx

LFM2.5-VL-450M checkpoints become the image+text qev-450m; LFM2.5-230M
checkpoints become the text-only qev-230m.

The foundation files are copied unchanged from the pinned Hugging Face revision
(bfloat16 safetensors that mlx-vlm loads directly). The PEFT decision LoRA is
renamed to MLX module paths and kept separate, so native generation still uses
the unmodified foundation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_TEXT_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "generation_config.json", "LICENSE")
# QevD checkpoint kind -> foundation.
FOUNDATIONS = {
    "lfm2_vl": {"base": "LiquidAI/LFM2.5-VL-450M", "revision": "fc6221ca597f3315e4f82fc2df606783267b34ba",
                "files": (*_TEXT_FILES, "processor_config.json"), "model_name": "qev-450m",
                "architecture": "Lfm2VlForConditionalGeneration", "modalities": ["text", "image"]},
    "lfm2": {"base": "LiquidAI/LFM2.5-230M", "revision": "40cb2ad3b3044d5a41eee083a6103c8b523afa45",
             "files": _TEXT_FILES, "model_name": "qev-230m",
             "architecture": "Lfm2ForCausalLM", "modalities": ["text"]},
}
NOTICE = """This model contains {base} (revision {revision}) unchanged in backbone/,
plus a decision LoRA adapter and pointer head trained by the Qev project
(https://github.com/loadchange/qev). The adapter and pointer head are new files; the
foundation weights were not modified. The whole directory is a Derivative Work distributed
under the LFM Open License v1.0 (LICENSE). Commercial use by a Legal Entity with annual
revenue of USD 10 million or more is not licensed.
"""


def convert_adapter(adapter_path):
    """PEFT lora_A/lora_B tensors -> {mlx_path.lora_a|lora_b: float32 array}."""
    import numpy as np
    from safetensors.numpy import load_file

    weights = {}
    for key, value in load_file(str(adapter_path)).items():
        if not key.startswith("base_model.model.layers.") or ".lora_" not in key:
            raise ValueError(f"Unexpected adapter tensor {key}")
        module, side = key.removeprefix("base_model.model.").rsplit(".lora_", 1)
        weights[f"language_model.model.{module}.{'lora_a' if side.startswith('A') else 'lora_b'}"] = \
            value.astype(np.float32)
    return weights


def export(checkpoint, output, *, model_name=None):
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((checkpoint / "qevd_config.json").read_text())
    foundation = FOUNDATIONS.get(source.get("kind"))
    if foundation is None or source.get("revision") != foundation["revision"]:
        raise ValueError("Expected a QevD checkpoint on "
                         + " or ".join(f["base"] for f in FOUNDATIONS.values()) + " at the pinned revision")
    if source.get("attention") != "causal":
        raise ValueError("The MLX runtime reads LFM2 decisions with causal attention")
    base = Path(snapshot_download(foundation["base"], revision=foundation["revision"]))
    (output / "backbone").mkdir(parents=True)
    for name in foundation["files"]:
        shutil.copy2(base / name, output / "backbone" / name)
    shutil.copy2(base / "LICENSE", output / "LICENSE")
    (output / "NOTICE").write_text(NOTICE.format(base=foundation["base"], revision=foundation["revision"]))
    save_file(convert_adapter(checkpoint / "adapter/adapter_model.safetensors"),
              str(output / "decision_adapters.safetensors"))
    shutil.copy2(checkpoint / "pointer.safetensors", output / "pointer.safetensors")
    config = {"format": "qev", "format_version": 3, "family": "lfm2", "runtime": "mlx", "backend": "mlx_vlm",
              "model_name": model_name or foundation["model_name"],
              "foundation_architecture": foundation["architecture"],
              "base_model": foundation["base"], "base_revision": foundation["revision"],
              "modalities": foundation["modalities"],
              "native_generation": "adapter_disabled", "question_isolation": "independent_rows",
              "hidden_size": source["hidden_size"], "pointer_dim": source["pointer_dim"], "pointer_bias": False,
              "lora_rank": source["lora_rank"], "lora_alpha": source["lora_alpha"],
              "lora_targets": source["lora_targets"], "max_length": source["max_length"],
              "max_state": source["max_state"], "multimodal_max_length": 8192,
              "temperature": source["temperature"], "training_label": source.get("label"),
              "mlx_dtype": "bfloat16"}
    (output / "qev_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", help="default: qev-450m or qev-230m by foundation")
    args = ap.parse_args(argv)
    print(export(args.checkpoint, args.output, model_name=args.model_name))


if __name__ == "__main__":
    main()
