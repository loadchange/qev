"""Full multimodal Qwen3.5 foundation plus a switchable decision adapter.

Each batch row is a complete, independent question. Qwen3.5 mixes recurrent
Gated DeltaNet layers with ordinary attention: a Kev-style block attention mask
would not isolate its recurrent state. We deliberately use neither packed
question branches nor a shared prefix cache.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


class PointerHead(nn.Module):
    """One shared candidate scorer, independent of the number of options."""

    def __init__(self, hidden_size: int, pointer_dim: int = 256, bias: bool = False):
        super().__init__()
        if hidden_size < 1 or pointer_dim < 1:
            raise ValueError("hidden_size and pointer_dim must be positive")
        self.q = nn.Linear(hidden_size, pointer_dim, bias=bias)
        self.k = nn.Linear(hidden_size, pointer_dim, bias=bias)
        self.scale = 1.0 / math.sqrt(pointer_dim)

    def forward(self, decision: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        # Explicitly keep the small readout in float32, including under bf16
        # autocast used to train the much larger backbone.
        with torch.autocast(device_type=decision.device.type, enabled=False):
            query = self.q(decision.float())
            keys = self.k(options.float())
            return (keys * query[:, None, :]).sum(dim=-1) * self.scale


def _load_foundation(base_model, revision, dtype, attn_implementation, local_files_only=False):
    """Keep vision, language, and the tied vocabulary head exactly as published."""
    from transformers import Qwen3_5ForConditionalGeneration

    foundation, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model, revision=revision, dtype=dtype,
        attn_implementation=attn_implementation, output_loading_info=True,
        local_files_only=local_files_only,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Incomplete multimodal Qwen3.5 checkpoint load: {loading}")
    return foundation


class QevModel(nn.Module):
    """A frozen Qwen3.5 base with trainable LoRA adapters and pointer head.

    ``forward`` returns float32 ``[batch, max_options]`` logits. Padded option
    slots are masked. Labels belong to the caller's loss function, allowing the
    same forward path for training, evaluation and serving.
    """

    def __init__(self, foundation: nn.Module, config: dict[str, Any]):
        super().__init__()
        self.foundation = foundation
        self.config = dict(config)
        self.head = PointerHead(
            int(config["hidden_size"]),
            int(config.get("pointer_dim", 256)),
            bool(config.get("pointer_bias", False)),
        )
        self.head.to(device=next(foundation.parameters()).device, dtype=torch.float32)

    @property
    def backbone(self):
        """The language adapter, retained as a training/integration convenience."""
        return self.foundation.model.language_model

    @classmethod
    def from_pretrained(
        cls,
        base_model: str = DEFAULT_BASE_MODEL,
        *,
        revision: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.05,
        lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
        pointer_dim: int = 256,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
        attn_implementation: str = "sdpa",
        max_length: int = 512,
        max_state: int = 320,
        local_files_only: bool = False,
    ) -> QevModel:
        from peft import LoraConfig, TaskType, get_peft_model

        if lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if not 0 <= lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not lora_targets:
            raise ValueError("At least one LoRA target is required")
        foundation = _load_foundation(
            base_model, revision, dtype, attn_implementation, local_files_only
        )
        foundation.requires_grad_(False)
        resolved_revision = getattr(foundation.config, "_commit_hash", None) or revision
        config = {
            "format": "qev",
            "format_version": 2,
            "foundation_architecture": "Qwen3_5ForConditionalGeneration",
            "modalities": ["text", "image", "video"],
            "native_generation": "adapter_disabled",
            "base_model": str(base_model),
            "base_revision": resolved_revision,
            "hidden_size": foundation.config.text_config.hidden_size,
            "pointer_dim": pointer_dim,
            "pointer_bias": False,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha if lora_alpha is not None else 2 * lora_rank,
            "lora_dropout": lora_dropout,
            "lora_targets": list(lora_targets),
            "question_isolation": "independent_rows",
            "max_length": max_length,
            "max_state": max_state,
        }
        foundation.model.language_model = get_peft_model(
            foundation.model.language_model,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=config["lora_alpha"],
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),
                bias="none",
            ),
        )
        if device is not None:
            foundation.to(device)
        return cls(foundation, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_positions: torch.Tensor,
        option_mask: torch.Tensor,
        decision_positions: torch.Tensor,
        **multimodal_inputs,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, length]")
        batch, length = input_ids.shape
        if not batch or not length:
            raise ValueError("Empty input batches are not supported")
        if option_positions.ndim != 2 or option_positions.shape[0] != batch:
            raise ValueError("option_positions must have shape [batch, options]")
        if option_mask.shape != option_positions.shape or decision_positions.shape != (batch,):
            raise ValueError("Invalid option_mask or decision_positions shape")
        valid_options = option_mask.bool()
        if not bool(valid_options.any(dim=-1).all()):
            raise ValueError("Every row must have at least one option")
        if bool(((decision_positions < 0) | (decision_positions >= length)).any()):
            raise ValueError("decision_positions are outside the sequence")
        if bool((((option_positions < 0) | (option_positions >= length)) & valid_options).any()):
            raise ValueError("Valid option_positions are outside the sequence")
        safe_positions = option_positions.masked_fill(~valid_options, 0)
        rows = torch.arange(batch, device=input_ids.device)
        if not bool(attention_mask[rows, decision_positions].bool().all()):
            raise ValueError("A decision marker points into padding")
        if bool((~attention_mask[rows[:, None], safe_positions].bool() & valid_options).any()):
            raise ValueError("An option marker points into padding")
        # No past_key_values are carried between rows/calls. The official model
        # creates its attention and recurrent padding masks from this 2D mask.
        hidden = self.foundation.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **multimodal_inputs,
        ).last_hidden_state
        logits = self.head(hidden[rows, decision_positions], hidden[rows[:, None], safe_positions])
        return logits.masked_fill(~valid_options, torch.finfo(torch.float32).min)

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def get_tokenizer(self, *, local_files_only: bool = False):
        from transformers import AutoTokenizer

        source = getattr(self, "_tokenizer_source", None)
        return AutoTokenizer.from_pretrained(
            source or self.config["base_model"],
            revision=None if source else self.config.get("base_revision"),
            local_files_only=local_files_only,
        )

    def get_processor(self, *, local_files_only: bool = False):
        """The complete original image/video/text processor, not just a tokenizer."""
        from .processing import load_processor

        source = getattr(self, "_processor_source", None)
        return load_processor(
            source or self.config["base_model"],
            revision=None if source else self.config.get("base_revision"),
            local_files_only=local_files_only,
        )

    @torch.inference_mode()
    def generate_native(self, *args, **kwargs):
        """Run the original full model with all decision adapters disabled.

        This preserves base text/image/video generation; it does not claim the
        decision-adapted language weights have unchanged generative behavior.
        Callers must serialize concurrent training/generation on this instance.
        """
        was_training = self.training
        try:
            self.eval()
            with self.backbone.disable_adapter():
                return self.foundation.generate(*args, **kwargs)
        finally:
            self.train(was_training)

    def enable_gradient_checkpointing(self):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.backbone.enable_input_require_grads()

    def save_pretrained(self, path, *, tokenizer=None, processor=None, extra_config=None):
        """Save adapter plus readout, not a second copy of the base checkpoint."""
        from peft import PeftModel
        from safetensors.torch import save_file

        if not isinstance(self.backbone, PeftModel):
            raise ValueError("Saving an adapter checkpoint requires an unmerged PEFT backbone")  # noqa: TRY004 -- invalid checkpoint state
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        config = dict(self.config)
        if extra_config:
            protected = set(config) - {"max_length", "max_state"}
            conflict = protected & set(extra_config)
            if conflict:
                raise ValueError(f"Cannot override model metadata: {sorted(conflict)}")
            config.update(extra_config)
        self.backbone.save_pretrained(output / "adapter", safe_serialization=True)
        save_file(
            {name: tensor.detach().float().cpu().contiguous() for name, tensor in self.head.state_dict().items()},
            str(output / "pointer.safetensors"),
        )
        (output / "qev_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if tokenizer is not None:
            tokenizer.save_pretrained(output / "tokenizer")
        if processor is not None:
            processor.save_pretrained(output / "processor")
        elif (output / "processor" / "processor_config.json").is_file():
            pass
        else:
            # The processor is a small, essential part of the full-modality
            # checkpoint. Offline training can pass an already loaded one.
            self.get_processor().save_pretrained(output / "processor")

    @classmethod
    def from_checkpoint(
        cls,
        path,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
        merge: bool = False,
        trainable: bool = False,
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ) -> QevModel:
        from peft import PeftModel
        from safetensors.torch import load_file

        if merge:
            raise ValueError("Merging decision LoRA would alter the native base; keep the switchable adapter")
        path = Path(path)
        config = json.loads((path / "qev_config.json").read_text())
        if config.get("format") != "qev" or config.get("format_version") != 2:
            raise ValueError("Unsupported Qev checkpoint format")
        if config.get("native_generation") != "adapter_disabled":
            raise ValueError("Checkpoint does not preserve adapter-disabled native generation")
        if config.get("question_isolation") != "independent_rows":
            raise ValueError("Only independent question rows are supported")
        foundation = _load_foundation(
            config["base_model"], config.get("base_revision"),
            dtype, attn_implementation, local_files_only,
        )
        foundation.requires_grad_(False)
        foundation.model.language_model = PeftModel.from_pretrained(
            foundation.model.language_model, path / "adapter", is_trainable=trainable
        )
        if device is not None:
            foundation.to(device)
        model = cls(foundation, config)
        model.head.load_state_dict(load_file(str(path / "pointer.safetensors")), strict=True)
        if (path / "tokenizer" / "tokenizer_config.json").is_file():
            model._tokenizer_source = str(path / "tokenizer")
        if (path / "processor").is_dir():
            model._processor_source = str(path / "processor")
        if trainable:
            model.train()
        else:
            model.eval()
            model.requires_grad_(False)
        return model
