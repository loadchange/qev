"""Backbone-agnostic Qev decision readout for the diffusion experiment.

The Qwen3.5 baseline keeps ``qev.model.QevModel``. This module puts the same
pointer head and LoRA recipe on smaller backbones and lets the decision path
choose its attention pattern:

``causal``  the backbone's own causal mask.
``block``   diffusion-style block attention. State tokens attend bidirectionally
            within the state; question tokens attend to the whole state and
            bidirectionally within their own question. Because the state never
            sees a question, several questions can share one state exactly.
``full``    MDLM-style bidirectional attention over the complete row.

Native generation is not part of this experiment; the foundation weights stay
frozen and the adapter remains separate, as in Qev.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qev.model import PointerHead

QWEN3_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BACKBONES = {
    "qwen3": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
              "family": "qwen", "kind": "qwen3", "lora_targets": QWEN3_TARGETS},
    # Same architecture and tensor names as Qwen3-0.6B; weights adapted with
    # masked diffusion (MDLM) by dLLM. Loaded into Qwen3ForCausalLM so the
    # attention pattern is chosen here rather than by remote code.
    "a2d-qwen3": {"repo": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
                  "revision": "c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
                  "family": "qwen", "kind": "qwen3", "lora_targets": QWEN3_TARGETS},
    "lfm2-230m": {"repo": "LiquidAI/LFM2.5-230M", "revision": "40cb2ad3b3044d5a41eee083a6103c8b523afa45",
                  "family": "lfm2", "kind": "lfm2",
                  "lora_targets": ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3")},
    "lfm2-vl": {"repo": "LiquidAI/LFM2.5-VL-450M", "revision": "fc6221ca597f3315e4f82fc2df606783267b34ba",
                "family": "lfm2", "kind": "lfm2_vl",
                "lora_targets": ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3")},
}
SPECIAL = {
    "qwen": {"state": "<|fim_prefix|>", "question": "<|fim_middle|>", "option": "<|box_start|>",
             "end_option": "<|box_end|>", "decision": "<|fim_suffix|>"},
    "lfm2": {"state": "<|fim_pre|>", "question": "<|fim_mid|>", "option": "<|tool_call_start|>",
             "end_option": "<|tool_call_end|>", "decision": "<|fim_suf|>"},
}
ATTENTION_MODES = ("causal", "block", "full")
_SPECIAL_RE = re.compile(r"<\|([^<>\r\n]*?)\|>")


def clean(text: str, family: str) -> str:
    """Prevent user text from creating structural or media special tokens."""
    text = _SPECIAL_RE.sub(r"<¦\1¦>", str(text))
    return text.replace("<image>", "<¦image¦>") if family == "lfm2" else text


def special_ids(tokenizer, family: str) -> dict[str, int]:
    ids = {}
    for name, token in SPECIAL[family].items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id or \
                tokenizer(token, add_special_tokens=False)["input_ids"] != [token_id]:
            raise ValueError(f"Tokenizer lacks the single-token delimiter {token!r}")
        ids[name] = int(token_id)
    if len(set(ids.values())) != len(ids):
        raise ValueError("Structural delimiter IDs must be distinct")
    return ids


def prefix_ids(tokenizer, family: str) -> list[int]:
    """LFM2 is trained with a leading BOS; Qwen (and Qev) use none."""
    return [int(tokenizer.bos_token_id)] if family == "lfm2" and tokenizer.bos_token_id is not None else []


def encode_question(state, question, tokenizer, ids, *, family, prefix=(), max_length=1024, max_state=384):
    """qev.tokenization.encode_question plus segment ids (0 = state, 1 = question).

    Only the state is truncated; ``max_state`` includes the state delimiter and
    excludes the backbone prefix, which counts toward ``max_length``.
    """
    options, instruction = question.get("options"), question.get("instr")
    if not isinstance(options, list) or not options or not all(isinstance(v, str) for v in options):
        raise ValueError("question.options must be a nonempty list of strings")
    if len(options) > 255 or not isinstance(instruction, str):
        raise ValueError("Expected an instruction and at most 255 options")
    user = lambda text: list(tokenizer(clean(text, family), add_special_tokens=False)["input_ids"])
    instruction_ids, spans = user(instruction), [user(option) for option in options]
    fixed = len(prefix) + 3 + len(instruction_ids) + sum(len(span) + 2 for span in spans)
    if fixed > max_length:
        raise ValueError(f"Question and all {len(options)} options need {fixed} tokens before state")
    state_ids = user(state)
    budget = min(max_state - 1, max_length - fixed)
    head = [*prefix, ids["state"], *state_ids[:budget]]
    body, positions = [ids["question"], *instruction_ids], []
    for span in spans:
        body.extend([ids["option"], *span, ids["end_option"]])
        positions.append(len(head) + len(body) - 1)
    body.append(ids["decision"])
    return {"ids": head + body, "segments": [0] * len(head) + [1] * len(body),
            "option_positions": positions, "decision_position": len(head) + len(body) - 1,
            "state_truncated": len(state_ids) > budget, "state_tokens": min(len(state_ids), budget)}


def pack_questions(encodings):
    """One row holding a shared state and every question, with Kev-style positions.

    Each question keeps the positions it has in its independent row, so block
    attention (or per-question causal attention) reproduces independent rows.
    """
    first = encodings[0]
    state_length = first["segments"].index(1)
    state = first["ids"][:state_length]
    if any(e["ids"][:state_length] != state or e["segments"].index(1) != state_length for e in encodings):
        raise ValueError("Packed questions must share an identical state segment")
    ids, segments, positions = list(state), [0] * state_length, list(range(state_length))
    questions = []
    for index, encoding in enumerate(encodings, start=1):
        offset = len(ids) - state_length
        body = encoding["ids"][state_length:]
        ids.extend(body)
        segments.extend([index] * len(body))
        positions.extend(range(state_length, state_length + len(body)))
        questions.append({"option_positions": [p + offset for p in encoding["option_positions"]],
                          "decision_position": encoding["decision_position"] + offset})
    return {"ids": ids, "segments": segments, "position_ids": positions, "questions": questions}


def collate(encodings, pad_token_id, device=None):
    """Right-pad independent rows; padding gets segment -1."""
    from qev.tokenization import collate_encodings

    batch = collate_encodings(encodings, pad_token_id, device)
    segments = torch.full(batch["input_ids"].shape, -1, dtype=torch.long, device=device)
    for row, encoding in enumerate(encodings):
        segments[row, :len(encoding["segments"])] = torch.tensor(encoding["segments"], device=device)
    batch["segments"] = segments
    return batch


def attention_mask_4d(attention_mask, segments, mode, position_ids=None):
    """Boolean [batch, 1, query, key] masks; True means attend."""
    length = attention_mask.shape[-1]
    allowed = attention_mask.bool()[:, None, None, :].expand(-1, 1, length, -1)
    same_or_state = (segments[:, None, None, :] == 0) | (segments[:, None, :, None] == segments[:, None, None, :])
    if mode == "block":
        allowed = allowed & same_or_state
    elif mode == "packed_causal":
        if position_ids is None:
            raise ValueError("packed_causal requires position_ids")
        allowed = allowed & same_or_state & (position_ids[:, None, None, :] <= position_ids[:, None, :, None])
    elif mode != "full":
        raise ValueError(f"Unsupported dense attention mode {mode!r}")
    # A padded query attends at least to itself, so no softmax row is empty.
    return allowed | torch.eye(length, dtype=torch.bool, device=allowed.device)[None, None]


def _load_qwen3(repo, revision, dtype, local_files_only):
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3Config, Qwen3ForCausalLM

    raw = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision,
                                          local_files_only=local_files_only)).read_text())
    raw = {k: v for k, v in raw.items()
           if k not in ("auto_map", "architectures", "model_type", "transformers_version", "dtype")}
    return Qwen3ForCausalLM.from_pretrained(
        repo, revision=revision, config=Qwen3Config(**raw), dtype=dtype, attn_implementation="sdpa",
        output_loading_info=True, local_files_only=local_files_only)


def load_foundation(name, *, dtype=torch.float32, local_files_only=False):
    spec = BACKBONES[name]
    if spec["kind"] == "qwen3":
        foundation, loading = _load_qwen3(spec["repo"], spec["revision"], dtype, local_files_only)
    elif spec["kind"] == "lfm2":
        from transformers import Lfm2ForCausalLM

        foundation, loading = Lfm2ForCausalLM.from_pretrained(
            spec["repo"], revision=spec["revision"], dtype=dtype, attn_implementation="sdpa",
            output_loading_info=True, local_files_only=local_files_only)
    else:
        from transformers import Lfm2VlForConditionalGeneration

        foundation, loading = Lfm2VlForConditionalGeneration.from_pretrained(
            spec["repo"], revision=spec["revision"], dtype=dtype, attn_implementation="sdpa",
            output_loading_info=True, local_files_only=local_files_only)
    missing = [key for key in loading.get("missing_keys", []) if not key.startswith("lm_head")]
    if missing or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Incomplete {name} checkpoint load: {loading}")
    return foundation


def load_tokenizer(name, *, local_files_only=False):
    from transformers import AutoTokenizer

    spec = BACKBONES[name]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"],
                                              local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_processor(name, *, local_files_only=False):
    from transformers import AutoProcessor

    spec = BACKBONES[name]
    return AutoProcessor.from_pretrained(spec["repo"], revision=spec["revision"],
                                         local_files_only=local_files_only)


class QevDModel(nn.Module):
    """Frozen backbone, switchable decision LoRA, float32 pointer head."""

    def __init__(self, foundation: nn.Module, config: dict[str, Any]):
        super().__init__()
        if config["attention"] not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}")
        self.foundation = foundation
        self.config = dict(config)
        self.head = PointerHead(int(config["hidden_size"]), int(config.get("pointer_dim", 256)))
        self.head.to(device=next(foundation.parameters()).device, dtype=torch.float32)

    @property
    def text_model(self):
        return self.foundation.model if self.config["kind"] in ("qwen3", "lfm2") else self.foundation.model.language_model

    def set_text_model(self, module):
        # Assign inside the foundation: nn.Module.__setattr__ would bypass a
        # property setter and register a second, detached submodule instead.
        if self.config["kind"] in ("qwen3", "lfm2"):
            self.foundation.model = module
        else:
            self.foundation.model.language_model = module

    @property
    def _base_text(self):
        model = self.text_model
        return model.get_base_model() if hasattr(model, "get_base_model") else model

    @classmethod
    def from_pretrained(cls, backbone, *, attention="causal", lora_rank=16, lora_alpha=None,
                        lora_dropout=0.05, pointer_dim=256, dtype=torch.float32, device=None,
                        max_length=1024, max_state=384, local_files_only=False):
        from peft import LoraConfig, TaskType, get_peft_model

        spec = BACKBONES[backbone]
        foundation = load_foundation(backbone, dtype=dtype, local_files_only=local_files_only)
        foundation.requires_grad_(False)
        text_config = getattr(foundation.config, "text_config", foundation.config)
        config = {"format": "qevd", "format_version": 1, "backbone": backbone, "repo": spec["repo"],
                  "revision": spec["revision"], "family": spec["family"], "kind": spec["kind"],
                  "attention": attention, "hidden_size": text_config.hidden_size,
                  "pointer_dim": pointer_dim, "lora_rank": lora_rank,
                  "lora_alpha": lora_alpha if lora_alpha is not None else 2 * lora_rank,
                  "lora_dropout": lora_dropout, "lora_targets": list(spec["lora_targets"]),
                  "max_length": max_length, "max_state": max_state}
        model = cls(foundation, config)
        model.set_text_model(get_peft_model(model.text_model, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=lora_rank, lora_alpha=config["lora_alpha"],
            lora_dropout=lora_dropout, target_modules=list(spec["lora_targets"]), bias="none")))
        if device is not None:
            model.to(device)
        return model

    def embed(self, input_ids, pixel_values=None, spatial_shapes=None, pixel_attention_mask=None):
        embeddings = self._base_text.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return embeddings
        vlm = self.foundation.model
        features = vlm.get_image_features(pixel_values=pixel_values, spatial_shapes=spatial_shapes,
                                          pixel_attention_mask=pixel_attention_mask, return_dict=True).pooler_output
        features = torch.cat(features, dim=0).to(embeddings.device, embeddings.dtype)
        placeholder = vlm.get_placeholder_mask(input_ids=input_ids, inputs_embeds=embeddings,
                                               image_features=features)
        return embeddings.masked_scatter(placeholder, features)

    def hidden_states(self, input_ids, attention_mask, segments=None, position_ids=None, mode=None, **media):
        mode = mode or self.config["attention"]
        embeddings = self.embed(input_ids, **media)
        if mode == "causal":
            mask = attention_mask
        else:
            if segments is None:
                raise ValueError(f"{mode} attention requires segment ids")
            if self.config["kind"] in ("lfm2", "lfm2_vl") and mode == "packed_causal":
                raise ValueError("LFM2 short convolutions cross packed segments; use independent rows")
            mask = {"full_attention": attention_mask_4d(attention_mask, segments, mode, position_ids)}
            if self.config["kind"] in ("lfm2", "lfm2_vl"):
                from transformers.masking_utils import create_recurrent_attention_mask

                mask["conv"] = create_recurrent_attention_mask(
                    config=self._base_text.config, inputs_embeds=embeddings, attention_mask=attention_mask)
        return self.text_model(inputs_embeds=embeddings, attention_mask=mask, position_ids=position_ids,
                               use_cache=False).last_hidden_state

    def forward(self, input_ids, attention_mask, option_positions, option_mask, decision_positions,
                segments=None, position_ids=None, **media):
        batch, length = input_ids.shape
        valid = option_mask.bool()
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("Every row must have at least one option")
        if bool(((decision_positions < 0) | (decision_positions >= length)).any()):
            raise ValueError("decision_positions are outside the sequence")
        safe = option_positions.masked_fill(~valid, 0)
        hidden = self.hidden_states(input_ids, attention_mask, segments, position_ids, **media)
        rows = torch.arange(batch, device=input_ids.device)
        logits = self.head(hidden[rows, decision_positions], hidden[rows[:, None], safe])
        return logits.masked_fill(~valid, torch.finfo(torch.float32).min)

    @torch.inference_mode()
    def forward_packed(self, packed, device):
        """Logits for every question of one packed request in a single forward."""
        ids = torch.tensor([packed["ids"]], device=device)
        segments = torch.tensor([packed["segments"]], device=device)
        positions = torch.tensor([packed["position_ids"]], device=device)
        mode = "packed_causal" if self.config["attention"] == "causal" else "block"
        if self.config["attention"] == "full":
            raise ValueError("full attention lets the state see questions; it cannot share a state")
        hidden = self.hidden_states(ids, torch.ones_like(ids), segments, positions, mode=mode)[0]
        return [self.head(hidden[q["decision_position"]][None], hidden[q["option_positions"]][None])[0]
                for q in packed["questions"]]

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def enable_gradient_checkpointing(self):
        self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.text_model.enable_input_require_grads()

    def save_pretrained(self, path, extra_config=None):
        from safetensors.torch import save_file

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.text_model.save_pretrained(output / "adapter", safe_serialization=True)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(output / "pointer.safetensors"))
        config = {**self.config, **(extra_config or {})}
        (output / "qevd_config.json").write_text(json.dumps(config, indent=2) + "\n")

    @classmethod
    def from_checkpoint(cls, path, *, dtype=torch.float32, device=None, local_files_only=False):
        from peft import PeftModel
        from safetensors.torch import load_file

        path = Path(path)
        config = json.loads((path / "qevd_config.json").read_text())
        if config.get("format") != "qevd":
            raise ValueError("Not a QevD checkpoint")
        foundation = load_foundation(config["backbone"], dtype=dtype, local_files_only=local_files_only)
        foundation.requires_grad_(False)
        model = cls(foundation, config)
        model.set_text_model(PeftModel.from_pretrained(model.text_model, path / "adapter", is_trainable=False))
        model.head.load_state_dict(load_file(str(path / "pointer.safetensors")), strict=True)
        if device is not None:
            model.to(device)
        model.eval().requires_grad_(False)
        return model


def encode_multimodal_question(state, question, processor, ids, *, images=(), family="lfm2",
                               max_length=8192):
    """One LFM2-VL image decision; placeholders expand inside the state segment."""
    tokenizer = processor.tokenizer
    prefix = "".join(tokenizer.convert_ids_to_tokens(prefix_ids(tokenizer, family)))
    special = SPECIAL[family]
    prompt = prefix + special["state"] + clean(state, family) + "<image>" * len(images)
    prompt += special["question"] + clean(question["instr"], family)
    for option in question["options"]:
        prompt += special["option"] + clean(option, family) + special["end_option"]
    prompt += special["decision"]
    kwargs = {"text": [prompt], "return_tensors": "pt", "add_special_tokens": False}
    if images:
        kwargs["images"] = [list(images)]
    inputs = dict(processor(**kwargs))
    token_ids = inputs["input_ids"][0].tolist()
    if len(token_ids) > max_length:
        raise ValueError(f"Expanded multimodal question needs {len(token_ids)} tokens")
    positions = [i for i, t in enumerate(token_ids) if t == ids["end_option"]]
    decisions = [i for i, t in enumerate(token_ids) if t == ids["decision"]]
    questions = [i for i, t in enumerate(token_ids) if t == ids["question"]]
    if len(positions) != len(question["options"]) or decisions != [len(token_ids) - 1] or len(questions) != 1:
        raise ValueError("Processor did not preserve the candidate structure")
    segments = torch.tensor([[0 if i < questions[0] else 1 for i in range(len(token_ids))]])
    inputs.update(option_positions=torch.tensor([positions]), option_mask=torch.ones((1, len(positions)), dtype=torch.bool),
                  decision_positions=torch.tensor(decisions), segments=segments)
    inputs.setdefault("attention_mask", torch.ones_like(inputs["input_ids"]))
    return inputs


def softmax(logits, temperature=1.0):
    z = [float(v) / temperature for v in logits]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    return [v / sum(e) for v in e]
