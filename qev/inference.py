"""Shared API inference for the trained Torch and native MLX checkpoints."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .api import (
    ChatCompletionRequest,
    SystemOneRequest,
    decode_chat_messages,
    decode_state,
    to_answers,
    to_record,
)
from .evaluation import inference_autocast, probabilities
from .tokenization import encode_question


class Agent:
    def __init__(self, checkpoint, *, backend="auto", device=None, batch_size=1, calibrated=True):
        self.path = Path(checkpoint).expanduser().resolve()
        self.config = json.loads((self.path / "qev_config.json").read_text())
        self.backend = self.config.get("runtime", "torch") if backend == "auto" else backend
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Keep the default Torch call shape independent of neighboring questions.
        # CUDA BF16 kernels can change probabilities with batching/padding even
        # when rows have no shared model state. Explicit batching trades this
        # stability for throughput; training/evaluation use their own batch size.
        self.batch_size = batch_size
        self.temperature = float(self.config.get("temperature", 1.)) if calibrated else 1.
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid checkpoint temperature")
        self.lock = threading.Lock()
        if self.backend == "mlx":
            from .mlx_runtime import MLXRuntime
            self.model = MLXRuntime.from_checkpoint(self.path)
            self.tokenizer = self.model.tokenizer
        elif self.backend == "torch":
            import torch

            from .model import QevModel
            self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            # FP32 master weights; CUDA forward uses the same bf16 autocast as
            # training (the fused recurrent kernels require reduced precision).
            # Keep adapters separate: native generation must be able to disable
            # them and recover the unchanged image/video/text foundation model.
            self.model = QevModel.from_checkpoint(self.path, device=self.device, merge=False)
            self.tokenizer = self.model.get_tokenizer()
        else:
            raise ValueError("backend must be auto, torch, or mlx")
        self.processor = None

    def _processor(self):
        if self.processor is None:
            if hasattr(self.model, "get_processor"):
                self.processor = self.model.get_processor()
            else:
                self.processor = getattr(self.model, "processor", None)
        if self.processor is None:
            raise NotImplementedError("This checkpoint does not include the original multimodal processor")
        return self.processor

    def _predict_media(self, decoded, record):
        from .tokenization import encode_multimodal_question
        processor = self._processor()
        encodings = [encode_multimodal_question(
            decoded.text, question, processor, images=decoded.images or None,
            videos=decoded.videos or None, video_fps=decoded.video_fps or None,
            max_length=self.config.get("multimodal_max_length", 8192), max_state=None,
        ) for question in record["questions"]]
        if self.backend == "mlx":
            packed = []
            for encoding in encodings:
                item = {key: value for key, value in encoding.items() if key != "inputs"}
                item["model_inputs"] = encoding["inputs"]
                packed.append(item)
            ps = self.model.predict_encoded(packed)
        else:
            import torch
            ps = []
            with torch.inference_mode():
                for encoding in encodings:
                    inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                              for key, value in encoding["inputs"].items()}
                    with inference_autocast(self.model):
                        logits = self.model(**inputs)[0].float().cpu().numpy()
                    ps.append(probabilities(logits[:len(encoding["option_positions"])], 1.0).tolist())
        # A temperature fitted on text classification is not validated for
        # images/videos, so these experimental decision outputs remain raw.
        return ps, encodings

    def _native_eos_ids(self):
        foundation = getattr(self.model, "foundation", None)
        settings = getattr(foundation, "generation_config", None) if foundation is not None else getattr(self.model, "generation_config", None)
        eos = settings.get("eos_token_id") if isinstance(settings, dict) else getattr(settings, "eos_token_id", None)
        if eos is None:
            base = foundation if foundation is not None else getattr(self.model, "backbone", None)
            config = getattr(base, "config", None)
            eos = getattr(getattr(config, "text_config", config), "eos_token_id", None)
        return set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])

    def predict(self, state, questions, *, model=None):
        request = SystemOneRequest(state=state, questions=questions, model=model or self.config.get("model_name", "qev-0.8b"))
        if len(request.questions) > 64:
            raise ValueError("At most 64 questions per request")
        rec, meta = to_record(request)
        start = time.perf_counter()
        decoded = decode_state(request.state)
        with self.lock:
            if decoded.has_media:
                ps, encodings = self._predict_media(decoded, rec)
            else:
                encodings = [encode_question(rec["state"], q, self.tokenizer,
                    self.config.get("max_length", 1024), self.config.get("max_state", 384)) for q in rec["questions"]]
            if not decoded.has_media and self.backend == "mlx":
                logits = self.model.predict_logits(encodings)
                ps = [probabilities(row, self.temperature).tolist() for row in logits]
            elif not decoded.has_media:
                import torch

                from .tokenization import collate_encodings
                ps = []
                with torch.inference_mode():
                    for start_index in range(0, len(encodings), self.batch_size):
                        chunk = encodings[start_index:start_index+self.batch_size]
                        with inference_autocast(self.model):
                            logits = self.model(**collate_encodings(chunk, self.tokenizer.pad_token_id, self.device)).float().cpu().numpy()
                        ps.extend(probabilities(z[:len(e['option_positions'])], self.temperature).tolist()
                                  for z, e in zip(logits, chunk))
        return {"model": request.model, "answers": to_answers(ps, meta),
                "usage": {"input_tokens": sum(len(e["ids"]) for e in encodings), "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "qev": {"backend": self.backend, "temperature": 1.0 if decoded.has_media else self.temperature,
                        "training_modalities": ["text"], "multimodal_decision_accuracy_validated": False,
                        "input_modalities": ["text"] + (["image"] if decoded.images else []) + (["video"] if decoded.videos else []),
                        "truncated_questions": [m["id"] for m, e in zip(meta, encodings) if e.get("state_truncated", False)]}}

    def chat_completions(self, request: ChatCompletionRequest | dict):
        """Generate with the full original foundation and all decision adapters off."""
        if not isinstance(request, ChatCompletionRequest):
            request = ChatCompletionRequest.model_validate(request)
        if not hasattr(self.model, "generate_native"):
            raise NotImplementedError("Native generation requires a full multimodal checkpoint")
        start = time.perf_counter()
        messages, budget, video_fps = decode_chat_messages(request.messages)
        with self.lock:
            processor = self._processor()
            video_metadata = []
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "video":
                        frames = len(item["video"])
                        video_metadata.append({"fps": video_fps[len(video_metadata)],
                                               "total_num_frames": frames, "frames_indices": list(range(frames))})
            video_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False,
                            "cap_pixels_per_frame": False} if video_metadata else {}
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=request.enable_thinking,
                processor_kwargs=video_kwargs,
            )
            prompt_length = int(inputs["input_ids"].shape[-1])
            if prompt_length + request.generation_budget > self.config.get("native_max_length", 16384):
                raise ValueError("Native prompt plus generation budget exceeds the context limit")
            if self.backend == "torch":
                inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            generation = {"max_new_tokens": request.generation_budget, "do_sample": request.temperature > 0}
            if request.temperature > 0:
                generation.update(temperature=request.temperature, top_p=request.top_p)
            if self.backend == "torch":
                with inference_autocast(self.model):
                    output = self.model.generate_native(**inputs, **generation)
            else:
                output = self.model.generate_native(**inputs, **generation)
            sequences = output.sequences if hasattr(output, "sequences") else output
            token_rows = sequences.tolist() if hasattr(sequences, "tolist") else sequences
            ids = token_rows[0] if token_rows and isinstance(token_rows[0], list) else token_rows
            continuation = ids[prompt_length:]
            text = processor.batch_decode([continuation], skip_special_tokens=True)[0]
            reached_eos = bool(continuation) and continuation[-1] in self._native_eos_ids()
        return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": request.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                    "finish_reason": "length" if len(continuation) >= request.generation_budget and not reached_eos else "stop"}],
                "usage": {"prompt_tokens": prompt_length, "completion_tokens": len(continuation),
                          "total_tokens": prompt_length + len(continuation)},
                "qev": {"backend": self.backend, "decision_adapter_enabled": False,
                        "media_frames": budget.frames, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}}

    system_one = predict
