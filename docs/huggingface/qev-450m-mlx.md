---
license: other
license_name: lfm-open-license-v1.0
license_link: LICENSE
base_model:
  - LiquidAI/LFM2.5-VL-450M
base_model_relation: adapter
language:
  - en
  - zh
tags:
  - qev
  - lfm2
  - mlx
  - mlx-vlm
  - apple-silicon
  - lora
  - structured-decisions
  - multimodal
---

[English](qev-450m-mlx.md) | [简体中文](qev-450m-mlx.zh-CN.md)

# Qev-450M-MLX — LFM2.5-VL decision checkpoint for Apple Silicon

This repository publishes **Qev v0.4.0's default model**: the unchanged **LiquidAI/LFM2.5-VL-450M** foundation in `backbone/`, a switchable decision LoRA (`decision_adapters.safetensors`) and a candidate pointer head (`pointer.safetensors`) trained by the [Qev project](https://github.com/loadchange/qev). Decision inference enables the LoRA; native text/image generation disables it and uses the unmodified foundation.

Qev scores supplied answer options through a Jev / TypeSafe-style API (`POST /v1/systemone`) with real probabilities, not generated JSON text. Compared with the earlier Qwen3.5-0.8B checkpoint ([twainsk/qev-0.8b-mlx](https://huggingface.co/twainsk/qev-0.8b-mlx)), this model is 3.4× smaller, answers Snake decisions 2.1× faster and generates chat text about 2× faster on the same Mac, at a measured cost of −2.8 points on the general text subset. The text-only companion is [twainsk/qev-230m-mlx](https://huggingface.co/twainsk/qev-230m-mlx).

**Use the Qev runtime.** A generic Transformers pipeline or `mlx_lm.generate` does not load the pointer head and switchable adapters.

## Quickstart on Apple Silicon

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull                 # downloads and verifies this repository (~1.0 GB) into ~/.qev
qev decide "Card declined twice at checkout" -i "Which team should handle this?" \
  --choice "billing=Invoices and refunds" "technical=Bugs and outages" sales
qev decide --image receipt.jpg -i "Is this a restaurant receipt?" --noul
qev serve                # POST /v1/systemone, /v1/chat/completions, web lab at /
```

From a source checkout: `uv sync --python 3.12 --extra mlx`, then `uv run hf download twainsk/qev-450m-mlx --local-dir models/qev-450m-mlx` and pass `--model models/qev-450m-mlx`. Requirements: Apple Silicon, macOS 14+, Python 3.12+. Validated with MLX 0.32.2 and mlx-vlm 0.7.1; images use mlx-vlm's numpy LFM2-VL processor, so no PyTorch is needed.

## Contents

- `qev_config.json` — Qev format version 3, `family: lfm2`, `runtime: mlx`, modalities `["text", "image"]`.
- `decision_adapters.safetensors` — LoRA (rank 64, alpha 128) over `q/k/v/out_proj`, `in_proj` and the MLP `w1/w2/w3` of all language layers, kept separate from the foundation.
- `pointer.safetensors` — the 256-dim candidate scorer.
- `backbone/` — LiquidAI/LFM2.5-VL-450M at revision `fc6221ca…34ba`, byte-identical to upstream (bfloat16 safetensors, tokenizer, processor, chat template, license).
- `reports/` — evaluation, provenance and MLX-parity summaries; `release_manifest.json` lists sizes and SHA-256 digests.

## Training

Supervised imitation on the Qev decision suite: 17,892 / 1,120 / 1,380 training / calibration / development questions (12,000 Snake plus 5,892 general training rows; English plus four bilingual synthetic rule families). One NVIDIA A100 40 GB, 4 epochs, batch 8 × accumulation 2 (4,476 updates, ~18 min), learning rate 1e-4 cosine, BF16 autocast with FP32 master weights. Foundation parameters stayed frozen; 24.5 M parameters (LoRA + pointer) were trained. Temperature `4.59479341998814` was fitted on the held-out calibration split. A 6-epoch / lr 2e-4 variant scored far worse (59.1% general) and was rejected.

## Evaluation

Development set (1,380 questions), PyTorch reference; MLX matches within 2 argmax flips (below).

| Subset | Accuracy |
|---|---|
| All | 86.1% |
| Snake (500) | 98.8% |
| General text (880) | 78.9% |
| Chinese (50) | 94.0% |

Closed-loop Snake, 20 seeded games: mean 41.4 food, **0 collisions**. Reversed-option probe: 85.1% (−1.0 pt), 7.2% of choices flip. Paired against the Qwen3.5-0.8B checkpoint on identical questions: general −2.8 pt (McNemar p=0.024), Snake parity (p=0.63); against the previous LFM2.5-VL recipe: general +3.2 pt (p=0.013).

**Image decisions are zero-shot transfer from text-only decision training**: on 500 A-OKVQA validation questions the decision head answered 69.4% with the image (37.2% without; chance 25%), the same as the Qwen3.5 checkpoint's 69.4%. Text calibration is not applied to image questions (temperature 1.0).

MLX vs PyTorch parity on all 1,380 development questions: `adapter` mode 2 argmax flips, max probability difference 0.051; `bf16` 2 flips, 0.038; net accuracy unchanged. Apple M4 16 GB: Snake decision p50 98 ms (`adapter`) / 71 ms (`bf16`, the Homebrew service default), peak MLX memory ≤1.9 GiB, native chat ≈100 tokens/s in `bf16`.

## Limitations

- The decision adapter was trained on text only; image decisions are measured zero-shot (69.4% A-OKVQA), and image probabilities are uncalibrated. Video input is accepted as ordered frames but was not evaluated for decisions.
- Chinese training coverage is four synthetic rule families; the 50-question Chinese subset is small.
- Option-order sensitivity (7.2% flips) is higher than the Qwen3.5 checkpoint (3.6%).
- API compatibility does not imply matching Jev's model quality or confidence values.

## License

The backbone and this Derivative Work are distributed under the **LFM Open License v1.0** (`LICENSE`): commercial use by a Legal Entity with annual revenue of USD 10 million or more is not licensed. The decision adapter and pointer head are new files; the foundation weights were not modified (`NOTICE`).
