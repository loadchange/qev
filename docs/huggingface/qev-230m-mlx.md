---
license: other
license_name: lfm-open-license-v1.0
license_link: LICENSE
base_model:
  - LiquidAI/LFM2.5-230M
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
---

[English](qev-230m-mlx.md) | [简体中文](qev-230m-mlx.zh-CN.md)

# Qev-230M-MLX — text-only fast decision checkpoint for Apple Silicon

This repository publishes Qev's **smallest, fastest, text-only model**: the unchanged **LiquidAI/LFM2.5-230M** foundation in `backbone/`, plus a switchable decision LoRA and candidate pointer head trained by the [Qev project](https://github.com/loadchange/qev). It answers a Snake decision in **43 ms** at **1.2 GiB** peak MLX memory on an Apple M4 and downloads in about **0.5 GB** — but it accepts **text only**: image or video requests are rejected with a clear error. The default multimodal model is [twainsk/qev-450m-mlx](https://huggingface.co/twainsk/qev-450m-mlx).

Qev scores supplied answer options through a Jev / TypeSafe-style API (`POST /v1/systemone`) with real probabilities, not generated JSON text. Decision inference enables the LoRA; native chat (`/v1/chat/completions`) disables it and uses the unmodified foundation (~170 tokens/s on an M4).

**Use the Qev runtime.** A generic Transformers pipeline or `mlx_lm.generate` does not load the pointer head and switchable adapters.

## Quickstart on Apple Silicon

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull qev-230m        # downloads and verifies this repository (~0.5 GB) into ~/.qev
QEV_MODEL=qev-230m qev decide "Customer asks for a duplicate charge refund" \
  -i "Is a refund requested?" --noul -q
QEV_MODEL=qev-230m qev serve
```

From a source checkout: `uv sync --python 3.12 --extra mlx`, then `uv run hf download twainsk/qev-230m-mlx --local-dir models/qev-230m-mlx` and pass `--model models/qev-230m-mlx`. Requirements: Apple Silicon, macOS 14+, Python 3.12+. Validated with MLX 0.32.2 and mlx-vlm 0.7.1; no PyTorch is needed.

## Contents

- `qev_config.json` — Qev format version 3, `family: lfm2`, `runtime: mlx`, modalities `["text"]`.
- `decision_adapters.safetensors` — LoRA (rank 64, alpha 128) over `q/k/v/out_proj`, `in_proj` and the MLP `w1/w2/w3` of all 14 layers, kept separate from the foundation.
- `pointer.safetensors` — the 256-dim candidate scorer.
- `backbone/` — LiquidAI/LFM2.5-230M at revision `40cb2ad3…fa45`, byte-identical to upstream.
- `reports/` — evaluation, provenance and MLX-parity summaries; `release_manifest.json` lists sizes and SHA-256 digests.

## Training

Same recipe and data as qev-450m: 17,892 / 1,120 / 1,380 training / calibration / development questions, one A100 40 GB, 4 epochs, batch 8 × accumulation 2 (4,476 updates, ~16 min), lr 1e-4 cosine, frozen foundation, 16.1 M trained parameters, temperature `4.59479341998814` fitted on the calibration split.

## Evaluation

Development set (1,380 questions), PyTorch reference; MLX matches within 6 argmax flips (below).

| Subset | Accuracy |
|---|---|
| All | 85.2% |
| Snake (500) | 99.0% |
| General text (880) | 77.4% |
| Chinese (50) | 90.0% |

Closed-loop Snake, 20 seeded games: mean 43.4 food, **0 collisions**. Paired against the Qwen3.5-0.8B checkpoint on identical questions: general −4.3 pt (McNemar p=0.002), Snake parity (p=1.0).

MLX vs PyTorch parity on all 1,380 development questions: `adapter` 6 argmax flips (max probability difference 0.059), `bf16` 6 flips (0.062), net accuracy unchanged. Apple M4 16 GB: Snake decision p50 60 ms (`adapter`) / 43 ms (`bf16`), peak MLX memory ≤1.2 GiB, native chat ≈170 tokens/s in `bf16`.

## Limitations

- **Text only.** Image and video inputs are rejected (`qev-230m accepts text only`); use qev-450m for pictures.
- **Higher option-order sensitivity**: reversing option order flips 16.4% of choices and costs 8.1 pt (85.3% → 77.2%), versus 7.2% / −1.0 pt for qev-450m and 3.6% / ±0 for the Qwen3.5 checkpoint. Prefer qev-450m when answer stability across option orderings matters.
- Chinese training coverage is four synthetic rule families; the 50-question Chinese subset is small.
- API compatibility does not imply matching Jev's model quality or confidence values.

## License

The backbone and this Derivative Work are distributed under the **LFM Open License v1.0** (`LICENSE`): commercial use by a Legal Entity with annual revenue of USD 10 million or more is not licensed. The decision adapter and pointer head are new files; the foundation weights were not modified (`NOTICE`).
