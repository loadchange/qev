---
license: other
license_name: apache-2.0-and-lfm1.0
license_link: https://huggingface.co/twainsk/qev-diffusion-experiment/blob/main/LICENSES.md
base_model:
- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3-0.6B
- dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
- LiquidAI/LFM2.5-VL-450M
library_name: peft
tags:
- qev
- decision-model
- lora
- diffusion
- experiment
---

# Qev diffusion experiment checkpoints

**English** | [简体中文](README.zh-CN.md)

Private research archive of the 2026-09-25 experiment that compared diffusion and autoregressive backbones for Qev's structured decisions. These are experiment artifacts, not a Qev release: they do not run in `qev snake`, `qev serve` or the MLX runtime.

Report, code and full results: [`experiments/diffusion`](https://github.com/loadchange/qev/tree/664e6764d218d2e1a2baa4a40355a88d6d4d8501/experiments/diffusion) at commit `664e6764d218d2e1a2baa4a40355a88d6d4d8501`.

## Checkpoints

Each folder holds a rank-16 LoRA adapter over the frozen base (`adapter/`), a 256-dimensional pointer head (`pointer.safetensors`), its config with the fitted temperature, and the arm's reports: development evaluation, GPU latency, provenance, closed-loop Snake summary, training log and per-question development rows. Base weights are not included; loading downloads them at the pinned revision.

| Folder | Base (revision) | Decision attention | Dev acc | Snake | General | License |
|---|---|---|---|---|---|---|
| `qwen35-causal` | Qwen/Qwen3.5-0.8B (`2fc0636`) | causal | 88.0% | 99.2% | 81.7% | Apache-2.0 |
| `qwen3-causal` | Qwen/Qwen3-0.6B (`c1899de`) | causal | 87.4% | 98.8% | 80.9% | Apache-2.0 |
| `a2d-block` | dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 (`c8d24a3`) | block (diffusion-style) | 85.7% | 99.2% | 78.0% | Apache-2.0 |
| `qwen3-block` | Qwen/Qwen3-0.6B (`c1899de`) | block | 75.4% | 98.8% | 62.0% | Apache-2.0 |
| `lfm2-causal` | LiquidAI/LFM2.5-VL-450M (`fc6221c`) | causal | 84.0% | 98.6% | 75.7% | LFM Open License v1.0 |
| `lfm2-block` | LiquidAI/LFM2.5-VL-450M (`fc6221c`) | block | 84.1% | 99.2% | 75.5% | LFM Open License v1.0 |

All arms: `data/snake-v1` (500 Snake + 880 general development questions), same recipe, one seed. `reports/` holds the cross-arm results: image-decision probe (A-OKVQA), option-order probe, exact McNemar tests, zero-shot DiffusionGemma (djev) reads, Apple Silicon latency and the summary digest.

## Loading

```bash
git clone https://github.com/loadchange/qev && cd qev
git checkout 664e6764d218d2e1a2baa4a40355a88d6d4d8501
uv sync --python 3.12 --extra dev
uv run hf download twainsk/qev-diffusion-experiment --local-dir runs/qevd-archive
```

```python
from experiments.diffusion.qevd import QevDModel
model = QevDModel.from_checkpoint("runs/qevd-archive/lfm2-causal")

from qev.model import QevModel  # the Qwen3.5 arm uses the Qev checkpoint format
baseline = QevModel.from_checkpoint("runs/qevd-archive/qwen35-causal")
```

`experiments/diffusion/order_probe.py` and `mm_probe.py` evaluate these folders directly.

## Licenses and limitations

The adapters on Qwen models are distributed under Apache-2.0. `lfm2-causal` and `lfm2-block` are Derivative Works of LiquidAI/LFM2.5-VL-450M under the LFM Open License v1.0 (`LICENSE` and `NOTICE` in each folder): commercial use by an organization with annual revenue of USD 10 million or more is not licensed. See [LICENSES.md](LICENSES.md).

One seed per arm; development questions come from task families seen in training; image results are zero-shot transfer from text-only decision training. Probabilities are calibrated with a single temperature fitted on the calibration split.
