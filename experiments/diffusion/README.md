# Diffusion vs. autoregressive backbones for Qev decisions

**English** | [简体中文](README.zh-CN.md)

Experiment run on 2026-09-25. Question: would a diffusion language model, as used by [djev](https://github.com/mmastrac/djev) (DiffusionGemma with one-step structured reads), make Qev better? And can Qev become smaller and faster while keeping multimodal input?

## Short answer

- **Diffusion did not help accuracy.** With the same size, data and training recipe, the masked-diffusion Qwen3-0.6B (dLLM MDLM weights, bidirectional block reads) scored 3.0 points below its autoregressive parent on general decisions (78.0% vs 80.9%, exact McNemar p = 0.017) and tied on Snake. Switching the autoregressive weights to bidirectional reads without diffusion pretraining lost 18.9 points.
- **The djev model itself is not better on Qev's tasks.** Zero-shot DiffusionGemma-26B-A4B (25.2B parameters, djev-run's read) tied the trained 0.8B Qev on general questions (81.2% vs 81.8%, p = 0.73) and trailed every trained arm on Snake by about 14 points (84.6% vs 98.6–99.2%).
- **What djev gets right is the single-pass read, and that does not need a diffusion model.** Packing all questions over one shared state in one forward reproduced independent rows exactly (fp32 difference ≤ 6e-12), but it only pays off when the shared state is long (images, documents): with Snake-length states a dense-mask packed forward was slower on GPU than an ordinary batch.
- **Smaller, faster and still multimodal: LFM2.5-VL-450M.** Half the parameters of Qwen3.5-0.8B, 2.1× faster on Apple Silicon at equal precision with merged LoRA (83 vs 178 ms per Snake decision, fp32) and 3.3× faster than today's production path (66 ms bf16 vs 218 ms). Snake play (98.6%, 43.1 food per game, no collisions) and image decisions (69.6% vs 70.2% on A-OKVQA) match; general text decisions drop 6.0 points (p = 8e-6), concentrated in reasoning-heavy tasks. Its license (LFM Open License v1.0) forbids commercial use by organizations with annual revenue of $10M or more.

## Setup

Every arm used the same data, pointer head and optimizer; only the backbone and the decision attention pattern changed.

- Data: `data/snake-v1`, 17,892 training / 1,120 calibration / 1,380 development questions (500 Snake + 880 general and bilingual rule questions).
- Recipe: rank-16 LoRA (alpha 32, dropout 0.05) on every language linear layer, frozen foundation, 256-dimensional pointer head, AdamW lr 1e-4, weight decay 0.01, 10% warmup then cosine, 2 epochs, batch 8 × 2 accumulation, fp32 master weights with bf16 autocast, seed 42, one temperature fitted on calibration.
- Attention: `causal` is the backbone's own mask. `block` is the diffusion-style read: state tokens attend bidirectionally within the state, and each question attends to the state and bidirectionally within itself, so the state never sees a question and can be shared exactly.
- Hardware: Colab A100-SXM4-40GB for training, GPU timings and probes; Apple M4 16 GB (MLX 0.32.2) for Mac latency.

| Arm | Backbone (pinned revision in `qevd.py`) | Attention | Question it answers |
|---|---|---|---|
| `qwen35-causal` | Qwen3.5-0.8B (current Qev) | causal | baseline under the shared recipe |
| `qwen3-causal` | Qwen3-0.6B autoregressive | causal | control |
| `a2d-block` | Qwen3-0.6B masked diffusion (`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`) | block | does a diffusion model help? |
| `qwen3-block` | Qwen3-0.6B autoregressive | block | is bidirectional reading alone enough? |
| `lfm2-causal` | LFM2.5-VL-450M | causal | smaller and multimodal |
| `lfm2-block` | LFM2.5-VL-450M | block | smaller, multimodal, diffusion-style read |

The Qwen3.5 baseline cannot use `block`: its Gated DeltaNet layers are recurrent.

## Results

Development split, calibrated. Snake closed loop: 20 held-out seeds (10000–10019), 8×8 board, 500-step cap, argmax actions, no guardrail — the published protocol.

| Arm | Params | Dev acc | Snake acc | General acc | NLL | ECE | Snake food / game | Collisions | GPU ms per decision (adapter / merged) |
|---|---|---|---|---|---|---|---|---|---|
| published `qev-snake-0.8b` (two-stage) | 0.86B | 87.8% | 98.4% | 81.7% | 0.355 | 0.029 | 42.9 | 0 | — |
| `qwen35-causal` | 0.86B | **88.0%** | 99.2% | **81.7%** | **0.319** | 0.021 | 43.3 | 0 | 101 / 64 |
| `qwen3-causal` | 0.61B | 87.4% | 98.8% | 80.9% | 0.347 | 0.019 | 43.5 | 0 | 84 / 43 |
| `a2d-block` | 0.61B | 85.7% | 99.2% | 78.0% | 0.371 | 0.018 | 43.9 | 0 | 88 / 48 |
| `qwen3-block` | 0.61B | 75.4% | 98.8% | 62.0% | 0.647 | 0.025 | 42.6 | 0 | 86 / 47 |
| `lfm2-causal` | 0.46B | 84.0% | 98.6% | 75.7% | 0.433 | 0.019 | 43.1 | 0 | **39 / 20** |
| `lfm2-block` | 0.46B | 84.1% | 99.2% | 75.5% | 0.416 | 0.016 | 42.7 | 0 | 38 / 20 |
| djev zero-shot¹ | 25.2B | 82.5% | 84.6% | 81.2% | 0.513 | 0.015 | — | — | 161 (HTTP) |

¹ DiffusionGemma-26B-A4B NVFP4 served by vLLM nightly with djev-run's flags and prompt, one read-only denoise step, top-20 logprobs, temperature fitted on calibration. It skips the 60 Banking77 questions (77 options exceed its 62 single-token labels), so its general accuracy covers the other 820.

Exact McNemar tests on identical questions (`paired_tests.py`):

| Comparison | General (880) | Snake (500) |
|---|---|---|
| diffusion `a2d-block` vs autoregressive `qwen3-causal` | 78.0% vs 80.9%, p = 0.017 | 99.2% vs 98.8%, p = 0.63 |
| `qwen3-block` vs `qwen3-causal` | 62.0% vs 80.9%, p = 2e-28 | 98.8% vs 98.8%, p = 1 |
| `a2d-block` vs `qwen3-block` (effect of diffusion pretraining) | 78.0% vs 62.0%, p = 3e-20 | p = 0.63 |
| `lfm2-causal` vs `qwen35-causal` | 75.7% vs 81.7%, p = 8e-6 | 98.6% vs 99.2%, p = 0.38 |
| `lfm2-block` vs `lfm2-causal` | 75.5% vs 75.7%, p = 0.91 | 99.2% vs 98.6%, p = 0.25 |
| `qwen3-causal` vs `qwen35-causal` | 80.9% vs 81.7%, p = 0.55 | 98.8% vs 99.2%, p = 0.50 |
| djev vs `qwen35-causal` (820 general without Banking77) | 81.2% vs 81.8%, p = 0.73 | 84.6% vs 99.2%, p = 2e-22 |

LFM2 answered 53 fewer general questions correctly than Qwen3.5: 19 in the programmatic rule families (100 questions; for example membership, English: 41.7% vs 100%), 9 in MNLI, 6 each in BoolQ and Banking77. Each rule family has only 12–13 development questions.

### Image decisions (zero-shot transfer)

No arm saw an image during decision training. A-OKVQA validation, first 500 questions, 4 options (`mm_probe.py`):

| Checkpoint | Decision head + image | Decision head, no image | Frozen foundation, letter logits | GPU ms per decision |
|---|---|---|---|---|
| `lfm2-causal` | 69.6% | 35.2% | 75.8% | 68 |
| `lfm2-block` | 66.6% | 30.8% | 75.8% | 68 |
| `qwen35-causal` | 70.2% | 33.8% | 71.4% | 153 |
| published `qev-snake-0.8b` | 69.4% | 32.6% | 71.4% | 151 |

The text-trained decision heads use the image (about +35 points over no image) and come within 1–9 points of the foundation's own multiple-choice reading.

### Candidate-order robustness

Every development question scored again with its options reversed (`order_probe.py`):

| Checkpoint | Choice changes | Accuracy original / reversed | Banking77 changes |
|---|---|---|---|
| `qwen35-causal` | **3.6%** | 88.0% / 88.0% | 13.3% |
| published `qev-snake-0.8b` | 5.4% | 87.8% / 87.2% | 20.0% |
| `qwen3-causal` | 8.3% | 87.4% / 87.0% | 23.3% |
| `a2d-block` | 7.0% | 85.7% / 85.0% | 26.7% |
| `qwen3-block` | 36.4% | 75.4% / 62.1% | 50.0% |
| `lfm2-causal` | 10.0% | 84.1% / 81.3% | 26.7% |
| `lfm2-block` | 7.6% | 84.1% / 82.5% | 11.7% |

Block reads made the small LFM2 model less sensitive to option order; they did not change its accuracy.

### Speed on Apple Silicon

One Snake decision, 40 development states, p50 ms, Apple M4 16 GB (`mlx_latency.py`, `mlx_qev_adapters.py`). Candidate timings treat LoRA as merged, which leaves compute unchanged; the Qev rows use the production runtime.

| Model | fp32 | bf16 | 8-bit |
|---|---|---|---|
| Qev Qwen3.5-0.8B, switchable LoRA (production) | 218 | 189 | 190 |
| Qev Qwen3.5-0.8B, adapters off (merged-LoRA floor) | 178 | 143 | — |
| Qwen3-0.6B | 130 | 105 | 107 |
| LFM2.5-VL-450M | **83** | **66** | 72 |

8-bit quantization did not speed up these ~350-token prefills. Block masks cost 0.6–8.9 ms more than causal masks.

### Several questions over one state (GPU)

One Snake state with 16 questions, merged LoRA, p50 ms: `qwen3-causal` sequential 699, batched 85, packed 135; `lfm2-causal` sequential 320, batched 43. Packing saved 18% of tokens and matched independent rows exactly, but its dense 4,320×4,320 mask made it slower than batching. On a fast GPU these small models are launch-overhead bound: merging LoRA cuts single-decision latency by 35–50%.

## What this means for Qev

1. Keep an autoregressive backbone. Adopt djev's single-pass idea only where the shared state is long — for example several questions over one image — using a shared-state cache or block-sparse attention rather than a dense packed mask.
2. For speed, merge the decision LoRA into a separate decision copy of the weights (native generation keeps the switchable adapter): 218 → 178 ms on Mac at fp32, with no retraining.
3. For a smaller multimodal model, LFM2.5-VL-450M is viable where Snake/game and image decisions matter more than general text rules, and if its license fits. Closing the 6-point text gap needs more or better text training; image decisions would benefit from real multimodal decision data.
4. If multimodal input were optional, Qwen3-0.6B matched Qwen3.5-0.8B (p = 0.55) and was 1.4× faster on the Mac at equal precision with merged LoRA, but it has no vision encoder.

## Limitations

- One seed per arm. Development questions come from task families seen in training; Chinese coverage is four programmatic rule families. Snake closed-loop play is saturated (no arm collided; every game but one `qwen3-block` starvation reached the step cap), so it cannot rank the arms.
- The published `qev-snake-0.8b` used two training stages; the arms here used one shared stage.
- Image results are zero-shot transfer on one dataset; no image decision training was done.
- MLX candidate timings use merged, untrained weights; they measure speed only. MLX export of the LFM2 decision adapter is not implemented yet.
- GPU timings came from three Colab A100 sessions with different host CPUs and are launch-overhead sensitive; compare arms trained on the same session with care.
- djev-run's read limits questions to 62 single-token labels and fills labels outside the top-20 logprobs with a floor, as djev-run does.

## Reproduce

```bash
uv run pytest -q tests/test_qevd_experiment.py
# One arm (Colab A100 or any CUDA GPU with bf16):
python experiments/diffusion/train_qevd.py --backbone lfm2-vl --attention causal \
  --label lfm2-causal --out runs/qevd/lfm2-causal
# Probes on trained checkpoints:
python experiments/diffusion/mm_probe.py --arm lfm2-causal=qevd:runs/qevd/lfm2-causal \
  --arm qev-snake=qev:models/qev-snake-0.8b --output runs/qevd/mm-probe.json
python experiments/diffusion/order_probe.py --arm lfm2-causal=runs/qevd/lfm2-causal --output runs/qevd/order.json
python experiments/diffusion/paired_tests.py --runs runs/qevd --output runs/qevd/paired.json
# Apple Silicon latency:
uv run python experiments/diffusion/mlx_latency.py --output runs/qevd/mlx-latency.json
uv run python experiments/diffusion/mlx_qev_adapters.py models/qev-snake-0.8b-mlx runs/qevd/mlx-qev-adapters.json
```

Colab orchestration used `package.py`, `colab_setup.py`, `colab_extract.py`, `colab_launch.py` (arm queue), `colab_poll.py`, `colab_collect.py` and `colab_djev.py` (vLLM nightly plus `djev_zeroshot.py`). The colab CLI (≤ 0.7.2) does not refresh the one-hour runtime token and prunes still-running sessions; run `colab_refresh.py` with the CLI's interpreter every ~20 minutes during long jobs.

Reports and per-question rows are in [`docs/results/diffusion`](../../docs/results/diffusion); `report.py` rebuilds the tables from them. The six checkpoints (LoRA adapters, pointer heads and per-arm reports) are archived in the private Hugging Face repository `twainsk/qev-diffusion-experiment`; they load only with the code in this directory (`QevDModel.from_checkpoint`) or, for `qwen35-causal`, with `qev.model.QevModel`. `publish_archive.py` rebuilds that archive and verifies every uploaded file ([record](../../docs/results/diffusion/huggingface_archive.json)).
