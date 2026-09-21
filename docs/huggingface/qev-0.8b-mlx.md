---
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-0.8B
base_model_relation: finetune
language:
  - en
  - zh
tags:
  - qev
  - qwen3.5
  - mlx
  - mlx-vlm
  - apple-silicon
  - lora
  - structured-decisions
  - multimodal
  - snake
---

# Qev-0.8B-MLX — full multimodal FP32 checkpoint

This repository publishes the **Qev v0.3.0 Snake continuation checkpoint for Apple Silicon**. It contains the complete converted Qwen3.5-0.8B foundation, separate decision LoRA weights, a candidate pointer head and text/image/video preprocessing assets. Its local training/export name is `qev-snake-0.8b-mlx`; `qev-0.8b` remains the API model alias. This is the newer continuation checkpoint, not the original v0.2 weights.

Qev scores supplied answer options through a Jev / TypeSafe-style API. Decision inference enables the LoRA adapter; native text, image and sampled-video generation disables it and uses the unchanged foundation. This is a **full FP32 export**, not a 4-bit model and not an adapter-only download. The corresponding PyTorch adapter checkpoint is [twainsk/qev-0.8b](https://huggingface.co/twainsk/qev-0.8b).

**Use the [Qev runtime](https://github.com/loadchange/qev).** The top-level repository is a custom Qev checkpoint: a generic Transformers pipeline, `mlx_lm.generate`, or `ollama run` does not load its pointer head and switchable adapters. MLX uses Apple Silicon's CPU/GPU and unified memory; the suffix does not mean Apple Neural Engine execution.

## Quickstart on Apple Silicon

Install [uv](https://docs.astral.sh/uv/) and Git, then run:

```bash
git clone https://github.com/loadchange/qev.git
cd qev
uv sync --python 3.12 --extra mlx
uv run hf download twainsk/qev-0.8b-mlx --local-dir models/qev-snake-0.8b-mlx

# Terminal Snake with direct model decisions
uv run qev snake --model models/qev-snake-0.8b-mlx

# Typed decision example and local HTTP server
uv run qev predict --model models/qev-snake-0.8b-mlx --request examples/request.json
uv run qev serve --model models/qev-snake-0.8b-mlx --port 8008
```

The Qev loader accepts local paths. Download the complete repository and retain its `backbone/` subdirectory. Run these commands from the cloned project directory, or provide absolute checkpoint and request paths. The MLX checkpoint contains the foundation and requires no separate base-model download for inference.

**中文：** 这是 Mac Apple Silicon 的完整 FP32 模型。下载后运行 `uv run qev snake`，终端默认优先查找 `models/qev-snake-0.8b-mlx`。空格暂停/继续，`N` 单步，`+` / `-` 调速，`Q` 或 Ctrl-C 退出。模型每步自行选择方向，界面展示候选概率和实际动作。

The server binds to `127.0.0.1` by default. Typed decisions use `POST /v1/systemone`; adapter-disabled native generation uses `POST /v1/chat/completions`. See the [API examples](https://github.com/loadchange/qev/blob/d68c468/docs/API.md). Checkpoint aliases do not switch the inference backend.

## Contents and requirements

- `qev_config.json`: Qev format version 2, `runtime: mlx`, `backend: mlx_vlm`, FP32 export metadata and temperature.
- `decision_adapters.safetensors`: separate converted LoRA tensors, not merged into the foundation.
- `pointer.safetensors`: the learned candidate scorer.
- `backbone/`: complete converted visual/language weights, model configuration, tokenizer, original image/video processor assets, foundation manifest and upstream license documents.
- `reports/`: data provenance and evaluation summaries; `release_manifest.json` lists release file sizes and SHA-256 hashes.
- Root license and attribution documents. Training rows are not included.

The checkpoint before publication metadata is approximately **3.48 GB** in decimal units, including a 3.41 GB foundation tensor file. Keep the complete folder structure. Apple Silicon macOS and Python 3.12 or later are required for this MLX path. Use the project's dependency lock: this export and its validation used MLX 0.32.2 and mlx-vlm 0.7.1. The package also installs Transformers/PEFT/PyTorch dependencies for shared processing and other Qev paths.

Gameplay was tested on an Apple M4 with 16 GiB unified memory. This is an observed test configuration, not a guaranteed minimum for all media sizes or concurrent applications. The full FP32 format intentionally uses more memory than low-bit exports. No new low-precision release is claimed here.

## Training and export

The source adapter was continued from the earlier Qev decision model with **12,000 / 500 / 500 Snake questions** for training/calibration/development, plus all **5,892 / 620 / 880 original general decision questions** in their existing splits. The combined counts are 17,892 / 1,120 / 1,380. Original foundation parameters stayed frozen; only language LoRA and the pointer head were optimized.

Training used an NVIDIA A100 40 GB, two epochs, batch 8, accumulation 2 and learning rate 3e-5: 2,238 updates and about 1,485 optimization seconds. FP32 master weights used CUDA BF16 autocast and an FP32 pointer. A separate calibration split fitted temperature `2.82842712474619`. This was supervised imitation and task replay, not reinforcement learning.

The foundation is `Qwen/Qwen3.5-0.8B`, pinned to `2fc06364715b967f1860aea9cf38778875588b17`. Export retains the complete visual encoder, language model and vocabulary projection. It reuses the verified FP32 foundation conversion and preserves LoRA as switchable tensors. All 372 converted adapter tensors were checked against the trained PyTorch checkpoint; the pointer file was byte-identical. See the [export integrity report](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/export_integrity.json).

**Fresh Torch CPU FP32 versus MLX FP32 logit parity was not run for this continuation.** The exported configuration deliberately records `parity_status: not_run`. The adapter conversion/integrity checks and actual MLX gameplay do not establish identical probabilities across runtimes.

Full training details and evidence are in the [Snake model report](https://github.com/loadchange/qev/blob/d68c468/docs/SNAKE_MODEL.md) and [training instructions](https://github.com/loadchange/qev/blob/d68c468/docs/TRAINING.md).

## Actual MLX gameplay

Apple M4, 16 GiB memory, FP32, one game at a time, 8×8 board, held-out seeds 10000–10004 and a 500-step limit per game:

| Measure | Parent checkpoint | This checkpoint |
| --- | ---: | ---: |
| Mean food collected | 3.0 | 42.6 |
| Food by seed | 4 / 1 / 5 / 1 / 4 | 37 / 45 / 44 / 38 / 49 |
| Collision games | 5 / 5 | 0 / 5 |
| Games reaching 500 steps | 0 / 5 | 5 / 5 |

The 2,500 new-model decisions had median latency 223.84 ms and P95 240.21 ms. Overall speed including game processing and trace logging was about 4.40 steps/second. Parent inference used local HTTP to the same MLX Agent; new inference called the local Agent directly, so this comparison is about gameplay rather than transport latency. See the [parent report](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/parent-mlx-spatial.json) and [new checkpoint report](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/trained-mlx-spatial.json).

The environment supplies explicit text features, including static BFS reachable space, tail connectivity, food path distance and recent visits. Runtime does not supply a teacher direction or preferred-action label, remove colliding candidates, or replace a model choice. Every executed action is the model argmax. **The Snake board is not a visual model input.** This evaluation therefore measures feature-assisted decision making, not pixel-based game understanding.

The source PyTorch checkpoint also improved from 3.10 to 42.90 mean food over 20 held-out 8×8 games and from 0.625 to 42.00 over eight 12×12 games, with no collisions in those 28 trained-model games. On the same A100, 500-question Snake teacher agreement improved from 73.60% to 98.40%, while accuracy on the original 880 development questions changed from 81.25% to 81.70%. These are PyTorch CUDA BF16 results, not a claim that the MLX export reproduces every prediction.

## Multimodal preservation and limitations

Native generation disables the decision adapter. Original foundation hashes were unchanged by training, and this MLX foundation tensor file is byte-identical to the parent MLX export. Same-A100 text/image/video probes produced identical generated token IDs before and after continuation. The new MLX checkpoint passed [19 real HTTP/SDK checks](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/service_mlx.json), including media processing, native generation, typed decisions and adapter restoration.

These checks establish implementation and limited regression evidence; they are **not a comprehensive vision/video quality benchmark**. Multimodal decision accuracy has not been measured, and media decision probabilities remain uncalibrated. General decision and Chinese training coverage are limited. Finite Snake results do not guarantee optimal play or absence of collisions; none of the reported games filled the board.

Qev provides Jev-compatible interface conventions, not private Jev weights or evidence of equivalent quality. No direct Laya superiority claim is made. Hardware, precision and batching can change decisions and trajectories. See the [complete conditions and limitations](https://github.com/loadchange/qev/blob/d68c468/docs/SNAKE_MODEL.md).

## License and attribution

Qev code, adapters and pointer weights are released under Apache-2.0. The included Qwen3.5 foundation remains subject to its upstream Apache-2.0 license and attribution; preserve `backbone/LICENSE` and `backbone/BASE_MODEL_README.md` along with the root `LICENSE` and `NOTICE`.

The public records replayed during training derive from `jaredpalmer/kev-suites` and multiple datasets with differing declared licenses, including share-alike, other and undeclared terms. The Apache software/model declaration does not relicense those datasets or settle their downstream terms. This repository does not redistribute the training dataset. See [NOTICE](https://github.com/loadchange/qev/blob/d68c468/NOTICE) and the [data provenance manifest](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/data_manifest.json). Qev is independent of the Jev, Qwen and MLX authors.
