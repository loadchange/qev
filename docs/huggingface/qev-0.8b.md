---
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-0.8B
base_model_relation: adapter
language:
  - en
  - zh
tags:
  - qev
  - qwen3.5
  - lora
  - peft
  - pytorch
  - structured-decisions
  - multimodal
  - snake
---

[English](qev-0.8b.md) | [简体中文](qev-0.8b.zh-CN.md)

# Qev-0.8B — PyTorch decision adapter

This repository publishes the **Qev v0.3.0 Snake continuation checkpoint**, trained from the earlier Qev decision adapter with Snake supervision and replay of the original general decision tasks. Its local training name is `qev-snake-0.8b`; `qev-0.8b` remains the API model alias. This is the newer checkpoint, not the original v0.2 weights.

Qev adds a language LoRA adapter and a candidate pointer head to the complete [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) multimodal foundation. It scores supplied answer options and returns typed decisions through a Jev / TypeSafe-style API. Native text, image and sampled-video generation disables the decision adapter and uses the frozen foundation.

**This is an adapter checkpoint, not a standalone Transformers pipeline model.** Use the [Qev runtime](https://github.com/loadchange/qev). The foundation is fetched separately at the pinned revision on first use. A normal `transformers.pipeline`, standalone PEFT loader, or `ollama run` does not load the complete Qev decision system.

The corresponding full Apple Silicon export is [twainsk/qev-0.8b-mlx](https://huggingface.co/twainsk/qev-0.8b-mlx).

## Quickstart

Install [uv](https://docs.astral.sh/uv/) and Git, then run:

```bash
git clone https://github.com/loadchange/qev.git
cd qev
uv sync --python 3.12
uv run hf download twainsk/qev-0.8b --local-dir models/qev-snake-0.8b

# Run terminal Snake. PyTorch automatically selects CUDA, MPS or CPU.
uv run qev snake --model models/qev-snake-0.8b

# Typed decision example and local HTTP server
uv run qev predict --model models/qev-snake-0.8b --request examples/request.json
uv run qev serve --model models/qev-snake-0.8b --port 8008
```

The Qev loader accepts local paths: download the repository before passing its local directory to `--model`. Run these commands from the cloned project directory, or provide absolute checkpoint and request paths. NVIDIA CUDA is the evaluated PyTorch path; Apple Silicon users should prefer the MLX release.

After downloading, run `uv run qev snake --model models/qev-snake-0.8b` to watch the model decide each step in the terminal. Space pauses/resumes, `N` advances one step, `+` / `-` adjusts speed, and `Q` or Ctrl-C quits. The first run downloads the pinned Qwen foundation.

The server binds to `127.0.0.1` by default. Typed decisions use `POST /v1/systemone`; adapter-disabled native generation uses `POST /v1/chat/completions`. See the [API examples](https://github.com/loadchange/qev/blob/main/docs/API.md). Checkpoint aliases do not switch the inference backend.

## Contents and requirements

- `qev_config.json`: Qev format version 2, architecture, pinned foundation reference and calibrated temperature.
- `adapter/adapter_model.safetensors` and `adapter/adapter_config.json`: unmerged rank-16 LoRA weights and configuration.
- `pointer.safetensors`: the learned 256-dimensional candidate pointer head.
- `tokenizer/` and `processor/`: text and original image/video preprocessing assets.
- `reports/`: data provenance and evaluation summaries; `release_manifest.json` lists release file sizes and SHA-256 hashes.
- Root `LICENSE` and `NOTICE`: license and attribution documents. Training rows and generated PEFT template cards are not included.

The published files total about 85.6 MB. They do **not** contain the frozen foundation weights. Qev loads `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, so allow additional download, storage and runtime memory for that model. For offline use, first populate the Hugging Face cache with that exact revision.

Use Python 3.12 or later and the Qev project's dependency lock. The validated software family is Transformers 5.17, PEFT 0.21 and PyTorch 2.10 or later; training used PyTorch 2.11.0 with CUDA 12.8. All decision adapters remain separate from the original foundation. Merging them would change the native generation path and is unsupported by Qev.

## Training

The original foundation's 852,985,920 parameters remained frozen. Qev continued training the existing 11,346,944 LoRA and pointer parameters using supervised cross-entropy; this was not reinforcement learning, and no private Jev labels were used.

| Item | Value |
| --- | --- |
| Snake train / calibration / development questions | 12,000 / 500 / 500 |
| Replayed original train / calibration / development questions | 5,892 / 620 / 880 |
| Combined train / calibration / development questions | 17,892 / 1,120 / 1,380 |
| Snake trajectory sizes | 6×6, 8×8, 12×12, 16×16 |
| Training | 2 epochs, batch 8, accumulation 2, learning rate 3e-5 |
| Compute | NVIDIA A100 40 GB, 2,238 updates, about 1,485 optimization seconds |
| Precision | FP32 master weights, CUDA BF16 autocast, FP32 pointer |
| Fitted calibration temperature | 2.82842712474619 |

Snake labels come from a deterministic teacher that uses the same explicit fields visible to the model. The environment supplies collision and food facts, static BFS reachable space, tail connectivity, food path distance and recent visit counts. No teacher direction, preferred-action marker or ranking is inserted in runtime input. The three non-reversing candidates include potential collisions, and runtime execution uses the model's argmax directly without a safety override. Snake is a **text-feature task**; the displayed board is not supplied as an image.

The dataset manifest SHA-256 is `f41151c68d6465b90bb8ea66ca0ea8611a6b37ed48596ba9484abc8ef4deda33`. Full provenance, source licenses, split isolation and reproduction commands are recorded in the [training documentation](https://github.com/loadchange/qev/blob/main/docs/TRAINING.md) and [Snake model report](https://github.com/loadchange/qev/blob/main/docs/SNAKE_MODEL.md).

## Evaluation

The following comparison uses the same A100 and CUDA BF16 arithmetic before and after continuation:

| Development measure | Questions | Parent | This checkpoint |
| --- | ---: | ---: | ---: |
| Snake teacher-action agreement on held-out game seeds | 500 | 73.60% | 98.40% |
| Original general decision accuracy | 880 | 81.25% | 81.70% |

Closed-loop results use held-out seeds, the same spatial features and a 500-step cap per game. Torch advances eight independent games per batch. The model chooses every executed action.

| Board / held-out seeds | Parent mean food | This checkpoint mean food | Parent collisions | This checkpoint collisions |
| --- | ---: | ---: | ---: | ---: |
| 8×8 / 10000–10019, 20 games | 3.10 | 42.90 | 20 / 20 | 0 / 20 |
| 12×12 / 10000–10007, 8 games | 0.625 | 42.00 | 8 / 8 | 0 / 8 |

All 28 new-model games reached the step cap; none filled the board. These are finite closed-loop tests, not a guarantee of collision-free or optimal play. The corresponding MLX export averaged 42.6 food over five 8×8 games on an Apple M4. Numerical precision, batch size and action-dependent trajectories can change results. See the [complete benchmark conditions and evidence](https://github.com/loadchange/qev/blob/main/docs/SNAKE_MODEL.md).

## Multimodal preservation and limitations

Full frozen-foundation hashes matched before and after training. On the same A100, adapter-disabled native generation produced identical token IDs for three fixed probes covering text, image and video. This is weight-preservation and limited regression evidence, **not a comprehensive multimodal quality evaluation**. A later zero-shot probe (500 A-OKVQA validation questions, 4 options) answered 69.4% correctly with the image and 32.6% without; video decisions have not been measured and multimodal decision probabilities remain uncalibrated.

The checkpoint is a small supervised decision experiment. General task coverage and Chinese task evidence are limited. It does not establish performance parity with Jev, TypeSafe or Laya; Jev compatibility refers to interface and answer types. It does not learn Snake geometry directly from pixels. The initial generic model, dataset provenance and additional limitations are documented in the [original model card](https://github.com/loadchange/qev/blob/main/docs/MODEL_CARD.md); v0.3.0 results above supersede the original checkpoint's metrics for this release.

## License and attribution

Qev code, adapters and pointer weights are released under Apache-2.0. The Qwen3.5 foundation is separately distributed by its authors under Apache-2.0. Preserve the included `LICENSE` and `NOTICE`, and the upstream attribution documents where present.

Replayed public training records originate from `jaredpalmer/kev-suites` and its source datasets, whose declared licenses differ and include share-alike, other and undeclared terms. The Apache software/model declaration does not relicense those datasets or settle their downstream terms. This model repository is not a redistribution of the training dataset. See [NOTICE](https://github.com/loadchange/qev/blob/d68c468/NOTICE) and the [data provenance manifest](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/data_manifest.json). Qev is an independent project and is not an official Jev or Qwen release.
