[English](TRAINING.md) | [简体中文](TRAINING.zh-CN.md)

# Training and reproduction

Training uses the complete official multimodal model pinned to `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`. All original parameters stay frozen; only language LoRA and the 256-dimensional pointer head are updated. Every frozen parameter is hashed with SHA-256 before and after training, and verification is recorded in `base_integrity.json` inside the model directory. Both historical v0.2 general decision training and v0.3.0 Snake continuation are complete. See [SNAKE_MODEL](SNAKE_MODEL.md) for the specialized model and evaluation; reproduction steps follow.

Run the commands below from the project root. Existing `data/v1` can be reused. To prepare data again or reproduce the complete Colab workflow, use a separate project working directory without copying existing `data/`, `models/` or `runs/` artifacts into it. Data preparation, checkpoint extraction and MLX export all refuse to overwrite existing targets. Do not delete the delivered model to free its path; local training and MLX export examples use the new `qev-repro` name.

## Historical v0.2 general decision data

```bash
uv run python -m qev.data prepare --out data/v1 \
  --source-root ../kev/evals/public-pool-v4 \
  --seed 20260920
```

Data preparation preserves the upstream train/calibration/development splits and does not read the locked test. Upstream files use pinned revisions and SHA checks; if local sources are absent, they are downloaded from the pinned sources in code. Candidate shuffling, none and distractor augmentation are fixed during preparation and do not alter calibration or development. Exact-state and group-identifier separation across splits is checked.

`data/v1` stores JSONL files and the manifest. `docs/results/data_manifest.json` preserves the sources, licenses and file hashes used in this run. Chinese data covers only four deterministic rule families: refunds, approvals, deadlines and membership. This decision fine-tuning run has no visual training samples; full visual understanding and generation are retained from the original foundation, while the visual decision head requires independent evaluation.

## Training from the foundation: local / NVIDIA

```bash
uv sync --python 3.12 --extra dev
uv run python -m qev.train --data data/v1 --out models/qev-repro \
  --epochs 2 --batch 4 --accum 4 --lr 5e-5 --device cuda
```

The effective batch usually contains 16 questions; the final partial batch is normalized by its actual sample count. Settings are AdamW, weight decay 0.01, 10% warmup plus cosine decay, gradient clipping at 1 and seed 42. LoRA uses rank 16 / alpha 32 / dropout 0.05. The context budget is 1024 and the state budget 384; oversized candidates or instructions raise errors instead of removing candidates. A checkpoint is saved each epoch, with training logs every 10 updates.

CUDA uses FP32 master weights, BF16 autocast and an FP32 pointer, with the same precision policy during evaluation. Installing FLA can accelerate Qwen3.5 linear attention. CPU / MLX operators and precision differ and require additional numerical validation.

A random-pointer baseline is reported before training; afterward, calibration fits a single temperature and development is used only for evaluation. Development contains new samples from seen task families, not evidence of new-task generalization or a final locked test. A scalar temperature does not change argmax, and confidence cannot automatically be interpreted as accuracy in any setting.

## Snake data and continued training

Specialized training continues the language LoRA and candidate pointer head from `models/qev-0.8b`, preserving the frozen original multimodal foundation. `--init-checkpoint` loads a trained checkpoint and creates a new optimizer and learning-rate schedule. This starts another supervised training run rather than resuming an interrupted optimizer state. The original model remains available; output must use another directory.

First generate the fixed dataset from the project root. This command loads only the local tokenizer, without loading model weights or using Jev or another external model for labels:

```bash
uv run python scripts/prepare_snake_data.py --out data/snake-v1 \
  --tokenizer models/qev-0.8b-mlx/backbone \
  --train 12000 --calibration 500 --development 500 \
  --seed 20260921 --sizes 6,8,12,16 \
  --episode-steps 600 --samples-per-episode 48 --exploration 0.10 \
  --replay-data data/v1
```

`--train` / `--calibration` / `--development` specify counts of new Snake samples. `--replay-data` additionally includes every original task record in its existing split. Existing `data/snake-v1` can be reused; select a new directory when regenerating, because the generator refuses to overwrite existing data.

Inputs directly reuse the service's `decision_request`: three non-reversing candidates contain collision, food distance, `reachable_space`, `tail_reachable`, `food_path_distance` and `recent_visits` features. The game engine computes geometry with BFS on the static board after a hypothetical step, providing explicit environmental assistance. The heuristic teacher reads only these visible fields and visible size, length and heading. It prioritizes tail connectivity and reachable space, then considers food paths and recent visits. It cannot access future food randomness and does not prove action optimality.

Trajectories follow teacher actions, with probability `--exploration` of randomly choosing a collision-free candidate to cover states off the teacher's trajectory. Supervision still uses the teacher's choice for that state. No training label is created when all candidates collide. Each split uses separate game seeds/episode groups; duplicate observations and identical states across splits are excluded. Candidate order is shuffled with a fixed seed. Seeds 10000–10019 are reserved for later closed-loop evaluation and do not appear in these training trajectories.

Artifacts include `train.jsonl`, `calibration.jsonl`, `development.jsonl`, `manifest.json`, `episodes.jsonl`, `token_audit.json` and the default `teacher_evaluation.json`. The manifest records source/data hashes, label provenance, original-task replay sources, splits and feature versions; the token audit checks input budgets and state truncation. `teacher_evaluation.json` measures the programmatic teacher and is not a Qev model result. The teacher does not choose actions in terminal or HTTP runtime; the model's raw top choice is executed directly.

Continue training on an NVIDIA GPU with BF16 support:

```bash
uv run python -m qev.train \
  --init-checkpoint models/qev-0.8b --data data/snake-v1 --out models/qev-snake-0.8b \
  --epochs 2 --batch 8 --accum 2 --lr 3e-5 --device cuda --no-checkpointing
```

The A100 40GB run used these settings. With less GPU memory, enable the default gradient checkpointing, reduce batch and increase accum while keeping the effective batch and learning rate unchanged.

By default, the parent checkpoint's results on the new development split are first saved as `initial_development.json`, followed by training and refitting temperature on calibration. Development includes teacher-imitation questions on new game seeds and replayed tasks. Offline classification accuracy does not measure food collection, survival or completing a real game. Save training logs, data hashes, frozen-parameter checks and final evaluation with the new checkpoint.

After training finishes completely, copy the full Torch checkpoint to an Apple Silicon Mac, export into a new MLX directory and run terminal evaluation:

```bash
uv run python -m qev.export --checkpoint models/qev-snake-0.8b \
  --output models/qev-snake-0.8b-mlx --dtype float32
uv run qev snake --model models/qev-snake-0.8b-mlx --observation spatial \
  --headless --fps 0 --seed 10000 --episodes 20 --size 12 --max-steps 2000 \
  --report runs/snake-trained-spatial.jsonl
```

Training and artifacts are complete; see the [specialized model report](SNAKE_MODEL.md) for quality and integrity results. To compare gains from specialized training, run parent and child checkpoints with the same `spatial` observation, seeds, board size and step cap, retaining separate reports. To compare environmental feature gains, hold the checkpoint fixed and compare `local` with `spatial`. Also check for regressions on the original general tasks. See the [demo documentation](DEMO.md) for terminal controls and report formats.

## Historical v0.2 Colab CLI

Historical v0.2 training used an L4 GPU. The helper scripts below reproduce general decision training; they do not automatically switch to Snake continuation. The example uses a fresh session name, `qev-08b-repro`. For another reproduction, choose a different unused session name and use it consistently in every command. Log in to the CLI and prepare `data/v1` as above, then run:

These helpers write to the fixed remote path `/content/qev/models/qev-0.8b`, and the extraction script writes to local `models/qev-0.8b`; neither provides a renaming argument. Therefore the Colab workflow requires a fresh local project working directory and a fresh remote session, with the local target directory absent.

```bash
colab new -s qev-08b-repro --gpu L4
colab exec -s qev-08b-repro -f scripts/colab_setup.py --timeout 600
uv run python scripts/package_colab.py
colab upload -s qev-08b-repro runs/qev-code.tar.gz /content/qev-code.tar.gz
colab exec -s qev-08b-repro -f scripts/colab_extract.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_launch_smoke.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

Wait for the smoke test to exit with code 0 before launching training:

```bash
colab exec -s qev-08b-repro -f scripts/colab_launch_train.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

Training and smoke tests run as background subprocesses; a returned CLI command does not mean training has finished. Repeat `colab_poll.py` until the relevant process exits, without rerunning launch scripts. Training must report `qev_job RETURN_CODE 0`, with a final `QEV_TRAINING_COMPLETE` in its log.

After training, validate native multimodal preservation and then real HTTP and the official SDK in the same session. Run stages sequentially to avoid loading multiple models onto the GPU:

```bash
colab exec -s qev-08b-repro -f scripts/colab_launch_validation.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

Poll again; after confirming `qev_validation RETURN_CODE 0` and `passed: true` in the report, run:

```bash
colab exec -s qev-08b-repro -f scripts/colab_setup_validation.py --timeout 120
colab exec -s qev-08b-repro -f scripts/colab_launch_service_validation.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

Repeat polling until `qev_service RETURN_CODE 0`, `passed: true` and no failed checks are confirmed. Then collect artifacts. The collection script prints the archive's `sha256` and includes `SHA256SUMS.json` for checkpoint files inside the archive:

```bash
colab exec -s qev-08b-repro -f scripts/colab_collect.py --timeout 120
colab download -s qev-08b-repro /content/qev-trained.tar.gz runs/qev-trained-repro.tar.gz
```

Replace `COLLECT_OUTPUT_SHA256` below with the full 64-character SHA-256 from the preceding collection output; do not use a hash from another training run:

```bash
uv run python scripts/unpack_training.py runs/qev-trained-repro.tar.gz \
  --sha256 'COLLECT_OUTPUT_SHA256'
```

The extraction script verifies the entire download, extracts it into `models/qev-0.8b` and `runs/`, then checks every checkpoint file. After `Verified ... checkpoint files`, confirm that local `runs/native_validation.json` and `runs/service_torch.json` both contain `passed: true`, and `models/qev-0.8b/base_integrity.json` contains `unchanged: true`. Stop only the session you created, after validation and download verification complete:

```bash
colab stop -s qev-08b-repro
```

Setup retains Colab's bundled CUDA PyTorch and removes the unused older torchao version that is incompatible with PEFT. Dependency versions are recorded in training provenance; `uv.lock` pins the Mac environment. The checkpoint stores the complete processor and foundation revision; the original Hugging Face foundation weights still require a local cache or first download.

## Exporting and validating MLX on Apple Silicon

Run MLX export on an Apple Silicon Mac. If downloading on another machine, first copy the project source, complete `models/qev-0.8b` directory and corresponding `data/v1` into a separate project working directory on the Mac. The pinned official foundation referenced by the Torch checkpoint is downloaded on first export. MLX export includes all converted visual and language weights and does not merge decision LoRA into the original foundation.

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run python -m qev.export --checkpoint models/qev-0.8b \
  --output models/qev-repro-mlx --dtype float32
uv run python scripts/verify_mlx_integration.py \
  --checkpoint models/qev-0.8b --output-model models/qev-repro-mlx \
  --data data/v1 --report runs/mlx-repro-integration.json
uv run python scripts/validate_service.py \
  --checkpoint models/qev-repro-mlx --backend mlx \
  --output runs/service_mlx_repro.json
```

If using the earlier local NVIDIA training result, change both occurrences of `--checkpoint models/qev-0.8b` to `--checkpoint models/qev-repro`. `models/qev-repro-mlx` must be a new output directory; the subsequent integration step reads that export without converting again. Use fresh model and report paths for another reproduction.

Wait for each command to finish in sequence and verify `passed: true` in both reports. Integration checks numerical agreement for 20 candidate questions and 4 native generation examples; HTTP/SDK checks cover interfaces, media inputs and request isolation. Passing these checks does not establish multimodal decision accuracy.
