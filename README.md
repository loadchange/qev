# Qev

**English** | [简体中文](README.zh-CN.md)

A structured decision model on a complete, unchanged multimodal foundation — **LFM2.5-VL-450M** by default, with Qwen3.5-0.8B and a text-only LFM2.5-230M also released — exposing a Jev / TypeSafe-style API and an Apple Silicon MLX runtime.

Qev adds a separate language LoRA decision adapter and a candidate pointer head while retaining the original vision encoder, language model, image/video processors, and generation head. Ordinary chat disables the decision adapter and uses the original foundation. Structured decisions enable the adapter and construct JSON directly from candidate probabilities, without generating and parsing answer text.

This release is a small supervised training experiment. See the [model card](docs/MODEL_CARD.md) for artifacts, measured results, and limitations. Jev compatibility refers to the API and output types; it does not imply reproducing Jev's private architecture or matching its quality.

## Install and run

On Apple Silicon (macOS 14+), install the command-line tool with Homebrew:

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull                     # download and verify the model into ~/.qev
qev decide "Card declined twice at checkout" -i "Which team?" --choice billing technical account
brew services start qev      # keep the model warm: CLI decisions then take a fraction of a second
```

See the [command-line guide](docs/CLI.md) for decisions with images, the HTTP service and speed options. From a source checkout:

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run hf download twainsk/qev-0.8b-mlx --local-dir models/qev-snake-0.8b-mlx
uv run qev snake
uv run qev predict --model models/qev-snake-0.8b-mlx --request examples/request.json
uv run qev serve --model models/qev-snake-0.8b-mlx --port 8008
```

`qev serve` records the two model API POST endpoints under `~/.qev/request-logs` (`$QEV_HOME/request-logs`) by default. Change the location with `--request-log-dir PATH` or disable logging with `--no-request-log`; see [local request logs](docs/API.md#local-request-logs).

The code is on [GitHub](https://github.com/loadchange/qev). Model weights are published separately on Hugging Face and are not included in a Git clone:

| Repository | Contents | Runtime |
| --- | --- | --- |
| [twainsk/qev-450m-mlx](https://huggingface.co/twainsk/qev-450m-mlx) | **v0.4.0 default.** About 1.00 GB: unchanged LFM2.5-VL-450M foundation (text+image), decision adapter, and pointer head | Apple Silicon / MLX |
| [twainsk/qev-230m-mlx](https://huggingface.co/twainsk/qev-230m-mlx) | About 0.53 GB: unchanged text-only LFM2.5-230M foundation, decision adapter, and pointer head | Apple Silicon / MLX |
| [twainsk/qev-0.8b](https://huggingface.co/twainsk/qev-0.8b) | About 85 MB: LoRA, pointer head, and processor; the pinned Qwen3.5 foundation is downloaded separately on first use | PyTorch / NVIDIA |
| [twainsk/qev-0.8b-mlx](https://huggingface.co/twainsk/qev-0.8b-mlx) | About 3.48 GB: complete FP32 multimodal foundation, adapters, and pointer head | Apple Silicon / MLX |

Both repositories contain the v0.3.0 models after Snake-specific continued training, with replay of the original general tasks and preserved native multimodal generation. They require the Qev runtime and cannot be used directly as a standard Transformers pipeline or Ollama model. Download the PyTorch version with `uv run hf download twainsk/qev-0.8b --local-dir models/qev-snake-0.8b`. Each repository includes a model card, license, evaluation summary, and a `release_manifest.json` file checksum manifest.

The Snake-specific GPU / PyTorch checkpoint is `models/qev-snake-0.8b`, and the Mac checkpoint is `models/qev-snake-0.8b-mlx`; the original `models/qev-0.8b[-mlx]` checkpoints remain available. Prefer MLX on Mac. MLX runs on Apple Silicon CPUs/GPUs with unified memory; this does not mean it uses the Apple Neural Engine or automatically improves model accuracy.

**Use `qev snake` as the primary Snake interface.** The terminal shows candidate probabilities, the executed action, score, and inference time beside the board. Space pauses/resumes, `N` steps once, `+` / `-` changes the speed, and `Q` or Ctrl-C exits. It prefers `models/qev-snake-0.8b-mlx` when present, otherwise `models/qev-0.8b-mlx`; use `--model` to select a checkpoint explicitly. Snake-specific training is complete, and the terminal loads the new weights by default while allowing explicit selection of the original weights.

The default `--observation spatial` supplies static BFS reachable space, tail connectivity, food path distance, and recent visit counts. These are engine-computed environment features, explicitly labeled in the interface as static BFS features with no action override. `--observation local` retains the historical collision/food-distance observations for comparisons with a fixed model, seed, and rules. The model still chooses and executes the direction directly; candidates that may collide are not removed.

```bash
# Historical model with historical local observations
uv run qev snake --model models/qev-0.8b-mlx --observation local --seed 7
# Run ten games without a UI, seeds 10000..10009, recording real requests and responses
uv run qev snake --headless --fps 0 --seed 10000 --episodes 10 \
  --report runs/snake-terminal.jsonl
```

The JSONL report is written one record at a time and refuses to overwrite an existing file. Without a TTY, the terminal automatically uses headless mode. See the [training guide](docs/TRAINING.md) for Snake data and training reproduction, and the [demo guide](docs/DEMO.md) for terminal/HTTP modes and all controls. The historical v0.2 checkpoint was not trained on Snake. In a CUDA comparison with the same 8×8 board, 500-step limit, and 20 held-out seeds, the new model improved mean food eaten from 3.1 to 42.9 and reduced games ending in collisions from 20 to 0. These are finite observed games, not a guarantee of collision-free play. See the [Snake model report](docs/SNAKE_MODEL.md) for complete conditions, Mac results, and multimodal preservation checks.

The service listens on `127.0.0.1:8008` by default. Structured decisions use `POST /v1/systemone`; native generation uses `POST /v1/chat/completions`. Multimodal HTTP requests carry inline images and sampled video frames; see the [API examples](docs/API.md).

After startup, open **http://127.0.0.1:8008** for the built-in lab, with no separate frontend service. On the first visit, the page chooses English or Simplified Chinese from the browser language. You can switch manually; the selection is saved in the current browser and survives refreshes.

- **Web Snake** provides another visualization with start, pause, single-step, and restart with a fixed seed. Inspect real candidate probabilities, timing, and requests for each step, and export the game record. It shares the terminal's game rules and model decision path.
- **[API playground](http://127.0.0.1:8008/playground)** offers three Choice scenarios: text only, an image in the background, and images in every option. Edit the background, question, and candidates; upload or paste background and candidate images; then send a real request. The advanced JSON editor stays synchronized with the form and shows responses, timing, and token usage. Each image area has its own paste target for Ctrl/Cmd+V, plus a clipboard button when browser clipboard reading is available.

All three scenarios call `POST /v1/systemone` and return `answers.<question>.choice` and `probabilities`, without generating answer text: `usage.output_tokens=0`. Background images belong in `state.content`; candidate images belong in `questions.<question>.criteria.<option>.content`. Each `content` contains `text` and inline `image_url` items. Built-in image examples can be sent immediately; see the [image decision protocol](docs/API.md#image-decisions). Other decision types, native generation, and service queries remain available under “More examples.”

Both web and terminal Snake use textual environment features. The displayed board is not sent to the model as an image and is not a visual decision evaluation.

```bash
curl http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

The artifact names `qev-0.8b` / `qev-0.8b-mlx` refer to Qev checkpoints, not general generation models that can be run directly with `ollama run`. The candidate pointer head and switchable adapter require the Qev runtime.

## How the model is trained

1. Pin the official foundation revision to `2fc06364715b967f1860aea9cf38778875588b17` and load its complete multimodal weights.
2. Freeze the original parameters, add rank-16 LoRA to the language attention, GatedDeltaNet, and MLP projections, and train a separate 256-dimensional pointer head.
3. Encode the state, question, and all candidates as an independent sequence. Compute logits from hidden vectors at candidate-end and decision positions, using supervised cross-entropy.
4. Fit one temperature parameter on an independent calibration split, then report development accuracy, NLL, Brier score, ECE, and classification metrics.
5. Save the separate adapter, pointer head, complete processor, and pinned foundation reference. MLX export includes the complete vision/language foundation and switchable adapters.

Qwen3.5's hybrid architecture includes recurrent state that ordinary attention masks cannot isolate. Each question is therefore encoded independently. Qev does not use Kev's shared-state branch packing or claim to compute the state only once for multiple questions.

The historical v0.2 training data contains 10 public task sources and executable English/Chinese rules: 5,892 training questions, 620 calibration questions, and 880 development evaluation questions. Full input checks found no truncated states or candidates. Fixed data, split checks, and license sources are recorded in `data/v1/manifest.json`. Upstream locked test sets were not read, and Jev was not queried for labels. Chinese samples cover only four programmatic rules and do not establish general Chinese-language ability. Snake-specific training continues from this checkpoint, using action labels from an explicit heuristic teacher and replay of the original tasks. The teacher does not participate in runtime action execution.

## Measurements and validation

The following results describe the historical v0.2 general decision checkpoint, not the new Snake-specific model. That version completed two training epochs on an NVIDIA L4 and produced Torch and MLX checkpoints. On the same 880 development questions, the random pointer head achieved 28.75% accuracy before training. Results after training:

| Runtime | Accuracy | Calibrated NLL | ECE |
| --- | --- | --- | --- |
| CUDA BF16, batch 4 | 81.36% | 0.4890 | 0.0432 |
| Delivered MLX FP32, one question at a time | 81.25% | 0.4887 | 0.0395 |

Both use the same calibration temperature, `T=1.464086`, without refitting on MLX or development data. One question changed argmax across precisions. See the [model card](docs/MODEL_CARD.md) for complete and grouped metrics.

Frozen foundation parameter hashes match before and after training. Native text, image, video, and mixed inputs passed finite comparison probes. Real HTTP / official SDK validation passed [17/17 on Torch](docs/results/service_torch.json) and [19/19 on MLX](docs/results/service_mlx.json). On an Apple M4 with 16 GiB memory and FP32, the median of five warm measurements for a fixed 40-token question was [41.82 ms](docs/results/benchmark_mlx.json).

Candidate order can affect the result: 27 of 28 candidate-reversal probes retained the same choice. One Banking77 question with 77 candidates changed its choice, with a maximum individual probability change of 0.585. The Chinese-language evidence in this run covers only four programmatic rule types.

Later probes of the published Snake checkpoint `qev-snake-0.8b` (CUDA BF16, 2026-09-25): reversing the options of all 1,380 development questions changed 5.4% of choices (accuracy 87.8% → 87.2%). On the first 500 A-OKVQA validation questions (4 options, chance 25%), its text-trained decision head answered 69.4% correctly with the image and 32.6% without it; this is zero-shot transfer, and media probabilities remain uncalibrated. The same experiment compared diffusion and autoregressive backbones: diffusion reads did not improve accuracy, and zero-shot DiffusionGemma-26B (the djev method) tied Qev on general questions but trailed it by about 14 points on Snake. See the [backbone experiment](experiments/diffusion/README.md).

The historical v0.2 artifacts were downloaded locally, and that Colab training session was stopped. File verification, 81 tests, CLI examples, and wheel build results for that release are recorded in the [release checks](docs/results/release_checks.json).

## Reproduce

First train on an NVIDIA GPU / Colab:

```bash
uv run python -m qev.train --data data/v1 --out models/qev-new \
  --epochs 2 --batch 4 --accum 4 --lr 5e-5 --device cuda
```

Copy the complete `models/qev-new` checkpoint directory to an Apple Silicon Mac, install the MLX dependencies, and export it:

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run python -m qev.export --checkpoint models/qev-new \
  --output models/qev-new-mlx --dtype float32
uv run pytest -q
```

This release uses FP32 by default for MLX. FP16 conversion exceeded the predefined error threshold in the numerical audit and was therefore not selected as the default delivery format. See the [training guide](docs/TRAINING.md) for data preparation and the Colab workflow. Training uses fixed random seeds and records software versions, source and data digests, learning rates, losses, compute resources, and frozen-weight hashes before and after training. Changes in numerical precision and hardware may change results.

The code is licensed under Apache-2.0. The foundation model, datasets, and dependencies retain their respective licenses; see [NOTICE](NOTICE).
