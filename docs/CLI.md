# Qev command line

**English** | [简体中文](CLI.zh-CN.md)

`qev` asks the model for typed decisions from the terminal, serves the HTTP API and web lab, and manages models. On Apple Silicon it runs on MLX without PyTorch.

## Install

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull            # download and verify the default model (1.0 GB) into ~/.qev
qev doctor          # check the chip, MLX/Metal, model files and a local server
```

Requirements: Apple Silicon, macOS 14 or newer. Homebrew may ask you to trust the tap first (`brew trust loadchange/qev`). From a source checkout, `uv sync --python 3.12 --extra mlx` and `uv run qev …` work the same way.

`qev pull --from DIR [--link]` imports a checkpoint you already downloaded (for example with `hf download twainsk/qev-450m-mlx`) after checking every runtime file against the release manifest shipped with qev.

Registry models (select with `QEV_MODEL`, `--model`, or `qev pull NAME`):

| Name | Foundation | Decides on | Download |
|---|---|---|---|
| `qev-450m` (default) | LFM2.5-VL-450M | text, images, video frames | 1.00 GB |
| `qev-230m` | LFM2.5-230M | text only (media is rejected) | 0.53 GB |
| `qev-0.8b` | Qwen3.5-0.8B | text, images, video | 3.48 GB |

The LFM2.5 models are distributed under the LFM Open License v1.0: commercial use by an entity with annual revenue of USD 10 million or more is not licensed.

## Decisions

```bash
qev decide "Card declined twice at checkout" -i "Which team should handle this?" \
  --choice "billing=Invoices, payments and refunds" "technical=Bugs and outages" sales
qev decide "The customer asks for the duplicate charge to be refunded" -i "Is a refund requested?" --noul -q
cat ticket.txt | qev decide - -i "How urgent is this?" --score low medium high
qev decide --image receipt.jpg -i "Is this a restaurant receipt?" --noul
qev decide --frame f1.png --frame f2.png --fps 2 -i "Does the door open?" --noul
qev decide --request request.json --json        # a full /v1/systemone request, several questions
```

| Option | Meaning |
|---|---|
| `STATE`, `-`, `--state-file`, `--state-json` | The state to judge: text, standard input, a text file, or a JSON object/array |
| `--image PATH` | Attach an image (repeatable). Large photos are rotated by EXIF and downscaled to the API limits |
| `--frame PATH` + `--fps` | Attach a video as ordered frames |
| `--choice KEY[=DESCRIPTION]…` / `--noul` / `--score LEVEL…` | Question type; `-i` sets the question and `--id` its name |
| `--request PATH` | Send a complete request instead |
| default / `--json` / `-q` | Readable probabilities, the full API response, or only the answer |
| `--exit-status` | With one yes/no question, exit 0 for yes and 1 for no |

Exit codes: 0 success, 1 “no” with `--exit-status`, 2 invalid input, 3 model not installed, 4 server error.

## Keep the model warm

Loading the model takes a few seconds. When a server answers on `$QEV_SERVER` (default `http://127.0.0.1:8008`) and runs the requested model, `qev decide`, `qev chat` and `qev snake` send requests to it instead:

```bash
brew services start qev      # launchd service: qev serve --no-request-log --decision-weights bf16
qev decide "…" --noul -q     # about 0.1–0.3 s end to end on an M4, versus about 5 s loading in-process
```

`--local` always loads the model in-process; `--server URL` requires a specific server.

## HTTP API

```bash
qev serve [--port 8008] [--host 127.0.0.1] [--decision-weights bf16]
```

It serves `POST /v1/systemone`, `POST /v1/chat/completions`, `GET /v1/models`, `GET /health` and the web lab at `/`. See [API](API.md). Interactive `qev serve` records requests under `$QEV_HOME/request-logs` unless `--no-request-log` is given; the Homebrew service does not record them.

## Speed on Apple Silicon

`--decision-weights` (or `$QEV_DECISION_WEIGHTS`) chooses how the MLX runtime computes decisions. Native generation always uses the unchanged foundation. For the default `qev-450m` (bfloat16 foundation) on an M4:

| Mode | Decision path | Snake decision, M4 | Peak MLX memory |
|---|---|---|---|
| `adapter` | LoRA computed separately (validated reference) | 98 ms | 1.78 GiB |
| `merged` | plus a merged float32 decision copy | 96 ms | 2.94 GiB |
| `merged-bf16` | plus a merged bfloat16 decision copy | 70 ms | 1.91 GiB |
| `bf16` | whole model in bfloat16, merged decision weights | 71 ms | 1.86 GiB |

On the 1,380 development questions every mode scored 86.2% (Snake 98.8%, general 79.1%) and differed from the PyTorch reference by the same 2 decisions ([results](results/qev-450m_decision_weights.json)). The command-line default is `adapter`; the Homebrew service uses `bf16`. The text-only `qev-230m` answers a Snake decision in 43 ms at 1.18 GiB in `bf16` ([results](results/qev-230m_decision_weights.json)); the previous `qev-0.8b` measurements (float32 foundation, 149–219 ms, up to 5.07 GB) remain in [decision_weights.json](results/decision_weights.json).

## Other commands

| Command | Purpose |
|---|---|
| `qev chat "prompt" [--image PATH]` | Native text/image generation with the decision adapter off |
| `qev list`, `qev pull NAME`, `qev rm NAME` | Registry models in `$QEV_HOME/models` (default `~/.qev`) |
| `qev doctor [--verify] [--json]` | Environment, model and server checks |
| `qev snake` | Terminal Snake; uses a running server or the installed model outside a checkout |
| `qev predict --model DIR --request FILE` | Run one request file and print JSON (kept for compatibility) |

Environment: `QEV_HOME` (models and logs), `QEV_MODEL` (default model), `QEV_SERVER` (server URL), `QEV_DECISION_WEIGHTS`, `QEV_PROCESSOR` (`hf` or `numpy` image/video processors; `auto` uses the Transformers ones when torch is installed).
