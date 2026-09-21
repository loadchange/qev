# Qev API

**English** | [简体中文](API.zh-CN.md)

Qev provides two endpoints: `/v1/systemone` returns probabilities over fixed candidates, while `/v1/chat/completions` uses the complete Qwen3.5 foundation's native text, image, and video generation. Native generation disables Qev's decision LoRA and uses the preserved vision modules, language model, and vocabulary output head.

The current decision adapter was trained only on text. Images and videos can participate in the decision forward pass, but **multimodal decision accuracy has not been validated**. Temperature fitted on text tasks is not applied to these probabilities. API compatibility does not imply matching Jev's model quality or confidence values.

## Startup and model names

```bash
uv run qev serve --model models/qev-snake-0.8b-mlx --backend mlx --port 8008
# Or select a complete Torch checkpoint:
uv run qev serve --model models/qev-0.8b --backend torch --port 8008
```

The service listens on `127.0.0.1` by default. The local API does not require an API key.

| Endpoint | Purpose |
|---|---|
| `GET /`, `GET /snake` | Built-in model lab / autonomous Snake page |
| `GET /playground` | API playground |
| `GET /health` | Service status and current backend |
| `GET /v1/models` | TypeSafe-style model list |
| `POST /v1/systemone` | Noul, Choice, and Score decisions |
| `POST /v1/chat/completions` | Non-streaming native Qwen generation |

Pages and model endpoints share the same origin and require no separate frontend build. On the first visit, the page chooses English or Simplified Chinese from the browser language; a manual language switch is saved in the current browser and survives refreshes. Interface language does not rewrite user input or API fields. Snake creation, stepping, export, and related endpoints live under `/api/snake/games`, with state and decisions maintained by the server. See the [demo guide](DEMO.md) for that protocol. These demo endpoints do not change the existing Jev / TypeSafe protocol.

Available aliases include `qev-latest`, `qev:0.8b`, `qev:0.8b-mlx`, `qev-0.8b`, `qev-0.8b-mlx`, and `jev-latest` for older clients. The generation endpoint defaults to `qev-native`. These names refer to the **currently loaded checkpoint**; a request does not switch weights or the Torch/MLX backend.

## Local request logs

`qev serve` logs `POST /v1/systemone` and `POST /v1/chat/completions` to `runs/request-logs` by default, relative to the working directory. Use `--request-log-dir PATH` to change the directory or `--no-request-log` to disable logging. Programmatic `create_app(agent)` does not enable file logging unless a `request_log` is supplied. GET requests and Snake demo endpoints are outside this log.

Each request gets a directory named with UTC time and a UUID. It contains the original `request.json`, including complete data URLs for replay; `response.json` for a completed response; and `metadata.json` with the request ID, time, endpoint, status, elapsed time and model information. Malformed JSON is also preserved verbatim and marked with `request_parse_error`. `media/` stores extracted original PNG/JPEG/WebP bytes, including sampled video frames, without re-encoding. Each media entry records `json_pointer`, `path`, `mime`, `bytes` and `sha256` so it can be matched to the request. A compact `index.jsonl` in the log root lists records. Request headers, including authentication credentials, are not collected, and these files are not served over HTTP.

Completed logged responses include `X-Qev-Request-Id` and `X-Qev-Log-Status: saved` or `error`; a log-writing failure leaves the model response unchanged. Disabled logging adds neither header. Bodies over the 12 MiB HTTP limit are rejected before logging and are not retained. An unhandled inference exception retains the request and exception state when logging succeeds; it may have no `response.json` or log headers because the outer error handler produces the final error response.

## TypeSafe SDK

Request serialization, model listing, and SDK decoding of all three answer types have been checked with the official `typesafe-sdk==0.7.0`. `instructions` is optional; the SDK's default `None` is accepted as empty instructions.

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8008",
    model="qev:0.8b",
)
result = client.system_one(
    state="I was charged twice. Please refund the duplicate.",
    questions={
        "team": Choice(instructions="Who handles this?", criteria={
            "billing": "Payments and refunds", "technical": "Software problems",
        }),
        "refund": Noul(instructions="Does the customer request a refund?"),
        "urgency": Score(instructions="How urgent?", criteria=["Low", "Medium", "High"]),
    },
)
print(result.choices["team"].probabilities)
print(result.nouls["refund"].noul)
print(result.scores["urgency"].score)
```

`state` continues to accept JSON strings, objects, arrays, and other JSON values. Ordinary structured JSON is rendered with field names and ordering, preserving the original text interface. Choice accepts 1–255 named candidates; Score accepts 2–255 ordered levels. Each request allows at most 64 questions.

Answers contain:

- Noul: `noul = P(true)`.
- Choice: `choice`, candidate probabilities, and `confidence`.
- Score: the expected zero-based level, `legend`, level probabilities, and `confidence`.

Probabilities retain numerical precision and are not produced by generating JSON text. `confidence` does not guarantee correctness. Score confidence is an approximate convention based on the expected distance from the modal level; TypeSafe's exact formula is not public. `qev.question_temperatures` reports the actual temperature by question name: text-only questions retain checkpoint calibration, while questions containing images or video use `1.0`. `qev.temperature` reports that value when all questions use the same temperature, otherwise `null`. Text-only questions in mixed requests still retain text calibration. The decision endpoint always returns `usage.output_tokens=0`.

The default HTTP service and Python `Agent` run one question at a time. Torch's default `batch_size=1` matches MLX's per-question behavior and prevents a question's computation shape from changing with other questions in the request. Python users may explicitly create `Agent(checkpoint, batch_size=4)` to improve Torch throughput. CUDA BF16 batch and padding changes can alter probabilities, so this optional mode is not guaranteed to match per-question results. Training and offline evaluation retain their respective batch-4 settings.

## Image decisions

The playground at <http://127.0.0.1:8008/playground> offers three `choice` scenarios: text only, an image in the background, and images in every option. All call `/v1/systemone`, returning `choice` and `probabilities` without generating answer text. Text-only backgrounds and candidate descriptions can still be strings. For background or candidate images, use an OpenAI-style content array or a `{"content": [...]}` wrapper. Images must be PNG, JPEG, or WebP base64 data URLs.

The background image area and each candidate image area support both upload and paste. Select the target area's paste input and press Ctrl/Cmd+V, or use its clipboard button when browser clipboard reading is available. Both paths convert images in the browser to the same inline protocol and obey the media budgets below. Images are submitted to the service only when the request is sent.

Background image example:

```python
import base64
import httpx
from pathlib import Path

# The client reads the file explicitly; the server never opens a client-supplied path.
image = "data:image/png;base64," + base64.b64encode(Path("photo.png").read_bytes()).decode()
state = [
    {"type": "text", "text": "Identify the main subject of the image."},
    {"type": "image_url", "image_url": {"url": image}},
]
response = httpx.post("http://127.0.0.1:8008/v1/systemone", json={
    "model": "qev:0.8b",
    "state": state,
    "questions": {
        "scene": {"type": "choice", "instructions": "Choose the main setting.",
                  "criteria": {"indoors": "Indoors", "outdoors": "Outdoors"}},
    },
}, timeout=120)
response.raise_for_status()
print(response.json())
```

When each candidate has an image, place it in the corresponding `criteria[key].content` and keep stable candidate keys. Candidate content supports text and images; sampled video frames still belong in `state`. The following client code reuses the imports above, reads two local PNGs, and submits their bytes to the service:

```python
def candidate(path, description):
    url = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    return {"content": [
        {"type": "text", "text": description},
        {"type": "image_url", "image_url": {"url": url}},
    ]}

response = httpx.post("http://127.0.0.1:8008/v1/systemone", json={
    "model": "qev:0.8b",
    "state": "Choose the red circle from the candidate images below.",
    "questions": {
        "shape": {
            "type": "choice",
            "instructions": "Choose the image that best matches the background requirement.",
            "criteria": {
                "a": candidate("candidate-a.png", "Candidate A"),
                "b": candidate("candidate-b.png", "Candidate B"),
            },
        },
    },
}, timeout=120)
response.raise_for_status()
answer = response.json()["answers"]["shape"]
print(answer["choice"], answer["probabilities"])
```

Candidate images are encoded inside their own candidate boundaries, retaining visual patches, image grids, and multimodal position encoding. Each question runs independently, and one question may contain both background and candidate images. Base64 is not treated as text, and captions do not replace real visual input. Questions containing media use `T=1`, and the response reports `multimodal_decision_accuracy_validated: false`. Current training supervision remains textual; this input capability does not establish validated visual selection accuracy.

## Native multimodal generation

```python
response = httpx.post("http://127.0.0.1:8008/v1/chat/completions", json={
    "model": "qev-native",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Please describe this image."},
        {"type": "image_url", "image_url": {"url": image}},
    ]}],
    "max_tokens": 128,
    "temperature": 0,
}, timeout=120)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

The response is an OpenAI-style `chat.completion`, including `choices`, `finish_reason`, and actual `prompt_tokens/completion_tokens/total_tokens`. `qev.decision_adapter_enabled` is `false`. This endpoint uses the native generation head, not the candidate pointer head. Decision training accuracy does not describe native generation quality.

`system/user/assistant` messages are supported. Set either `max_tokens` or `max_completion_tokens`, with a range of 1–2048 and a default of 256. `temperature=0` uses greedy generation; values above 0 support `top_p`. Set `enable_thinking: true` to use the native thinking template. Only `stream: false` is currently supported; tool calling and audio protocols are not implemented.

## Video

Video uses Qev's explicit sampled-frame extension. Sample frames on the client, then submit chronologically ordered image data URLs with identical dimensions:

```python
video_content = {
    "type": "video",
    "frames": [first_frame_data_url, second_frame_data_url],
    "fps": 2.0,
}
# Native generation: messages=[{"role":"user","content":[
#   {"type":"text","text":"Describe the change between these two frames."}, video_content]}]
# Candidate decisions: state={"content":[video_content]}
```

`fps` is the **sampling frame rate of the submitted frame sequence**, not the original file's frame rate. The native processor uses it to calculate frame times, with additional frame sampling disabled. The API does not accept local video paths, remote `video_url` values, or raw MP4 bytes. This is an API input convention; Qwen3.5's video modules have not been removed.

## Input budgets and errors

Images/video frames in `state` and candidate images in every question's `criteria` share one request-wide media budget: at most 8 images, at most 2 MiB of file bytes per image after base64 decoding, and at most 8 MiB total. Each image allows at most 4 million pixels, with at most 8 million pixels total. The HTTP body limit is 12 MiB. Frame rate must be greater than 0 and no more than 60. Only single-frame PNG/JPEG/WebP files are accepted; animations must be split into video frames.

Expanded multimodal decisions allow at most 8192 tokens by default. Native generation allows at most 16384 tokens for the prompt plus generation budget. Excess input is rejected explicitly; images, video frames, and candidates are not silently dropped. Existing text-only decisions retain the checkpoint's text length budget and report state truncation in `qev.truncated_questions`.

Invalid media, exceeded content budgets, and unsupported options return 422. Excessive HTTP bodies return 413, unknown models return 404, and older checkpoints/backends without complete multimodal capabilities return 501. The server does not download image URLs, open `file://` resources, or execute client-supplied local paths.
