"""Exercise a real Qev checkpoint over loopback HTTP and the official SDK.

    python scripts/validate_service.py --checkpoint models/qev-0.8b \
        --backend torch --device cuda --output runs/service_torch.json
    python scripts/validate_service.py --checkpoint models/qev-0.8b-mlx \
        --backend mlx --output runs/service_mlx.json

The default starts and stops its own server process. With --base-url, use an
already running service instead. Media probes check functioning and isolation,
not image/video decision accuracy. No mocked transports or random models are
used by this acceptance script.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

STATE = {"customer": "Mira", "message": "I was charged twice. Please refund the duplicate today.",
         "context": {"language": "English", "客户编号": 42}}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Invoices, charges and refunds", "technical": "Software bugs", "sales": None}},
    "refund": {"type": "noul", "instructions": {"rule": "Does the customer request money back?"}},
    "urgency": {"type": "score", "instructions": "How urgent is the request?",
                "criteria": ["No time pressure", "Soon", "Today or a blocking issue"]},
}
TEXT_CHAT = {"model": "qev-native", "messages": [{"role": "user", "content": "Reply with the single word ready."}],
             "temperature": 0, "max_tokens": 16, "enable_thinking": False}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def fixtures():
    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).rectangle((48, 48, 176, 176), fill="red")
    frames = []
    for x in (48, 88, 128, 168):
        frame = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(frame).ellipse((x - 28, 84, x + 28, 140), fill="blue")
        frames.append(data_url(frame))
    return {"type": "image_url", "image_url": {"url": data_url(image)}}, {"type": "video", "frames": frames, "fps": 2.0}


def validate_decision(result, questions, backend, *, modalities=("text",)):
    require(set(result["answers"]) == set(questions), "Response question IDs differ from the request")
    require(result["usage"]["input_tokens"] > 0 and result["usage"]["output_tokens"] == 0,
            "Typed decisions must report input tokens and zero generated tokens")
    require(result["qev"]["backend"] == backend, "Unexpected decision backend")
    require(set(result["qev"]["input_modalities"]) == set(modalities), "Incorrect modality metadata")
    if len(modalities) > 1:
        require(result["qev"]["temperature"] == 1.0, "Text temperature was applied to media")
        require(result["qev"]["multimodal_decision_accuracy_validated"] is False,
                "Text training cannot establish media decision accuracy")
    for key, question in questions.items():
        answer = result["answers"][key]
        require(answer["type"] == question["type"], f"Wrong answer type for {key}")
        if question["type"] == "noul":
            require(math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1, "Invalid P(true)")
            continue
        expected_keys = (list(question["criteria"]) if question["type"] == "choice"
                         else [str(i) for i in range(len(question["criteria"]))])
        probabilities = answer["probabilities"]
        require(list(probabilities) == expected_keys, f"Option identity/order changed for {key}")
        values = list(probabilities.values())
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in values), "Invalid probability")
        require(math.isclose(math.fsum(values), 1.0, abs_tol=1e-6), "Probabilities do not sum to one")
        require(math.isfinite(answer["confidence"]) and -1e-6 <= answer["confidence"] <= 1,
                "Invalid confidence")
        if question["type"] == "choice":
            require(answer["choice"] == max(probabilities, key=probabilities.get), "Choice differs from argmax")
        else:
            require(answer["legend"] == dict(zip(expected_keys, question["criteria"])), "Score legend changed")
            require(math.isclose(answer["score"], sum(i * p for i, p in enumerate(values)), abs_tol=1e-6),
                    "Score is not the expected rubric level")
    return result


def validate_native(result, backend, *, frames=0, max_tokens=16):
    require(result["object"] == "chat.completion", "Incorrect native response object")
    require(result["qev"]["backend"] == backend, "Unexpected native backend")
    require(result["qev"]["decision_adapter_enabled"] is False, "Native generation used decision adapters")
    require(result["qev"]["media_frames"] == frames, "Incorrect decoded frame count")
    require(len(result["choices"]) == 1, "Expected one native continuation")
    choice = result["choices"][0]
    require(choice["message"]["role"] == "assistant" and isinstance(choice["message"]["content"], str),
            "Invalid native assistant message")
    require(choice["finish_reason"] in {"stop", "length"}, "Invalid completion reason")
    usage = result["usage"]
    require(usage["prompt_tokens"] > 0 and 1 <= usage["completion_tokens"] <= max_tokens, "Invalid native usage")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Incorrect total tokens")
    return result


def compare_answers(expected, observed, atol):
    require(set(expected) == set(observed), "Question identities changed between calls")
    largest = 0.0
    for name, before in expected.items():
        after = observed[name]
        require(before["type"] == after["type"], "Answer type changed")
        if before["type"] == "noul":
            pairs = [(before["noul"], after["noul"])]
        else:
            require(before["probabilities"].keys() == after["probabilities"].keys(), "Option identities changed")
            pairs = [(value, after["probabilities"][key]) for key, value in before["probabilities"].items()]
            if before["type"] == "choice":
                require(before["choice"] == after["choice"], "Selected decision changed between calls")
        largest = max(largest, *(abs(a - b) for a, b in pairs))
    require(largest <= atol, f"Repeated decision probability drift {largest} exceeds {atol}")
    return {"max_probability_error": largest, "atol": atol}


def native_signature(result):
    return {"choices": result["choices"], "usage": result["usage"]}


@contextmanager
def service(args, report):
    if args.base_url:
        yield args.base_url.rstrip("/")
        return
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "qev.cli", "serve", "--model", str(Path(args.checkpoint).resolve()),
               "--backend", args.backend, "--host", "127.0.0.1", "--port", str(port)]
    if args.device:
        command.extend(["--device", args.device])
    log_path = Path(args.output).with_suffix(".server.log").resolve()
    report["server_log"] = str(log_path)
    report["server_command"] = command
    with log_path.open("w") as server_log:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                   stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.startup_timeout
            with httpx.Client(timeout=2.0, trust_env=False) as probe:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited with {process.returncode}; inspect {log_path}")
                    try:
                        if probe.get(base_url + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Server startup timed out; inspect {log_path}")
                    time.sleep(0.2)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def run_checks(args, report, base_url):
    import typesafe_sdk as sdk

    backend = report["backend"]
    image, video = fixtures()
    initial = {}
    with httpx.Client(base_url=base_url, timeout=args.request_timeout, trust_env=False) as client:
        def run(name, operation):
            started = time.perf_counter()
            try:
                value = operation()
                report["checks"][name] = {"passed": True, "seconds": time.perf_counter() - started, "result": value}
            except Exception as exc:  # noqa: BLE001 -- collect each failed acceptance check into the final report
                report["checks"][name] = {"passed": False, "seconds": time.perf_counter() - started,
                                          "error": f"{type(exc).__name__}: {exc}"}
            print("QEV_SERVICE_CHECK", name, report["checks"][name]["passed"], flush=True)

        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()

        def post(path, body):
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()

        def decision(state=STATE, questions=QUESTIONS, model="jev-latest", modalities=("text",)):
            return validate_decision(post("/v1/systemone", {"state": state, "questions": questions, "model": model}),
                                     questions, backend, modalities=modalities)

        def health():
            result = get("/health")
            require(result == {"status": "ok", "backend": backend}, "Incorrect health/backend")
            return result

        run("health", health)
        run("models_http", lambda: get("/v1/models"))

        def initial_decision():
            result = decision()
            initial["decision"] = result
            return result
        run("text_decision", initial_decision)

        def sdk_roundtrip():
            with sdk.TypeSafeClient(api_key="local-acceptance", base_url=base_url, model="jev-latest",
                                    timeout=args.request_timeout) as official:
                models = official.models.list()
                require("qev-native" in {model.name for model in models.models}, "SDK model list missing native alias")
                result = official.system_one(state=STATE, questions={
                    name: {"choice": sdk.Choice, "noul": sdk.Noul, "score": sdk.Score}[question["type"]](
                        **{key: value for key, value in question.items() if key != "type"})
                    for name, question in QUESTIONS.items()
                })
                require(set(result.choices) == {"department"} and set(result.nouls) == {"refund"}
                        and set(result.scores) == {"urgency"}, "Official SDK typed accessors failed")
                require(result.scores["urgency"].legend == {0: "No time pressure", 1: "Soon", 2: "Today or a blocking issue"},
                        "Official SDK score legend coercion failed")
                require(result.usage.output_tokens == 0 and result.model == "jev-latest", "SDK usage/model serialization failed")
                serialized = result.model_dump(mode="json")
                compare = compare_answers(initial["decision"]["answers"], serialized["answers"], args.probability_atol)
                return {"response": serialized, "comparison": compare, "typed_accessors": True}
        run("official_sdk_over_http", sdk_roundtrip)

        def aliases():
            results = {}
            for alias in ("qev-latest", "qev:0.8b", "qev:0.8b-mlx", "qev-0.8b", "qev-0.8b-mlx"):
                result = decision(model=alias)
                require(result["model"] == alias, "Response did not preserve requested model alias")
                results[alias] = compare_answers(initial["decision"]["answers"], result["answers"], args.probability_atol)
            return results
        run("typed_model_aliases", aliases)

        def question_isolation():
            results = {}
            for name, question in QUESTIONS.items():
                single = decision(questions={name: question})
                results[name] = compare_answers({name: initial["decision"]["answers"][name]}, single["answers"], args.isolation_atol)
            reverse = decision(questions=dict(reversed(list(QUESTIONS.items()))))
            results["reversed"] = compare_answers(initial["decision"]["answers"], reverse["answers"], args.isolation_atol)
            return results
        run("independent_question_rows", question_isolation)

        def native_text_initial():
            result = validate_native(post("/v1/chat/completions", TEXT_CHAT), backend)
            initial["native"] = result
            return result
        run("native_text_before_media", native_text_initial)

        # Sampling uses MLX random state in addition to the model's arrays.
        # Exercise it over the worker-thread HTTP path as well as greedy mode.
        run("native_sampling", lambda: validate_native(post("/v1/chat/completions", {
            **TEXT_CHAT, "temperature": 0.7, "top_p": 0.9,
        }), backend))

        color_questions = {"color": {"type": "choice", "instructions": "What color is the square?",
                                     "criteria": {"red": None, "blue": None, "green": None}}}
        motion_questions = {"motion": {"type": "choice", "instructions": "In which direction does the blue circle move?",
                                       "criteria": {"left to right": None, "right to left": None, "stationary": None}}}
        run("image_decision", lambda: decision(state=[{"type": "text", "text": "Inspect the attached image."}, image],
            questions=color_questions, modalities=("text", "image")))
        run("video_decision", lambda: decision(state={"content": [video]}, questions=motion_questions,
            modalities=("text", "video")))
        run("mixed_image_video_decision", lambda: decision(state={"content": [image, video]}, questions=color_questions,
            modalities=("text", "image", "video")))

        def native_media(media, prompt, frames, max_tokens):
            body = {"model": "qev-native", "messages": [{"role": "user", "content": [media, {"type": "text", "text": prompt}]}],
                    "max_completion_tokens": max_tokens, "temperature": 0, "enable_thinking": False}
            return validate_native(post("/v1/chat/completions", body), backend, frames=frames, max_tokens=max_tokens)
        run("native_image", lambda: native_media(image, "What color is the shape in the image? Answer in one word.", 1, 16))
        run("native_video", lambda: native_media(video, "Describe the motion of the blue circle in one short sentence.", 4, 32))

        def native_repeat():
            result = validate_native(post("/v1/chat/completions", TEXT_CHAT), backend)
            require(native_signature(result) == native_signature(initial["native"]), "Native text changed after media requests")
            return {"identical_continuation_and_usage": True, "response": result}
        run("native_text_after_media", native_repeat)
        run("decision_adapter_restored", lambda: compare_answers(initial["decision"]["answers"], decision()["answers"], args.probability_atol))

        def concurrent_modes():
            with ThreadPoolExecutor(max_workers=2) as pool:
                typed = pool.submit(decision)
                native = pool.submit(lambda: validate_native(post("/v1/chat/completions", TEXT_CHAT), backend))
                typed_result, native_result = typed.result(), native.result()
            comparison = compare_answers(initial["decision"]["answers"], typed_result["answers"], args.probability_atol)
            require(native_signature(native_result) == native_signature(initial["native"]), "Concurrent mode switching changed native text")
            return {"decision": comparison, "native_identical": True}
        run("concurrent_adapter_modes", concurrent_modes)

        def invalid_requests():
            binary = {"answer": {"type": "noul"}}
            cases = {
                "unknown_model": ("/v1/systemone", {"model": "unrecognized", "state": "hello", "questions": binary}, 404),
                "remote_image": ("/v1/systemone", {"state": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}], "questions": binary}, 422),
                "invalid_fps": ("/v1/systemone", {"state": [{**video, "fps": 0}], "questions": binary}, 422),
                "too_many_frames": ("/v1/systemone", {"state": [image] * 9, "questions": binary}, 422),
                "invalid_score": ("/v1/systemone", {"state": "hello", "questions": {"q": {"type": "score", "criteria": ["only"]}}}, 422),
                "too_many_questions": ("/v1/systemone", {"state": "hello", "questions": {str(i): {"type": "noul"} for i in range(65)}}, 422),
                "streaming": ("/v1/chat/completions", {**TEXT_CHAT, "stream": True}, 422),
                "conflicting_budgets": ("/v1/chat/completions", {**TEXT_CHAT, "max_completion_tokens": 16}, 422),
                "native_path": ("/v1/chat/completions", {**TEXT_CHAT, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "/private/image.png"}}]}]}, 422),
            }
            results = {}
            for name, (path, body, status) in cases.items():
                response = client.post(path, json=body)
                require(response.status_code == status, f"{name}: expected {status}, got {response.status_code}")
                results[name] = {"status": response.status_code, "detail": response.json().get("detail")}
            response = client.post("/v1/systemone", content=b" " * (12 * 1024 * 1024 + 1),
                                   headers={"Content-Type": "application/json"})
            require(response.status_code == 413, "Oversized request was not rejected before JSON parsing")
            results["oversized_body"] = {"status": response.status_code}
            return results
        run("invalid_inputs", invalid_requests)
        run("healthy_after_rejections", health)
        run("inference_after_rejections", lambda: compare_answers(
            initial["decision"]["answers"], decision()["answers"], args.probability_atol))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto")
    parser.add_argument("--device")
    parser.add_argument("--base-url", help="Test an already running server instead of starting one")
    parser.add_argument("--output", required=True)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--probability-atol", type=float, default=1e-5)
    parser.add_argument("--isolation-atol", type=float, default=0.005,
                        help="Batch-vs-single-row tolerance; bf16 kernels can round different shapes differently")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in (args.startup_timeout, args.request_timeout)):
        parser.error("Timeouts must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.probability_atol, args.isolation_atol)):
        parser.error("Probability tolerance must be finite and nonnegative")
    checkpoint = Path(args.checkpoint).resolve()
    config_path = checkpoint / "qev_config.json"
    config = json.loads(config_path.read_text())
    report = {"checkpoint": str(checkpoint), "backend": config.get("runtime", "torch") if args.backend == "auto" else args.backend,
              "base_model": config.get("base_model"), "base_revision": config.get("base_revision"),
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "checks": {}, "passed": False,
              "scope": "Real checkpoint via HTTP and the official TypeSafe SDK. Checks schemas, media plumbing, question isolation, adapter restoration, and repeated greedy text. These are functional checks, not held-out accuracy or cross-runtime native parity."}
    for filename in ("pointer.safetensors", "adapter/adapter_model.safetensors", "decision_adapters.safetensors"):
        path = checkpoint / filename
        if path.is_file():
            report.setdefault("artifact_sha256", {})[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    for package in ("typesafe-sdk", "httpx", "transformers", "torch", "mlx", "mlx-vlm"):
        try:
            report.setdefault("versions", {})[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with service(args, report) as base_url:
            report["base_url"] = base_url
            run_checks(args, report, base_url)
        report["passed"] = bool(report["checks"]) and all(check["passed"] for check in report["checks"].values())
    except Exception as exc:  # noqa: BLE001 -- persist failure evidence, then exit unsuccessfully
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("QEV_SERVICE_VALIDATION", json.dumps({"passed": report["passed"], "output": str(output),
          "checks": len(report["checks"]), "failed": [name for name, check in report["checks"].items() if not check["passed"]]}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
