"""Measure three fixed Choice requests against an already running Qev service.

No checkpoint is loaded and no server is started. Each scenario's first observed
request is recorded separately from subsequent serial rounds; server startup
warmup and earlier traffic mean that first observed is not necessarily cold.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageDraw


def fixtures(model):
    """Small deterministic PNGs are built before any measured request."""
    pictures = []
    for shape, color in (("circle", "red"), ("square", "blue"), ("triangle", "green")):
        image = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(image)
        if shape == "circle":
            draw.ellipse((18, 18, 78, 78), fill=color)
        elif shape == "square":
            draw.rectangle((18, 18, 78, 78), fill=color)
        else:
            draw.polygon(((48, 15), (81, 81), (15, 81)), fill=color)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        pictures.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        }})
    def request(state, instructions, criteria):
        return {"model": model, "state": state, "questions": {"pick": {
            "type": "choice", "instructions": instructions, "criteria": criteria,
        }}}
    return {
        "text": request("A customer was charged twice and requests a refund.",
                        "Select the department.", {
                            "billing": "Charges and refunds", "technical": "Software problems",
                            "sales": "New orders",
                        }),
        "state_image": request({"content": [
            {"type": "text", "text": "Identify the shape in this image."}, pictures[0],
        ]}, "Choose the matching shape.", {
            "circle": "Red circle", "square": "Blue square", "triangle": "Green triangle",
        }),
        "candidate_images": request("Select the image of a red circle.",
                                    "Choose one candidate image.", {
                                        key: {"content": [
                                            {"type": "text", "text": f"Candidate {key}"}, picture,
                                        ]} for key, picture in zip(("a", "b", "c"), pictures, strict=True)
                                    }),
    }


def distribution(values):
    if not values:
        return {"n": 0, "median_ms": None, "p95_ms": None}
    ordered = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1]}


def finite_json(response):
    body = response.json()
    json.dumps(body, allow_nan=False)
    return body


def validate_response(response, request):
    answer = response["answers"]["pick"]
    probabilities = answer["probabilities"]
    if answer.get("type") != "choice" or set(probabilities) != set(request["questions"]["pick"]["criteria"]):
        raise ValueError("Response must contain the requested Choice candidate keys")
    values = list(probabilities.values())
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Candidate probabilities must be finite and in [0,1]")
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Candidate probabilities must sum to one")
    if answer.get("choice") not in probabilities or probabilities[answer["choice"]] != max(values):
        raise ValueError("Choice must be a maximum-probability candidate")
    if response["usage"]["output_tokens"] != 0 or response["usage"]["input_tokens"] < 1:
        raise ValueError("Expected real decision inputs and zero generated tokens")


def measure(client, request, round_index):
    sample = {"round": round_index, "ok": False}
    started = time.perf_counter()
    try:
        response = client.post("/v1/systemone", json=request)
        sample.update(http_ms=(time.perf_counter() - started) * 1000,
                      http_status=response.status_code)
        try:
            body = finite_json(response)
        except ValueError:
            sample["response_text"] = response.text[:2000]
            raise ValueError("Service returned invalid or non-finite JSON") from None
        sample["response"] = body
        response.raise_for_status()
        validate_response(body, request)
        timings = body.get("qev", {}).get("timings_ms")
        if timings is not None and not isinstance(timings, dict):
            raise ValueError("Server timings_ms must be an object")
        if timings is not None and any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                                       for v in timings.values()):
            raise ValueError("Server timings_ms must contain finite nonnegative numbers")
        sample["server_timings_ms"] = timings
        latency = body.get("latency_ms")
        if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            raise ValueError("Server latency_ms must be finite and nonnegative")
        sample["server_latency_ms"] = latency
        total = timings.get("total", latency) if timings else latency
        sample["http_minus_server_total_ms"] = None if total is None else sample["http_ms"] - total
        sample["ok"] = True
    except Exception as error:  # noqa: BLE001 -- preserve the failed measurement; never retry.
        sample.setdefault("http_ms", (time.perf_counter() - started) * 1000)
        sample["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    return sample


def summarize(samples):
    valid = [sample for sample in samples if sample["ok"]]
    keys = sorted({key for sample in valid for key in (sample.get("server_timings_ms") or {})})
    return {
        "successful_samples": len(valid), "failed_samples": len(samples) - len(valid),
        "http": distribution([sample["http_ms"] for sample in valid]),
        "server_latency": distribution([sample["server_latency_ms"] for sample in valid
                                        if sample.get("server_latency_ms") is not None]),
        "server_timings": {key: distribution([sample["server_timings_ms"][key] for sample in valid
                                               if key in (sample.get("server_timings_ms") or {})]) for key in keys},
        "http_minus_server_total": distribution([sample["http_minus_server_total_ms"] for sample in valid
                                                  if sample.get("http_minus_server_total_ms") is not None]),
    }


def run(client, args, save):
    report = {
        "format": "qev-http-benchmark-v1", "created_utc": datetime.now(UTC).isoformat(),
        "base_url": args.base_url, "requested_model": args.model, "label": args.label,
        "client": {"platform": platform.platform(), "python": platform.python_version(), "httpx": httpx.__version__},
        "protocol": {"subsequent_rounds": args.repeats, "interval_seconds": args.interval,
                     "order": "text, state_image, candidate_images in each serial round",
                     "fixture_version": 1, "image_size": [96, 96], "model_warmup_calls_by_script": 0,
                     "http_connections": "one persistent client; no retries or environment proxies",
                     "p95_method": "nearest rank; with fewer than 20 samples P95 equals the maximum"},
        "scope": "Fixed small Choice requests, not general model performance or evidence of response caching. "
                 "First observed requests may already be warm due to startup or prior traffic. "
                 "HTTP minus server total also includes HTTP handling and serialization, not just network time. "
                 "Server model aliases do not establish checkpoint identity; label is operator-supplied.",
        "service": {}, "scenarios": {}, "complete": False,
    }
    for name, request in fixtures(args.model).items():
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        report["scenarios"][name] = {"request": request, "request_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                                     "first_observed": None, "subsequent": [], "summary": summarize([])}
    save(report)
    try:
        for name, endpoint in (("health", "/health"), ("models", "/v1/models")):
            response = client.get(endpoint)
            response.raise_for_status()
            report["service"][name] = finite_json(response)
        for round_index in range(args.repeats + 1):
            if round_index and args.interval:
                time.sleep(args.interval)
            for name, scenario in report["scenarios"].items():
                sample = measure(client, scenario["request"], round_index)
                if round_index == 0:
                    scenario["first_observed"] = sample
                else:
                    scenario["subsequent"].append(sample)
                    scenario["summary"] = summarize(scenario["subsequent"])
                save(report)
                print(f"{name} round={round_index} HTTP={sample['http_ms']:.2f} ms ok={sample['ok']}",
                      file=sys.stderr, flush=True)
                if not sample["ok"]:
                    raise RuntimeError(f"{name} round {round_index} failed; no retry: {sample['error']}")
        report["complete"] = True
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- always preserve partial results.
        report["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    finally:
        save(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008", help="Existing service root URL")
    parser.add_argument("--model", default="qev-latest", help="Model alias served by that process")
    parser.add_argument("--repeats", type=int, default=5, help="Subsequent samples per scenario, after its first request (minimum 3)")
    parser.add_argument("--interval", type=float, default=0, help="Idle seconds between rounds; no delay within each round")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--label", help="Optional description of server checkpoint, startup flags and memory conditions")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are not overwritten")
    args = parser.parse_args(argv)
    parsed = urlsplit(args.base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        parser.error("base-url must be an http(s) service root without credentials, path, query or fragment")
    args.base_url = args.base_url.rstrip("/")
    if args.repeats < 3:
        parser.error("repeats must be at least 3 (in addition to the first observed request)")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.interval) or args.interval < 0:
        parser.error("timeout must be finite and positive; interval must be finite and nonnegative")
    if args.output.exists():
        parser.error(f"Refusing to overwrite report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        def save(report):
            stream.seek(0)
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.truncate()
            stream.flush()
        with httpx.Client(base_url=args.base_url, timeout=args.timeout, trust_env=False) as client:
            report = run(client, args, save)
    print(json.dumps({"output": str(args.output.resolve()), "complete": report["complete"],
                      "scenarios": {name: {"first_http_ms": (scenario["first_observed"] or {}).get("http_ms"),
                                            **scenario["summary"]} for name, scenario in report["scenarios"].items()},
                      "error": report.get("error")}, ensure_ascii=False, allow_nan=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
