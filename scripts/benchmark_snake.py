r"""Closed-loop Snake evaluation with original argmax actions and replay traces.

Run parent and candidate separately on the same environment, then pair by seed::

    python scripts/benchmark_snake.py --model models/parent --output results/parent.json
    python scripts/benchmark_snake.py --model models/new --output results/new.json \
        --compare results/parent.json
    python scripts/benchmark_snake.py --base-url http://127.0.0.1:8000 \
        --model-name qev-0.8b --output results/http.json

--batch 8 evaluates up to eight independent live games in one local Torch forward.
This reports batch throughput, not single-step latency. No teacher, safety filter,
action replacement, retry, or generated decision is used. HTTP mode imports no
Torch model and runs this same local game, sending only decision requests.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

# Allow the documented command in a fresh source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qev import snake, snake_features
from qev.snake import SnakeGame, decision_request, model_choice

FORMAT = "qev-snake-benchmark-v1"


def parse_seeds(value):
    """Comma-separated signed seeds or an inclusive start:end range."""
    try:
        if ":" in value:
            start, end = map(int, value.split(":"))
            if not start <= end < start + 10000:
                raise ValueError("seed range must contain 1..10000 seeds")
            seeds = list(range(start, end + 1))
        else:
            seeds = [int(part.strip()) for part in value.split(",")]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(not -(2**31) <= seed < 2**31 for seed in seeds):
            raise ValueError("seeds must be signed 32-bit integers")
        return seeds
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def protocol(args):
    return {
        "seeds": args.seeds, "size": args.size, "initial_length": 3,
        "max_steps": args.max_steps, "starvation_limit": 2 * args.size**2,
        "observation": args.observation,
        "feature_version": (snake_features.FEATURE_VERSION
                            if args.observation == "spatial" else "local-v1"),
        "execution": "model_argmax_direct", "guardrail": False,
        "game_source_sha256": fingerprint(inspect.getsource(SnakeGame)),
        "prompt_source_sha256": fingerprint(inspect.getsource(decision_request)),
        "feature_source_sha256": fingerprint(inspect.getsource(snake_features)),
        "action_validation_sha256": fingerprint(inspect.getsource(model_choice)),
        "direction_order": list(snake.DIRECTIONS),
        "direction_vectors": {key: list(value) for key, value in snake.VECTORS.items()},
        "opposites": snake.OPPOSITE,
    }


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "max": max(values)}


class Predictor:
    """One Agent instance, or serial HTTP requests; never a second policy."""

    def __init__(self, args):
        self.batch_size = args.batch
        self.timeout = args.timeout
        if args.base_url:
            if args.batch != 1:
                raise ValueError("HTTP requests use --batch 1; batching is local Torch only")
            url = urlsplit(args.base_url.rstrip("/"))
            if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
                raise ValueError("base URL must be an http(s) URL without query or fragment")
            if url.username or url.password:
                raise ValueError("Use --api-key-env instead of credentials in the URL")
            path = url.path.rstrip("/")
            if not path.endswith("/systemone"):
                path += "/systemone" if path.endswith("/v1") else "/v1/systemone"
            self.url = urlunsplit((url.scheme, url.netloc, path, "", ""))
            self.model_name = args.model_name or "qev-0.8b"
            self.headers = {"Content-Type": "application/json"}
            if args.api_key_env:
                key = os.environ.get(args.api_key_env)
                if not key:
                    raise ValueError(f"Environment variable {args.api_key_env} is empty")
                self.headers["Authorization"] = f"Bearer {key}"
            self.source = {"kind": "http", "endpoint": self.url,
                           "model_name": self.model_name, "label": args.label,
                           "precision": "server-defined; inspect raw response metadata"}
            self.agent = None
        else:
            from qev.inference import Agent

            self.agent = Agent(args.model, backend=args.backend, device=args.device)
            if args.batch > 1 and self.agent.backend != "torch":
                raise ValueError("--batch > 1 is supported only for local Torch checkpoints")
            self.model_name = args.model_name or self.agent.config.get("model_name", "qev-0.8b")
            self.source = {"kind": "local", "checkpoint": str(self.agent.path),
                           "backend": self.agent.backend, "model_name": self.model_name,
                           "label": args.label, "temperature": self.agent.temperature,
                           "checkpoint_config": self.agent.config,
                           "config_sha256": hashlib.sha256(
                               (self.agent.path / "qev_config.json").read_bytes()).hexdigest()}
            if self.agent.backend == "torch":
                from qev.evaluation import prediction_precision

                self.source["precision"] = prediction_precision(self.agent.model)
            else:
                self.source["precision"] = self.agent.config.get("mlx_dtype")

    def predict(self, requests):
        if self.agent is None:
            assert len(requests) == 1
            body = json.dumps(requests[0], allow_nan=False).encode()
            with urlopen(Request(self.url, data=body, headers=self.headers),
                         timeout=self.timeout) as response:
                return [json.load(response)]
        if self.batch_size == 1:
            assert len(requests) == 1
            return [self.agent.predict(**requests[0])]

        # Match Agent's text path: identical encoder, precision context,
        # candidate slicing, temperature, softmax and typed-answer conversion.
        import torch

        from qev.api import to_answers, to_record
        from qev.evaluation import inference_autocast, probabilities
        from qev.tokenization import collate_encodings, encode_question

        encodings, metadata = [], []
        for request in requests:
            record, meta = to_record(request)
            if len(record["questions"]) != 1:
                raise ValueError("Snake batch requires exactly one question per game")
            encodings.append(encode_question(
                record["state"], record["questions"][0], self.agent.tokenizer,
                self.agent.config.get("max_length", 1024),
                self.agent.config.get("max_state", 384)))
            metadata.append(meta)
        with self.agent.lock, torch.inference_mode(), inference_autocast(self.agent.model):
            logits = self.agent.model(**collate_encodings(
                encodings, self.agent.tokenizer.pad_token_id, self.agent.device)).float().cpu().numpy()
        responses = []
        for request, encoding, meta, row in zip(requests, encodings, metadata, logits):
            values = probabilities(row[:len(encoding["option_positions"])], self.agent.temperature)
            responses.append({"model": request["model"],
                              "answers": to_answers([values.tolist()], meta),
                              "usage": {"input_tokens": len(encoding["ids"]), "output_tokens": 0},
                              "qev": {"backend": "torch", "temperature": self.agent.temperature,
                                      "benchmark_batch_size": len(requests),
                                      "truncated_questions": ["move"] if encoding["state_truncated"] else []}})
        return responses


def validate_response(response, request):
    if response.get("qev", {}).get("truncated_questions"):
        raise ValueError("Snake observation was truncated; no action executed")
    return model_choice(response, request["questions"]["move"]["criteria"])


def episode_result(game):
    return {"seed": game.seed, "food": game.score, "score": game.score, "steps": game.step,
            "length": len(game.body), "status": game.status,
            "terminal_reason": game.terminal_reason,
            "collision": game.terminal_reason in ("wall", "body"),
            "starvation": game.terminal_reason == "starvation", "won": game.status == "won",
            "argmax_direct": True, "interventions": 0}


def paired_comparison(report, reference):
    if reference.get("format") != FORMAT or reference.get("protocol") != report["protocol"]:
        raise ValueError("Comparison requires identical format, seeds, rules, observation and prompt")
    if not reference.get("complete") or not report.get("complete"):
        raise ValueError("Comparison requires two complete runs without model errors")
    if reference["execution"]["batch_size"] != report["execution"]["batch_size"]:
        raise ValueError("Comparison requires equal batch sizes; padding can change model probabilities")
    earlier = {episode["seed"]: episode for episode in reference["episodes"]}
    rows = []
    for episode in report["episodes"]:
        before = earlier[episode["seed"]]
        rows.append({"seed": episode["seed"],
                     "reference_food": before["food"], "candidate_food": episode["food"],
                     "food_delta": episode["food"] - before["food"],
                     "steps_delta": episode["steps"] - before["steps"],
                     "reference_reason": before["terminal_reason"],
                     "candidate_reason": episode["terminal_reason"]})
    return {"reference_source": reference["source"], "paired_episodes": rows,
            "mean_food_delta": statistics.mean(row["food_delta"] for row in rows),
            "more_food": sum(row["food_delta"] > 0 for row in rows),
            "equal_food": sum(row["food_delta"] == 0 for row in rows),
            "less_food": sum(row["food_delta"] < 0 for row in rows),
            "scope": "Same seeded environment; model choices cause trajectories and food locations to diverge. "
                     "This measures closed-loop behavior, not same-state classification accuracy."}


def run_benchmark(args, predictor, trace):
    """Drive real requests; dependency injection is only for lightweight verification."""
    report = {"format": FORMAT, "created_utc": datetime.now(UTC).isoformat(),
              "protocol": protocol(args), "source": predictor.source,
              "execution": {"batch_size": args.batch, "warmup_calls": args.warmup,
                            "timing": ("serial decision request wall time" if args.batch == 1
                                       else "multi-game prediction batch wall time, not per-step latency"),
                            "batch_padding_caveat": "Batching/padding can change floating-point probabilities."},
              "machine": {"platform": platform.platform(), "python": platform.python_version()},
              "episodes": [], "errors": [], "complete": False}

    def write(kind, **fields):
        trace.write(json.dumps({"type": kind, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        trace.flush()

    def make_game(seed):
        return SnakeGame(seed=seed, size=args.size, max_steps=args.max_steps,
                         observation=args.observation)

    write("metadata", **report)
    warmup_game = make_game(args.seeds[0])
    warmup_request = decision_request(warmup_game, predictor.model_name)[0]
    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        responses = predictor.predict([warmup_request] * min(args.batch, len(args.seeds)))
        for response in responses:
            validate_response(response, warmup_request)
    report["warmup_seconds"] = time.perf_counter() - warmup_started
    active, next_seed, call_times, call_sizes, token_counts = [], 0, [], [], []
    positions = Counter()
    direct_steps = requests_count = missed_food = safe_move_rejected = moved_away = 0
    started = time.perf_counter()
    fatal_error = False
    while active or next_seed < len(args.seeds):
        while len(active) < args.batch and next_seed < len(args.seeds):
            game = make_game(args.seeds[next_seed])
            next_seed += 1
            active.append(game)
            write("episode_start", game=game.snapshot())
        requests, candidate_facts, snapshots = [], [], []
        for game in active:
            request, facts = decision_request(game, predictor.model_name)
            requests.append(request)
            candidate_facts.append(facts)
            snapshots.append(game.snapshot())
        call_started = time.perf_counter()
        try:
            responses = predictor.predict(requests)
            if len(responses) != len(requests):
                raise ValueError("Predictor must return one response per game")
        except Exception as error:  # noqa: BLE001 -- Record the failed real call; never retry.
            responses = [None] * len(active)
            fatal_error = True
            call_error = {"type": type(error).__name__, "message": str(error)[:1000]}
        elapsed_ms = (time.perf_counter() - call_started) * 1000
        call_times.append(elapsed_ms)
        call_sizes.append(len(active))
        requests_count += len(active)
        for game, request, facts, before, response in zip(
                active, requests, candidate_facts, snapshots, responses):
            row = {"seed": game.seed, "step_before": game.step, "request": request,
                   "response": response, "candidate_facts": facts, "before": before,
                   "prediction_call": len(call_times), "prediction_batch_size": len(active),
                   "prediction_batch_ms": elapsed_ms, "proposed": None,
                   "executed": None, "intervened": False}
            try:
                if fatal_error:
                    raise RuntimeError(f"Prediction call failed: {call_error}")
                try:
                    json.dumps(response, allow_nan=False)
                except (TypeError, ValueError):
                    row["response"] = None
                    row["unserializable_response_repr"] = repr(response)[:10000]
                    raise ValueError("Model response is not valid finite JSON") from None
                chosen, probabilities = validate_response(response, request)
                chosen_fact = next(move for move in facts if move["direction"] == chosen)
                safe = [move for move in facts if move["collision"] is None]
                safe_move_rejected += bool(safe) and chosen_fact["collision"] is not None
                missed_food += any(move["eats_food"] for move in safe) and not chosen_fact["eats_food"]
                moved_away += (any(move["manhattan_change"] < 0 for move in safe)
                               and chosen_fact["manhattan_change"] > 0)
                positions[list(request["questions"]["move"]["criteria"]).index(chosen)] += 1
                row.update(proposed=chosen, executed=chosen, probabilities=probabilities,
                           ate=game.advance(chosen), argmax_direct=True)
                direct_steps += 1
                tokens = response.get("usage", {}).get("input_tokens")
                if isinstance(tokens, int):
                    token_counts.append(tokens)
            except Exception as error:  # noqa: BLE001 -- Malformed output terminates without fallback.
                game.status, game.terminal_reason = "finished", "model_error"
                row["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
                report["errors"].append({"seed": game.seed, "step": game.step, **row["error"]})
            row.update(step_after=game.step, after=game.snapshot())
            if args.batch == 1:
                row["request_wall_ms"] = elapsed_ms
            write("step", **row)
            if game.status != "running":
                result = episode_result(game)
                report["episodes"].append(result)
                write("episode_end", **result)
        active = [game for game in active if game.status == "running"]
        if fatal_error:
            break
    seconds = time.perf_counter() - started
    order = {seed: index for index, seed in enumerate(args.seeds)}
    report["episodes"].sort(key=lambda row: order[row["seed"]])
    episodes = report["episodes"]
    report["complete"] = len(episodes) == len(args.seeds) and not report["errors"]
    report["aggregate"] = {
        "episodes": len(episodes), "food": distribution([row["food"] for row in episodes]),
        "total_food": sum(row["food"] for row in episodes),
        "steps": sum(row["steps"] for row in episodes),
        "collisions": sum(row["collision"] for row in episodes),
        "starvations": sum(row["starvation"] for row in episodes),
        "wins": sum(row["won"] for row in episodes),
        "terminal_reasons": dict(Counter(row["terminal_reason"] for row in episodes)),
        "argmax_direct_steps": direct_steps, "interventions": 0,
        "selected_candidate_positions": dict(sorted(positions.items())),
        "collision_with_safe_alternative": safe_move_rejected,
        "missed_immediate_food": missed_food,
        "distance_increased_with_safe_closer_alternative": moved_away,
        "local_diagnostic_note": "Moving away from food can be valid with spatial planning; this is not an error label.",
    }
    prediction_seconds = sum(call_times) / 1000
    report["timing"] = {
        "seconds_excluding_load_and_warmup": seconds,
        "executed_steps_per_second_including_game_and_trace": direct_steps / seconds if seconds else 0,
        "prediction_calls": len(call_times), "decision_requests": requests_count,
        "prediction_call_ms": distribution(call_times),
        "prediction_batch_sizes": dict(Counter(call_sizes)),
        "prediction_decisions_per_second": requests_count / prediction_seconds if prediction_seconds else 0,
        "input_tokens_per_request": distribution(token_counts),
    }
    if args.batch == 1:
        report["timing"]["single_request_wall_ms"] = distribution(call_times)
    write("run_end", **report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Local Qev checkpoint")
    source.add_argument("--base-url", help="HTTP server root, /v1, or /v1/systemone")
    parser.add_argument("--model-name", help="API model alias; defaults to local config or qev-0.8b")
    parser.add_argument("--label", help="Human-readable checkpoint identity")
    parser.add_argument("--backend", choices=("auto", "torch", "mlx"), default="auto")
    parser.add_argument("--device")
    parser.add_argument("--api-key-env", help="Name of environment variable holding optional HTTP bearer key")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--batch", type=int, default=1, help="Independent simultaneous games; >1 requires local Torch")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("10000:10019"))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--observation", choices=("local", "spatial"), default="spatial")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Raw JSONL; defaults to OUTPUT stem + .steps.jsonl")
    parser.add_argument("--compare", type=Path, help="Earlier report with identical protocol and batch size")
    args = parser.parse_args(argv)
    if not 1 <= args.batch <= 64 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("batch must be 1..64, warmup nonnegative, timeout finite and positive")
    SnakeGame(seed=args.seeds[0], size=args.size, max_steps=args.max_steps, observation=args.observation)
    args.trace = args.trace or args.output.with_suffix(".steps.jsonl")
    paths = [args.output.resolve(), args.trace.resolve()]
    if args.compare:
        paths.append(args.compare.resolve())
    if len(set(paths)) != len(paths):
        parser.error("output, trace and comparison paths must be distinct")
    reference = json.loads(args.compare.read_text()) if args.compare else None
    if reference and (reference.get("format") != FORMAT or reference.get("protocol") != protocol(args)
                      or reference.get("execution", {}).get("batch_size") != args.batch
                      or not reference.get("complete")):
        parser.error("reference must be complete and use identical rules, seeds, prompt and batch size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    predictor = Predictor(args)
    load_seconds = time.perf_counter() - load_started
    with args.trace.open("w") as trace:
        report = run_benchmark(args, predictor, trace)
    report.update(load_seconds=load_seconds, trace=str(args.trace.resolve()))
    if reference and report["complete"]:
        report["comparison"] = paired_comparison(report, reference)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "trace": str(args.trace),
                      "complete": report["complete"], "aggregate": report["aggregate"],
                      "timing": report["timing"], "comparison": report.get("comparison")},
                     ensure_ascii=False, allow_nan=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
