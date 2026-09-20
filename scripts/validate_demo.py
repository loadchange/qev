"""Audit an already running Qev demo over real HTTP; never load/start a model.

    python scripts/validate_demo.py --base-url http://127.0.0.1:8008 \
        --seeds 7,11,42 --max-steps 32 --output docs/results/demo_validation.json

Valid model-selected collisions are game outcomes, not acceptance failures.
Malformed probabilities, truncated observations, and model errors fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def distribution(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": ordered[0], "max": ordered[-1],
            "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]}


def step_traces(steps):
    for entry in steps:
        snapshot = entry.get("snapshot")
        trace = snapshot.get("last_decision") if isinstance(snapshot, dict) else None
        if isinstance(trace, dict):
            yield trace


def report_json(value):
    """Retain evidence of illegal JSON numbers without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": repr(value)}
    if isinstance(value, dict):
        return {key: report_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [report_json(item) for item in value]
    return value


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        path = attributes.get("src" if tag == "script" else "href" if tag == "link" else "")
        if path and path.startswith("/assets/"):
            self.paths.add(path)


def audit_step(before, after, *, backend):
    """Check the action against the original request, response, and environment."""
    require(after["id"] == before["id"], "A step changed the session identity")
    require(after["policy"] == {"mode": "model_direct", "feature_assisted": True,
                                "planner": False, "guardrail": False}, "Unexpected game policy")
    trace = after["last_decision"]
    require(isinstance(trace, dict), "Missing decision trace")
    require("error" not in trace and after["terminal_reason"] != "model_error",
            f"Model failure cannot pass acceptance: {trace.get('error')}")
    request, response = trace["request"], trace["response"]
    require(isinstance(request["state"], str) and request["state"], "Missing model observation")
    require(set(request["questions"]) == {"move"}, "Unexpected decision questions")
    question = request["questions"]["move"]
    require(question["type"] == "choice", "Snake must use a choice question")
    require(trace["candidate_facts"] == before["candidates"], "Trace facts differ from the observed board")
    expected_criteria = {
        fact["direction"]: {key: "none" if key == "collision" and value is None else value
                            for key, value in fact.items() if key != "direction"}
        for fact in before["candidates"]
    }
    require(question["criteria"] == expected_criteria, "Request changed the observed candidate facts")
    directions = before["available_directions"]
    require(len(directions) == 3 and list(question["criteria"]) == directions,
            "All three non-reversing candidates must reach the model")
    require(response["model"] == request["model"], "Model identity differs between request and response")
    require(response["qev"]["backend"] == backend, "Decision backend differs from /health")
    require(response["qev"]["truncated_questions"] == [], "Model observation was truncated")
    require(response["qev"]["input_modalities"] == ["text"], "Snake must use text observations")
    require(response["usage"]["input_tokens"] > 0 and response["usage"]["output_tokens"] == 0,
            "Invalid decision token usage")
    answer = response["answers"]["move"]
    probabilities = answer["probabilities"]
    require(answer["type"] == "choice" and list(probabilities) == directions,
            "Response candidate identities/order differ from the request")
    require(all(finite_number(p) and 0 <= p <= 1 for p in probabilities.values()),
            "Non-finite or out-of-range model probability")
    require(math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-6, rel_tol=0),
            "Model probabilities do not sum to one")
    require(trace["probabilities"] == probabilities, "Displayed probabilities differ from the model output")
    argmax = max(probabilities, key=probabilities.get)
    require(argmax == answer["choice"] == trace["proposed"] == trace["executed"],
            "Model argmax, reported choice, proposed action, and execution must agree")
    require(trace["intervened"] is False, "A safety intervention replaced the model action")
    require(trace["step_before"] == before["step"] and trace["step_after"] == after["step"]
            and trace["step"] == after["step"] == before["step"] + 1, "Incorrect step accounting")
    require(after["history_count"] == before["history_count"] + 1, "Incorrect history count")
    require(after["direction"] == argmax, "Game direction differs from the executed action")
    fact = next(item for item in before["candidates"] if item["direction"] == argmax)
    if fact["collision"]:
        require(after["status"] == "dead" and after["terminal_reason"] == fact["collision"],
                "A selected collision was hidden or substituted")
        require(after["body"] == before["body"] and after["score"] == before["score"],
                "Collision changed the body or score")
        require(trace["ate"] is False, "A collision cannot eat food")
    else:
        ate = fact["eats_food"]
        body = [fact["next_cell"], *before["body"]]
        if not ate:
            body.pop()
        require(after["body"] == body, "Body transition does not match the chosen next cell")
        require(after["score"] == before["score"] + int(ate) and trace["ate"] == ate,
                "Food/score accounting differs from the executed move")
        require(after["length"] == before["length"] + int(ate), "Incorrect snake growth")
        require(after["steps_since_food"] == (0 if ate else before["steps_since_food"] + 1),
                "Incorrect starvation counter")
        if not ate:
            require(after["food"] == before["food"], "Food moved without being eaten")
        if after["food"] is not None:
            require(after["food"] not in after["body"], "Food spawned on the snake")
        expected_reason = ("board_filled" if after["length"] == after["size"] ** 2
                           else "max_steps" if after["step"] >= after["max_steps"]
                           else "starvation" if after["steps_since_food"] >= after["starvation_limit"]
                           else None)
        expected_status = "won" if expected_reason == "board_filled" else "finished" if expected_reason else "running"
        require(after["terminal_reason"] == expected_reason and after["status"] == expected_status,
                "Terminal status does not match the board/episode limits")
    require(after["done"] == (after["status"] != "running") and after["won"] == (after["status"] == "won")
            and after["alive"] == (after["status"] != "dead"), "Inconsistent terminal flags")
    require(trace["terminal_reason"] == after["terminal_reason"], "Trace terminal reason changed")
    for key in ("inference_ms", "decision_ms"):
        require(finite_number(trace[key]) and trace[key] >= 0, f"Invalid {key}")
    require(finite_number(response["latency_ms"]) and response["latency_ms"] >= 0,
            "Invalid model response latency")
    return trace


class Audit:
    def __init__(self, client, report, args):
        self.client, self.report, self.args = client, report, args
        self.created = set()
        self.backend = None

    def request(self, method, path, expected=200, **kwargs):
        started = time.perf_counter()
        response = self.client.request(method, path, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000
        self.report["http"].append({"method": method, "path": path, "status": response.status_code,
                                   "elapsed_ms": round(elapsed, 3)})
        require(response.status_code == expected,
                f"{method} {path}: expected {expected}, got {response.status_code}: {response.text[:1000]}")
        return response

    def create(self, seed):
        snapshot = self.request("POST", "/api/snake/games", 201,
                                json={"seed": seed, "size": self.args.size,
                                      "max_steps": self.args.max_steps}).json()
        self.created.add(snapshot["id"])
        require(snapshot["seed"] == seed and snapshot["size"] == self.args.size
                and snapshot["max_steps"] == self.args.max_steps, "New game ignored the requested configuration")
        require(snapshot["step"] == 0 and snapshot["score"] == 0 and snapshot["history_count"] == 0
                and snapshot["last_decision"] is None and snapshot["status"] == "running",
                "New game did not start from an empty history")
        return snapshot

    def delete(self, game_id):
        self.request("DELETE", f"/api/snake/games/{game_id}", 204)
        self.created.remove(game_id)

    def step(self, before):
        started = time.perf_counter()
        after = self.request("POST", f"/api/snake/games/{before['id']}/step",
                             json={"expected_step": before["step"]}).json()
        return after, round((time.perf_counter() - started) * 1000, 3)

    def check(self, name, operation):
        started = time.perf_counter()
        try:
            result = operation()
            self.report["checks"][name] = {"passed": True, "result": result}
        except Exception as exc:  # noqa: BLE001 -- Persist every acceptance failure for review.
            self.report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        self.report["checks"][name]["seconds"] = round(time.perf_counter() - started, 3)
        print("QEV_DEMO_CHECK", name, self.report["checks"][name]["passed"], flush=True)

    def health(self):
        result = self.request("GET", "/health").json()
        require(result["status"] == "ok", "Service is not healthy")
        self.backend = result["backend"]
        if self.args.expected_backend:
            require(self.backend == self.args.expected_backend, "Unexpected service backend")
        self.report["backend"] = self.backend
        self.report["models"] = self.request("GET", "/v1/models").json()
        return result

    def static_assets(self):
        parser = AssetParser()
        for path in ("/", "/snake", "/playground"):
            page = self.request("GET", path)
            require("text/html" in page.headers.get("content-type", ""), f"{path} is not HTML")
            require("Qev" in page.text, f"{path} does not identify the Qev UI")
            parser.feed(page.text)
        pending = parser.paths
        assets = {}
        while pending:
            path = pending.pop()
            if path in assets:
                continue
            response = self.request("GET", path)
            require(bool(response.content), f"Empty asset: {path}")
            kind = response.headers.get("content-type", "")
            require("text/html" not in kind, f"Static asset returned an HTML fallback: {path}")
            assets[path] = {"bytes": len(response.content), "content_type": kind,
                            "sha256": hashlib.sha256(response.content).hexdigest()}
            if path.endswith(".js"):
                require("javascript" in kind, f"Incorrect JavaScript MIME type: {path}")
                for imported in re.findall(r"(?:from\s*|import\s*)['\"]([^'\"]+)['\"]", response.text):
                    if imported.startswith("."):
                        resolved = urljoin(path, imported)
                        require(resolved.startswith("/assets/"), f"Unexpected module path: {resolved}")
                        pending.add(resolved)
        require({"/assets/app.js", "/assets/snake.js", "/assets/playground.js", "/assets/style.css"}
                <= set(assets), "The HTML/module graph does not include all demo assets")
        return assets

    def lifecycle(self):
        first, isolated = self.create(7), self.create(11)
        first_id, isolated_id = first["id"], isolated["id"]
        require(first_id != isolated_id, "Independent sessions share an ID")
        require(self.request("GET", f"/api/snake/games/{first_id}").json() == first,
                "GET changed a newly created game")
        advanced, roundtrip = self.step(first)
        # Keep the complete real trace even if a model output fails validation.
        self.report["lifecycle_step"] = {"before": first, "after": advanced, "http_ms": roundtrip}
        audit_step(first, advanced, backend=self.backend)
        self.request("POST", f"/api/snake/games/{first_id}/step", 409, json={"expected_step": 0})
        require(self.request("GET", f"/api/snake/games/{first_id}").json() == advanced,
                "Rejected duplicate step changed the game")
        history = self.request("GET", f"/api/snake/games/{first_id}?history=true").json()
        require(history.pop("history") == [advanced["last_decision"]] and history == advanced,
                "History differs from the original step response")
        require(self.request("GET", f"/api/snake/games/{isolated_id}").json() == isolated,
                "Stepping one session changed another")
        self.delete(first_id)
        for method, path, payload in (("GET", f"/api/snake/games/{first_id}", {}),
                                      ("POST", f"/api/snake/games/{first_id}/step", {"expected_step": 1}),
                                      ("DELETE", f"/api/snake/games/{first_id}", {})):
            self.request(method, path, 404, **({"json": payload} if method == "POST" else {}))
        reset = self.create(7)
        require(reset["id"] != first_id, "Reset reused the deleted session ID")
        require({key: value for key, value in reset.items() if key != "id"}
                == {key: value for key, value in first.items() if key != "id"},
                "Reset with the same seed/config did not reproduce the initial state")
        require(self.request("GET", f"/api/snake/games/{isolated_id}").json() == isolated,
                "Resetting another game affected an independent session")
        self.delete(reset["id"])
        self.delete(isolated_id)
        return {"create_get_step_history": True, "duplicate_step_status": 409,
                "deleted_get_step_delete_status": 404, "reset_reproduces_seed": True,
                "session_isolation": True}

    def episode(self, seed):
        result = {"seed": seed, "passed": False, "steps": []}
        self.report["episodes"].append(result)
        current = self.create(seed)
        game_id = current["id"]
        result["initial"] = current
        try:
            while not current["done"]:
                require(len(result["steps"]) < self.args.max_steps, "Episode exceeded its declared step limit")
                before = current
                current, elapsed = self.step(before)
                result["steps"].append({"snapshot": current, "http_ms": elapsed})
                audit_step(before, current, backend=self.backend)
            require(current["terminal_reason"] in {"wall", "body", "board_filled", "starvation", "max_steps"},
                    f"Unacceptable termination: {current['terminal_reason']}")
            history = self.request("GET", f"/api/snake/games/{current['id']}?history=true").json()
            traces = [entry["snapshot"]["last_decision"] for entry in result["steps"]]
            require(history.pop("history") == traces and history == current,
                    "Exported history differs from the complete live trace")
            ended, _ = self.step(current)
            require(ended == current, "Stepping a terminal game caused another model decision or transition")
            result["passed"] = True
        finally:
            final = current if isinstance(current, dict) else {}
            result["outcome"] = {key: final.get(key) for key in
                                 ("score", "step", "length", "status", "terminal_reason", "won")}
            traces = list(step_traces(result["steps"]))
            result["latency_ms"] = {
                "inference": distribution([trace["inference_ms"] for trace in traces
                                            if finite_number(trace.get("inference_ms"))]),
                "http_roundtrip": distribution([entry["http_ms"] for entry in result["steps"]]),
            }
            self.delete(game_id)
        return {"seed": seed, "outcome": result["outcome"], "latency_ms": result["latency_ms"]}

    def cleanup(self):
        removed, failures = [], []
        for game_id in tuple(self.created):
            try:
                self.delete(game_id)
                removed.append(game_id)
            except Exception as exc:  # noqa: BLE001 -- Continue cleaning our other sessions.
                failures.append(f"{game_id}: {exc}")
        require(not failures, f"Could not clean up validation sessions: {failures}")
        return {"removed_remaining_sessions": removed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Existing HTTP service; no server is started")
    parser.add_argument("--expected-backend", choices=("mlx", "torch"))
    parser.add_argument("--seeds", default="7,11,42")
    parser.add_argument("--size", type=int, default=12, choices=range(6, 21))
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("docs/results/demo_validation.json"))
    args = parser.parse_args()
    parsed = urlparse(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        parser.error("--base-url must be an HTTP(S) origin without a path/query/fragment")
    try:
        seeds = [int(item.strip()) for item in args.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must contain comma-separated integers")
    if not seeds or len(set(seeds)) != len(seeds) or any(not -(2**31) <= seed < 2**31 for seed in seeds):
        parser.error("--seeds must be distinct signed 32-bit integers")
    if not 1 <= args.max_steps <= 2000 or not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--max-steps must be 1..2000 and --request-timeout positive and finite")
    report = {"format": "qev-demo-validation-v1", "started_at": datetime.now(UTC).isoformat(),
              "base_url": args.base_url.rstrip("/"), "server_started_by_script": False,
              "model_loaded_by_script": False, "seeds": seeds, "size": args.size,
              "max_steps": args.max_steps, "checks": {}, "episodes": [], "http": [],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "acceptance": "HTTP/assets, isolated lifecycle, intact model probabilities, exact argmax execution, "
                            "untruncated observations, and complete traces; game survival is not a pass condition."}
    with httpx.Client(base_url=report["base_url"], timeout=args.request_timeout, trust_env=False) as client:
        audit = Audit(client, report, args)
        try:
            audit.check("health_and_model_identity", audit.health)
            audit.check("pages_and_static_module_graph", audit.static_assets)
            audit.check("session_lifecycle_and_isolation", audit.lifecycle)
            for seed in seeds:
                audit.check(f"real_game_seed_{seed}", lambda seed=seed: audit.episode(seed))
        finally:
            audit.check("validation_session_cleanup", audit.cleanup)
    report["passed"] = all(check["passed"] for check in report["checks"].values())
    report["finished_at"] = datetime.now(UTC).isoformat()
    traces = [trace for episode in report["episodes"] for trace in step_traces(episode["steps"])]
    report["inference_latency_ms"] = distribution([trace["inference_ms"] for trace in traces
                                                   if finite_number(trace.get("inference_ms"))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"passed": report["passed"], "report": str(args.output),
                      "games": [episode.get("outcome") for episode in report["episodes"]]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
