"""Benchmark behavior and auditability without downloaded model weights."""

import copy
import io
import json
import threading
from types import SimpleNamespace

import pytest

from qev.snake import SnakeGame, decision_request
from scripts import benchmark_snake as benchmark


def settings(**changes):
    values = {"seeds": [7, 11], "size": 6, "max_steps": 20,
              "observation": "spatial", "batch": 1, "warmup": 0}
    return SimpleNamespace(**(values | changes))


class RecordingPredictor:
    """A declared test policy that intentionally keeps driving into a wall."""

    def __init__(self, direction="RIGHT", model_name="fixture"):
        self.direction, self.model_name = direction, model_name
        self.source = {"kind": "test_fixture", "model_name": model_name}
        self.calls, self.responses = [], []

    def predict(self, requests):
        self.calls.append(copy.deepcopy(requests))
        responses = []
        for request in requests:
            criteria = request["questions"]["move"]["criteria"]
            assert self.direction in criteria
            probabilities = {key: .625 if key == self.direction else .375 / (len(criteria) - 1)
                             for key in criteria}
            responses.append({"model": self.model_name, "answers": {"move": {
                "type": "choice", "choice": self.direction, "probabilities": probabilities}},
                "usage": {"input_tokens": 123, "output_tokens": 0}})
        self.responses.append(copy.deepcopy(responses))
        return responses


def run(args, predictor):
    trace = io.StringIO()
    report = benchmark.run_benchmark(args, predictor, trace)
    return report, [json.loads(line) for line in trace.getvalue().splitlines()]


def test_collision_argmax_is_executed_with_original_probabilities_and_no_rescue():
    predictor = RecordingPredictor()
    report, events = run(settings(), predictor)
    steps = [event for event in events if event["type"] == "step"]
    assert report["complete"]  # A model losing is a completed measurement, not a failed run.
    assert report["aggregate"]["collisions"] == 2
    assert report["aggregate"]["steps"] == report["aggregate"]["argmax_direct_steps"] == 6
    assert len(predictor.calls) == 6  # Three rightward attempts from x=3 on a six-cell board.
    for step, responses in zip(steps, predictor.responses):
        assert step["response"] == responses[0]
        assert step["probabilities"] == responses[0]["answers"]["move"]["probabilities"]
        assert step["proposed"] == step["executed"] == "RIGHT"
        assert not step["intervened"]
    fatal_steps = [step for step in steps if step["after"]["terminal_reason"] == "wall"]
    assert len(fatal_steps) == 2
    for step in fatal_steps:
        criteria = step["request"]["questions"]["move"]["criteria"]
        assert criteria["RIGHT"]["collision"] == "wall"
        assert any(fact["collision"] == "none" for fact in criteria.values())
        assert step["before"]["body"][0] == [5, 3]
        assert step["after"]["status"] == "dead"
    assert report["aggregate"]["collision_with_safe_alternative"] == 2


@pytest.mark.parametrize("failure", [
    "exception", "missing_response", "missing_candidate", "negative_probability",
    "wrong_sum", "nan", "choice_not_argmax", "truncation",
])
def test_invalid_inference_never_advances_or_retries_and_cannot_pass(failure):
    class InvalidPredictor(RecordingPredictor):
        def predict(self, requests):
            responses = super().predict(requests)
            response = responses[0]
            answer = response["answers"]["move"]
            if failure == "exception":
                raise RuntimeError("test inference failure")
            if failure == "missing_response":
                return []
            if failure == "missing_candidate":
                answer["probabilities"].pop("UP")
            elif failure == "negative_probability":
                answer["probabilities"] = {"UP": -.1, "DOWN": .4, "RIGHT": .7}
            elif failure == "wrong_sum":
                answer["probabilities"]["RIGHT"] = .5
            elif failure == "nan":
                answer["probabilities"]["RIGHT"] = float("nan")
            elif failure == "choice_not_argmax":
                answer["choice"] = "UP"
            elif failure == "truncation":
                response["qev"] = {"truncated_questions": ["move"]}
            return responses

    predictor = InvalidPredictor()
    report, events = run(settings(seeds=[7]), predictor)
    steps = [event for event in events if event["type"] == "step"]
    assert len(predictor.calls) == len(steps) == 1
    assert not report["complete"] and report["errors"]
    assert report["aggregate"]["steps"] == report["aggregate"]["argmax_direct_steps"] == 0
    assert report["episodes"][0]["terminal_reason"] == "model_error"
    step = steps[0]
    assert step["executed"] is None and not step["intervened"]
    assert step["before"]["body"] == step["after"]["body"]
    assert step["before"]["food"] == step["after"]["food"]
    assert step["step_before"] == step["step_after"] == 0
    assert step["error"] and events[-1]["complete"] is False


@pytest.mark.parametrize("section,key,value", [
    ("protocol", "seeds", [11, 7]),
    ("protocol", "size", 8),
    ("protocol", "max_steps", 500),
    ("protocol", "starvation_limit", 999),
    ("protocol", "observation", "local"),
    ("protocol", "prompt_source_sha256", "different-prompt"),
    ("execution", "batch_size", 8),
])
def test_paired_comparison_rejects_different_experimental_conditions(section, key, value):
    report, _ = run(settings(max_steps=1), RecordingPredictor())
    reference = copy.deepcopy(report)
    reference[section][key] = value
    with pytest.raises(ValueError, match="identical|equal"):
        benchmark.paired_comparison(report, reference)


def test_paired_comparison_refuses_incomplete_runs():
    report, _ = run(settings(max_steps=1), RecordingPredictor())
    incomplete = copy.deepcopy(report)
    incomplete["complete"] = False
    for current, reference in ((incomplete, report), (report, incomplete)):
        with pytest.raises(ValueError, match="complete"):
            benchmark.paired_comparison(current, reference)


@pytest.mark.parametrize("observation", ["local", "spatial"])
def test_same_seeds_start_identically_and_logged_input_replays_actual_model_trajectory(observation):
    args = settings(max_steps=2, warmup=2, observation=observation)
    recordings = []
    for direction, name in (("RIGHT", "parent-fixture"), ("DOWN", "candidate-fixture")):
        predictor = RecordingPredictor(direction, name)
        report, events = run(args, predictor)
        steps = [event for event in events if event["type"] == "step"]
        actual_calls = [request for batch in predictor.calls[args.warmup:] for request in batch]
        assert [step["request"] for step in steps] == actual_calls
        assert len(actual_calls) == report["aggregate"]["steps"] == 4
        games = {seed: SnakeGame(seed=seed, size=args.size, max_steps=args.max_steps,
                                 observation=observation) for seed in args.seeds}
        for step, actual_request in zip(steps, actual_calls):
            replay = games[step["seed"]]
            assert replay.snapshot() == step["before"]
            expected_request, expected_facts = decision_request(replay, name)
            assert actual_request == expected_request
            assert step["candidate_facts"] == expected_facts
            replay.advance(step["response"]["answers"]["move"]["choice"])
            assert replay.snapshot() == step["after"]
        recordings.append((report, events, steps))
    first, second = recordings
    initial = lambda events: [event["game"] for event in events if event["type"] == "episode_start"]
    assert initial(first[1]) == initial(second[1])
    assert first[2][0]["after"]["body"] != second[2][0]["after"]["body"]
    comparison = benchmark.paired_comparison(second[0], first[0])
    assert [row["seed"] for row in comparison["paired_episodes"]] == args.seeds


def test_multi_game_timing_counts_each_call_once_and_never_claims_single_step_latency(monkeypatch):
    ticks = iter(range(100))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: float(next(ticks)))
    report, events = run(settings(batch=2, max_steps=1), RecordingPredictor())
    timing = report["timing"]
    steps = [event for event in events if event["type"] == "step"]
    assert timing["prediction_calls"] == 1 and timing["decision_requests"] == 2
    assert timing["prediction_call_ms"]["mean"] == 1000
    assert timing["prediction_decisions_per_second"] == 2
    assert "single_request_wall_ms" not in timing
    assert all("request_wall_ms" not in step for step in steps)
    assert all(step["prediction_batch_size"] == 2 for step in steps)
    assert len({step["prediction_call"] for step in steps}) == 1


def test_tiny_torch_batch_preserves_agent_probabilities_and_per_game_candidate_mapping():
    import torch

    from qev.inference import Agent
    from qev.tokenization import SPECIAL

    class TinyTokenizer:
        pad_token_id, unk_token_id = 0, -1

        def __init__(self):
            self.special = {token: index + 1 for index, token in enumerate(SPECIAL.values())}

        def __call__(self, text, **kwargs):
            return {"input_ids": ([self.special[text]] if text in self.special
                                  else [10 + ord(char) % 100 for char in text])}

        def convert_tokens_to_ids(self, text):
            return self.special.get(text, -1)

    class TinyScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.batch_sizes = []

        def forward(self, input_ids, option_positions, **kwargs):
            self.batch_sizes.append(len(input_ids))
            # Each candidate score depends on its own row's full preceding text.
            # Padding and a neighboring game's text must not change that score.
            prefix = input_ids.cumsum(dim=1).gather(1, option_positions)
            return (prefix % 17).float() * self.weight

    agent = Agent.__new__(Agent)
    agent.config = {"max_length": 4096, "max_state": 2048}
    agent.backend, agent.device, agent.batch_size = "torch", "cpu", 1
    agent.temperature, agent.lock = 1.464, threading.Lock()
    agent.tokenizer, agent.model = TinyTokenizer(), TinyScorer().eval()
    predictor = benchmark.Predictor.__new__(benchmark.Predictor)
    predictor.agent, predictor.batch_size = agent, 8
    requests = [decision_request(SnakeGame(seed=seed, size=size, observation="local"), "fixture")[0]
                for seed, size in ((7, 6), (11, 8), (42, 10))]
    batched = predictor.predict(requests)
    assert agent.model.batch_sizes == [3]  # A real multi-row Torch forward, not serial calls.
    for request, response in zip(requests, batched):
        serial = agent.predict(**request)
        assert response["answers"] == serial["answers"]
        assert response["usage"] == serial["usage"]
        answer = response["answers"]["move"]
        assert set(answer["probabilities"]) == set(request["questions"]["move"]["criteria"])
        assert answer["probabilities"][answer["choice"]] == max(answer["probabilities"].values())
        assert "confidence" in answer and response["qev"]["temperature"] == 1.464
        assert "latency_ms" not in response
    assert batched[0]["answers"] != batched[1]["answers"]  # Row swapping would be observable.
