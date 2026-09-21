"""Exercise terminal control and audit trails without loading model weights."""

import argparse
import copy
import io
import json
import os
import re
import select
import sys
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from qev import cli, snake_cli
from qev.snake import SnakeGame
from qev.snake_terminal import cell_width


class StubAgent:
    backend = "stub"

    def __init__(self, *, failure=False):
        self.config = {"model_name": "test-checkpoint"}
        self.calls = []
        self.failure = failure

    def predict(self, state, questions, *, model):
        self.calls.append(copy.deepcopy({"state": state, "questions": questions, "model": model}))
        if self.failure:
            raise RuntimeError("deliberate inference failure")
        choices = questions["move"]["criteria"]
        return {"model": model, "answers": {"move": {"type": "choice", "choice": "RIGHT",
                "confidence": 1.0, "probabilities": {key: float(key == "RIGHT") for key in choices}}},
                "usage": {"input_tokens": 123, "output_tokens": 0}, "qev": {"backend": "stub"}}


def arguments(*extra):
    return snake_cli.add_arguments(argparse.ArgumentParser()).parse_args(
        ["--headless", "--fps", "0", *extra])


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("observation", ["local", "spatial"])
def test_headless_episodes_preserve_requests_actions_and_cleanup(tmp_path, observation):
    agent = StubAgent()
    driver = snake_cli.LocalDriver(agent, "stub")
    report, output = tmp_path / "run.jsonl", io.StringIO()
    code = snake_cli.run(arguments("--episodes", "2", "--max-steps", "1", "--report", str(report),
                                   "--observation", observation), driver=driver, stream=output)
    summary = json.loads(output.getvalue())
    events = records(report)
    steps = [event for event in events if event["type"] == "step"]
    assert code == 0 and len(agent.calls) == len(steps) == 2
    assert [entry["seed"] for entry in summary["episodes"]] == [7, 8]
    assert all(entry["terminal_reason"] == "max_steps" for entry in summary["episodes"])
    assert summary["total_steps"] == summary["observed_decisions"] == 2
    assert summary["policy"]["observation"] == observation
    assert summary["policy"]["feature_search"] == (observation == "spatial")
    assert not driver.store._sessions and "\033" not in output.getvalue()
    for event, request in zip(steps, agent.calls):
        game, trace = event["game"], event["game"]["last_decision"]
        assert game["policy"] == summary["policy"]
        assert trace["request"] == request
        assert trace["proposed"] == trace["executed"] == "RIGHT" and not trace["intervened"]
        assert trace["probabilities"]["RIGHT"] == 1.0
        assert trace["response"]["usage"]["output_tokens"] == 0
    assert events[-1]["type"] == "run_end" and events[-1]["exit_code"] == 0


def test_model_error_stops_without_fabricating_an_action(tmp_path):
    agent, output = StubAgent(failure=True), io.StringIO()
    driver = snake_cli.LocalDriver(agent, "stub")
    report = tmp_path / "error.jsonl"
    code = snake_cli.run(arguments("--report", str(report)), driver=driver, stream=output)
    summary = json.loads(output.getvalue())
    assert code == 1 and len(agent.calls) == 1
    assert summary["total_steps"] == 0 and summary["observed_decisions"] == 1
    assert summary["episodes"][0]["terminal_reason"] == "model_error"
    step = next(event for event in records(report) if event["type"] == "step")
    assert step["game"]["last_decision"]["executed"] is None
    assert not driver.store._sessions


class Keys:
    def __init__(self, values):
        self.values, self.entered, self.exited = iter(values), False, False

    def __enter__(self):
        self.entered = True
        return self

    def read(self):
        return next(self.values)

    def __exit__(self, *_):
        self.exited = True


def test_pause_single_step_and_quit_make_only_one_model_call():
    keys, agent, output = Keys([" ", "n", "q"]), StubAgent(), io.StringIO()
    driver = snake_cli.LocalDriver(agent, "stub")
    code = snake_cli.run(arguments(), driver=driver, stream=output, keyboard=keys, sleep=lambda _: None)
    assert code == 0 and len(agent.calls) == 1
    assert keys.entered and keys.exited
    assert json.loads(output.getvalue())["episodes"][0]["stop_reason"] == "user_quit"
    assert not driver.store._sessions


def test_quit_remains_responsive_at_extremely_low_fps():
    keys, agent, output = Keys(["", "", "q"]), StubAgent(), io.StringIO()
    driver, now, waits = snake_cli.LocalDriver(agent, "stub"), [0.0], []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    code = snake_cli.run(arguments("--fps", "0.001"), driver=driver, stream=output,
                         keyboard=keys, clock=lambda: now[0], sleep=sleep)
    assert code == 0 and len(agent.calls) == 1
    assert waits == [0.05] and now[0] == 0.05


@pytest.mark.parametrize("error,expected_code", [(KeyboardInterrupt, 130), (RuntimeError, 1)])
def test_interrupted_or_lost_step_resyncs_once_without_retry(tmp_path, error, expected_code):
    class InterruptedDriver(snake_cli.LocalDriver):
        def step(self, *args):
            super().step(*args)
            raise error("response lost after action")

        def get(self, game_id, *, timeout=None):
            assert timeout == 3
            return super().get(game_id)

    agent, output, keys = StubAgent(), io.StringIO(), Keys([""])
    driver, report = InterruptedDriver(agent, "stub"), tmp_path / "interrupted.jsonl"
    code = snake_cli.run(arguments("--report", str(report)), driver=driver, stream=output, keyboard=keys)
    summary = json.loads(output.getvalue())
    assert code == expected_code and len(agent.calls) == 1
    assert summary["total_steps"] == 1 and not driver.store._sessions
    assert summary["interrupted"] == (error is KeyboardInterrupt)
    assert keys.exited
    assert sum(event["type"] == "resync" for event in records(report)) == 1
    assert records(report)[-1]["exit_code"] == expected_code


class Response(io.BytesIO):
    def __init__(self, payload=None, status=200):
        super().__init__(json.dumps(payload).encode())
        self.status = status


def test_http_driver_preserves_settings_expected_step_and_methods():
    calls = []

    def opener(request, *, timeout):
        calls.append((request.get_method(), request.full_url,
                      json.loads(request.data) if request.data else None, timeout))
        return Response({"id": "owned"}) if request.get_method() != "DELETE" else Response(status=204)

    driver = snake_cli.HTTPDriver("http://127.0.0.1:8008/", timeout=17, opener=opener)
    assert driver.create(seed=7, size=12, max_steps=500, observation="spatial") == {"id": "owned"}
    driver.step("owned", 4)
    driver.get("owned", timeout=3)
    assert driver.delete("owned", timeout=3) is None
    root = "http://127.0.0.1:8008/api/snake/games"
    assert calls == [("POST", root, {"seed": 7, "size": 12, "max_steps": 500, "observation": "spatial"}, 17),
                     ("POST", root + "/owned/step", {"expected_step": 4}, 17),
                     ("GET", root + "/owned", None, 3), ("DELETE", root + "/owned", None, 3)]


def test_http_driver_never_retries_a_failed_post():
    calls = []

    def opener(request, **_):
        calls.append(request)
        raise URLError("connection lost")

    with pytest.raises(RuntimeError, match="not retried"):
        snake_cli.HTTPDriver("http://localhost:8008", opener=opener).step("owned", 2)
    assert len(calls) == 1


@pytest.mark.parametrize("observation,disclosure", [
    ("spatial", "静态BFS环境特征·无动作接管"), ("local", "碰撞与食物距离环境特征·无动作接管")])
def test_renderer_shows_real_snapshot_probabilities_and_feature_disclosure(observation, disclosure):
    driver = snake_cli.LocalDriver(StubAgent(), "stub")
    game = driver.create(observation=observation)
    game = driver.step(game["id"], 0)
    plain = snake_cli.render_frame(game, color=False)
    assert "\033" not in plain and "100.0%" in plain and "MOVE RIGHT" in plain
    assert "INPUT 123 tokens / OUTPUT 0" in plain and disclosure in plain
    assert "\033[" in snake_cli.render_frame(game, color=True)


@pytest.mark.parametrize("columns,rows,size", [
    (120, 44, 12), (80, 24, 12), (60, 24, 20), (80, 24, 20),
    (40, 24, 12), (20, 24, 12), (12, 10, 20),
])
def test_renderer_fits_terminal_cells_without_mutating_snapshot(columns, rows, size):
    game = SnakeGame(size=size).snapshot()
    original = copy.deepcopy(game)
    plain = snake_cli.render_frame(game, columns=columns, rows=rows)
    colored = snake_cli.render_frame(game, columns=columns, rows=rows, color=True)
    stripped = re.sub(r"\x1b\[[0-9;]*m", "", colored)
    assert stripped == plain
    assert len(plain.splitlines()) <= rows
    assert all(cell_width(line) == columns - 1 for line in plain.splitlines())
    assert game == original
    if columns >= size + 5 and rows >= size + 4:
        assert "◆" in plain  # Food remains visible even in the narrow-board layout.
        # The inner board must retain both complete horizontal boundaries;
        # merely seeing its food would not detect a cropped bottom row.
        horizontal_borders = [line for line in plain.splitlines() if "╭" in line or "╰" in line]
        assert len(horizontal_borders) == 4


def test_renderer_reports_actual_usage_and_sanitizes_server_error_text():
    game = SnakeGame().snapshot()
    game.update(status="finished", terminal_reason="model_error", last_decision={
        "response": {"usage": {"input_tokens": 321, "output_tokens": 7}},
        "error": {"message": "失败\x1b]0;title\x07\x1b[2J e\u0301\u202e\nnext\tline"},
    })
    plain = snake_cli.render_frame(game, columns=80, rows=24)
    assert "MODEL ERROR" in plain and "LIVE · MODEL CONTROL" not in plain
    assert "INPUT 321 tokens / OUTPUT 7" in plain
    assert "失败 e\u0301 next line" in plain
    assert all(char not in plain for char in ("\033", "\x07", "\t", "\u202e"))
    assert all(cell_width(line) == 79 for line in plain.splitlines())


@pytest.mark.parametrize("columns,rows", [(40, 24), (20, 12)])
def test_narrow_terminal_retains_quit_hint(columns, rows):
    frame = snake_cli.render_frame(SnakeGame().snapshot(), columns=columns, rows=rows)
    assert "Q" in frame.splitlines()[-2] or "Q" in frame.splitlines()[-1]


def test_full_height_ansi_frame_overwrites_without_scrolling(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(snake_cli.shutil, "get_terminal_size", lambda: os.terminal_size((80, 24)))
    stream = io.StringIO()
    display = snake_cli.Terminal(stream, enabled=True, color=False, alternate=True)
    display.draw(SnakeGame().snapshot())
    output = stream.getvalue()
    assert output.startswith("\033[H") and "\033[H\033[J" not in output
    assert output.endswith("\033[K\033[J")
    assert output.count("\r\n") == 23


def test_terminal_restores_cursor_and_alternate_screen_after_exception(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = io.StringIO()
    with pytest.raises(RuntimeError), snake_cli.Terminal(stream, enabled=True, color=False, alternate=True):
        raise RuntimeError("drawing interrupted")
    assert stream.getvalue().startswith("\033[?1049h\033[?25l")
    assert stream.getvalue().endswith("\033[0m\033[?25h\033[?1049l")


@pytest.mark.skipif(os.name != "posix", reason="POSIX cbreak terminal only")
def test_keyboard_reads_without_enter_and_restores_terminal_after_exception(monkeypatch):
    import termios

    master, slave = os.openpty()
    try:
        with os.fdopen(os.dup(slave)) as input_stream:
            monkeypatch.setattr(sys, "stdin", input_stream)
            before = termios.tcgetattr(slave)
            with pytest.raises(RuntimeError), snake_cli.Keyboard() as keys:
                assert not termios.tcgetattr(slave)[3] & termios.ICANON
                os.write(master, b"Q")
                assert select.select([slave], [], [], 1)[0]
                assert keys.read() == "q"
                raise RuntimeError("interrupted with terminal open")
            after = termios.tcgetattr(slave)
            # macOS may set PENDIN when returning to canonical input; that is
            # kernel bookkeeping, not a change to the user's terminal mode.
            after[3] &= ~getattr(termios, "PENDIN", 0)
            before[3] &= ~getattr(termios, "PENDIN", 0)
            assert after == before
    finally:
        os.close(master)
        os.close(slave)


def test_existing_report_is_preserved_before_model_loading(tmp_path, monkeypatch):
    report = tmp_path / "existing.jsonl"
    report.write_text("preserve me")
    monkeypatch.setattr(snake_cli, "make_driver", lambda _: pytest.fail("model loaded before report check"))
    with pytest.raises(FileExistsError):
        snake_cli.run(arguments("--report", str(report)), stream=io.StringIO())
    assert report.read_text() == "preserve me"


@pytest.mark.parametrize("extra", [
    ["--size", "5"], ["--max-steps", "0"], ["--episodes", "101"], ["--fps", "nan"],
    ["--fps", "241"], ["--seed", str(2**31 - 1), "--episodes", "2"],
    ["--request-timeout", "0"], ["--base-url", "file:///tmp/service"],
    ["--base-url", "http://localhost", "--backend", "mlx"],
])
def test_invalid_settings_are_rejected_before_model_loading(extra, monkeypatch):
    monkeypatch.setattr(snake_cli, "make_driver", lambda _: pytest.fail("invalid run loaded a model"))
    with pytest.raises(ValueError):
        snake_cli.run(arguments(*extra), stream=io.StringIO())


def test_default_checkpoint_preference_and_explicit_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded = []

    def agent(model, **_):
        loaded.append(model)
        return StubAgent()

    monkeypatch.setitem(sys.modules, "qev.inference", SimpleNamespace(Agent=agent))
    legacy = tmp_path / "models/qev-0.8b-mlx"
    legacy.mkdir(parents=True)
    (legacy / "qev_config.json").write_text("{}")
    snake_cli.make_driver(arguments())
    preferred = tmp_path / "models/qev-snake-0.8b-mlx"
    preferred.mkdir()
    snake_cli.make_driver(arguments())  # An incomplete download must not hide a usable checkpoint.
    (preferred / "qev_config.json").write_text("{}")
    snake_cli.make_driver(arguments())
    explicit = tmp_path / "models/original-comparison"
    explicit.mkdir()
    (explicit / "qev_config.json").write_text("{}")
    snake_cli.make_driver(arguments("--model", "models/original-comparison"))
    assert loaded == ["models/qev-0.8b-mlx", "models/qev-0.8b-mlx",
                      "models/qev-snake-0.8b-mlx", "models/original-comparison"]


def test_missing_default_checkpoint_explains_download_before_importing_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "qev.inference", None)
    with pytest.raises(FileNotFoundError) as failure:
        snake_cli.make_driver(arguments())
    message = str(failure.value)
    assert "models/qev-snake-0.8b-mlx/qev_config.json" in message
    assert "uv run hf download twainsk/qev-0.8b-mlx --local-dir models/qev-snake-0.8b-mlx" in message
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("backend,repository", [("auto", "twainsk/qev-0.8b-mlx"),
                                                ("torch", "twainsk/qev-0.8b")])
def test_missing_explicit_checkpoint_does_not_use_available_default(tmp_path, monkeypatch, backend, repository):
    monkeypatch.chdir(tmp_path)
    preferred = tmp_path / "models/qev-snake-0.8b-mlx"
    preferred.mkdir(parents=True)
    (preferred / "qev_config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "qev.inference", None)
    with pytest.raises(FileNotFoundError) as failure:
        snake_cli.make_driver(arguments("--model", "models/chosen checkpoint", "--backend", backend))
    message = str(failure.value)
    assert "models/chosen checkpoint/qev_config.json" in message
    assert f"uv run hf download {repository} --local-dir 'models/chosen checkpoint'" in message
    assert not (tmp_path / "models/chosen checkpoint").exists()


def test_http_and_cli_dispatch_do_not_construct_an_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "qev.inference", None)
    assert isinstance(snake_cli.make_driver(arguments("--base-url", "http://localhost:8008")), snake_cli.HTTPDriver)
    monkeypatch.setattr(snake_cli, "run", lambda args: 17 if args.observation == "local" else 0)
    assert cli.main(["snake", "--observation", "local"]) == 17
    with pytest.raises(SystemExit) as exc:
        cli.main(["snake", "--help"])
    assert exc.value.code == 0
