"""Terminal Snake with original Qev decisions, locally or through its HTTP API."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import shlex
import shutil
import statistics
import sys
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .snake import POLICY, GameStore
from .snake_features import FEATURE_VERSION
from .snake_terminal import render_frame


def add_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="Local checkpoint; prefer models/qev-snake-0.8b-mlx when present")
    source.add_argument("--base-url", help="Use an existing Qev service, e.g. http://127.0.0.1:8008")
    parser.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto")
    parser.add_argument("--device", help="Torch device for local inference")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=12, help="Square board size, 6..20")
    parser.add_argument("--max-steps", type=int, default=500, help="Per-game limit, 1..2000")
    parser.add_argument("--observation", choices=["spatial", "local"], default="spatial",
                        help="Spatial: static BFS environment facts; local: original collision/distance facts")
    parser.add_argument("--fps", type=float, default=8, help="Maximum decisions per second; 0 runs unpaced")
    parser.add_argument("--episodes", type=int, default=1, help="Number of games, 1..100; each increments the seed")
    parser.add_argument("--headless", action="store_true", help="No board; print a JSON run summary")
    parser.add_argument("--report", type=Path, help="New JSONL file with every real request, response and board")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors (also honors NO_COLOR)")
    parser.add_argument("--no-alt-screen", action="store_true", help="Keep the final board in terminal scrollback")
    parser.add_argument("--request-timeout", type=float, default=120, help="HTTP timeout in seconds; steps are never retried")
    return parser


def validate_arguments(args):
    if not 6 <= args.size <= 20:
        raise ValueError("--size must be in 6..20")
    if not 1 <= args.max_steps <= 2000:
        raise ValueError("--max-steps must be in 1..2000")
    if not 1 <= args.episodes <= 100:
        raise ValueError("--episodes must be in 1..100")
    if not -(2**31) <= args.seed or args.seed + args.episodes - 1 >= 2**31:
        raise ValueError("All episode seeds must fit signed 32-bit integers")
    if not math.isfinite(args.fps) or not 0 <= args.fps <= 240:
        raise ValueError("--fps must be finite and in 0..240")
    if not math.isfinite(args.request_timeout) or not 0 < args.request_timeout <= 600:
        raise ValueError("--request-timeout must be finite and in (0,600]")
    if args.base_url:
        parsed = urlparse(args.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("--base-url must be an HTTP(S) service URL without a query or fragment")
        if args.backend != "auto" or args.device:
            raise ValueError("--backend and --device select local inference; omit them with --base-url")


class LocalDriver:
    def __init__(self, agent, model):
        self.store = GameStore(agent, max_games=1, ttl_seconds=86400)
        self.metadata = {"mode": "local", "model": str(model), "backend": agent.backend}

    def create(self, **settings):
        return self.store.create(**settings)

    def step(self, game_id, expected_step):
        return self.store.step(game_id, expected_step)

    def get(self, game_id, *, timeout=None):
        return self.store.get(game_id)

    def delete(self, game_id, *, timeout=None):
        return self.store.delete(game_id)


class HTTPDriver:
    """No implicit POST retries: a lost response might already have moved the snake."""

    def __init__(self, base_url, timeout=120, opener=urlopen):
        self.base_url, self.timeout, self.opener = base_url.rstrip("/"), timeout, opener
        self.metadata = {"mode": "http", "base_url": self.base_url}

    def _request(self, method, path, body=None, *, timeout=None):
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.base_url + "/api/snake/games" + path, data=encoded,
                          headers={"Content-Type": "application/json", "Accept": "application/json"}, method=method)
        try:
            with self.opener(request, timeout=self.timeout if timeout is None else timeout) as response:
                return None if response.status == 204 else json.load(response)
        except HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"Qev HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Qev HTTP request failed; the step was not retried: {exc}") from exc

    def create(self, **settings):
        return self._request("POST", "", settings)

    def step(self, game_id, expected_step):
        return self._request("POST", f"/{game_id}/step", {"expected_step": expected_step})

    def get(self, game_id, *, timeout=None):
        return self._request("GET", f"/{game_id}", timeout=timeout)

    def delete(self, game_id, *, timeout=None):
        return self._request("DELETE", f"/{game_id}", timeout=timeout)


def make_driver(args):
    if args.base_url:
        return HTTPDriver(args.base_url, args.request_timeout)

    preferred, legacy = "models/qev-snake-0.8b-mlx", "models/qev-0.8b-mlx"
    model = args.model or next((path for path in (preferred, legacy)
                               if (Path(path) / "qev_config.json").is_file()), preferred)
    config_path = Path(model).expanduser() / "qev_config.json"
    if not config_path.is_file():
        repository = "twainsk/qev-0.8b" if args.backend == "torch" else "twainsk/qev-0.8b-mlx"
        raise FileNotFoundError(
            f"Qev checkpoint is missing: {config_path}\n"
            f"Download a checkpoint to {model}:\n"
            f"  uv run hf download {repository} --local-dir {shlex.quote(str(model))}\n"
            "Or use --model with an existing Qev checkpoint directory."
        )
    from .inference import Agent

    print(f"Loading local Qev checkpoint: {model}", file=sys.stderr, flush=True)
    return LocalDriver(Agent(model, backend=args.backend, device=args.device), model)


class Keyboard(AbstractContextManager):
    def __init__(self, enabled=True):
        self.enabled, self.saved, self.fd = enabled, None, None

    def __enter__(self):
        if self.enabled and sys.stdin.isatty() and os.name == "posix":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self):
        if self.saved is not None and select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 128).decode(errors="ignore").lower()
        return ""

    def __exit__(self, *_):
        if self.saved is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


class Terminal(AbstractContextManager):
    def __init__(self, stream, *, enabled, color, alternate):
        self.stream, self.enabled, self.color, self.alternate = stream, enabled, color, alternate
        self.ansi = enabled and os.environ.get("TERM") != "dumb"

    def __enter__(self):
        if self.ansi:
            self.stream.write(("\033[?1049h" if self.alternate else "") + "\033[?25l")
            self.stream.flush()
        return self

    def draw(self, game, **kwargs):
        if self.enabled:
            size = shutil.get_terminal_size()
            frame = render_frame(game, color=self.color, columns=size.columns, rows=size.lines, **kwargs)
            if self.ansi:
                # Overwrite in place without a final newline: a frame that fills
                # the viewport must not scroll its own header off the screen.
                self.stream.write("\033[H" + frame.replace("\n", "\033[K\r\n") + "\033[K\033[J")
            else:
                self.stream.write(frame + "\n")
            self.stream.flush()

    def __exit__(self, *_):
        if self.ansi:
            self.stream.write("\033[0m\033[?25h" + ("\033[?1049l" if self.alternate else ""))
            if not self.alternate:
                self.stream.write("\n")
            self.stream.flush()


class Report(AbstractContextManager):
    """Flush each JSONL event, retaining finished decisions after an interruption."""

    def __init__(self, path):
        self.path, self.file = path, None

    def __enter__(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("x", encoding="utf-8")
        return self

    def write(self, kind, **value):
        if self.file:
            self.file.write(json.dumps({"type": kind, **value}, ensure_ascii=False, allow_nan=False) + "\n")
            self.file.flush()

    def __exit__(self, *_):
        if self.file:
            self.file.close()


def run(args, *, driver=None, stream=None, keyboard=None, clock=time.perf_counter, sleep=time.sleep):
    validate_arguments(args)
    stream = stream or sys.stdout
    headless = args.headless or not stream.isatty()
    color = not headless and not args.no_color and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    policy = {**POLICY, "observation": args.observation,
              "feature_version": FEATURE_VERSION if args.observation == "spatial" else "local-v1",
              "feature_search": args.observation == "spatial"}
    summary = {"format": "qev-snake-run-v1", "policy": policy, "episodes": [],
               "interrupted": False, "errors": [], "total_steps": 0, "observed_decisions": 0}
    exit_code, paused, quit_requested = 0, False, False
    fps, latencies = args.fps, []
    with Report(args.report) as report:
        report.write("metadata", format="qev-snake-terminal-v1", created_utc=datetime.now(UTC).isoformat(),
                     settings=settings, policy=policy,
                     scope=("静态BFS环境特征·无动作接管" if args.observation == "spatial"
                            else "碰撞与食物距离环境特征·无动作接管"))
        load_started = clock()
        driver = driver if driver is not None else make_driver(args)
        summary["source"] = driver.metadata
        summary["load_seconds"] = clock() - load_started
        started = clock()
        report.write("source", **driver.metadata, load_seconds=summary["load_seconds"])
        with Terminal(stream, enabled=not headless, color=color, alternate=not args.no_alt_screen) as display, (keyboard or Keyboard(not headless)) as keys:
            for episode in range(1, args.episodes + 1):
                game, game_started = None, clock()
                episode_error, reason, last_step_started, state_confirmed = None, None, None, True
                try:
                    game = driver.create(seed=args.seed + episode - 1, size=args.size,
                                         max_steps=args.max_steps, observation=args.observation)
                    summary["policy"] = game.get("policy", policy)
                    report.write("episode_start", episode=episode, game=game, elapsed_seconds=clock() - started)
                    display.draw(game, episode=episode, episodes=args.episodes, paused=paused, fps=fps)
                    while game["status"] == "running" and not quit_requested:
                        pressed = keys.read()
                        if "\x03" in pressed:
                            summary["interrupted"], quit_requested, reason, exit_code = True, True, "keyboard_interrupt", 130
                            break
                        if "q" in pressed:
                            quit_requested, reason = True, "user_quit"
                            break
                        if " " in pressed:
                            paused = not paused
                        if "+" in pressed or "=" in pressed:
                            fps = min(240, max(1, fps) + 1)
                        if "-" in pressed:
                            fps = max(1, fps - 1)
                        if paused and "n" not in pressed:
                            if pressed:
                                display.draw(game, episode=episode, episodes=args.episodes, paused=True, fps=fps)
                            sleep(0.05)
                            continue
                        # Poll controls while pacing, even at a very low FPS. A
                        # paused N is immediate and still makes exactly one call.
                        if fps and last_step_started is not None and not (paused and "n" in pressed):
                            remaining = 1 / fps - (clock() - last_step_started)
                            if remaining > 0:
                                if pressed:
                                    display.draw(game, episode=episode, episodes=args.episodes, paused=paused, fps=fps)
                                sleep(min(remaining, 0.05))
                                continue
                        previous_step = game["step"]
                        last_step_started = clock()
                        game = driver.step(game["id"], previous_step)
                        if game["status"] == "running" and game["step"] != previous_step + 1:
                            raise RuntimeError("Server did not advance exactly one step; stopping without retry")
                        decision = game.get("last_decision") or {}
                        if decision.get("inference_ms") is not None:
                            latencies.append(decision["inference_ms"])
                        report.write("step", episode=episode, game=game, elapsed_seconds=clock() - started)
                        display.draw(game, episode=episode, episodes=args.episodes, paused=paused, fps=fps)
                        if game["terminal_reason"] == "model_error":
                            episode_error = decision.get("error", {}).get("message", "Model decision failed")
                            exit_code, quit_requested = 1, True
                except KeyboardInterrupt:
                    summary["interrupted"], quit_requested, reason, exit_code = True, True, "keyboard_interrupt", 130
                except Exception as exc:  # noqa: BLE001 -- Stop and record; never retry an uncertain model step.
                    episode_error, quit_requested, exit_code = f"{type(exc).__name__}: {exc}", True, 1
                finally:
                    # A step may have completed before its response was lost or
                    # interrupted. Fetch once; never retry a state-changing POST.
                    if game is not None and (episode_error or summary["interrupted"]):
                        try:
                            game = driver.get(game["id"], timeout=3)
                            report.write("resync", episode=episode, game=game, elapsed_seconds=clock() - started)
                        except Exception as sync_error:  # noqa: BLE001 -- Preserve both the request and recovery failures.
                            state_confirmed = False
                            summary["errors"].append(f"State recovery failed: {sync_error}")
                    if game is not None:
                        result = {"episode": episode, "id": game["id"], "seed": game["seed"],
                                  "score": game["score"], "steps": game["step"], "length": game["length"],
                                  "status": game["status"], "terminal_reason": game["terminal_reason"],
                                  "stop_reason": reason or game["terminal_reason"] or "request_error",
                                  "state_confirmed": state_confirmed,
                                  "observed_decisions": game.get("history_count", game["step"]),
                                  "seconds": clock() - game_started}
                        if episode_error:
                            result["error"] = episode_error
                        summary["episodes"].append(result)
                        summary["total_steps"] += game["step"]
                        summary["observed_decisions"] += result["observed_decisions"]
                        report.write("episode_end", **result, game=game)
                        try:
                            driver.delete(game["id"], timeout=3)
                        except Exception as cleanup_error:  # noqa: BLE001 -- Cleanup must not hide a completed run or Ctrl-C.
                            summary["errors"].append(f"Session cleanup failed: {cleanup_error}")
                    elif episode_error:
                        summary["errors"].append(episode_error)
                if quit_requested:
                    break
        summary["seconds"] = clock() - started
        summary["steps_per_second"] = summary["total_steps"] / summary["seconds"] if summary["seconds"] else 0.0
        summary["inference_p50_ms"] = statistics.median(latencies) if latencies else None
        summary["exit_code"] = exit_code
        report.write("run_end", **summary)
    if headless:
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), file=stream)
    else:
        print(f"Qev Snake: {len(summary['episodes'])} game(s), {summary['total_steps']} steps, "
              f"{summary['observed_decisions']} observed model decisions.", file=stream)
        for result in summary["episodes"]:
            print(f"  seed={result['seed']} score={result['score']} steps={result['steps']} stop={result['stop_reason']}", file=stream)
        if args.report:
            print(f"Decision report: {args.report}", file=stream)
    return exit_code


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"qev snake: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
