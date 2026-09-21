"""Deterministic Snake rules and bounded sessions driven by real Qev decisions.

The model receives explicit environment features, with optional static BFS
space/path facts. No teacher action or safety replacement is used at runtime.
Every executed step records the model's original choice and exact request.
"""

from __future__ import annotations

import copy
import json
import math
import random
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .snake_features import FEATURE_VERSION, RECENT_WINDOW, candidate_features

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
POLICY = {"mode": "model_direct", "feature_assisted": True, "planner": False, "guardrail": False}


class GameNotFound(Exception):
    pass


class GameConflict(Exception):
    pass


class GameCapacity(Exception):
    pass


class SnakeGame:
    """Head-first coordinates; x increases rightward and y increases downward."""

    def __init__(self, *, seed=7, size=12, max_steps=500, observation="spatial"):
        if type(seed) is not int or not -(2**31) <= seed < 2**31:
            raise ValueError("seed must be a signed 32-bit integer")
        if type(size) is not int or not 6 <= size <= 20:
            raise ValueError("size must be an integer between 6 and 20")
        if type(max_steps) is not int or not 1 <= max_steps <= 2000:
            raise ValueError("max_steps must be an integer between 1 and 2000")
        if observation not in ("local", "spatial"):
            raise ValueError("observation must be local or spatial")
        self.observation = observation
        self.seed, self.size, self.max_steps = seed, size, max_steps
        self.starvation_limit = 2 * size * size
        self.rng = random.Random(seed)
        middle = size // 2
        self.body = [(middle, middle), (middle - 1, middle), (middle - 2, middle)]
        self.recent_heads = [self.body[0]]
        self.direction = "RIGHT"
        self.step = self.score = self.steps_since_food = 0
        self.status, self.terminal_reason = "running", None
        self.food = self._spawn_food()

    def _spawn_food(self):
        occupied = set(self.body)
        free = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied]
        return self.rng.choice(free) if free else None

    def candidates(self):
        if self.status != "running":
            return []
        hx, hy = self.body[0]
        distance = abs(hx - self.food[0]) + abs(hy - self.food[1])
        result = []
        for direction in DIRECTIONS:
            if direction == OPPOSITE[self.direction]:
                continue
            dx, dy = VECTORS[direction]
            target = (hx + dx, hy + dy)
            eats = target == self.food
            # Entering the old tail is legal when that tail leaves this tick.
            occupied = self.body if eats else self.body[:-1]
            collision = ("wall" if not (0 <= target[0] < self.size and 0 <= target[1] < self.size)
                         else "body" if target in occupied else None)
            new_distance = abs(target[0] - self.food[0]) + abs(target[1] - self.food[1])
            result.append({"direction": direction, "next_cell": list(target), "collision": collision,
                           "eats_food": eats, "manhattan_distance": new_distance,
                           "manhattan_change": new_distance - distance})
        return (candidate_features(self.body, self.food, self.size, result, self.recent_heads)
                if self.observation == "spatial" else result)

    def advance(self, direction):
        if self.status != "running":
            raise GameConflict("Game has ended")
        moves = {move["direction"]: move for move in self.candidates()}
        if direction not in moves:
            raise ValueError("Action must be a non-reversing direction")
        move = moves[direction]
        self.step += 1
        self.direction = direction
        if move["collision"]:
            self.status, self.terminal_reason = "dead", move["collision"]
            return False
        self.body.insert(0, tuple(move["next_cell"]))
        self.recent_heads.append(self.body[0])
        self.recent_heads = self.recent_heads[-RECENT_WINDOW:]
        if move["eats_food"]:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:
                self.status, self.terminal_reason = "won", "board_filled"
        else:
            self.body.pop()
            self.steps_since_food += 1
        if self.status == "running":
            if self.step >= self.max_steps:
                self.status, self.terminal_reason = "finished", "max_steps"
            elif self.steps_since_food >= self.starvation_limit:
                self.status, self.terminal_reason = "finished", "starvation"
        return move["eats_food"]

    def snapshot(self):
        candidates = self.candidates()
        return {"seed": self.seed, "size": self.size, "width": self.size, "height": self.size,
                "body": [list(cell) for cell in self.body], "food": list(self.food) if self.food is not None else None,
                "direction": self.direction, "step": self.step, "score": self.score, "length": len(self.body),
                "status": self.status, "terminal_reason": self.terminal_reason,
                "alive": self.status != "dead", "won": self.status == "won", "done": self.status != "running",
                "max_steps": self.max_steps, "starvation_limit": self.starvation_limit,
                "steps_since_food": self.steps_since_food,
                "available_directions": [move["direction"] for move in candidates], "candidates": candidates,
                "recent_heads": [list(cell) for cell in self.recent_heads],
                "policy": {**POLICY, "observation": self.observation,
                           "feature_version": FEATURE_VERSION if self.observation == "spatial" else "local-v1",
                           "feature_search": self.observation == "spatial"}}


def decision_request(game, model):
    """Exact observable facts for both training and play; never a 'best' label."""
    head, food = game.body[0], game.food
    state = (f"Snake on a {game.size} by {game.size} board. x increases right, y increases down. "
             f"Head=({head[0]},{head[1]}), food=({food[0]},{food[1]}), heading={game.direction}. "
             f"Length={len(game.body)}. Steps without food={game.steps_since_food}. "
             + ("Candidates include static BFS space/path facts and recent visits, not a chosen route plan."
                if game.observation == "spatial" else
                "The candidates contain immediate next-cell facts, not a route plan."))
    facts = game.candidates()
    criteria = {move["direction"]: {key: ("none" if key == "collision" and value is None else value)
                                   for key, value in move.items() if key != "direction"}
                for move in facts}
    request = {"model": model, "state": state, "questions": {"move": {
        "type": "choice",
        "instructions": (("Choose a Snake move. Avoid collisions. Prefer tail_reachable and enough reachable_space "
                         "for the snake length. Then follow a short food_path_distance, penalizing recent_visits "
                         "to avoid loops. An empty food_path_distance means no static path. Prefer straight ahead on ties. "
                         "All three non-reversing moves remain selectable.")
                         if game.observation == "spatial" else
                         ("Choose one Snake move. Avoid wall and body collisions first, then eat food or "
                         "move toward it. A negative manhattan_change reduces distance. "
                         "Immediate reverse is excluded; collision moves are still selectable.")),
        "criteria": criteria,
    }}}
    return request, facts


def model_choice(response, directions):
    """Reject malformed outputs; never choose a fallback or replace the model action."""
    if not isinstance(response, dict):
        raise TypeError("Model response must be an object")
    # The HTTP audit must retain the actual serializable response, never NaN.
    json.dumps(response, allow_nan=False)
    answer = response.get("answers", {}).get("move", {})
    probabilities = answer.get("probabilities")
    if answer.get("type") != "choice" or not isinstance(probabilities, dict) or set(probabilities) != set(directions):
        raise ValueError("Model must return a choice probability for every candidate")
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Model probabilities must be finite values in [0,1]")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-5, rel_tol=0):
        raise ValueError("Model probabilities must sum to one")
    chosen = answer.get("choice")
    if chosen not in probabilities or probabilities[chosen] != max(probabilities.values()):
        raise ValueError("Model choice must be a maximum-probability candidate")
    return chosen, dict(probabilities)


@dataclass
class GameSession:
    id: str
    game: SnakeGame
    last_access: float
    lock: Any = field(default_factory=threading.Lock)
    history: list[dict] = field(default_factory=list)


class GameStore:
    """Bounded, process-local sessions; one in-flight operation per game.

    A game lock spans observation, the single predict call, and transition.
    expected_step protects retries; a busy game fails immediately with conflict.
    Expiration is idle-time based and never removes an in-flight model call.
    """

    def __init__(self, agent, *, max_games=8, ttl_seconds=1800, clock=time.monotonic):
        if type(max_games) is not int or not 1 <= max_games <= 16:
            raise ValueError("max_games must be in 1..16")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive and finite")
        self.agent, self.max_games, self.ttl_seconds, self.clock = agent, max_games, ttl_seconds, clock
        self._sessions, self._lock = {}, threading.Lock()

    def _prune(self):
        now = self.clock()
        expired = [key for key, session in self._sessions.items()
                   if not session.lock.locked() and now - session.last_access >= self.ttl_seconds]
        for key in expired:
            del self._sessions[key]

    def _snapshot(self, session, *, history=False):
        value = {"id": session.id, **session.game.snapshot(),
                 "last_decision": session.history[-1] if session.history else None,
                 "history_count": len(session.history), "expires_in_seconds": self.ttl_seconds}
        if history:
            value["history"] = session.history
        return copy.deepcopy(value)

    @contextmanager
    def _session(self, game_id):
        with self._lock:
            self._prune()
            session = self._sessions.get(game_id)
            if session is None:
                raise GameNotFound("Unknown or expired game")
            if not session.lock.acquire(blocking=False):
                raise GameConflict("Game has another operation in progress")
            session.last_access = self.clock()
        try:
            yield session
        finally:
            with self._lock:
                session.last_access = self.clock()
                session.lock.release()

    def create(self, *, seed=7, size=12, max_steps=500, observation="spatial"):
        game = SnakeGame(seed=seed, size=size, max_steps=max_steps, observation=observation)
        with self._lock:
            self._prune()
            if len(self._sessions) >= self.max_games:
                raise GameCapacity(f"At most {self.max_games} games may be active; delete a game or wait for expiry")
            session = GameSession(uuid.uuid4().hex, game, self.clock())
            self._sessions[session.id] = session
            return self._snapshot(session)

    def get(self, game_id, *, history=False):
        with self._session(game_id) as session:
            return self._snapshot(session, history=history)

    def delete(self, game_id):
        with self._session(game_id), self._lock:
            del self._sessions[game_id]

    def step(self, game_id, expected_step):
        if type(expected_step) is not int or not 0 <= expected_step <= 2000:
            raise ValueError("expected_step must be an integer between 0 and 2000")
        with self._session(game_id) as session:
            game = session.game
            if expected_step != game.step:
                raise GameConflict(f"Stale expected_step {expected_step}; current step is {game.step}")
            if game.status != "running":
                return self._snapshot(session)
            model = getattr(self.agent, "config", {}).get("model_name", "qev-latest")
            request, facts = decision_request(game, model)
            trace = {"step": game.step, "step_before": game.step, "step_after": game.step,
                     "request": copy.deepcopy(request), "response": None, "candidate_facts": facts,
                     "probabilities": {}, "proposed": None, "executed": None, "intervened": False}
            started = time.perf_counter()
            try:
                response = self.agent.predict(request["state"], request["questions"], model=request["model"])
                trace["inference_ms"] = round((time.perf_counter() - started) * 1000, 3)
                json.dumps(response, allow_nan=False)
                trace["response"] = copy.deepcopy(response)
                if isinstance(response, dict) and response.get("qev", {}).get("truncated_questions"):
                    raise ValueError("Snake observation was truncated; no action executed")
                chosen, probabilities = model_choice(response, request["questions"]["move"]["criteria"])
                trace.update(probabilities=probabilities, proposed=chosen)
            except Exception as error:  # noqa: BLE001 -- Persist model failures without inventing an action.
                # A broken model is visible and ends the run, without a fabricated
                # move or an automatic retry that might look like AI control.
                game.status, game.terminal_reason = "finished", "model_error"
                trace["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
            else:
                ate = game.advance(chosen)
                trace.update(executed=chosen, ate=ate, step=game.step, step_after=game.step)
            trace.setdefault("inference_ms", round((time.perf_counter() - started) * 1000, 3))
            trace["decision_ms"] = round((time.perf_counter() - started) * 1000, 3)
            trace["terminal_reason"] = game.terminal_reason
            session.history.append(trace)
            return self._snapshot(session)
