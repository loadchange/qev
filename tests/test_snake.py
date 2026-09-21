import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qev.demo import create_demo_router
from qev.snake import GameCapacity, GameConflict, GameNotFound, GameStore, SnakeGame


class StubAgent:
    def __init__(self, direction="RIGHT"):
        self.config = {"model_name": "test-checkpoint"}
        self.direction, self.calls = direction, []

    def predict(self, state, questions, *, model):
        self.calls.append(copy.deepcopy({"state": state, "questions": questions, "model": model}))
        choices = questions["move"]["criteria"]
        assert self.direction in choices
        return {"model": model, "answers": {"move": {"type": "choice", "choice": self.direction,
                "confidence": 1.0, "probabilities": {key: float(key == self.direction) for key in choices}}},
                "usage": {"input_tokens": 123, "output_tokens": 0}, "qev": {"backend": "stub"}}


def test_seed_controls_food_and_reset_layout_without_global_randomness():
    a, b = SnakeGame(seed=19), SnakeGame(seed=19)
    assert a.snapshot() == b.snapshot()
    assert a.food not in a.body
    a.food = b.food = (a.body[0][0] + 1, a.body[0][1])
    assert a.advance("RIGHT") and b.advance("RIGHT")
    assert a.snapshot() == b.snapshot()
    assert a.score == 1 and len(a.body) == 4 and a.food not in a.body
    before = a.snapshot()
    with pytest.raises(ValueError, match="non-reversing"):
        a.advance("LEFT")
    assert a.snapshot() == before


def test_tail_leaves_before_collision_check_but_body_does_not():
    game = SnakeGame(size=6)
    game.body = [(2, 2), (2, 3), (1, 3), (1, 2)]
    game.direction, game.food = "UP", (5, 5)
    moves = {move["direction"]: move for move in game.candidates()}
    assert moves["LEFT"]["collision"] is None
    assert not game.advance("LEFT")
    assert game.body == [(1, 2), (2, 2), (2, 3), (1, 3)]
    game.body = [(2, 2), (2, 3), (1, 3), (1, 2), (1, 1)]
    game.direction = "UP"
    assert {m["direction"]: m for m in game.candidates()}["LEFT"]["collision"] == "body"
    game.advance("LEFT")
    assert game.status == "dead" and game.terminal_reason == "body"


def test_max_steps_and_starvation_stop_valid_moves():
    limited = SnakeGame(size=6, max_steps=1)
    limited.food = (0, 0)
    limited.advance("RIGHT")
    assert limited.step == 1 and limited.terminal_reason == "max_steps"
    starving = SnakeGame(size=6, max_steps=2000)
    starving.food = (0, 0)
    for index in range(starving.starvation_limit):
        starving.advance(("UP", "LEFT", "DOWN", "RIGHT")[index % 4])
    assert starving.step == 72 and starving.terminal_reason == "starvation"
    with pytest.raises(GameConflict):
        starving.advance("UP")


def test_filling_final_free_cell_finishes_without_spawning_food():
    game = SnakeGame(size=6)
    path = [(0, 0)]
    for y in range(6):
        path.extend((x, y) for x in (range(1, 6) if y % 2 == 0 else range(5, 0, -1)))
    path.extend((0, y) for y in range(5, 0, -1))
    game.body = list(reversed(path[1:]))
    game.direction, game.food = "UP", (0, 0)
    assert game.advance("UP")
    assert game.status == "won" and game.terminal_reason == "board_filled"
    assert game.food is None and len(game.body) == 36
    assert game.snapshot()["candidates"] == []


def test_real_model_request_and_exact_action_are_auditable_even_on_collision():
    agent = StubAgent("RIGHT")
    store = GameStore(agent)
    created = store.create(size=6)
    session = store._sessions[created["id"]]
    session.game.body = [(5, 3), (4, 3), (3, 3)]
    result = store.step(created["id"], 0)
    assert len(agent.calls) == 1
    request = agent.calls[0]
    assert set(request["questions"]["move"]["criteria"]) == {"UP", "DOWN", "RIGHT"}
    assert request["questions"]["move"]["criteria"]["RIGHT"]["collision"] == "wall"
    assert "route plan" in request["state"]
    assert result["status"] == "dead" and result["terminal_reason"] == "wall"
    trace = result["last_decision"]
    assert trace["request"] == request
    assert trace["response"]["usage"] == {"input_tokens": 123, "output_tokens": 0}
    assert trace["proposed"] == trace["executed"] == "RIGHT" and not trace["intervened"]
    assert result["policy"] == {"mode": "model_direct", "feature_assisted": True, "planner": False,
        "guardrail": False, "observation": "spatial", "feature_version": "snake-observable-bfs-v1",
        "feature_search": True}
    assert store.step(created["id"], 1)["step"] == 1
    assert len(agent.calls) == 1  # No inference after termination.
    with pytest.raises(GameConflict):
        store.step(created["id"], 0)
    history = store.get(created["id"], history=True)["history"]
    assert history == [trace]
    history[0]["request"]["state"] = "changed by caller"
    assert store.get(created["id"], history=True)["history"][0]["request"] == request


def test_spatial_observations_expose_search_facts_without_teacher_action():
    spatial, local = SnakeGame(), SnakeGame(observation="local")
    assert spatial.body == local.body and spatial.food == local.food
    extra = {"reachable_space", "tail_reachable", "food_path_distance", "recent_visits"}
    for rich, old in zip(spatial.candidates(), local.candidates()):
        assert extra == rich.keys() - old.keys()
        assert {key: rich[key] for key in old} == old
    assert local.snapshot()["policy"]["feature_search"] is False
    spatial.advance("DOWN")
    assert spatial.recent_heads[-1] == spatial.body[0]


def test_duplicate_and_inflight_steps_do_not_call_model_twice():
    entered, release = threading.Event(), threading.Event()

    class BlockingAgent(StubAgent):
        def predict(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return super().predict(*args, **kwargs)

    agent = BlockingAgent()
    now = [0.0]
    store = GameStore(agent, max_games=1, ttl_seconds=10, clock=lambda: now[0])
    game_id = store.create()["id"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(store.step, game_id, 0)
        assert entered.wait(3)
        try:
            now[0] = 20.0
            with pytest.raises(GameCapacity):
                store.create()  # Expiry must not remove an in-flight inference.
            with pytest.raises(GameConflict, match="in progress"):
                store.step(game_id, 0)
            with pytest.raises(GameConflict):
                store.delete(game_id)
        finally:
            release.set()
        assert first.result(timeout=3)["step"] == 1
    with pytest.raises(GameConflict, match="Stale"):
        store.step(game_id, 0)
    assert len(agent.calls) == 1


@pytest.mark.parametrize("failure", ["exception", "probabilities", "choice", "truncation"])
def test_model_failure_is_visible_without_a_fabricated_action(failure):
    class BadAgent(StubAgent):
        def predict(self, *args, **kwargs):
            result = super().predict(*args, **kwargs)
            if failure == "exception":
                raise RuntimeError("inference failed")
            if failure == "probabilities":
                result["answers"]["move"]["probabilities"]["RIGHT"] = float("nan")
            elif failure == "truncation":
                result["qev"]["truncated_questions"] = ["move"]
            else:
                result["answers"]["move"]["choice"] = "LEFT"
            return result
    agent = BadAgent()
    store = GameStore(agent)
    original = store.create()
    result = store.step(original["id"], 0)
    assert result["terminal_reason"] == "model_error" and result["status"] == "finished"
    assert result["body"] == original["body"] and result["step"] == 0
    assert result["last_decision"]["executed"] is None
    assert "error" in result["last_decision"]
    if failure == "truncation":
        assert "truncated" in result["last_decision"]["error"]["message"]
        assert result["last_decision"]["response"]["qev"]["truncated_questions"] == ["move"]
    store.step(original["id"], 0)
    assert len(agent.calls) == 1


def test_capacity_expiry_and_delete_reset_do_not_retain_sessions():
    now = [0.0]
    store = GameStore(StubAgent(), max_games=1, ttl_seconds=10, clock=lambda: now[0])
    original = store.create(seed=29)
    with pytest.raises(GameCapacity):
        store.create()
    now[0] = 11
    with pytest.raises(GameNotFound):
        store.get(original["id"])
    replacement = store.create(seed=29)
    assert replacement["id"] != original["id"]
    assert replacement["body"] == original["body"] and replacement["food"] == original["food"]
    store.delete(replacement["id"])
    with pytest.raises(GameNotFound):
        store.get(replacement["id"])
    assert store.create(seed=29)["step"] == 0


def test_router_contract_bounds_history_and_error_codes():
    agent = StubAgent()
    app = FastAPI()
    app.include_router(create_demo_router(agent, store=GameStore(agent, max_games=1)))
    client = TestClient(app)
    for body in ({"size": 5}, {"size": 21}, {"max_steps": 2001}, {"seed": True}, {"action": "UP"}):
        assert client.post("/api/snake/games", json=body).status_code == 422
    created = client.post("/api/snake/games", json={"seed": 13, "size": 6, "max_steps": 1})
    assert created.status_code == 201
    path = "/api/snake/games/" + created.json()["id"]
    assert client.post("/api/snake/games", json={}).status_code == 429
    assert client.get("/api/snake/games/missing").status_code == 404
    assert client.post(path + "/step", json={"expected_step": True}).status_code == 422
    stepped = client.post(path + "/step", json={"expected_step": 0})
    assert stepped.status_code == 200 and stepped.json()["done"]
    assert client.post(path + "/step", json={"expected_step": 0}).status_code == 409
    assert client.post(path + "/step", json={"expected_step": 1}).status_code == 200
    assert len(agent.calls) == 1
    assert client.get(path + "?history=true").json()["history_count"] == 1
    assert client.delete(path).status_code == 204
    assert client.get(path).status_code == 404
