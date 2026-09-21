import copy
import itertools
import json
from collections import Counter

import pytest

from qev.data import digest, load_split, materialize, verify_partitions
from qev.snake import SnakeGame, decision_request
from qev.snake_features import candidate_features, teacher_choice
from scripts.prepare_snake_data import (
    RESERVED_BENCHMARK_SEEDS,
    SOURCE,
    generate_partitions,
    observation_hash,
    prepare,
    token_audit,
    visible_teacher,
)


def test_geometric_features_handle_moving_tail_food_and_collision_without_mutation():
    game = SnakeGame(size=6, observation="local")
    game.body = [(1, 1), (1, 2), (0, 2), (0, 1)]
    game.direction, game.food = "UP", (2, 1)
    before = game.snapshot()
    candidates = game.candidates()
    original = copy.deepcopy(candidates)
    features = {move["direction"]: move for move in candidate_features(
        game.body, game.food, 6, candidates, [(0, 1)] * 40)}
    assert candidates == original and game.snapshot() == before
    # The old tail moves away on LEFT, so this is a legal four-cell transition.
    assert features["LEFT"]["collision"] is None
    assert features["LEFT"]["reachable_space"] == 33
    assert features["LEFT"]["tail_reachable"] is True
    assert features["LEFT"]["recent_visits"] == 32
    # Eating retains the old tail, giving five occupied cells, not four.
    assert features["RIGHT"]["eats_food"] is True
    assert features["RIGHT"]["food_path_distance"] == 0
    assert features["RIGHT"]["reachable_space"] == 32
    blocked = [{"direction": "LEFT", "next_cell": [-1, 1], "collision": "wall", "eats_food": False}]
    collision = candidate_features(game.body, game.food, 6, blocked)[0]
    assert collision["reachable_space"] == 0 and not collision["tail_reachable"]
    assert collision["food_path_distance"] is None


def test_static_space_exposes_a_one_cell_pocket_instead_of_calling_it_safe():
    game = SnakeGame(size=6, observation="local")
    game.body = [(2, 2), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1),
                 (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4)]
    game.direction, game.food = "UP", (5, 5)
    left = next(move for move in candidate_features(game.body, game.food, 6, game.candidates())
                if move["direction"] == "LEFT")
    assert left["collision"] is None
    assert left["reachable_space"] == 1
    assert left["tail_reachable"] is False and left["food_path_distance"] is None


def test_teacher_consumes_visible_fields_and_ignores_candidate_order():
    request, _ = decision_request(SnakeGame(seed=11, size=8), "qev-0.8b")
    expected = visible_teacher(request)
    criteria = request["questions"]["move"]["criteria"]
    for order in itertools.permutations(criteria):
        changed = copy.deepcopy(request)
        changed["questions"]["move"]["criteria"] = {direction: criteria[direction] for direction in order}
        changed["_meta"] = {"hidden_body": "must never influence the teacher", "teacher_hint": "WRONG"}
        assert visible_teacher(changed) == expected
    local_request, _ = decision_request(SnakeGame(observation="local"), "qev-0.8b")
    with pytest.raises(ValueError, match="observable Snake features"):
        visible_teacher(local_request)
    colliding = [{"direction": direction, "collision": "wall"} for direction in ("UP", "LEFT", "RIGHT")]
    assert teacher_choice(colliding, length=3, size=8, heading="UP") is None


def test_deterministic_disjoint_seeded_trajectories_and_correct_permuted_labels():
    options = {"counts": {"train": 36, "calibration": 12, "development": 12}, "seed": 91,
               "sizes": (6, 8), "episode_steps": 24, "samples_per_episode": 6, "exploration": 0.2}
    partitions, episodes, excluded = generate_partitions(**options)
    assert (partitions, episodes, excluded) == generate_partitions(**options)
    verify_partitions(partitions)
    seeds = [episode["seed"] for episode in episodes]
    assert len(seeds) == len(set(seeds))
    assert not set(seeds) & set(RESERVED_BENCHMARK_SEEDS)
    assert all(episode["selected_records"] <= 6 for episode in episodes)
    assert sum(episode["split"] == "train" for episode in episodes) >= 6
    rows = {row["_meta"]["id"]: row for values in partitions.values() for row in values}
    identities = [row["_meta"]["semantic_sha256"] for row in rows.values()]
    assert len(identities) == len(set(identities))
    positions = Counter()
    for row in rows.values():
        q = materialize(row)["questions"][0]
        assert q["keys"][q["label"]] == visible_teacher(row)
        positions[q["label"]] += 1
        # Label and environment audit metadata never appear in model input.
        poisoned = copy.deepcopy(row)
        poisoned["_meta"]["environment"] = {"body": "private metadata replaced"}
        assert materialize(poisoned) == materialize(row)
    assert set(positions) == {0, 1, 2}
    # Replay the actual behavior actions, including off-teacher legal moves.
    for episode in episodes:
        game = SnakeGame(seed=episode["seed"], size=episode["size"], max_steps=episode["max_steps"])
        for step, action in enumerate(episode["actions"]):
            identifier = f"qev-snake/{game.size}/{game.seed}/{step}"
            if identifier in rows:
                row = rows[identifier]
                served, _ = decision_request(game, "qev-0.8b")
                assert observation_hash(served) == row["_meta"]["semantic_sha256"]
                assert [list(cell) for cell in game.body] == row["_meta"]["environment"]["body"]
                assert action == row["_meta"]["behavior_action"]
            game.advance(action)
        assert game.score == episode["score"] and game.terminal_reason == episode["terminal_reason"]


def test_prepare_preserves_replay_splits_and_is_loadable_with_original_schema(tmp_path):
    parent = tmp_path / "replay"
    parent.mkdir()
    manifest = {"partitions": {}, "license": "fixture license"}
    originals = {}
    for split in ("train", "calibration", "development"):
        row = {"state": f"Replay fixture unique to {split}", "questions": {
            "answer": {"type": "choice", "criteria": {"a": "yes", "b": "no"}, "label": "a"}},
            "_meta": {"id": f"replay-{split}", "group_id": f"replay-{split}", "source": "replay-fixture"}}
        path = parent / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        manifest["partitions"][split] = {"records": 1, "sha256": digest(path)}
        originals[split] = row
    (parent / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "mixed"
    result = prepare(out, counts={"train": 8, "calibration": 4, "development": 4},
                     sizes=(6,), episode_steps=12, samples_per_episode=4, teacher_probe_seeds=0,
                     replay_data=parent)
    loaded = {split: load_split(out, split) for split in originals}
    verify_partitions(loaded)
    for split, rows in loaded.items():
        assert originals[split] in rows
        assert all(originals[other] not in rows for other in originals if other != split)
        assert result["partitions"][split]["sources"][SOURCE] == len(rows) - 1
        assert result["replay"]["partitions"][split]["questions"] == 1
    assert result["token_audit"]["status"] == "not_run"
    with pytest.raises(FileExistsError):
        prepare(out)


def test_token_audit_covers_every_question_in_replay(monkeypatch):
    seen = []

    def encode(state, question, *args):
        seen.append(question["qid"])
        return {"ids": [0] * 10, "state_tokens": 2, "state_truncated": False}

    monkeypatch.setattr("qev.tokenization.encode_question", encode)
    rows = [{"state": "fixture", "questions": {
        "one": {"type": "noul", "label": True}, "two": {"type": "noul", "label": False}},
        "_meta": {"id": "multi"}}]
    result = token_audit({"train": rows}, object())
    assert seen == ["one", "two"]
    assert result["partitions"]["train"]["questions"] == 2
