"""Generate reproducible Snake imitation data from the actual serving request.

    python scripts/prepare_snake_data.py --out data/snake-v1 \
        --tokenizer models/qev-0.8b-mlx/backbone

Only a local tokenizer is loaded. The declared programmatic teacher consumes
the same visible candidate fields as the model. Optional replay keeps every
original record in its original partition. No model is trained or loaded here.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

from qev.api import render
from qev.data import (
    SPLITS,
    canonical_hash,
    digest,
    load_split,
    materialize,
    verify_partitions,
    write_json,
)
from qev.snake import SnakeGame, decision_request
from qev.snake_features import FEATURE_VERSION, RECENT_WINDOW, teacher_choice

FEATURE_FIELDS = {"reachable_space", "tail_reachable", "food_path_distance", "recent_visits"}
SOURCE = "synthetic_snake_observable_teacher"
RESERVED_BENCHMARK_SEEDS = tuple(range(10000, 10020))


def visible_teacher(request):
    """No game/body/RNG access: classify exactly the visible candidate values."""
    criteria = request["questions"]["move"]["criteria"]
    candidates = []
    for direction, facts in criteria.items():
        if not isinstance(facts, dict) or not FEATURE_FIELDS <= set(facts):
            raise ValueError("Serving decision_request lacks the agreed observable Snake features")
        candidates.append({"direction": direction, **facts})
    state = request["state"]
    board = re.search(r"Snake on a (\d+) by (\d+) board", state)
    length = re.search(r"Length=(\d+)\.", state)
    heading = re.search(r"heading=(UP|DOWN|LEFT|RIGHT)\.", state)
    if not board or board[1] != board[2] or not length or not heading:
        raise ValueError("Serving observation does not expose size, length, and heading")
    return teacher_choice(candidates, length=int(length[1]), size=int(board[1]), heading=heading[1])


def snapshot_observation(game):
    request, _ = decision_request(game, "qev-0.8b")
    choice = visible_teacher(request)
    return request, choice


def observation_hash(request):
    """Ignore option presentation order, not any model-visible feature value."""
    return canonical_hash({"state": request["state"], "questions": request["questions"]})


def normalize_state(state):
    return " ".join(render(state).casefold().split())


def roll_episode(*, seed, size, max_steps, split, rng, exploration):
    """Collect labels on valid trajectories, occasionally taking another legal move.

    Exploration is declared behavior-policy variation; gold labels always come
    from the same deterministic teacher. All-collision states have no safe gold
    action and are excluded, then the actual collision is recorded as an outcome.
    """
    game = SnakeGame(seed=seed, size=size, max_steps=max_steps)
    rows, actions, unlabelled = [], [], 0
    while game.status == "running":
        request, choice = snapshot_observation(game)
        criteria = request["questions"]["move"]["criteria"]
        legal = [direction for direction, facts in criteria.items() if facts["collision"] in (None, "none")]
        if choice is None:
            unlabelled += 1
            executed = next(iter(criteria))
        else:
            executed = rng.choice(legal) if rng.random() < exploration else choice
            row = copy.deepcopy(request)
            row.pop("model", None)
            row["questions"]["move"].update(label=choice, src=SOURCE)
            group = f"qev-snake/{size}/{seed}"
            row["_meta"] = {
                "id": f"{group}/{game.step}", "group_id": group, "source": SOURCE,
                "language": "en", "variant": "clean", "qev_parent_partition": split,
                "license": "Apache-2.0", "label_source": "deterministic visible-feature heuristic teacher",
                "feature_version": FEATURE_VERSION, "semantic_sha256": observation_hash(request),
                "episode_seed": seed, "size": size, "step": game.step,
                "behavior_action": executed, "behavior_differs_from_teacher": executed != choice,
                # Metadata never enters materialize()/the model input. It lets
                # reviewers independently reconstruct every geometric feature.
                "environment": {"body": [list(cell) for cell in game.body], "food": list(game.food),
                                "heading": game.direction, "length": len(game.body),
                                "steps_since_food": game.steps_since_food,
                                "recent_heads": [list(cell) for cell in game.recent_heads]},
            }
            rows.append(row)
        actions.append(executed)
        game.advance(executed)
    return rows, {"split": split, "seed": seed, "size": size, "max_steps": max_steps,
                  "score": game.score, "steps": game.step, "length": len(game.body),
                  "status": game.status, "terminal_reason": game.terminal_reason,
                  "unlabelled_all_collision_states": unlabelled, "actions": actions}


def generate_partitions(counts, *, seed=20260921, sizes=(6, 8, 12, 16), episode_steps=600,
                        samples_per_episode=48, exploration=0.10):
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("All three partitions require positive integer question counts")
    if not sizes or len(set(sizes)) != len(sizes) or any(type(n) is not int or not 6 <= n <= 20 for n in sizes):
        raise ValueError("Board sizes must be distinct integers in 6..20")
    if not 1 <= episode_steps <= 2000 or not 1 <= samples_per_episode <= episode_steps:
        raise ValueError("Require 1 <= samples_per_episode <= episode_steps <= 2000")
    if not 0 <= exploration <= 1:
        raise ValueError("exploration must be in [0,1]")
    partitions, episodes = {}, []
    used_seeds, observation_labels, state_partitions = set(RESERVED_BENCHMARK_SEEDS), {}, {}
    duplicates = Counter()
    for split in SPLITS:
        seed_rng = random.Random(f"qev-snake-seeds:{seed}:{split}")
        sample_rng = random.Random(f"qev-snake-samples:{seed}:{split}")
        rows = []
        episode_index = 0
        while len(rows) < counts[split]:
            if episode_index >= counts[split] * 10 + 100:
                raise RuntimeError(f"Could not generate enough disjoint observations for {split}")
            game_seed = seed_rng.randrange(-(2**31), 2**31)
            if game_seed in used_seeds:
                continue
            used_seeds.add(game_seed)
            size = sizes[episode_index % len(sizes)]
            episode_rng = random.Random(f"qev-snake-trajectory:{seed}:{split}:{game_seed}")
            candidates, summary = roll_episode(seed=game_seed, size=size, max_steps=episode_steps,
                                               split=split, rng=episode_rng, exploration=exploration)
            sample_rng.shuffle(candidates)
            selected = 0
            for row in candidates:
                semantic = row["_meta"]["semantic_sha256"]
                label = row["questions"]["move"]["label"]
                if semantic in observation_labels:
                    if observation_labels[semantic] != label:
                        raise ValueError("Identical visible observation received conflicting teacher labels")
                    duplicates["repeated_observation"] += 1
                    continue
                state = normalize_state(row["state"])
                if state in state_partitions and state_partitions[state] != split:
                    duplicates["cross_partition_state"] += 1
                    continue
                criteria = row["questions"]["move"]["criteria"]
                order = list(criteria)
                # One frozen order per independent observation in every split;
                # this does not create duplicated augmented validation copies.
                random.Random(f"qev-snake-options:{seed}:{split}:{semantic}").shuffle(order)
                row["questions"]["move"]["criteria"] = {key: criteria[key] for key in order}
                row["_meta"]["option_order"] = "frozen random permutation"
                observation_labels[semantic] = label
                state_partitions[state] = split
                rows.append(row)
                selected += 1
                if selected >= samples_per_episode or len(rows) == counts[split]:
                    break
            summary["selected_records"] = selected
            episodes.append(summary)
            episode_index += 1
        # Do not leave neighboring observations from an episode adjacent in a
        # sequential training file; all randomness is fixed before training.
        sample_rng.shuffle(rows)
        partitions[split] = rows
    verify_partitions(partitions)
    return partitions, episodes, dict(duplicates)


def teacher_evaluation(*, seeds=range(20), sizes=(8, 12), max_steps=2000):
    episodes = []
    for size in sizes:
        for seed in seeds:
            _, result = roll_episode(seed=seed, size=size, max_steps=max_steps, split="teacher_probe",
                                     rng=random.Random(0), exploration=0)
            result.pop("actions")
            episodes.append(result)
    summary = {}
    for size in sizes:
        group = [row for row in episodes if row["size"] == size]
        summary[str(size)] = {"episodes": len(group), "mean_score": sum(row["score"] for row in group) / len(group),
                              "min_score": min(row["score"] for row in group),
                              "max_score": max(row["score"] for row in group),
                              "mean_steps": sum(row["steps"] for row in group) / len(group),
                              "terminal_reasons": dict(Counter(row["terminal_reason"] for row in group))}
    return {"policy": "programmatic visible-feature teacher, no model", "feature_version": FEATURE_VERSION,
            "max_steps": max_steps, "summary": summary, "episodes": episodes}


def token_audit(partitions, tokenizer, *, max_length=1024, max_state=384):
    from qev.tokenization import encode_question

    if tokenizer is None:
        return {"status": "not_run", "reason": "No tokenizer supplied to the Python API"}
    results = {}
    for split, rows in partitions.items():
        lengths, states = [], []
        for row in rows:
            rendered = materialize(row)
            for question in rendered["questions"]:
                encoded = encode_question(rendered["state"], question, tokenizer, max_length, max_state)
                if encoded["state_truncated"]:
                    raise ValueError(f"State would truncate: {row['_meta']['id']}")
                lengths.append(len(encoded["ids"]))
                states.append(encoded["state_tokens"])
        results[split] = {"questions": len(lengths), "max_tokens": max(lengths),
                          "max_state_tokens": max(states), "state_truncations": 0}
    return {"status": "passed", "max_length": max_length, "max_state": max_state, "partitions": results}


def prepare(out, *, counts=None, seed=20260921, sizes=(6, 8, 12, 16), episode_steps=600,
            samples_per_episode=48, exploration=0.10, tokenizer=None, teacher_probe_seeds=20,
            replay_data=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {out}")
    counts = counts or {"train": 12000, "calibration": 500, "development": 500}
    partitions, episodes, duplicates = generate_partitions(
        counts, seed=seed, sizes=sizes, episode_steps=episode_steps,
        samples_per_episode=samples_per_episode, exploration=exploration)
    replay = None
    if replay_data is not None:
        replay_root = Path(replay_data)
        parent = json.loads((replay_root / "manifest.json").read_text())
        replay = {"directory": str(replay_root), "manifest_sha256": digest(replay_root / "manifest.json"),
                  "partitions": {}, "preserve_original_rows_and_partitions": True,
                  "license": parent.get("license", "See source manifest"),
                  "source_datasets": parent.get("source_datasets", [])}
        for split in SPLITS:
            original = load_split(replay_root, split)
            partitions[split].extend(original)
            random.Random(f"qev-snake-replay-order:{seed}:{split}").shuffle(partitions[split])
            replay["partitions"][split] = {"sha256": digest(replay_root / f"{split}.jsonl"),
                                           "records": len(original),
                                           "questions": sum(len(row["questions"]) for row in original)}
        verify_partitions(partitions)
    audit = token_audit(partitions, tokenizer)
    probe = teacher_evaluation(seeds=range(teacher_probe_seeds), sizes=sizes) if teacher_probe_seeds else None
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "format": "qev-data-v1", "seed": seed, "partitions": {},
        "license": "Mixed replay dataset licenses plus Apache-2.0 Snake data" if replay else "Apache-2.0",
        "replay": replay,
        "synthetic": {"source": SOURCE, "feature_version": FEATURE_VERSION,
                      "code_sha256": {name: digest(root / name) for name in
                                      ("scripts/prepare_snake_data.py", "qev/snake.py", "qev/snake_features.py")},
                      "label_source": "Explicit deterministic teacher over model-visible feature values only",
                      "external_model_labels": False, "optimal_play_ground_truth": False},
        "protocol": {"locked_test_read": False, "splits": "Independent game seeds/groups; disjoint normalized states and observations",
                     "sizes": list(sizes), "episode_steps": episode_steps, "samples_per_episode": samples_per_episode,
                     "exploration_probability": exploration, "exploration_actions": "uniform over collision-free candidates",
                     "recent_head_window": RECENT_WINDOW, "candidate_permutation": "fixed per observation in all splits",
                     "replay_mixed": replay is not None, "duplicates_excluded": duplicates,
                     "reserved_closed_loop_benchmark_seeds": list(RESERVED_BENCHMARK_SEEDS),
                     "teacher_inputs": "request criteria plus visible size, length, heading; no hidden body, food RNG, or future rollout",
                     "limitations": ["Heuristic imitation labels, not optimal-action proofs",
                                     "Static BFS features are environment assistance, not model-learned geometry",
                                     "Teacher can starve or exhaust its step budget", "Held-out game seeds do not establish novel-rule generalization"]},
        "token_audit": audit,
    }
    out.mkdir(parents=True)
    for split, rows in partitions.items():
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows))
        snake_rows = [row for row in rows if row["_meta"]["source"] == SOURCE]
        labels = [row["questions"]["move"]["label"] for row in snake_rows]
        positions = [list(row["questions"]["move"]["criteria"]).index(label) for row, label in zip(snake_rows, labels, strict=True)]
        question_types = Counter(q["type"] for row in rows for q in row["questions"].values())
        manifest["partitions"][split] = {"sha256": digest(path), "records": len(rows), "questions": sum(question_types.values()),
                                         "question_types": dict(question_types),
                                         "sources": dict(Counter(row["_meta"]["source"] for row in rows)),
                                         "snake_records": len(snake_rows), "snake_label_directions": dict(Counter(labels)),
                                         "snake_label_positions": dict(Counter(positions)),
                                         "episode_seeds": [episode["seed"] for episode in episodes if episode["split"] == split]}
    episode_path = out / "episodes.jsonl"
    episode_path.write_text("".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in episodes))
    manifest["episodes"] = {"file": episode_path.name, "sha256": digest(episode_path), "count": len(episodes)}
    write_json(out / "token_audit.json", audit)
    if probe is not None:
        write_json(out / "teacher_evaluation.json", probe)
        manifest["teacher_evaluation"] = {"file": "teacher_evaluation.json", "sha256": digest(out / "teacher_evaluation.json"),
                                           "summary": probe["summary"]}
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="models/qev-0.8b-mlx/backbone", help="Local tokenizer directory; never downloads weights")
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--calibration", type=int, default=500)
    parser.add_argument("--development", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--sizes", default="6,8,12,16")
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--samples-per-episode", type=int, default=48)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--teacher-probe-seeds", type=int, default=20)
    parser.add_argument("--replay-data", help="Keep every original row in its original train/calibration/development split")
    args = parser.parse_args()
    try:
        sizes = tuple(int(value) for value in args.sizes.split(","))
    except ValueError:
        parser.error("--sizes must contain comma-separated integers")
    if args.teacher_probe_seeds < 0:
        parser.error("--teacher-probe-seeds must be nonnegative")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = prepare(args.out, counts={split: getattr(args, split) for split in SPLITS}, seed=args.seed,
                     sizes=sizes, episode_steps=args.episode_steps, samples_per_episode=args.samples_per_episode,
                     exploration=args.exploration, tokenizer=tokenizer, teacher_probe_seeds=args.teacher_probe_seeds,
                     replay_data=args.replay_data)
    print(json.dumps({"out": args.out, "partitions": result["partitions"], "token_audit": result["token_audit"]}, indent=2))


if __name__ == "__main__":
    main()
