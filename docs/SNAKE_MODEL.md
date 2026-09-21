[English](SNAKE_MODEL.md) | [简体中文](SNAKE_MODEL.zh-CN.md)

# Qev Snake model

`qev snake` loads a real Qev model in the terminal and executes its highest-probability action at each step. It prefers `models/qev-snake-0.8b-mlx` by default; the original `models/qev-0.8b[-mlx]` remains available for comparison. See [DEMO](DEMO.md) for all controls and arguments.

## Why it was revised

Historical v0.2 connected the game interface but contained no Snake training samples. Inputs included only local collision facts and Manhattan distance, and real games missed food, looped and hit walls. In addition to the new terminal interface, this version continues supervised training from the existing Qev adapter/head and adds explicit, visible spatial features from the environment.

The local Laya-MLX `fc1df62` Snake demo inserts the planner's “Best progress toward food” into candidates in `laya_mlx/snake/policy.py`; its default safety shield can also replace the model's choice. Its demo scores therefore cannot be directly compared with Qev scores without overrides. This evaluation primarily compares **Qev before and after training under the same game environment, features, seeds and inference settings**; it does not claim superiority over Laya.

## Inputs, supervision and execution

`spatial` mode supplies three non-reversing directions, including directions that may collide. Candidates include collision, distance change, food consumption, static BFS reachable space, tail connectivity, food path distance and visit counts over the last 32 steps. The environment computes BFS, so the model need not infer geometry from board pixels; the game board is still not an image input.

An explicit heuristic teacher reads only fields visible to the model to generate labels. It prioritizes avoiding collisions and preserving tail connectivity and space, then considers food paths and visits. Training input contains no “best action” marker, teacher direction, teacher ranking or hidden board metadata. Runtime neither calls the teacher, removes colliding candidates nor replaces model actions. `local` mode preserves the old observation for ablation.

New Snake train / calibration / development counts are 12,000 / 500 / 500. Training uses 251 independent game seeds and sizes 6, 8, 12 and 16, sampling at most 48 states per episode, with 10% probability of a legal random deviation and shuffled candidate order. All 5,892 / 620 / 880 original questions are replayed in their existing splits. Combined totals are 17,892 / 1,120 / 1,380 questions, with no truncation in the per-question token audit. Game seeds and normalized observations are isolated across splits; 10000–10019 are excluded from training trajectories.

Training continues the original adapter/head for two epochs on an A100 40GB, with batch 8, accumulation 2 and learning rate 3e-5: 2238 updates and about 1484.69 optimization seconds. The complete original Qwen3.5 foundation stays frozen; the new calibration split fits temperature `T=2.82842712474619`. See the [records directory](results/snake_training) for provenance, data hashes, frozen checks and complete evaluation.

## Evaluation under matched conditions

Before/after results on the same A100, CUDA BF16 and development set:

| Measure | Count | Parent model | Specialized model |
| --- | ---: | ---: | ---: |
| Teacher-action agreement on unseen game seeds | 500 | 73.60% | 98.40% |
| Original general classification and programmatic rule accuracy | 880 | 81.25% | 81.70% |

Teacher agreement is not whole-game success. The following closed-loop evaluation uses the same `spatial` features and a 500-step cap, executing model argmax directly without safety overrides. Torch advances independent games in batches of 8.

| Board / held-out seeds | Parent mean food | Specialized mean food | Parent collision games | Specialized collision games |
| --- | ---: | ---: | ---: | ---: |
| 8×8 / 10000–10019, 20 games | 3.10 | 42.90 | 20 | 0 |
| 12×12 / 10000–10007, 8 games | 0.625 | 42.00 | 8 | 0 |

All 28 new-model games reached the 500-step cap; none filled the board. Identical seeds guarantee identical initial states, but different actions lead to different later food positions and trajectories. This measures closed-loop behavior rather than classification on the same state at every step. Floating-point differences between batched and single-game inference, or CUDA BF16 and MLX FP32, can alter choices and subsequent trajectories.

### Local MLX

Apple M4, 16 GiB memory, MLX FP32, 8×8 board, the same held-out seeds 10000–10004, one game at a time and a 500-step limit per game:

| Measure | Parent model | Specialized model |
| --- | ---: | ---: |
| Mean food collected | 3.0 | 42.6 |
| Food by game | 4 / 1 / 5 / 1 / 4 | 37 / 45 / 44 / 38 / 49 |
| Collision games | 5 / 5 | 0 / 5 |
| Games reaching the 500-step cap | 0 / 5 | 5 / 5 |

All five seeds improved; the new model neither starved nor filled the board. Across 2500 real decisions, median latency was 223.84 ms and P95 was 240.21 ms; speed including game processing and step logging was about 4.40 steps/second. The parent called the same local MLX Agent over HTTP; the new model called the local MLX Agent directly. This compares gameplay and does not compare transport latency. Protocol checks verified identical rules, inputs, candidate order and feature source. Raw probabilities, choices and executed actions are recorded at every step; all executed argmax without overrides. See the [parent record](results/snake_training/parent-mlx-spatial.json) and [specialized model record](results/snake_training/trained-mlx-spatial.json).

## Multimodal preservation and release validation

The complete Qwen3.5 foundation remains frozen throughout training. Native generation disables the decision adapter and uses the original language/visual weights and generation head. Loading old and new checkpoints sequentially on the same A100 produced identical output token IDs for three fixed text, image and video examples; full frozen-parameter hashes also matched. See the [native comparison](results/snake_training/snake-native.json). This provides limited regression and weight-preservation evidence, not a comprehensive multimodal accuracy evaluation.

MLX export reused the complete multimodal foundation, byte-identical to the previous version. All 372 adapter tensors were checked individually against the new Torch checkpoint, and the pointer-head file also matched; see [export verification](results/snake_training/export_integrity.json). Torch CPU FP32 versus MLX FP32 logit parity was not rerun, and the export configuration accurately retains `parity_status: not_run`; identical CUDA BF16 and MLX FP32 probabilities are not claimed.

The new MLX checkpoint passed [19 real HTTP / TypeSafe SDK checks](results/snake_training/service_mlx.json), covering text, image, video and mixed-media decisions, native generation, adapter restoration and invalid requests. Terminal pause, single-step, quit and cursor restoration passed [real PTY validation](results/snake_training/terminal_controls.json); the default path also loaded the new model in an actual smoke test. See [release checks](results/snake_training/release_checks.json) for 148 tests, build results and artifact hashes. Old and new checkpoints are stored separately without overwriting original files, and the Colab A100 session has stopped.

## Reproduction

```bash
uv run qev snake
uv run qev snake --size 8 --seed 10000 --max-steps 500
uv run python scripts/benchmark_snake.py --model models/qev-0.8b-mlx \
  --seeds 10000:10004 --size 8 --max-steps 500 --observation spatial \
  --output runs/parent-snake.json
uv run python scripts/benchmark_snake.py --model models/qev-snake-0.8b-mlx \
  --seeds 10000:10004 --size 8 --max-steps 500 --observation spatial \
  --output runs/trained-snake.json --compare runs/parent-snake.json
```

See [TRAINING](TRAINING.md) for complete data preparation and training commands. Benchmarks preserve actual requests, raw probabilities and actions, verify protocol consistency and compare each seed. Large raw trajectories remain locally in `runs/snake-training/`. This is imitation learning with explicit feature assistance; it does not demonstrate learning game rules from raw pixels, optimal planning or collision-free behavior in all situations.
