# Qev 贪吃蛇专项模型

`qev snake` 在终端加载真实 Qev 模型并逐步执行其最大概率动作。默认优先使用 `models/qev-snake-0.8b-mlx`；原 `models/qev-0.8b[-mlx]` 保留作对照。完整键位和参数见 [DEMO](DEMO.md)。

## 为什么重做

历史 v0.2 只接通了游戏接口：训练集没有 Snake 样本，输入仅含局部碰撞和曼哈顿距离，真实游戏会漏吃、绕圈和撞墙。新增终端界面之外，本次从既有 Qev adapter/head 继续监督训练，并增加明确可见的环境空间特征。

本地 Laya-MLX `fc1df62` 的 Snake 演示在 `laya_mlx/snake/policy.py` 中把规划器判定的 “Best progress toward food” 写进候选，默认安全 shield 还可以替换模型选择。因此它的演示分数不能直接与 Qev 的无接管模型分数比较。本次主要比较**相同 Qev 游戏环境、特征、种子和推理设置下的训练前后模型**，不宣称超过 Laya。

## 输入、监督和执行

`spatial` 模式给出三个非反向方向，包括可能发生碰撞的方向。候选包含碰撞、距离变化、吃食物，以及静态 BFS 可达空间、尾部连通性、食物路径距离和最近 32 步的访问次数。BFS 由环境计算，模型不必从棋盘像素自行推导几何；游戏棋盘仍不是图像输入。

明确的启发式教师只读取模型可见字段生成标签。它优先避碰、保留尾连通和空间，再结合食物路径和访问次数。训练时不输入 “最佳动作” 标记、教师方向、教师排名或隐藏棋盘元数据；运行时不调用教师、不剔除碰撞候选，也不替换模型动作。`local` 模式保留旧观察用于消融。

新增 Snake 训练 / 校准 / 开发样本数为 12,000 / 500 / 500。训练来自 251 个独立游戏种子，覆盖 6、8、12、16 四种尺寸，每局最多抽 48 个状态，行为含 10% 概率的合法随机偏离；候选顺序重排。原任务的 5,892 / 620 / 880 题全部按原分区回放。合并后共 17,892 / 1,120 / 1,380 题，逐题 token 审计无截断。游戏种子和规范化观察跨分区隔离；10000–10019 未参与训练轨迹。

A100 40GB 上从原 adapter/head 继续两轮，batch 8、accumulation 2、学习率 3e-5，2238 次更新，优化约 1484.69 秒。原始完整 Qwen3.5 基座冻结；新的 calibration 分区拟合温度 `T=2.82842712474619`。来源、数据摘要、冻结校验和完整评估见 [记录目录](results/snake_training)。

## 同条件评估

同一 A100、CUDA BF16、同一开发集的训练前后结果：

| 问题 | 数量 | 原模型 | 专项模型 |
| --- | ---: | ---: | ---: |
| 未见游戏种子的教师动作一致率 | 500 | 73.60% | 98.40% |
| 原有通用分类及程序规则准确率 | 880 | 81.25% | 81.70% |

教师一致率不等于整局成功率。以下闭环使用相同 `spatial` 特征，500 步上限，直接执行模型 argmax，无安全接管；Torch 以 8 局为批量推进独立游戏。

| 棋盘 / 保留种子 | 原模型平均食物 | 专项模型平均食物 | 原模型碰撞局数 | 专项模型碰撞局数 |
| --- | ---: | ---: | ---: | ---: |
| 8×8 / 10000–10019，共 20 局 | 3.10 | 42.90 | 20 | 0 |
| 12×12 / 10000–10007，共 8 局 | 0.625 | 42.00 | 8 | 0 |

新模型这 28 局全部走到 500 步上限，未填满棋盘。相同种子保证初始状态相同；动作不同会导致后续食物位置与轨迹不同。这里检验的是闭环行为，不是每步相同状态下的分类。批处理与单局、CUDA BF16 与 MLX FP32 的浮点差异可能改变选择与后续轨迹。

### 本机 MLX

Apple M4、16 GiB 内存、MLX FP32，8×8、相同保留种子 10000–10004、逐局推理、每局 500 步上限：

| 指标 | 原模型 | 专项模型 |
| --- | ---: | ---: |
| 平均吃到的食物 | 3.0 | 42.6 |
| 各局食物数 | 4 / 1 / 5 / 1 / 4 | 37 / 45 / 44 / 38 / 49 |
| 碰撞局数 | 5 / 5 | 0 / 5 |
| 走到 500 步上限 | 0 / 5 | 5 / 5 |

五个种子均有提高，新模型没有饥饿结束或填满棋盘。2500 次真实决策的延迟中位数为 223.84 ms、P95 为 240.21 ms，包含游戏和逐步记录的速度约 4.40 步/秒。原模型通过本机 HTTP 调用同一个 MLX Agent，新模型直接调用本机 MLX Agent；比较的是游戏行为，不据此比较传输延迟。协议核对了相同规则、输入、候选次序与特征源码。原始概率、选择与执行动作逐步留档，全部执行 argmax，无接管。见[原模型记录](results/snake_training/parent-mlx-spatial.json)和[专项模型记录](results/snake_training/trained-mlx-spatial.json)。

## 多模态保留与交付验证

训练始终冻结完整 Qwen3.5 基座。原生生成关闭决策 adapter，继续使用原始语言、视觉权重和生成头。在同一 A100 上顺序加载旧、新 checkpoint，文字、图片、视频三个固定样例的输出 token IDs 均完全一致；冻结参数的完整哈希也一致。见[原生能力对照](results/snake_training/snake-native.json)。这提供有限样例和权重保留证据，不是完整多模态准确率评测。

MLX 导出复用了与旧版字节相同的完整多模态基座；372 个 adapter 张量逐个与新 Torch checkpoint 核对相等，指针头文件也一致，见[导出校验](results/snake_training/export_integrity.json)。本次没有重新执行 Torch CPU FP32 与 MLX FP32 的 logits 数值对照，导出配置如实保留 `parity_status: not_run`；不宣称 CUDA BF16 与 MLX FP32 概率完全一致。

新 MLX checkpoint 通过了 [19 项真实 HTTP / TypeSafe SDK 检查](results/snake_training/service_mlx.json)，覆盖文字、图片、视频、混合媒体决策、原生生成、适配器恢复及错误请求。终端暂停、单步、退出和光标恢复经过[真实 PTY 验证](results/snake_training/terminal_controls.json)；默认路径也实际加载新模型完成冒烟运行。148 项测试、构建与产物摘要见[交付检查](results/snake_training/release_checks.json)。新旧 checkpoint 分开保存，原文件未覆盖，Colab A100 会话已停止。

## 复现

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

完整数据准备与训练命令见 [TRAINING](TRAINING.md)。基准会保存实际请求、原始概率与动作，检查协议一致后逐种子比较；原始轨迹体积较大，保存在本地 `runs/snake-training/`。这是一轮显式特征辅助的模仿学习，不能证明模型具备从原始像素学习游戏规则、最优规划或永不碰撞的能力。
