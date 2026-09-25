# 扩散与自回归骨干在 Qev 决策上的对比

[English](README.md) | **简体中文**

实验于 2026-09-25 完成。要回答的问题：像 [djev](https://github.com/mmastrac/djev) 那样使用扩散语言模型（DiffusionGemma 加一步结构化读取），能否让 Qev 效果更好？Qev 能否在保留多模态输入的同时变得更小、更快？

## 结论

- **扩散没有带来准确率提升。** 在相同尺寸、数据和训练配方下，掩码扩散版 Qwen3-0.6B（dLLM 的 MDLM 权重，双向块读取）在通用决策上比它的自回归原版低 3.0 个百分点（78.0% vs 80.9%，精确 McNemar p = 0.017），贪吃蛇持平。不经扩散预训练、直接让自回归权重做双向读取，会下降 18.9 个百分点。
- **djev 的模型本身在 Qev 的任务上也不更好。** 零样本 DiffusionGemma-26B-A4B（25.2B 参数，按 djev-run 的读取方式）在通用题上与训练过的 0.8B Qev 持平（81.2% vs 81.8%，p = 0.73），贪吃蛇比所有训练过的组低约 14 个百分点（84.6% vs 98.6%–99.2%）。
- **djev 真正做对的是"一次前向读出所有答案"，而这不需要扩散模型。** 把多道题打包在同一个共享 state 上一次前向，结果与逐题独立计算完全一致（fp32 差异 ≤ 6e-12）。但只有共享 state 很长（图片、长文档）时才划算；贪吃蛇这种长度的 state，用稠密掩码打包在 GPU 上反而比普通批处理慢。
- **更小、更快且仍然多模态：LFM2.5-VL-450M。** 参数量约为 Qwen3.5-0.8B 的一半。在 Apple Silicon 上、同等精度且合并 LoRA 时快 2.1 倍（每步贪吃蛇决策 fp32 83ms vs 178ms），比现在的线上路径快 3.3 倍（bf16 66ms vs 218ms）。贪吃蛇表现（98.6%，每局 43.1 个食物，0 碰撞）和图像决策（A-OKVQA 69.6% vs 70.2%）持平；通用文本决策低 6.0 个百分点（p = 8e-6），集中在偏推理的任务上。它的许可证（LFM Open License v1.0）禁止年收入 1000 万美元及以上的机构商用。

## 实验设置

所有组使用相同的数据、指针头和优化器，只改变骨干和决策时的注意力模式。

- 数据：`data/snake-v1`，训练 17,892 / 校准 1,120 / 开发 1,380 题（500 道贪吃蛇 + 880 道通用题和中英文规则题）。
- 配方：在全部语言线性层上加 rank-16 LoRA（alpha 32，dropout 0.05），冻结底模，256 维指针头，AdamW lr 1e-4，weight decay 0.01，10% warmup 后余弦衰减，2 个 epoch，batch 8 × 累积 2，fp32 主权重加 bf16 autocast，随机种子 42，在校准集上拟合一个温度。
- 注意力：`causal` 是骨干自带的因果掩码。`block` 是扩散式读取：state 内部双向注意；每道题可以看到整个 state，题内部双向注意。state 永远看不到题目，所以可以被多道题精确共享。
- 硬件：训练、GPU 计时和探针用 Colab A100-SXM4-40GB；Mac 延迟用 Apple M4 16GB（MLX 0.32.2）。

| 组 | 骨干（固定版本见 `qevd.py`） | 注意力 | 要回答的问题 |
|---|---|---|---|
| `qwen35-causal` | Qwen3.5-0.8B（现有 Qev） | 因果 | 统一配方下的基线 |
| `qwen3-causal` | Qwen3-0.6B 自回归 | 因果 | 对照组 |
| `a2d-block` | Qwen3-0.6B 掩码扩散（`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`） | 块双向 | 扩散模型有没有帮助？ |
| `qwen3-block` | Qwen3-0.6B 自回归 | 块双向 | 只开双向读取够不够？ |
| `lfm2-causal` | LFM2.5-VL-450M | 因果 | 更小且多模态 |
| `lfm2-block` | LFM2.5-VL-450M | 块双向 | 更小、多模态、扩散式读取 |

Qwen3.5 基线不能用 `block`：它的 Gated DeltaNet 层是循环结构。

## 结果

开发集，已校准。闭环贪吃蛇：20 个预留种子（10000–10019），8×8 棋盘，500 步上限，直接执行 argmax，无安全护栏——与已发布结果的协议相同。

| 组 | 参数量 | 开发集 | 贪吃蛇 | 通用 | NLL | ECE | 每局食物 | 碰撞 | GPU 每次决策 ms（分离 adapter / 合并） |
|---|---|---|---|---|---|---|---|---|---|
| 已发布 `qev-snake-0.8b`（两阶段） | 0.86B | 87.8% | 98.4% | 81.7% | 0.355 | 0.029 | 42.9 | 0 | — |
| `qwen35-causal` | 0.86B | **88.0%** | 99.2% | **81.7%** | **0.319** | 0.021 | 43.3 | 0 | 101 / 64 |
| `qwen3-causal` | 0.61B | 87.4% | 98.8% | 80.9% | 0.347 | 0.019 | 43.5 | 0 | 84 / 43 |
| `a2d-block` | 0.61B | 85.7% | 99.2% | 78.0% | 0.371 | 0.018 | 43.9 | 0 | 88 / 48 |
| `qwen3-block` | 0.61B | 75.4% | 98.8% | 62.0% | 0.647 | 0.025 | 42.6 | 0 | 86 / 47 |
| `lfm2-causal` | 0.46B | 84.0% | 98.6% | 75.7% | 0.433 | 0.019 | 43.1 | 0 | **39 / 20** |
| `lfm2-block` | 0.46B | 84.1% | 99.2% | 75.5% | 0.416 | 0.016 | 42.7 | 0 | 38 / 20 |
| djev 零样本¹ | 25.2B | 82.5% | 84.6% | 81.2% | 0.513 | 0.015 | — | — | 161（HTTP） |

¹ DiffusionGemma-26B-A4B NVFP4，用 vLLM nightly 按 djev-run 的参数和提示词部署，一步只读去噪，取 top-20 logprobs，温度在校准集上拟合。跳过了 60 道 Banking77 题（77 个选项超过它 62 个单 token 标签的上限），所以它的通用准确率覆盖其余 820 题。

相同题目上的精确 McNemar 检验（`paired_tests.py`）：

| 对比 | 通用（880） | 贪吃蛇（500） |
|---|---|---|
| 扩散 `a2d-block` vs 自回归 `qwen3-causal` | 78.0% vs 80.9%，p = 0.017 | 99.2% vs 98.8%，p = 0.63 |
| `qwen3-block` vs `qwen3-causal` | 62.0% vs 80.9%，p = 2e-28 | 98.8% vs 98.8%，p = 1 |
| `a2d-block` vs `qwen3-block`（扩散预训练的作用） | 78.0% vs 62.0%，p = 3e-20 | p = 0.63 |
| `lfm2-causal` vs `qwen35-causal` | 75.7% vs 81.7%，p = 8e-6 | 98.6% vs 99.2%，p = 0.38 |
| `lfm2-block` vs `lfm2-causal` | 75.5% vs 75.7%，p = 0.91 | 99.2% vs 98.6%，p = 0.25 |
| `qwen3-causal` vs `qwen35-causal` | 80.9% vs 81.7%，p = 0.55 | 98.8% vs 99.2%，p = 0.50 |
| djev vs `qwen35-causal`（不含 Banking77 的 820 道通用题） | 81.2% vs 81.8%，p = 0.73 | 84.6% vs 99.2%，p = 2e-22 |

LFM2 比 Qwen3.5 少答对 53 道通用题：19 道在程序化规则题（共 100 道；例如英文会员规则 41.7% vs 100%），MNLI 9 道，BoolQ 和 Banking77 各 6 道。每个规则族只有 12–13 道开发题。

### 图像决策（零样本迁移）

所有组的决策训练都没见过图片。A-OKVQA 验证集前 500 题，4 选 1（`mm_probe.py`）：

| 检查点 | 决策头 + 图片 | 决策头、无图 | 冻结底模字母 logits | GPU 每次决策 ms |
|---|---|---|---|---|
| `lfm2-causal` | 69.6% | 35.2% | 75.8% | 68 |
| `lfm2-block` | 66.6% | 30.8% | 75.8% | 68 |
| `qwen35-causal` | 70.2% | 33.8% | 71.4% | 153 |
| 已发布 `qev-snake-0.8b` | 69.4% | 32.6% | 71.4% | 151 |

只用文本训练的决策头确实在利用图片（比无图高约 35 个百分点），与底模自身做选择题的能力相差 1–9 个百分点。

### 候选顺序鲁棒性

把每道开发题的选项顺序反转后重新打分（`order_probe.py`）：

| 检查点 | 选择改变 | 准确率 原顺序 / 反转 | Banking77 改变 |
|---|---|---|---|
| `qwen35-causal` | **3.6%** | 88.0% / 88.0% | 13.3% |
| 已发布 `qev-snake-0.8b` | 5.4% | 87.8% / 87.2% | 20.0% |
| `qwen3-causal` | 8.3% | 87.4% / 87.0% | 23.3% |
| `a2d-block` | 7.0% | 85.7% / 85.0% | 26.7% |
| `qwen3-block` | 36.4% | 75.4% / 62.1% | 50.0% |
| `lfm2-causal` | 10.0% | 84.1% / 81.3% | 26.7% |
| `lfm2-block` | 7.6% | 84.1% / 82.5% | 11.7% |

块读取让小模型 LFM2 对选项顺序更不敏感，但没有改变它的准确率。

### Apple Silicon 上的速度

每步贪吃蛇决策，40 个开发集状态，p50 毫秒，Apple M4 16GB（`mlx_latency.py`、`mlx_qev_adapters.py`）。候选模型按 LoRA 已合并计时（计算量不变）；Qev 各行使用线上运行时。

| 模型 | fp32 | bf16 | 8bit |
|---|---|---|---|
| Qev Qwen3.5-0.8B，可切换 LoRA（线上） | 218 | 189 | 190 |
| Qev Qwen3.5-0.8B，关闭 adapter（合并 LoRA 的下限） | 178 | 143 | — |
| Qwen3-0.6B | 130 | 105 | 107 |
| LFM2.5-VL-450M | **83** | **66** | 72 |

8bit 量化对这种约 350 token 的预填充没有提速。块掩码比因果掩码多 0.6–8.9ms。

### 同一 state 上的多道题（GPU）

一个贪吃蛇 state 配 16 道题，合并 LoRA，p50 毫秒：`qwen3-causal` 逐题 699、批处理 85、打包 135；`lfm2-causal` 逐题 320、批处理 43。打包节省了 18% 的 token，结果与独立计算完全一致，但 4,320×4,320 的稠密掩码让它比批处理更慢。在高速 GPU 上这些小模型受 kernel 启动开销主导：合并 LoRA 能把单次决策延迟降低 35%–50%。

## 对 Qev 的意义

1. 保留自回归骨干。只在共享 state 很长的场景（例如同一张图上问多道题）借鉴 djev 的一次读取思路，用共享 state 缓存或块稀疏注意力实现，而不是稠密的打包掩码。
2. 提速：把决策 LoRA 合并进一份单独的决策权重（原生生成仍保留可切换 adapter），Mac 上 fp32 从 218ms 降到 178ms，无需重新训练。
3. 更小的多模态模型：如果贪吃蛇/游戏和图像决策比通用文本规则更重要，并且许可证合适，LFM2.5-VL-450M 可行。要补上 6 个百分点的文本差距需要更多或更好的文本训练；图像决策可以加入真正的多模态决策数据来提升。
4. 如果多模态不是硬性要求，Qwen3-0.6B 与 Qwen3.5-0.8B 持平（p = 0.55），在 Mac 上同等精度、合并 LoRA 时快 1.4 倍，但它没有视觉编码器。

## 局限

- 每组只有一个随机种子。开发题来自训练中见过的任务族；中文只覆盖四个程序化规则族。闭环贪吃蛇已经饱和（没有任何组碰撞；除了 `qwen3-block` 有一局饿死，其余全部跑满步数），无法区分各组。
- 已发布的 `qev-snake-0.8b` 用了两阶段训练；这里各组是同一套单阶段配方。
- 图像结果是单一数据集上的零样本迁移，没有做图像决策训练。
- MLX 候选计时使用合并后、未训练的权重，只衡量速度。LFM2 决策 adapter 的 MLX 导出尚未实现。
- GPU 计时来自三个宿主 CPU 不同的 Colab A100 会话，对 kernel 启动开销敏感；跨会话比较需谨慎。
- djev-run 的读取方式把每题限制在 62 个单 token 标签以内，top-20 以外的标签用下限值填充，这里完全照搬。

## 复现

```bash
uv run pytest -q tests/test_qevd_experiment.py
# 训练一组（Colab A100 或任何支持 bf16 的 CUDA GPU）：
python experiments/diffusion/train_qevd.py --backbone lfm2-vl --attention causal \
  --label lfm2-causal --out runs/qevd/lfm2-causal
# 在训练好的检查点上跑探针：
python experiments/diffusion/mm_probe.py --arm lfm2-causal=qevd:runs/qevd/lfm2-causal \
  --arm qev-snake=qev:models/qev-snake-0.8b --output runs/qevd/mm-probe.json
python experiments/diffusion/order_probe.py --arm lfm2-causal=runs/qevd/lfm2-causal --output runs/qevd/order.json
python experiments/diffusion/paired_tests.py --runs runs/qevd --output runs/qevd/paired.json
# Apple Silicon 延迟：
uv run python experiments/diffusion/mlx_latency.py --output runs/qevd/mlx-latency.json
uv run python experiments/diffusion/mlx_qev_adapters.py models/qev-snake-0.8b-mlx runs/qevd/mlx-qev-adapters.json
```

Colab 编排用到 `package.py`、`colab_setup.py`、`colab_extract.py`、`colab_launch.py`（组队列）、`colab_poll.py`、`colab_collect.py` 和 `colab_djev.py`（vLLM nightly 加 `djev_zeroshot.py`）。colab CLI（≤ 0.7.2）不会刷新 1 小时有效的运行时 token，会把仍在运行的会话清理掉；长任务期间请每 20 分钟左右用 CLI 自带的解释器运行一次 `colab_refresh.py`。

报告和逐题结果在 [`docs/results/diffusion`](../../docs/results/diffusion)，`report.py` 可以据此重建上面的表格。6 个检查点（LoRA adapter、指针头和各组报告）存档在 Hugging Face 私有仓库 `twainsk/qev-diffusion-experiment`；只能用本目录的代码（`QevDModel.from_checkpoint`）加载，`qwen35-causal` 则用 `qev.model.QevModel` 加载。`publish_archive.py` 可以重建该存档并核验每个上传文件（[发布记录](../../docs/results/diffusion/huggingface_archive.json)）。
