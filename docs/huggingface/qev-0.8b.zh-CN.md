---
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-0.8B
base_model_relation: adapter
language:
  - en
  - zh
tags:
  - qev
  - qwen3.5
  - lora
  - peft
  - pytorch
  - structured-decisions
  - multimodal
  - snake
---

[English](qev-0.8b.md) | [简体中文](qev-0.8b.zh-CN.md)

# Qev-0.8B — PyTorch 决策适配器

> [!NOTE]
> **自 Qev v0.4.0 起，默认模型改为 [twainsk/qev-450m-mlx](https://huggingface.co/twainsk/qev-450m-mlx)**（LFM2.5-VL-450M：下载 1.0 GB，Mac 决策快 2.1 倍，零样本图片准确率持平，通用文本 −2.8 个百分点），并同时提供纯文本的 [twainsk/qev-230m-mlx](https://huggingface.co/twainsk/qev-230m-mlx)。本 Qwen3.5 检查点继续发布、可用 `qev pull qev-0.8b` 安装；它仍是准确率最高、选项顺序最稳健的 Qev 模型，也是唯一接受视频决策输入的模型。


本仓库发布 **Qev v0.3.0 贪吃蛇专项续训 checkpoint**：它从早期 Qev 决策适配器出发，使用贪吃蛇监督数据和原通用决策任务回放继续训练。本地训练名称为 `qev-snake-0.8b`，API 模型别名仍为 `qev-0.8b`。这里提供的是新版 checkpoint，而非最初的 v0.2 权重。

Qev 在完整的 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 多模态基座上增加语言 LoRA 适配器和候选指针头，对给定答案选项评分，并通过 Jev / TypeSafe 风格 API 返回类型化决策。原生文字、图片和采样视频生成会关闭决策适配器，使用冻结基座。

**这是适配器 checkpoint，不是可独立运行的 Transformers pipeline 模型。** 请使用 [Qev 运行时](https://github.com/loadchange/qev)。首次运行会另行获取固定版本的基座。普通 `transformers.pipeline`、独立 PEFT 加载器或 `ollama run` 都无法加载完整的 Qev 决策系统。

对应的 Apple Silicon 完整导出为 [twainsk/qev-0.8b-mlx](https://huggingface.co/twainsk/qev-0.8b-mlx)。

## 快速开始

安装 [uv](https://docs.astral.sh/uv/) 和 Git，然后运行：

```bash
git clone https://github.com/loadchange/qev.git
cd qev
uv sync --python 3.12
uv run hf download twainsk/qev-0.8b --local-dir models/qev-snake-0.8b

# 运行终端贪吃蛇，PyTorch 自动选择 CUDA、MPS 或 CPU。
uv run qev snake --model models/qev-snake-0.8b

# 类型化决策示例与本地 HTTP 服务
uv run qev predict --model models/qev-snake-0.8b --request examples/request.json
uv run qev serve --model models/qev-snake-0.8b --port 8008
```

Qev 加载器接受本地路径：先下载仓库，再把本地目录传给 `--model`。请在克隆的项目目录中运行这些命令，或提供 checkpoint 与请求文件的绝对路径。已评估的 PyTorch 路径为 NVIDIA CUDA；Apple Silicon 用户建议使用 MLX 版本。

下载后运行 `uv run qev snake --model models/qev-snake-0.8b` 即可在终端观看模型逐步决策。空格暂停/继续，`N` 单步，`+` / `-` 调速，`Q` 或 Ctrl-C 退出。首次运行需要下载固定版本的 Qwen 基座。

服务默认监听 `127.0.0.1`。类型化决策使用 `POST /v1/systemone`；关闭适配器的原生生成使用 `POST /v1/chat/completions`。参见 [API 示例](https://github.com/loadchange/qev/blob/main/docs/API.zh-CN.md)。checkpoint 别名不会切换推理后端。

## 内容与运行要求

- `qev_config.json`：Qev 格式版本 2、架构、固定基座引用和校准温度。
- `adapter/adapter_model.safetensors` 和 `adapter/adapter_config.json`：未合并的 rank-16 LoRA 权重与配置。
- `pointer.safetensors`：训练得到的 256 维候选指针头。
- `tokenizer/` 和 `processor/`：文字及原生图片/视频预处理资源。
- `reports/`：数据来源与评估摘要；`release_manifest.json` 列出发布文件大小和 SHA-256 摘要。
- 根目录 `LICENSE` 与 `NOTICE`：许可及署名文档。不包含训练记录或自动生成的 PEFT 模板模型卡。

发布文件合计约 85.6 MB，**不包含**冻结基座权重。Qev 加载的基座为 `Qwen/Qwen3.5-0.8B`，revision 为 `2fc06364715b967f1860aea9cf38778875588b17`，请为该模型预留额外下载、存储和运行内存。离线使用前，应先把这一确切 revision 下载到 Hugging Face 缓存。

请使用 Python 3.12 或更高版本及 Qev 项目的依赖锁。已验证的软件系列为 Transformers 5.17、PEFT 0.21 和 PyTorch 2.10 或更高版本；训练使用 PyTorch 2.11.0 与 CUDA 12.8。所有决策适配器均与原基座分开保存。合并会改变原生生成路径，Qev 不支持这种用法。

## 训练

原基座的 852,985,920 个参数始终冻结。Qev 使用监督交叉熵继续训练已有的 11,346,944 个 LoRA 与指针参数；没有使用强化学习或 Jev 私有标签。

| 项目 | 数值 |
| --- | --- |
| Snake 训练 / 校准 / 开发问题数 | 12,000 / 500 / 500 |
| 回放的原训练 / 校准 / 开发问题数 | 5,892 / 620 / 880 |
| 合并后的训练 / 校准 / 开发问题数 | 17,892 / 1,120 / 1,380 |
| Snake 轨迹棋盘尺寸 | 6×6, 8×8, 12×12, 16×16 |
| 训练配置 | 2 轮，batch 8，累积 2，学习率 3e-5 |
| 计算资源 | NVIDIA A100 40 GB，2,238 次更新，约 1,485 秒优化时间 |
| 精度 | FP32 master 权重，CUDA BF16 autocast，FP32 pointer |
| 拟合的校准温度 | 2.82842712474619 |

Snake 标签来自确定性教师，它读取的显式字段与模型可见字段相同。环境提供碰撞和食物事实、静态 BFS 可达空间、尾部连通性、食物路径距离和近期访问次数。运行时输入不会加入教师方向、优选动作标记或排名。三个非反向候选包含可能碰撞的方向，运行时直接执行模型 argmax，没有安全接管。Snake 是**文字特征任务**；显示的棋盘不会作为图片传给模型。

数据清单 SHA-256 为 `f41151c68d6465b90bb8ea66ca0ea8611a6b37ed48596ba9484abc8ef4deda33`。完整来源、原数据许可、分区隔离与复现命令见 [训练文档](https://github.com/loadchange/qev/blob/main/docs/TRAINING.zh-CN.md)和 [Snake 模型说明](https://github.com/loadchange/qev/blob/main/docs/SNAKE_MODEL.zh-CN.md)。

## 评估

以下结果在续训前后使用相同 A100 和 CUDA BF16 运算：

| 开发集指标 | 问题数 | 父 checkpoint | 本 checkpoint |
| --- | ---: | ---: | ---: |
| 保留游戏种子上的 Snake 教师动作一致率 | 500 | 73.60% | 98.40% |
| 原有通用决策准确率 | 880 | 81.25% | 81.70% |

闭环评估使用保留种子、相同空间特征和每局 500 步上限。Torch 每批推进八个独立游戏。所有实际动作均由模型选择。

| 棋盘 / 保留种子 | 父 checkpoint 平均食物 | 本 checkpoint 平均食物 | 父 checkpoint 碰撞局数 | 本 checkpoint 碰撞局数 |
| --- | ---: | ---: | ---: | ---: |
| 8×8 / 10000–10019，共 20 局 | 3.10 | 42.90 | 20 / 20 | 0 / 20 |
| 12×12 / 10000–10007，共 8 局 | 0.625 | 42.00 | 8 / 8 | 0 / 8 |

新模型的 28 局均走到步数上限，没有填满棋盘。这些有限闭环测试不能保证永不碰撞或最优游戏策略。对应 MLX 导出在 Apple M4 上的五局 8×8 游戏平均吃到 42.6 个食物。数值精度、批大小和由动作决定的轨迹都可能改变结果。参见 [完整基准条件与证据](https://github.com/loadchange/qev/blob/main/docs/SNAKE_MODEL.zh-CN.md)。

## 多模态保留与限制

训练前后冻结基座的完整哈希一致。在同一 A100 上，关闭适配器的原生生成对文字、图片、视频三个固定探针给出完全相同的 token IDs。这提供权重保留和有限回归证据，**不是完整多模态质量评估**。之后的零样本探测（500 道 A-OKVQA 验证题，4 选 1）带图答对 69.4%，无图 32.6%；视频决策尚未测量，多模态决策概率仍未校准。

这是小规模监督决策实验，通用任务覆盖和中文任务证据有限。它没有证明与 Jev、TypeSafe 或 Laya 的效果相当；Jev 兼容指接口和答案类型。模型也没有直接从像素学习 Snake 几何。原通用模型、数据来源和其他限制见 [原始模型说明](https://github.com/loadchange/qev/blob/main/docs/MODEL_CARD.zh-CN.md)；本次发布应采用上面的 v0.3.0 成绩，而非原 checkpoint 的指标。

## 许可与署名

Qev 代码、适配器和指针权重使用 Apache-2.0 发布。Qwen3.5 基座由其作者另行按照 Apache-2.0 分发。请保留随附的 `LICENSE`、`NOTICE` 以及已有的上游署名文档。

回放的公开训练记录来自 `jaredpalmer/kev-suites` 及其来源数据集，其许可声明不同，含相同方式共享、other 和未声明条款。Apache 软件/模型声明不会为这些数据重新授予许可，也不解决其下游条款。本模型仓库不再分发训练数据集。参见 [NOTICE](https://github.com/loadchange/qev/blob/d68c468/NOTICE) 和 [数据来源清单](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/data_manifest.json)。Qev 是独立项目，不是 Jev 或 Qwen 官方发布。
