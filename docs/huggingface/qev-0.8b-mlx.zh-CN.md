---
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-0.8B
base_model_relation: finetune
language:
  - en
  - zh
tags:
  - qev
  - qwen3.5
  - mlx
  - mlx-vlm
  - apple-silicon
  - lora
  - structured-decisions
  - multimodal
  - snake
---

[English](qev-0.8b-mlx.md) | [简体中文](qev-0.8b-mlx.zh-CN.md)

# Qev-0.8B-MLX — 完整多模态 FP32 checkpoint

本仓库发布 **面向 Apple Silicon 的 Qev v0.3.0 贪吃蛇专项续训 checkpoint**。它包含完整转换的 Qwen3.5-0.8B 基座、独立决策 LoRA 权重、候选指针头和文字/图片/视频预处理资源。本地训练/导出名称为 `qev-snake-0.8b-mlx`，API 模型别名仍为 `qev-0.8b`。这里提供的是较新的续训 checkpoint，而非最初的 v0.2 权重。

Qev 通过 Jev / TypeSafe 风格 API 对给定答案选项评分。决策推理启用 LoRA 适配器；原生文字、图片和采样视频生成关闭适配器，使用未改动的基座。这是**完整 FP32 导出**，不是 4-bit 模型，也不是仅含适配器的下载。对应的 PyTorch 适配器 checkpoint 为 [twainsk/qev-0.8b](https://huggingface.co/twainsk/qev-0.8b)。

**请使用 [Qev 运行时](https://github.com/loadchange/qev)。** 仓库顶层是自定义 Qev checkpoint：通用 Transformers pipeline、`mlx_lm.generate` 或 `ollama run` 无法加载其指针头和可切换适配器。MLX 使用 Apple Silicon 的 CPU/GPU 和统一内存；这一后缀不表示 Apple Neural Engine 执行。

## Apple Silicon 快速开始

安装 [uv](https://docs.astral.sh/uv/) 和 Git，然后运行：

```bash
git clone https://github.com/loadchange/qev.git
cd qev
uv sync --python 3.12 --extra mlx
uv run hf download twainsk/qev-0.8b-mlx --local-dir models/qev-snake-0.8b-mlx

# 直接执行模型决策的终端贪吃蛇
uv run qev snake --model models/qev-snake-0.8b-mlx

# 类型化决策示例与本地 HTTP 服务
uv run qev predict --model models/qev-snake-0.8b-mlx --request examples/request.json
uv run qev serve --model models/qev-snake-0.8b-mlx --port 8008
```

Qev 加载器接受本地路径。请下载完整仓库并保留 `backbone/` 子目录。在克隆的项目目录中运行这些命令，或提供 checkpoint 和请求文件的绝对路径。MLX checkpoint 已包含基座，推理无需另行下载基座模型。

这是 Mac Apple Silicon 的完整 FP32 模型。下载后运行 `uv run qev snake`，终端默认优先查找 `models/qev-snake-0.8b-mlx`。空格暂停/继续，`N` 单步，`+` / `-` 调速，`Q` 或 Ctrl-C 退出。模型每步自行选择方向，界面展示候选概率和实际动作。

服务默认监听 `127.0.0.1`。类型化决策使用 `POST /v1/systemone`；关闭适配器的原生生成使用 `POST /v1/chat/completions`。参见 [API 示例](https://github.com/loadchange/qev/blob/main/docs/API.zh-CN.md)。checkpoint 别名不会切换推理后端。

## 内容与运行要求

- `qev_config.json`：Qev 格式版本 2、`runtime: mlx`、`backend: mlx_vlm`、FP32 导出元数据与温度。
- `decision_adapters.safetensors`：单独转换的 LoRA 张量，未合并到基座。
- `pointer.safetensors`：训练得到的候选评分器。
- `backbone/`：完整转换的视觉/语言权重、模型配置、tokenizer、原生图片/视频 processor 资源、基座清单和上游许可文档。
- `reports/`：数据来源与评估摘要；`release_manifest.json` 列出发布文件大小和 SHA-256 摘要。
- 根目录许可和署名文档。不包含训练记录。

加入发布元数据前，checkpoint 约为十进制 **3.48 GB**，其中基座张量文件为 3.41 GB。请保留完整目录结构。此 MLX 路径要求 Apple Silicon macOS 和 Python 3.12 或更高版本。请使用项目的依赖锁：本次导出和验证使用 MLX 0.32.2 与 mlx-vlm 0.7.1。软件包还会安装 Transformers/PEFT/PyTorch，用于共用预处理和其他 Qev 路径。

游戏在 Apple M4、16 GiB 统一内存上测试。这是已测配置，不是对所有媒体大小或并行应用场景保证的最低要求。完整 FP32 格式有意使用比低比特导出更多的内存。本次不宣称发布了新的低精度版本。

## 训练与导出

源适配器从早期 Qev 决策模型继续训练，新增训练/校准/开发集 **12,000 / 500 / 500 条 Snake 问题**，并按原分区回放全部 **5,892 / 620 / 880 条原通用决策问题**。合并后为 17,892 / 1,120 / 1,380 条。原始基座参数始终冻结，只优化语言 LoRA 和指针头。

训练使用 NVIDIA A100 40 GB、两轮、batch 8、累积 2、学习率 3e-5，共 2,238 次更新，约 1,485 秒优化时间。FP32 master 权重搭配 CUDA BF16 autocast 与 FP32 pointer。单独的校准集拟合温度 `2.82842712474619`。这是监督模仿与任务回放，没有使用强化学习。

基座为 `Qwen/Qwen3.5-0.8B`，固定 revision 为 `2fc06364715b967f1860aea9cf38778875588b17`。导出保留完整视觉编码器、语言模型和词表投影，复用已验证的 FP32 基座转换，并把 LoRA 保留为可切换张量。全部 372 个转换后的适配器张量均与训练后的 PyTorch checkpoint 核对，指针文件逐字节一致。参见 [导出完整性报告](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/export_integrity.json)。

**本次续训没有重新执行 Torch CPU FP32 与 MLX FP32 的 logit 数值对照。** 导出配置明确记录 `parity_status: not_run`。适配器转换/完整性检查和真实 MLX 游戏结果不能证明不同运行时的概率完全一致。

完整训练详情与证据见 [Snake 模型说明](https://github.com/loadchange/qev/blob/main/docs/SNAKE_MODEL.zh-CN.md)和 [训练指南](https://github.com/loadchange/qev/blob/main/docs/TRAINING.zh-CN.md)。

## 真实 MLX 游戏结果

Apple M4、16 GiB 内存、FP32、逐局推理、8×8 棋盘、保留种子 10000–10004、每局 500 步上限：

| 指标 | 父 checkpoint | 本 checkpoint |
| --- | ---: | ---: |
| 平均吃到的食物 | 3.0 | 42.6 |
| 各种子的食物数 | 4 / 1 / 5 / 1 / 4 | 37 / 45 / 44 / 38 / 49 |
| 碰撞局数 | 5 / 5 | 0 / 5 |
| 走到 500 步上限的局数 | 0 / 5 | 5 / 5 |

新模型的 2,500 次决策延迟中位数为 223.84 ms，P95 为 240.21 ms。包含游戏处理和轨迹记录的整体速度约 4.40 步/秒。父模型通过本机 HTTP 调用同一个 MLX Agent，新模型直接调用本机 Agent，因此比较的是游戏行为，不是传输延迟。参见 [父模型报告](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/parent-mlx-spatial.json)和 [新 checkpoint 报告](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/trained-mlx-spatial.json)。

环境提供显式文字特征，包括静态 BFS 可达空间、尾部连通性、食物路径距离和近期访问次数。运行时不提供教师方向或优选动作标签，不删除碰撞候选，也不替换模型选择。所有实际动作都是模型 argmax。**Snake 棋盘不是模型的视觉输入。** 因此，这一评估测量的是显式特征辅助下的决策能力，不是基于像素的游戏理解。

源 PyTorch checkpoint 在 20 局保留种子的 8×8 游戏中，平均食物数从 3.10 提升到 42.90；在八局 12×12 游戏中从 0.625 提升到 42.00，训练后模型的这 28 局都没有碰撞。在同一 A100 上，500 条 Snake 问题的教师动作一致率从 73.60% 提升到 98.40%，原有 880 条开发问题的准确率从 81.25% 变为 81.70%。这些是 PyTorch CUDA BF16 结果，不表示 MLX 导出复现了每一项预测。

## 多模态保留与限制

原生生成会关闭决策适配器。训练没有改变原基座哈希，MLX 基座张量文件与父版本 MLX 导出逐字节相同。同一 A100 上的文字/图片/视频探针在续训前后生成相同 token IDs。新 MLX checkpoint 通过了 [19 项真实 HTTP/SDK 检查](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/service_mlx.json)，包括媒体处理、原生生成、类型化决策和适配器恢复。

这些检查提供实现和有限回归证据，**不是完整视觉/视频质量基准**。多模态决策准确率尚未测量，媒体决策概率仍未校准。通用决策与中文训练覆盖有限。有限 Snake 成绩不能保证最优游戏策略或永不碰撞；所有报告的游戏都未填满棋盘。

Qev 提供 Jev 兼容的接口约定，不包含 Jev 私有权重，也没有效果相当的证据。本次没有直接宣称优于 Laya。硬件、精度和批处理都可能改变决策与轨迹。参见 [完整条件与限制](https://github.com/loadchange/qev/blob/main/docs/SNAKE_MODEL.zh-CN.md)。

## 许可与署名

Qev 代码、适配器和指针权重使用 Apache-2.0 发布。包含的 Qwen3.5 基座仍受其上游 Apache-2.0 许可和署名要求约束；请同时保留 `backbone/LICENSE`、`backbone/BASE_MODEL_README.md` 及根目录 `LICENSE`、`NOTICE`。

训练回放的公开记录来自 `jaredpalmer/kev-suites` 及多个许可声明各异的数据集，含相同方式共享、other 和未声明条款。Apache 软件/模型声明不会为这些数据集重新授予许可，也不解决其下游条款。本仓库不再分发训练数据集。参见 [NOTICE](https://github.com/loadchange/qev/blob/d68c468/NOTICE)和 [数据来源清单](https://github.com/loadchange/qev/blob/d68c468/docs/results/snake_training/data_manifest.json)。Qev 独立于 Jev、Qwen 和 MLX 作者。
