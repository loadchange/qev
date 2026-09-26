---
license: other
license_name: lfm-open-license-v1.0
license_link: LICENSE
base_model:
  - LiquidAI/LFM2.5-VL-450M
base_model_relation: adapter
language:
  - en
  - zh
tags:
  - qev
  - lfm2
  - mlx
  - mlx-vlm
  - apple-silicon
  - lora
  - structured-decisions
  - multimodal
---

[English](qev-450m-mlx.md) | [简体中文](qev-450m-mlx.zh-CN.md)

# Qev-450M-MLX — 面向 Apple Silicon 的 LFM2.5-VL 决策检查点

本仓库发布 **Qev v0.4.0 的默认模型**：`backbone/` 中是逐字节未改动的 **LiquidAI/LFM2.5-VL-450M** 底模，外加 [Qev 项目](https://github.com/loadchange/qev)训练的可切换决策 LoRA（`decision_adapters.safetensors`）和候选指针头（`pointer.safetensors`）。决策推理启用 LoRA；原生文本/图像生成关闭它并使用未修改的底模。

Qev 通过 Jev / TypeSafe 风格的 API（`POST /v1/systemone`）对给定候选项输出真实概率，而不是生成 JSON 文本。与早先的 Qwen3.5-0.8B 检查点（[twainsk/qev-0.8b-mlx](https://huggingface.co/twainsk/qev-0.8b-mlx)）相比，本模型在同一台 Mac 上体积缩小到约 30%，贪吃蛇决策快 2.1 倍，聊天生成约快 2 倍，代价是通用文本子集实测 −2.8 个百分点。纯文本的姊妹模型是 [twainsk/qev-230m-mlx](https://huggingface.co/twainsk/qev-230m-mlx)。

**请使用 Qev 运行时。** 通用 Transformers pipeline 或 `mlx_lm.generate` 不会加载指针头和可切换 adapter。

## Apple Silicon 快速开始

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull                 # 下载并校验本仓库（约 1.0 GB）到 ~/.qev
qev decide "结账时卡被拒两次" -i "哪个部门处理？" \
  --choice "billing=账单与退款" "technical=故障和宕机" sales
qev decide --image receipt.jpg -i "这是餐厅小票吗？" --noul
qev serve                # POST /v1/systemone、/v1/chat/completions，网页实验室在 /
```

源码方式：`uv sync --python 3.12 --extra mlx`，然后 `uv run hf download twainsk/qev-450m-mlx --local-dir models/qev-450m-mlx` 并传 `--model models/qev-450m-mlx`。要求：Apple Silicon、macOS 14+、Python 3.12+。在 MLX 0.32.2 与 mlx-vlm 0.7.1 上验证；图片走 mlx-vlm 的 numpy LFM2-VL 处理器，无需 PyTorch。

## 内容

- `qev_config.json` — Qev 格式版本 3，`family: lfm2`，`runtime: mlx`，模态 `["text", "image"]`。
- `decision_adapters.safetensors` — LoRA（rank 64，alpha 128），作用于所有语言层的 `q/k/v/out_proj`、`in_proj` 和 MLP `w1/w2/w3`，与底模分开存放。
- `pointer.safetensors` — 256 维候选打分头。
- `backbone/` — LiquidAI/LFM2.5-VL-450M，固定在 revision `fc6221ca…34ba`，与上游逐字节一致（bfloat16 safetensors、tokenizer、processor、聊天模板、许可证）。
- `reports/` — 评测、来源与 MLX 一致性摘要；`release_manifest.json` 列出文件大小和 SHA-256。

## 训练

在 Qev 决策数据集上做监督模仿：训练/校准/开发 17,892 / 1,120 / 1,380 道题（12,000 贪吃蛇 + 5,892 通用训练行；英文为主，另有四个中英双语合成规则族）。单张 NVIDIA A100 40 GB，4 个 epoch，batch 8 × 累积 2（4,476 步，约 18 分钟），学习率 1e-4 余弦衰减，BF16 autocast + FP32 主权重。底模参数全程冻结；只训练 24.5M 参数（LoRA + 指针头）。温度 `4.59479341998814` 在独立校准集上拟合。6 epoch / lr 2e-4 的变体明显更差（通用 59.1%），已弃用。

## 评测

开发集（1,380 题），PyTorch 参考；MLX 与其最多相差 2 次 argmax 翻转（见下）。

| 子集 | 准确率 |
|---|---|
| 全部 | 86.1% |
| 贪吃蛇（500） | 98.8% |
| 通用文本（880） | 78.9% |
| 中文（50） | 94.0% |

闭环贪吃蛇 20 局固定种子：平均吃到 41.4 个食物，**0 次碰撞**。选项反转探针：85.1%（−1.0pt），7.2% 的选择翻转。与 Qwen3.5-0.8B 检查点在完全相同题目上配对检验：通用 −2.8pt（McNemar p=0.024），贪吃蛇持平（p=0.63）；与上一版 LFM2.5-VL 配方相比：通用 +3.2pt（p=0.013）。

**图片决策是纯文本决策训练的零样本迁移**：在 500 道 A-OKVQA 验证题上，决策头带图回答正确 69.4%（不带图 37.2%；随机 25%），与 Qwen3.5 检查点的 69.4% 持平。文本温度不作用于图片题（温度 1.0）。

MLX 与 PyTorch 在全部 1,380 道开发题上的一致性：`adapter` 模式 2 次 argmax 翻转、最大概率差 0.051；`bf16` 2 次、0.038；净准确率不变。Apple M4 16 GB：贪吃蛇决策 p50 98 ms（`adapter`）/ 71 ms（`bf16`，Homebrew 服务默认），MLX 峰值内存 ≤1.9 GiB，原生聊天 `bf16` 约 100 token/s。

## 局限

- 决策 adapter 只用文本训练；图片决策为零样本实测（A-OKVQA 69.4%），且图片概率未校准。视频以有序帧输入可用，但未做决策评测。
- 中文训练覆盖只有四个合成规则族；50 题中文子集样本很小。
- 选项顺序敏感度（翻转 7.2%）高于 Qwen3.5 检查点（3.6%）。
- API 兼容不代表达到 Jev 模型的质量或置信度。

## 许可证

底模与本衍生作品按 **LFM Open License v1.0**（`LICENSE`）分发：年营收达到或超过 1,000 万美元的法律实体不获商用许可。决策 adapter 和指针头是新增文件；底模权重未被修改（`NOTICE`）。
