---
license: other
license_name: lfm-open-license-v1.0
license_link: LICENSE
base_model:
  - LiquidAI/LFM2.5-230M
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
---

[English](qev-230m-mlx.md) | [简体中文](qev-230m-mlx.zh-CN.md)

# Qev-230M-MLX — 面向 Apple Silicon 的纯文本极速决策检查点

本仓库发布 Qev **最小、最快的纯文本模型**：`backbone/` 中是逐字节未改动的 **LiquidAI/LFM2.5-230M** 底模，外加 [Qev 项目](https://github.com/loadchange/qev)训练的可切换决策 LoRA 和候选指针头。在 Apple M4 上一步贪吃蛇决策 **43 ms**，MLX 峰值内存 **1.2 GiB**，下载约 **0.5 GB**——但它**只接受文本**：图片或视频请求会得到明确报错。默认的多模态模型是 [twainsk/qev-450m-mlx](https://huggingface.co/twainsk/qev-450m-mlx)。

Qev 通过 Jev / TypeSafe 风格的 API（`POST /v1/systemone`）对给定候选项输出真实概率，而不是生成 JSON 文本。决策推理启用 LoRA；原生聊天（`/v1/chat/completions`）关闭它并使用未修改的底模（M4 上约 170 token/s）。

**请使用 Qev 运行时。** 通用 Transformers pipeline 或 `mlx_lm.generate` 不会加载指针头和可切换 adapter。

## Apple Silicon 快速开始

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull qev-230m        # 下载并校验本仓库（约 0.5 GB）到 ~/.qev
QEV_MODEL=qev-230m qev decide "客户要求退还重复扣款" -i "是否要求退款？" --noul -q
QEV_MODEL=qev-230m qev serve
```

源码方式：`uv sync --python 3.12 --extra mlx`，然后 `uv run hf download twainsk/qev-230m-mlx --local-dir models/qev-230m-mlx` 并传 `--model models/qev-230m-mlx`。要求：Apple Silicon、macOS 14+、Python 3.12+。在 MLX 0.32.2 与 mlx-vlm 0.7.1 上验证；无需 PyTorch。

## 内容

- `qev_config.json` — Qev 格式版本 3，`family: lfm2`，`runtime: mlx`，模态 `["text"]`。
- `decision_adapters.safetensors` — LoRA（rank 64，alpha 128），作用于全部 14 层的 `q/k/v/out_proj`、`in_proj` 和 MLP `w1/w2/w3`，与底模分开存放。
- `pointer.safetensors` — 256 维候选打分头。
- `backbone/` — LiquidAI/LFM2.5-230M，固定在 revision `40cb2ad3…fa45`，与上游逐字节一致。
- `reports/` — 评测、来源与 MLX 一致性摘要；`release_manifest.json` 列出文件大小和 SHA-256。

## 训练

与 qev-450m 相同的配方和数据：训练/校准/开发 17,892 / 1,120 / 1,380 道题，单张 A100 40 GB，4 个 epoch，batch 8 × 累积 2（4,476 步，约 16 分钟），学习率 1e-4 余弦衰减，底模冻结，训练 16.1M 参数，温度 `4.59479341998814` 在校准集上拟合。

## 评测

开发集（1,380 题），PyTorch 参考；MLX 与其最多相差 6 次 argmax 翻转（见下）。

| 子集 | 准确率 |
|---|---|
| 全部 | 85.2% |
| 贪吃蛇（500） | 99.0% |
| 通用文本（880） | 77.4% |
| 中文（50） | 90.0% |

闭环贪吃蛇 20 局固定种子：平均吃到 43.4 个食物，**0 次碰撞**。与 Qwen3.5-0.8B 检查点在完全相同题目上配对检验：通用 −4.3pt（McNemar p=0.002），贪吃蛇持平（p=1.0）。

MLX 与 PyTorch 在全部 1,380 道开发题上的一致性：`adapter` 6 次 argmax 翻转（最大概率差 0.059），`bf16` 6 次（0.062），净准确率不变。Apple M4 16 GB：贪吃蛇决策 p50 60 ms（`adapter`）/ 43 ms（`bf16`），MLX 峰值内存 ≤1.2 GiB，原生聊天 `bf16` 约 170 token/s。

## 局限

- **仅文本。** 图片和视频输入会被拒绝（`qev-230m accepts text only`）；需要图片请用 qev-450m。
- **选项顺序敏感度较高**：反转选项顺序会翻转 16.4% 的选择、损失 8.1pt（85.3% → 77.2%），而 qev-450m 为 7.2% / −1.0pt，Qwen3.5 检查点为 3.6% / 持平。对选项顺序稳定性有要求时请选 qev-450m。
- 中文训练覆盖只有四个合成规则族；50 题中文子集样本很小。
- API 兼容不代表达到 Jev 模型的质量或置信度。

## 许可证

底模与本衍生作品按 **LFM Open License v1.0**（`LICENSE`）分发：年营收达到或超过 1,000 万美元的法律实体不获商用许可。决策 adapter 和指针头是新增文件；底模权重未被修改（`NOTICE`）。
