---
license: other
license_name: apache-2.0-and-lfm1.0
license_link: https://huggingface.co/twainsk/qev-diffusion-experiment/blob/main/LICENSES.md
base_model:
- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3-0.6B
- dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
- LiquidAI/LFM2.5-VL-450M
library_name: peft
tags:
- qev
- decision-model
- lora
- diffusion
- experiment
---

# Qev 扩散实验检查点

[English](README.md) | **简体中文**

2026-09-25 实验的私有研究存档。该实验比较了扩散骨干与自回归骨干在 Qev 结构化决策上的表现。这些是实验产物，不是 Qev 正式版本：不能在 `qev snake`、`qev serve` 或 MLX 运行时中使用。

报告、代码和完整结果：commit `664e6764d218d2e1a2baa4a40355a88d6d4d8501` 下的 [`experiments/diffusion`](https://github.com/loadchange/qev/tree/664e6764d218d2e1a2baa4a40355a88d6d4d8501/experiments/diffusion)。

## 检查点

每个目录包含冻结底模之上的 rank-16 LoRA adapter（`adapter/`）、256 维指针头（`pointer.safetensors`）、含拟合温度的配置，以及该组的报告：开发集评估、GPU 延迟、来源记录、闭环贪吃蛇汇总、训练日志和逐题开发集结果。不包含底模权重；加载时会按固定版本下载。

| 目录 | 底模（版本） | 决策注意力 | 开发集 | 贪吃蛇 | 通用 | 许可证 |
|---|---|---|---|---|---|---|
| `qwen35-causal` | Qwen/Qwen3.5-0.8B（`2fc0636`） | 因果 | 88.0% | 99.2% | 81.7% | Apache-2.0 |
| `qwen3-causal` | Qwen/Qwen3-0.6B（`c1899de`） | 因果 | 87.4% | 98.8% | 80.9% | Apache-2.0 |
| `a2d-block` | dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1（`c8d24a3`） | 块双向（扩散式） | 85.7% | 99.2% | 78.0% | Apache-2.0 |
| `qwen3-block` | Qwen/Qwen3-0.6B（`c1899de`） | 块双向 | 75.4% | 98.8% | 62.0% | Apache-2.0 |
| `lfm2-causal` | LiquidAI/LFM2.5-VL-450M（`fc6221c`） | 因果 | 84.0% | 98.6% | 75.7% | LFM Open License v1.0 |
| `lfm2-block` | LiquidAI/LFM2.5-VL-450M（`fc6221c`） | 块双向 | 84.1% | 99.2% | 75.5% | LFM Open License v1.0 |

所有组都使用 `data/snake-v1`（开发集 500 道贪吃蛇 + 880 道通用题）、同一套配方、一个随机种子。`reports/` 是跨组结果：图像决策探针（A-OKVQA）、选项顺序探针、精确 McNemar 检验、零样本 DiffusionGemma（djev）读取、Apple Silicon 延迟和汇总。

## 加载

```bash
git clone https://github.com/loadchange/qev && cd qev
git checkout 664e6764d218d2e1a2baa4a40355a88d6d4d8501
uv sync --python 3.12 --extra dev
uv run hf download twainsk/qev-diffusion-experiment --local-dir runs/qevd-archive
```

```python
from experiments.diffusion.qevd import QevDModel
model = QevDModel.from_checkpoint("runs/qevd-archive/lfm2-causal")

from qev.model import QevModel  # Qwen3.5 这一组使用 Qev 检查点格式
baseline = QevModel.from_checkpoint("runs/qevd-archive/qwen35-causal")
```

`experiments/diffusion/order_probe.py` 和 `mm_probe.py` 可以直接评测这些目录。

## 许可证与局限

基于 Qwen 系列模型的 adapter 按 Apache-2.0 分发。`lfm2-causal` 和 `lfm2-block` 是 LiquidAI/LFM2.5-VL-450M 的衍生作品，按 LFM Open License v1.0 分发（见各目录中的 `LICENSE` 和 `NOTICE`）：年收入 1000 万美元及以上的机构商用不在授权范围内。详见 [LICENSES.md](LICENSES.md)。

每组一个随机种子；开发题来自训练中见过的任务族；图像结果是从纯文本决策训练出发的零样本迁移。概率用在校准集上拟合的单一温度校准。
