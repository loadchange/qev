# Qev

以完整 **Qwen3.5-0.8B 多模态模型**为基座的结构化决策模型，提供 Jev / TypeSafe 风格接口和 Apple Silicon MLX 运行时。

Qev 增加独立的语言 LoRA 决策适配器与候选指针头，保留原始视觉编码器、语言模型、图像/视频处理器和生成头。普通对话关闭决策适配器，使用原始基座；结构化决策开启适配器，用候选概率直接构造 JSON，不靠生成文本再解析。

当前版本是小规模监督训练实验。模型产物、实测指标与限制以 [模型说明](docs/MODEL_CARD.md) 为准。Jev 兼容指 API 和输出类型，不代表复现 Jev 的私有架构或达到相同质量。

## 安装与运行

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run qev predict --model models/qev-0.8b-mlx --request examples/request.json
uv run qev serve --model models/qev-0.8b-mlx --port 8008
```

GPU / PyTorch 版本使用 `models/qev-0.8b`。Mac 优先使用 MLX；MLX 通过 Apple Silicon CPU/GPU 和统一内存运行，不表示调用 Apple Neural Engine，也不会自动提高模型准确率。

服务默认只监听 `127.0.0.1:8008`。结构化接口为 `POST /v1/systemone`，原生生成接口为 `POST /v1/chat/completions`。多模态 HTTP 请求使用内嵌图片和采样视频帧，格式见 [接口示例](docs/API.md)。

```bash
curl http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

产物名称 `qev-0.8b` / `qev-0.8b-mlx` 是 Qev checkpoint，不是可直接 `ollama run` 的通用生成模型。候选指针头和可切换适配器需要 Qev 运行时。

## 模型如何训练

1. 固定官方基座 revision：`2fc06364715b967f1860aea9cf38778875588b17`，加载完整多模态权重。
2. 冻结原始参数，在语言注意力、GatedDeltaNet 和 MLP 投影中添加 rank 16 LoRA，另训练 256 维指针头。
3. 把状态、问题和所有候选项编码为独立序列，从候选结束位置与决策位置的隐藏向量计算 logits，使用监督交叉熵。
4. 在独立 calibration 分区拟合一个温度参数，随后报告 development 的准确率、NLL、Brier、ECE 和分类指标。
5. 保存独立适配器、指针头、完整 processor 与固定基座引用；MLX 导出完整视觉/语言基座和可开关适配器。

Qwen3.5 的混合架构包含循环状态，普通 attention mask 不能隔离这些状态。因此每个问题独立编码，不使用 Kev 的共享状态分支打包，也不声称多个问题只计算一次状态。

训练数据包含 10 个公开任务来源及中英可执行规则：5,892 条训练问题、620 条校准问题、880 条开发评估问题。全量输入检查未截断状态或候选。固定数据、分区校验与许可证来源在 `data/v1/manifest.json`；没有读取上游 locked test，也没有调用 Jev 获取标签。中文样本只覆盖四种程序规则，不能据此声称通用中文能力。

## 实测与验证

已完成 NVIDIA L4 上的两轮训练，并生成 Torch 与 MLX checkpoint。相同的 880 条 development 问题上，训练前随机指针头准确率为 28.75%；训练后结果如下：

| 运行路径 | 准确率 | 校准 NLL | ECE |
| --- | --- | --- | --- |
| CUDA BF16，batch 4 | 81.36% | 0.4890 | 0.0432 |
| 交付 MLX FP32，逐题 | 81.25% | 0.4887 | 0.0395 |

两者沿用同一 calibration 温度 `T=1.464086`，没有在 MLX 或 development 上重新拟合。跨精度比较有 1 题的 argmax 改变。完整指标与分组结果见 [模型说明](docs/MODEL_CARD.md)。

原始基座训练前后冻结参数哈希一致；原生文字、图像、视频和混合输入通过有限样例对照。真实 HTTP / 官方 SDK 验收：[Torch 17/17](docs/results/service_torch.json)、[MLX 19/19](docs/results/service_mlx.json)。在 Apple M4、16 GiB 内存、FP32 下，固定 40-token 单题的 5 次热启动测量中位数为 [41.82 ms](docs/results/benchmark_mlx.json)。

候选顺序会影响结果：28 道候选反转探针中，27 道保持同选；一题包含 77 个候选的 Banking77 问题发生翻转，单候选概率最大变化为 0.585。多模态决策准确率尚未评估；本轮中文证据只覆盖四类程序规则。

产物已下载到本地，Colab 训练会话已停止。文件校验、81 项测试、CLI 示例与 wheel 构建结果汇总在 [交付检查](docs/results/release_checks.json)。

## 复现

先在 NVIDIA GPU / Colab 上训练：

```bash
uv run python -m qev.train --data data/v1 --out models/qev-new \
  --epochs 2 --batch 4 --accum 4 --lr 5e-5 --device cuda
```

把完整 `models/qev-new` checkpoint 目录复制到 Apple Silicon Mac，安装 MLX 依赖后导出：

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run python -m qev.export --checkpoint models/qev-new \
  --output models/qev-new-mlx --dtype float32
uv run pytest -q
```

本版 MLX 默认使用 FP32。FP16 转换在本次基础数值审计中超过预设误差门槛，因此没有作为默认交付格式。数据准备和 Colab 流程见 [训练说明](docs/TRAINING.md)。训练使用固定随机种子并记录软件版本、源码摘要、数据摘要、学习率、损失、算力与冻结权重前后哈希。数值精度和硬件变化可能导致结果差异。

代码许可为 Apache-2.0。基座、数据和依赖遵循各自许可，见 [NOTICE](NOTICE)。
