# Qev

[English](README.md) | **简体中文**

以完整 **Qwen3.5-0.8B 多模态模型**为基座的结构化决策模型，提供 Jev / TypeSafe 风格接口和 Apple Silicon MLX 运行时。

Qev 增加独立的语言 LoRA 决策适配器与候选指针头，保留原始视觉编码器、语言模型、图像/视频处理器和生成头。普通对话关闭决策适配器，使用原始基座；结构化决策开启适配器，用候选概率直接构造 JSON，不靠生成文本再解析。

当前版本是小规模监督训练实验。模型产物、实测指标与限制以 [模型说明](docs/MODEL_CARD.zh-CN.md) 为准。Jev 兼容指 API 和输出类型，不代表复现 Jev 的私有架构或达到相同质量。

## 安装与运行

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run hf download twainsk/qev-0.8b-mlx --local-dir models/qev-snake-0.8b-mlx
uv run qev snake
uv run qev predict --model models/qev-snake-0.8b-mlx --request examples/request.json
uv run qev serve --model models/qev-snake-0.8b-mlx --port 8008
```

`qev serve` 默认把两个模型 API POST 端点的请求记录到 `runs/request-logs`。可用 `--request-log-dir PATH` 更改位置，或用 `--no-request-log` 关闭；详见 [本地请求日志](docs/API.zh-CN.md#本地请求日志)。

代码位于 [GitHub](https://github.com/loadchange/qev)，模型权重单独发布到 Hugging Face，Git clone 不包含权重：

| 下载仓库 | 内容 | 适用环境 |
| --- | --- | --- |
| [twainsk/qev-0.8b](https://huggingface.co/twainsk/qev-0.8b) | 约 85 MB，LoRA、指针头与处理器；首次运行另外下载固定版本 Qwen3.5 基座 | PyTorch / NVIDIA |
| [twainsk/qev-0.8b-mlx](https://huggingface.co/twainsk/qev-0.8b-mlx) | 约 3.48 GB，完整 FP32 多模态基座、适配器与指针头 | Apple Silicon / MLX |

两个仓库均为 v0.3.0 贪吃蛇专项续训后的模型，保留原通用任务回放训练和原生多模态生成。它们需要 Qev 运行时，不能直接当作标准 Transformers pipeline 或 Ollama 模型使用。PyTorch 版下载命令为 `uv run hf download twainsk/qev-0.8b --local-dir models/qev-snake-0.8b`。每个仓库包含模型卡、许可证、评测摘要与 `release_manifest.json` 文件校验清单。

本次专项模型的 GPU / PyTorch 版本为 `models/qev-snake-0.8b`，Mac 版本为 `models/qev-snake-0.8b-mlx`；原 `models/qev-0.8b[-mlx]` 保留。Mac 优先使用 MLX；MLX 通过 Apple Silicon CPU/GPU 和统一内存运行，不表示调用 Apple Neural Engine，也不会自动提高模型准确率。

**贪吃蛇首选终端入口 `qev snake`。** 棋盘旁显示模型候选概率、实际动作、得分和推理耗时；空格暂停/继续，`N` 单步，`+` / `-` 调速，`Q` 或 Ctrl-C 退出。默认优先加载存在的 `models/qev-snake-0.8b-mlx`，否则加载 `models/qev-0.8b-mlx`；使用 `--model` 可固定 checkpoint。专项训练已完成；终端默认加载新权重，原权重仍可显式选择。

默认 `--observation spatial` 提供静态 BFS 可达空间、尾部连通性、食物路径距离和近期访问次数；这些是引擎计算的环境特征，画面明确标注“静态BFS环境特征·无动作接管”。`--observation local` 保留历史碰撞/食物距离观测，便于固定模型、种子和规则做对照。模型仍直接选择并执行方向，可能碰撞的候选不会被剔除。

```bash
# 历史模型 + 历史局部观测
uv run qev snake --model models/qev-0.8b-mlx --observation local --seed 7
# 无界面连续运行十局，种子依次为 10000..10009，逐步保存真实请求与响应
uv run qev snake --headless --fps 0 --seed 10000 --episodes 10 \
  --report runs/snake-terminal.jsonl
```

JSONL 报告逐条写入且拒绝覆盖已有文件；没有 TTY 时自动使用 headless 模式。专项数据与训练复现见 [训练说明](docs/TRAINING.zh-CN.md)，终端/HTTP 模式与完整操作见 [演示说明](docs/DEMO.zh-CN.md)。历史 v0.2 checkpoint 未接受贪吃蛇训练。新模型在相同 8×8、500 步、20 个保留种子的 CUDA 对照中，平均吃食从 3.1 提升到 42.9，碰撞从 20 局降至 0 局；这是有限回合实测，不是永不碰撞保证。完整条件、Mac 结果与多模态保留验证见 [专项模型说明](docs/SNAKE_MODEL.zh-CN.md)。

服务默认只监听 `127.0.0.1:8008`。结构化接口为 `POST /v1/systemone`，原生生成接口为 `POST /v1/chat/completions`。多模态 HTTP 请求使用内嵌图片和采样视频帧，格式见 [接口示例](docs/API.zh-CN.md)。

启动后打开 **http://127.0.0.1:8008** 即可使用内置实验室，无需额外前端服务。网页首次按浏览器语言选择英文或简体中文，也可手动切换；手动选择会保存在当前浏览器，刷新后仍然生效：

- **网页贪吃蛇**：作为另一种可视化界面，支持开始、暂停、单步、固定种子重开，查看每步真实候选概率、耗时及请求，导出本局记录；与终端共用游戏规则和模型决策链路。
- **[接口测试](http://127.0.0.1:8008/playground)**：提供“纯文本选择”“背景带图片”“所有选项带图片”三个 Choice 场景，编辑背景、问题和候选，通过上传或粘贴添加、替换背景和候选图片，再发送真实请求；高级 JSON 与表单同步，并可查看响应、耗时和 token 使用。每个图片区域提供独立粘贴入口，可用 Ctrl/Cmd+V；浏览器支持剪贴板读取时，也可使用剪贴板按钮。

三个场景都调用 `POST /v1/systemone`，返回 `answers.<问题名>.choice` 与 `probabilities`，不生成回答文字，`usage.output_tokens=0`。背景图片放在 `state.content`，候选图片放在 `questions.<问题名>.criteria.<候选名>.content`；每个 `content` 包含 `text` 和内联 `image_url`。接口页内置图片示例可直接运行，完整协议见 [API 示例](docs/API.zh-CN.md#图像决策)。其他类型的决策、原生生成及服务查询保留在“更多示例”。

贪吃蛇网页与终端都使用文字环境特征；显示的棋盘不是模型图像输入，不属于视觉决策评估。

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

历史 v0.2 训练数据包含 10 个公开任务来源及中英可执行规则：5,892 条训练问题、620 条校准问题、880 条开发评估问题。全量输入检查未截断状态或候选。固定数据、分区校验与许可证来源在 `data/v1/manifest.json`；没有读取上游 locked test，也没有调用 Jev 获取标签。中文样本只覆盖四种程序规则，不能据此声称通用中文能力。新增贪吃蛇专项训练从该 checkpoint 继续，使用显式启发式教师生成的动作标签和原任务数据回放；教师不参与运行时的动作执行。

## 实测与验证

以下是历史 v0.2 通用决策 checkpoint 的结果，不是新贪吃蛇专项模型的结果。该版本已完成 NVIDIA L4 上的两轮训练，并生成 Torch 与 MLX checkpoint。相同的 880 条 development 问题上，训练前随机指针头准确率为 28.75%；训练后结果如下：

| 运行路径 | 准确率 | 校准 NLL | ECE |
| --- | --- | --- | --- |
| CUDA BF16，batch 4 | 81.36% | 0.4890 | 0.0432 |
| 交付 MLX FP32，逐题 | 81.25% | 0.4887 | 0.0395 |

两者沿用同一 calibration 温度 `T=1.464086`，没有在 MLX 或 development 上重新拟合。跨精度比较有 1 题的 argmax 改变。完整指标与分组结果见 [模型说明](docs/MODEL_CARD.zh-CN.md)。

原始基座训练前后冻结参数哈希一致；原生文字、图像、视频和混合输入通过有限样例对照。真实 HTTP / 官方 SDK 验收：[Torch 17/17](docs/results/service_torch.json)、[MLX 19/19](docs/results/service_mlx.json)。在 Apple M4、16 GiB 内存、FP32 下，固定 40-token 单题的 5 次热启动测量中位数为 [41.82 ms](docs/results/benchmark_mlx.json)。

候选顺序会影响结果：28 道候选反转探针中，27 道保持同选；一题包含 77 个候选的 Banking77 问题发生翻转，单候选概率最大变化为 0.585。本轮中文证据只覆盖四类程序规则。

之后对已发布贪吃蛇检查点 `qev-snake-0.8b` 的探测（CUDA BF16，2026-09-25）：把 1,380 道开发题的选项全部反转后，5.4% 的选择发生改变（准确率 87.8% → 87.2%）。在 A-OKVQA 验证集前 500 题上（4 选 1，随机 25%），只用文本训练的决策头带图答对 69.4%，无图 32.6%；这是零样本迁移，媒体决策概率仍未校准。同一实验还比较了扩散与自回归骨干：扩散式读取没有提升准确率；零样本 DiffusionGemma-26B（djev 的做法）在通用题上与 Qev 持平，在贪吃蛇上低约 14 个百分点。详见[骨干对照实验](experiments/diffusion/README.zh-CN.md)。

历史 v0.2 产物已下载到本地，其 Colab 训练会话已停止。该次文件校验、81 项测试、CLI 示例与 wheel 构建结果汇总在 [交付检查](docs/results/release_checks.json)。

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

本版 MLX 默认使用 FP32。FP16 转换在本次基础数值审计中超过预设误差门槛，因此没有作为默认交付格式。数据准备和 Colab 流程见 [训练说明](docs/TRAINING.zh-CN.md)。训练使用固定随机种子并记录软件版本、源码摘要、数据摘要、学习率、损失、算力与冻结权重前后哈希。数值精度和硬件变化可能导致结果差异。

代码许可为 Apache-2.0。基座、数据和依赖遵循各自许可，见 [NOTICE](NOTICE)。
