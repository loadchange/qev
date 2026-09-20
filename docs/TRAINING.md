# 训练与复现

本次训练使用固定的官方 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` 完整多模态模型。原始参数全部冻结，只更新语言 LoRA 和 256 维指针头。训练前后对每一个冻结参数计算 SHA-256，校验结果写入模型目录的 `base_integrity.json`。

下面的命令从项目根目录执行。已有的 `data/v1` 可以直接复用；重新准备数据或完整复现 Colab 流程时，请使用独立的项目工作目录，不把已有 `data/`、`models/`、`runs/` 产物复制进去。数据准备、checkpoint 解包和 MLX 导出均拒绝覆盖既有目标。不要删除当前交付的模型来腾出同名路径；本地训练和 MLX 导出示例使用新的 `qev-repro` 名称。

## 数据

```bash
uv run python -m qev.data prepare --out data/v1 \
  --source-root ../kev/evals/public-pool-v4 \
  --seed 20260920
```

数据准备命令保留上游 train/calibration/development 分区，不读取 locked test。上游文件使用固定 revision 和 SHA 校验；若本地源不存在，按代码中的固定来源下载。训练选项随机重排、none 与干扰项增强在准备时固定，不改动校准和开发分区。精确状态和 group 标识的分区隔离会检查。

`data/v1` 中保存 JSONL 和 manifest。`docs/results/data_manifest.json` 留存本次使用的来源、许可和文件摘要。中文仅为退款、审批、期限、会员四类确定性规则。本轮决策微调没有视觉训练样本；完整视觉理解和生成保留自原始基座，视觉决策头需要独立评估。

## 本地 / NVIDIA

```bash
uv sync --python 3.12 --extra dev
uv run python -m qev.train --data data/v1 --out models/qev-repro \
  --epochs 2 --batch 4 --accum 4 --lr 5e-5 --device cuda
```

有效批量通常为 16 个问题，最后不足批按实际样本数归一化。AdamW、weight decay 0.01、10% warmup 加 cosine 衰减、梯度裁剪 1、seed 42。LoRA rank 16 / alpha 32 / dropout 0.05。上下文预算 1024，状态预算 384；超长候选或指令报错，不删除候选。每轮保存 checkpoint，训练日志每 10 次更新记录一次。

CUDA 使用 FP32 master 权重、BF16 autocast 和 FP32 pointer，评估保持同一精度策略。安装 FLA 可以加速 Qwen3.5 线性注意力。CPU / MLX 的算子和精度不同，需额外数值验证。

训练前报告随机指针头 baseline，训练后用 calibration 拟合单一温度，development 只用于评估。development 属于已见任务家族的新样本，不是新任务泛化或最终 locked test。标量温度不会改变 argmax，不能把 confidence 自动解释为任意场景下的正确率。

## Colab CLI

本次训练使用 L4 GPU。下面用新的会话名 `qev-08b-repro` 示范；再次复现应选择另一个未使用的会话名，并在所有命令中保持一致。先完成 CLI 登录，按前文准备 `data/v1`，再执行：

这些辅助脚本固定写入远端 `/content/qev/models/qev-0.8b`，解包脚本固定写入本地 `models/qev-0.8b`，不提供重命名参数。因此 Colab 流程需要新的本地项目工作目录及新的远端会话；本地目标目录应尚不存在。

```bash
colab new -s qev-08b-repro --gpu L4
colab exec -s qev-08b-repro -f scripts/colab_setup.py --timeout 600
uv run python scripts/package_colab.py
colab upload -s qev-08b-repro runs/qev-code.tar.gz /content/qev-code.tar.gz
colab exec -s qev-08b-repro -f scripts/colab_extract.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_launch_smoke.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

等待 smoke 退出码为 0，再启动训练：

```bash
colab exec -s qev-08b-repro -f scripts/colab_launch_train.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

训练和 smoke 以后台子进程执行；CLI 命令返回不代表训练完成。需要重复运行 `colab_poll.py`，直到相应进程退出；不要重复运行 launch 脚本。训练必须得到 `qev_job RETURN_CODE 0`，训练日志中应有最终 `QEV_TRAINING_COMPLETE`。

训练完成后，在同一个会话中先验证原生多模态保留，再验证真实 HTTP 与官方 SDK。各阶段串行运行，避免同时加载多份模型占用 GPU：

```bash
colab exec -s qev-08b-repro -f scripts/colab_launch_validation.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

重复 poll，确认 `qev_validation RETURN_CODE 0` 和报告 `passed: true` 后，再运行：

```bash
colab exec -s qev-08b-repro -f scripts/colab_setup_validation.py --timeout 120
colab exec -s qev-08b-repro -f scripts/colab_launch_service_validation.py --timeout 60
colab exec -s qev-08b-repro -f scripts/colab_poll.py --timeout 60
```

重复 poll，确认 `qev_service RETURN_CODE 0`、`passed: true` 且没有失败检查。随后收集产物；收集脚本会输出压缩包的 `sha256`，并在包内保存 checkpoint 各文件的 `SHA256SUMS.json`：

```bash
colab exec -s qev-08b-repro -f scripts/colab_collect.py --timeout 120
colab download -s qev-08b-repro /content/qev-trained.tar.gz runs/qev-trained-repro.tar.gz
```

将下面的 `COLLECT_OUTPUT_SHA256` 替换为上一条收集输出中的完整 64 位 SHA-256，不要使用其他训练轮次的摘要：

```bash
uv run python scripts/unpack_training.py runs/qev-trained-repro.tar.gz \
  --sha256 'COLLECT_OUTPUT_SHA256'
```

解包脚本先核对整个下载文件，再解包到 `models/qev-0.8b` 和 `runs/`，并逐个核对 checkpoint 文件。看到 `Verified ... checkpoint files` 后，确认本地 `runs/native_validation.json`、`runs/service_torch.json` 均为 `passed: true`，且 `models/qev-0.8b/base_integrity.json` 为 `unchanged: true`。验证及下载校验完成后才关闭自己创建的会话：

```bash
colab stop -s qev-08b-repro
```

setup 保留 Colab 原装 CUDA PyTorch，移除与 PEFT 不兼容且不使用的旧 torchao。软件依赖版本写入训练 provenance；Mac 环境由 `uv.lock` 固定。checkpoint 内保存完整 processor 和基座 revision，HF 原始基座权重仍需要本地缓存或首次下载。

## 在 Apple Silicon Mac 上导出与验证 MLX

MLX 导出在 Apple Silicon Mac 上执行。如果前面的下载在另一台机器完成，先把项目源码、完整 `models/qev-0.8b` 目录和对应的 `data/v1` 复制到 Mac 上的独立项目工作目录。Torch checkpoint 引用的固定官方基座会在首次导出时下载；MLX 导出包含转换后的完整视觉和语言权重，不把决策 LoRA 合并进原基座。

```bash
uv sync --python 3.12 --extra mlx --extra dev
uv run python -m qev.export --checkpoint models/qev-0.8b \
  --output models/qev-repro-mlx --dtype float32
uv run python scripts/verify_mlx_integration.py \
  --checkpoint models/qev-0.8b --output-model models/qev-repro-mlx \
  --data data/v1 --report runs/mlx-repro-integration.json
uv run python scripts/validate_service.py \
  --checkpoint models/qev-repro-mlx --backend mlx \
  --output runs/service_mlx_repro.json
```

若使用前文的本地 NVIDIA 训练结果，将两处 `--checkpoint models/qev-0.8b` 改为 `--checkpoint models/qev-repro`。`models/qev-repro-mlx` 应为新的输出目录；后续 integration 步骤读取刚导出的目录，不再次转换。再次复现时也应选择新的模型和报告路径。

按顺序等待每条命令完成，并检查两份报告均为 `passed: true`。integration 验证 20 条候选问题的数值一致性和 4 个原生生成样例；HTTP/SDK 验证接口、媒体输入和请求隔离。这些通过条件不等于多模态决策准确率已经得到验证。
