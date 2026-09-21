# 训练与复现

训练使用固定的官方 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` 完整多模态模型。原始参数全部冻结，只更新语言 LoRA 和 256 维指针头。训练前后对每一个冻结参数计算 SHA-256，校验结果写入模型目录的 `base_integrity.json`。历史 v0.2 通用决策训练已完成；下文另列正在进行的贪吃蛇专项续训流程，暂不宣称其最终结果或新模型已经交付。

下面的命令从项目根目录执行。已有的 `data/v1` 可以直接复用；重新准备数据或完整复现 Colab 流程时，请使用独立的项目工作目录，不把已有 `data/`、`models/`、`runs/` 产物复制进去。数据准备、checkpoint 解包和 MLX 导出均拒绝覆盖既有目标。不要删除当前交付的模型来腾出同名路径；本地训练和 MLX 导出示例使用新的 `qev-repro` 名称。

## 历史 v0.2 通用决策数据

```bash
uv run python -m qev.data prepare --out data/v1 \
  --source-root ../kev/evals/public-pool-v4 \
  --seed 20260920
```

数据准备命令保留上游 train/calibration/development 分区，不读取 locked test。上游文件使用固定 revision 和 SHA 校验；若本地源不存在，按代码中的固定来源下载。训练选项随机重排、none 与干扰项增强在准备时固定，不改动校准和开发分区。精确状态和 group 标识的分区隔离会检查。

`data/v1` 中保存 JSONL 和 manifest。`docs/results/data_manifest.json` 留存本次使用的来源、许可和文件摘要。中文仅为退款、审批、期限、会员四类确定性规则。本轮决策微调没有视觉训练样本；完整视觉理解和生成保留自原始基座，视觉决策头需要独立评估。

## 从基座训练：本地 / NVIDIA

```bash
uv sync --python 3.12 --extra dev
uv run python -m qev.train --data data/v1 --out models/qev-repro \
  --epochs 2 --batch 4 --accum 4 --lr 5e-5 --device cuda
```

有效批量通常为 16 个问题，最后不足批按实际样本数归一化。AdamW、weight decay 0.01、10% warmup 加 cosine 衰减、梯度裁剪 1、seed 42。LoRA rank 16 / alpha 32 / dropout 0.05。上下文预算 1024，状态预算 384；超长候选或指令报错，不删除候选。每轮保存 checkpoint，训练日志每 10 次更新记录一次。

CUDA 使用 FP32 master 权重、BF16 autocast 和 FP32 pointer，评估保持同一精度策略。安装 FLA 可以加速 Qwen3.5 线性注意力。CPU / MLX 的算子和精度不同，需额外数值验证。

训练前报告随机指针头 baseline，训练后用 calibration 拟合单一温度，development 只用于评估。development 属于已见任务家族的新样本，不是新任务泛化或最终 locked test。标量温度不会改变 argmax，不能把 confidence 自动解释为任意场景下的正确率。

## 贪吃蛇专项数据与续训

专项训练从 `models/qev-0.8b` 的语言 LoRA 和候选指针头继续，保留冻结的原始多模态基座。`--init-checkpoint` 读取已训练 checkpoint，重新创建优化器和学习率调度；这是新一轮监督训练，不是从中断的优化器状态恢复。原模型路径保持可用，输出必须使用另一个目录。

先在项目根目录生成固定数据。以下命令只加载本地 tokenizer，不加载模型权重，也不调用 Jev 或其他外部模型提供标签：

```bash
uv run python scripts/prepare_snake_data.py --out data/snake-v1 \
  --tokenizer models/qev-0.8b-mlx/backbone \
  --train 12000 --calibration 500 --development 500 \
  --seed 20260921 --sizes 6,8,12,16 \
  --episode-steps 600 --samples-per-episode 48 --exploration 0.10 \
  --replay-data data/v1
```

`--train` / `--calibration` / `--development` 指新增的 Snake 样本数；`--replay-data` 另外加入全部原任务记录，并保留它们原有的分区。已有 `data/snake-v1` 可直接复用；重新生成时选择新目录，生成器拒绝覆盖既有数据。

数据输入直接复用服务的 `decision_request`：三个非反向候选拥有碰撞、食物距离，以及 `reachable_space`、`tail_reachable`、`food_path_distance`、`recent_visits` 特征。几何字段由游戏引擎在假设执行一步后的静态棋盘上计算 BFS，属于明确提供的环境辅助。启发式教师只读取这些可见字段和可见的尺寸、长度、朝向，优先考虑尾连通与可达空间，再结合食物路径和近期访问；它不访问未来食物随机数，也不提供最优动作证明。

轨迹以教师动作推进，并按 `--exploration` 的概率从无碰撞候选中随机选择动作来覆盖偏离教师的状态；监督标签仍是该状态的教师选择。所有候选都碰撞时不创建训练标签。每个分区使用独立游戏种子/局组，排除重复观测与跨分区相同状态；候选顺序以固定随机种子重排。10000–10019 保留为后续闭环评估种子，不参与这些训练轨迹。

产物包括 `train.jsonl`、`calibration.jsonl`、`development.jsonl`、`manifest.json`、`episodes.jsonl`、`token_audit.json` 和默认生成的 `teacher_evaluation.json`。manifest 记录源码/数据摘要、标签来源、原任务回放来源、分区和特征版本；token 审计检查输入预算和状态截断。`teacher_evaluation.json` 测量的是程序教师，不能作为 Qev 模型成绩。教师不参与终端或 HTTP 运行时的动作选择，模型原始首选直接执行。

在支持 BF16 的 NVIDIA GPU 上续训：

```bash
uv run python -m qev.train \
  --init-checkpoint models/qev-0.8b --data data/snake-v1 --out models/qev-snake-0.8b \
  --epochs 2 --batch 8 --accum 2 --lr 3e-5 --device cuda --no-checkpointing
```

本次 A100 40GB 运行使用上述参数。显存较少时可启用默认梯度检查点并减小 batch、增大 accum；有效批量与学习率保持不变。

默认先保存父 checkpoint 在新 development 分区上的 `initial_development.json`，再训练并在 calibration 上重新拟合温度。development 包含新游戏种子上的教师模仿问题及回放任务；离线分类准确率不等于真实整局的吃食、存活或通关能力。训练日志、数据摘要、冻结参数校验和最终评估应与新 checkpoint 一起保存。

等待训练完整结束后，把完整 Torch checkpoint 复制到 Apple Silicon Mac，导出新的 MLX 目录，再运行终端评估：

```bash
uv run python -m qev.export --checkpoint models/qev-snake-0.8b \
  --output models/qev-snake-0.8b-mlx --dtype float32
uv run qev snake --model models/qev-snake-0.8b-mlx --observation spatial \
  --headless --fps 0 --seed 10000 --episodes 20 --size 12 --max-steps 2000 \
  --report runs/snake-trained-spatial.jsonl
```

本轮训练与产物已完成，具体质量与完整性结果见 [专项模型说明](SNAKE_MODEL.md)。比较专项训练收益时，用同样的 `spatial` 观测、种子、棋盘尺寸和步数上限分别运行父/子 checkpoint，并分别保留报告；比较环境特征收益时，固定 checkpoint 再比较 `local` 与 `spatial`。同时检查原通用任务是否退化。终端键位与报告格式见 [演示说明](DEMO.md)。

## 历史 v0.2 Colab CLI

历史 v0.2 训练使用 L4 GPU。下面的辅助脚本复现通用决策训练，不会自动切换为上面的 Snake 续训。以新的会话名 `qev-08b-repro` 示范；再次复现应选择另一个未使用的会话名，并在所有命令中保持一致。先完成 CLI 登录，按前文准备 `data/v1`，再执行：

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
