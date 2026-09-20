# Qev-0.8B 模型说明

Qev 在完整的 `Qwen/Qwen3.5-0.8B` 多模态基座上增加独立的语言 LoRA 和候选指针头，用于文字条件下的 `choice`、`noul`、`score` 结构化决策。它保留基座的原始文字、图像、视频理解与语言生成路径。它是小规模监督学习实验，Jev / TypeSafe 兼容指接口和输出类型，不代表官方关系、私有权重复现或同等效果。

本页的表格由 `scripts/update_model_card.py` 从真实产物生成；缺失报告标为“待验证”，不以单元测试或未训练模型的审计结果替代训练后评估。

## 基座、结构与生成能力

- 基座固定为 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，实例化完整 `Qwen3_5ForConditionalGeneration`。
- 保留原始视觉编码器、图像/视频 processor、语言模型、词表和 LM head。24 个语言层采用 GatedDeltaNet 与全注意力混合结构，语言隐藏维度为 1024。
- 语言投影层加入 rank 16、alpha 32、dropout 0.05 的 LoRA；候选结束位置和决策位置的隐藏向量分别投影到 256 维，以点积得到候选 logits。仅 LoRA 和指针头参与决策训练，全部原始参数冻结。
- `choice` 返回概率最大的候选；`noul` 返回 yes 的概率；`score` 返回有序候选等级的概率期望。JSON 由程序构造，没有自由生成再解析的过程。
- 每个问题单独编码。GatedDeltaNet 的循环与卷积状态不能仅靠普通 attention mask 隔离，所以没有采用 Kev 的共享前缀多题分支，也不声称多个问题只计算一次状态。

**原生生成明确关闭决策 LoRA，保留原始权重路径。** 适配器以 sidecar 保存、不合并到基座；生成结束后恢复适配器状态。训练前后的冻结参数 SHA-256 用于核对基座是否改变。MLX 导出也保留完整视觉/语言模型及可切换适配器。权重冻结和适配器关闭保证的是原始计算路径；不同框架、精度、算子及生成配置仍可能导致数值或 token 差异，实际对照结果见下表。

本轮所有决策监督均为文本。图片/视频能进入模型并输出有限概率，只证明多模态输入链路可运行，**多模态决策准确率尚未评估**。原生图像/视频生成能力继承自 Qwen 基座，不能据此宣称新的指针头已经学会同等水平的视觉决策。文字拟合的温度不应用于多模态决策，多模态决策使用 `T=1`。

## 数据与划分

公开数据来自 `jaredpalmer/kev-suites@a3318ddc1f630c5673232efacd8123a84de3f480` 的 `public-pool-v4`，涵盖 Banking77、BoolQ、AG News、MNLI、SST5、Yelp、TREC、DBpedia14、Amazon 和 IMDb。另由可执行程序生成中英退款、审批、期限、会员四类规则标签，没有调用 Jev 或其他外部模型提供标签。

保留上游 train / calibration / development 划分，不读取 locked test。训练增强在准备阶段按固定种子冻结，包含候选重排、none 与干扰项；calibration 和 development 不增强。检查跨分区规范化后的精确 state、group 标识以及合成规则的语义实例隔离。

<!-- BEGIN GENERATED DATA -->
| 分区 | Requests | 问题数 | Choice | Noul | Score | 最大 token 数 | State 截断数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4692 | 5892 | 2318 | 2126 | 1448 | 881 | 0 |
| calibration | 500 | 620 | 226 | 226 | 168 | 810 | 0 |
| development | 700 | 880 | 326 | 326 | 228 | 850 | 0 |

数据种子：`20260920`。训练增强前 4500 条 request，新增 192 条 none 配对记录。来源：[data_manifest.json](results/data_manifest.json)；完整 token 审计：[data_audit.json](results/data_audit.json)。
<!-- END GENERATED DATA -->

数据审计只覆盖精确重复、分组和所列合成实例；没有做模糊去重或基座预训练污染审计。development 的来源与标签平衡抽样参与了数据选择，它不是最终 locked test。中文开发样本仍来自训练中已有的四个规则家族，只能作为同家族新实例证据，不能推断广泛中文能力或未见规则家族的泛化。

## 训练与产物

训练目标为候选标签的监督交叉熵，不使用强化学习、Jev 蒸馏或 JEPA 潜在表示预测。配置为 AdamW、weight decay 0.01、10% warmup 后 cosine 衰减、梯度裁剪 1、seed 42，本轮训练 2 轮。每个 microbatch 4 条问题，累积 4 次，通常有效批量 16；最后不足批按实际问题数归一化。

文字序列预算为 1024 token，state 预算 384 token。不能容纳的候选或指令明确报错，不静默删除选项。CUDA 使用 FP32 master 权重、BF16 autocast、FP32 指针头，训练后评估沿用该策略。开发集没有用于梯度更新；单一温度只在 calibration 上以 NLL 选取。

<!-- BEGIN GENERATED TRAINING -->
| 项目 | 实际记录 |
| --- | --- |
| 已完成 epoch / 配置 epoch | 2 / 2 |
| 训练设备 | NVIDIA L4 |
| 训练精度 | fp32 master weights; bf16 autocast; fp32 pointer |
| 可训练参数 | 11,346,944 |
| microbatch / 梯度累积 | 4 / 4 |
| 初始学习率 / seed | 5e-05 / 42 |
| 训练问题 / 校准问题 / 开发问题 | 5892 / 620 / 880 |
| optimizer 更新次数 | 738 |
| 训练循环耗时（秒，不含评估） | 1,718.64 |
| 累计前向 token（不含 padding） | 1,954,128 |

软件版本：`numpy==2.5.3`, `peft==0.21.0`, `safetensors==0.8.0`, `torch==2.11.0+cu128`, `transformers==5.17.0`。完整参数、源码及数据摘要：[provenance.json](results/provenance.json)。
<!-- END GENERATED TRAINING -->

Torch checkpoint 保存决策适配器、指针头、完整 processor、配置及固定的基座引用，首次运行仍需获取相应官方基座。MLX 产物另含完整转换后的基座权重。`qev-0.8b` 与 `qev-0.8b-mlx` 需要 Qev 运行时，并非可直接交给 Ollama 的通用生成权重。

## 实测评估

主要报告 development 的候选 argmax 准确率、NLL、Brier、ECE 和 Score MAE。每条问题等权；多问题 request 会贡献多个样本。Brier 是每题全部候选的平方误差之和，ECE 使用 10 个等宽最大概率区间。Score MAE 与 HTTP score 一样，以从 0 开始的候选等级索引为尺度。

随机指针头 baseline 的 LoRA 为初始状态，衡量新决策头训练前的表现，不代表 Qwen 原生问答能力。标量温度不改变候选 argmax；它在指定校准分布上拟合，不保证任意领域、语言或模态上的 confidence 等于正确率。表中的高置信错误率指“最大概率至少 0.9 且预测错误”的问题占全部问题的比例。

<!-- BEGIN GENERATED EVALUATION -->
| 评估 | 问题数 | 准确率 | NLL | Brier | ECE | 高置信错误率 | Score MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 训练前随机指针头 · development | 880 | 28.75% | 2.1275 | 0.9291 | 0.3112 | 14.66% | 1.1697 |
| CUDA BF16 · development · T=1 | 880 | 81.36% | 0.5512 | 0.2672 | 0.0695 | 3.86% | 0.4492 |
| CUDA BF16 · development · 校准 | 880 | 81.36% | 0.4890 | 0.2578 | 0.0432 | 2.39% | 0.4677 |
| 交付 MLX float32 · development · T=1 | 880 | 81.25% | 0.5506 | 0.2670 | 0.0710 | 3.86% | 0.4491 |
| 交付 MLX float32 · development · 沿用校准 | 880 | 81.25% | 0.4887 | 0.2577 | 0.0395 | 2.50% | 0.4676 |

训练前记录：[untrained_development.json](results/untrained_development.json)。

温度 `T=1.464086`，拟合分区：`calibration`。报告：[evaluation.json](results/evaluation.json)。评估设备 `cuda:0`，权重 `float32`，autocast `bfloat16`，pointer `float32`。

交付 MLX 使用 `float32` 逐题推理，完整评估 880 条相同 development 问题：[evaluation_mlx.json](results/evaluation_mlx.json)。MLX 沿用上述 CUDA calibration 温度，没有在 development 或 MLX 输出上重新拟合。CUDA BF16 批量评估与 MLX FP32 逐题评估的算术精度、算子和批量形状不同，指标及部分 argmax 可以不同。逐条身份对齐后，在 880 条问题中，argmax 改变 1 条；校准概率最大差值 0.023412，每题最大差值的中位数 0.000732。这是跨精度诊断，不等同于下文 CPU FP32 → MLX FP32 导出对照的验收门槛。

校准拟合分区（用于选温度，不视作独立效果证据）：

| 分区 | 问题数 | NLL · T=1 | NLL · 校准 |
| --- | --- | --- | --- |
| calibration | 620 | 0.4586 | 0.4247 |

Development 按输出类型划分（两条路径均应用上述温度）：

| 输出类型 | 问题数 | CUDA 准确率 | MLX 准确率 | CUDA NLL | MLX NLL | CUDA ECE | MLX ECE | CUDA Score MAE | MLX Score MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| choice | 326 | 88.04% | 88.04% | 0.4618 | 0.4614 | 0.0622 | 0.0621 | — | — |
| noul | 326 | 86.81% | 86.81% | 0.3043 | 0.3039 | 0.0388 | 0.0387 | — | — |
| score | 228 | 64.04% | 63.60% | 0.7919 | 0.7921 | 0.0737 | 0.0725 | 0.4677 | 0.4676 |

Development 按语言划分（两条路径均应用上述温度）：

| 语言 | 问题数 | CUDA 准确率 | MLX 准确率 | CUDA NLL | MLX NLL | CUDA ECE | MLX ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 830 | 80.24% | 80.12% | 0.5154 | 0.5152 | 0.0462 | 0.0422 |
| zh | 50 | 100.00% | 100.00% | 0.0494 | 0.0488 | 0.0412 | 0.0407 |

Development 按数据来源划分（两条路径均应用上述温度）：

| 数据来源 | 问题数 | CUDA 准确率 | MLX 准确率 | CUDA NLL | MLX NLL | CUDA ECE | MLX ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agnews | 60 | 85.00% | 85.00% | 0.4986 | 0.4977 | 0.0865 | 0.0869 |
| agnews_yn | 120 | 90.83% | 90.83% | 0.2225 | 0.2215 | 0.0215 | 0.0212 |
| amazon | 60 | 50.00% | 50.00% | 1.0234 | 1.0234 | 0.1029 | 0.1019 |
| banking77 | 60 | 76.67% | 76.67% | 0.9680 | 0.9676 | 0.1492 | 0.1484 |
| boolq | 60 | 70.00% | 70.00% | 0.5804 | 0.5812 | 0.0912 | 0.0920 |
| dbpedia14 | 60 | 96.67% | 96.67% | 0.2534 | 0.2547 | 0.0395 | 0.0393 |
| imdb | 60 | 85.00% | 85.00% | 0.3962 | 0.3958 | 0.0787 | 0.0784 |
| mnli | 60 | 80.00% | 80.00% | 0.6285 | 0.6268 | 0.1230 | 0.1227 |
| sst5 | 60 | 51.67% | 51.67% | 1.0321 | 1.0328 | 0.0909 | 0.1011 |
| synthetic_approval_en | 13 | 100.00% | 100.00% | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| synthetic_approval_zh | 13 | 100.00% | 100.00% | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| synthetic_deadline_en | 12 | 91.67% | 91.67% | 0.1520 | 0.1511 | 0.1154 | 0.1149 |
| synthetic_deadline_zh | 12 | 100.00% | 100.00% | 0.2029 | 0.2008 | 0.1688 | 0.1670 |
| synthetic_membership_en | 12 | 100.00% | 100.00% | 0.0094 | 0.0089 | 0.0091 | 0.0086 |
| synthetic_membership_zh | 12 | 100.00% | 100.00% | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| synthetic_refund_en | 13 | 100.00% | 100.00% | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| synthetic_refund_zh | 13 | 100.00% | 100.00% | 0.0011 | 0.0011 | 0.0011 | 0.0011 |
| trec | 60 | 96.67% | 96.67% | 0.1604 | 0.1602 | 0.0321 | 0.0321 |
| yelp | 60 | 63.33% | 61.67% | 0.8805 | 0.8814 | 0.0989 | 0.1146 |
| yelp_yn | 60 | 91.67% | 91.67% | 0.2311 | 0.2307 | 0.0625 | 0.0493 |
<!-- END GENERATED EVALUATION -->

`noul` 的分类准确率使用 yes/no argmax；`score` 的分类准确率使用等级 argmax，与期望分值的 MAE 分别报告。此处没有与 Jev、Kev、Laya 或其他模型在同一独立测试集上的性能比较。

## 原生保留、MLX 与服务验证

冻结哈希、训练前后 native token 对照、Torch/MLX 候选概率对照与 HTTP/SDK 验收分别验证不同条件。有限样例通过不等于所有输入逐值相同；服务通过也不等于任务准确率已验证。未训练的非零 LoRA 数值审计只能验证实现，不能充当本 checkpoint 的最终指标。

<!-- BEGIN GENERATED VALIDATION -->
| 验证 | 状态 | 证据 |
| --- | --- | --- |
| 冻结基座参数 | 通过 | [base_integrity.json](results/base_integrity.json) |
| Torch / MLX 原生生成 token 对照 | 通过（4/4） | [mlx-integration.json](results/mlx-integration.json) |
| Torch / MLX 候选概率 | 通过 | [mlx-integration.json](results/mlx-integration.json) |
| 训练后原生生成保留 | 通过 | [native_validation.json](results/native_validation.json) |
| 多题隔离 · mlx | 通过 | [probe_mlx.json](results/probe_mlx.json) |
| HTTP / SDK · mlx | 通过（19/19） | [service_mlx.json](results/service_mlx.json) |
| HTTP / SDK · torch | 通过（17/17） | [service_torch.json](results/service_torch.json) |

冻结参数数：852,985,920；训练前 SHA-256：`1021eaeed8a1528b435a90a690d42245f07e1b65c7c2dc8a37ec9516a8cf4bee`；训练后 SHA-256：`1021eaeed8a1528b435a90a690d42245f07e1b65c7c2dc8a37ec9516a8cf4bee`。

固定短请求的热启动顺序延迟（[benchmark_mlx.json](results/benchmark_mlx.json)，backend `mlx`，每种问题数测量 5 次）：

| 问题数 | 累计输入 token | 中位耗时 ms | 问题/秒 |
| --- | --- | --- | --- |
| 1 | 40 | 41.82 | 23.91 |
| 4 | 160 | 166.82 | 23.98 |
| 16 | 640 | 669.53 | 23.90 |

上述延迟平台：`macOS-27.0-arm64-arm-64bit`，架构 `arm64`；模型加载耗时 6.87 秒，不计入热请求延迟。

设备芯片：`Apple M4`；统一内存：16.0 GiB；权重精度：`float32`。软件版本完整记录于该报告的 `packages` 字段。

[mlx-integration.json](results/mlx-integration.json) 的原生生成样例：`text`、`image`、`video`、`mixed_image_video`。这是固定输入的贪心 token 对照，不是多模态准确率评估。

[mlx-integration.json](results/mlx-integration.json)：20 条问题，原始概率最大误差 0.00000509，argmax 改变 0，容差 0.00200000，MLX dtype `float32`。

其中 17 条文字、3 条多模态；参考路径：`Torch fp32 full foundation + unmerged LoRA, independent rows`。

同一报告的校准概率最大误差 0.00000441，argmax 改变 0。

[native_validation.json](results/native_validation.json) 的图像 fixture token 对照：通过。视频生成与视觉指针输出为功能探针，不是视觉准确率基准。

[probe_mlx.json](results/probe_mlx.json)：28 条开发问题候选反转后同选比例 96.43%。

其中 27/28 条保持同选，单候选概率最大变化 0.584656。该探针不支持候选顺序不变性的保证。

发生翻转的 `banking77` 问题包含 77 个候选，原选择 `unable_to_verify_identity`，反转后为 `verify_my_identity`，单候选概率最大变化 0.584656。

单题、同请求多题及重复请求之间的最大概率差值：0.00000000；此项隔离验证与改变候选顺序是不同测试。
<!-- END GENERATED VALIDATION -->

MLX 使用 Apple Silicon 的 CPU/GPU 与统一内存，不表示调用 Apple Neural Engine，也不会自动改善模型准确率。延迟报告若存在，仅覆盖指定硬件和固定短输入的热启动顺序调用，不应外推到长文本、图像、视频或并发吞吐。

本轮 [交付检查](results/release_checks.json) 汇总了 Ruff、81 项测试、真实 CLI 示例、wheel 构建与源码一致性校验。权重文件大小与 SHA-256 见 [产物清单](results/artifact_inventory.json)；[Colab 关闭记录](results/colab_shutdown.json) 确认训练会话已终止且服务器无活动会话。

默认服务与 `Agent` 逐题推理（Torch `batch_size=1`，与 MLX 的逐题行为一致），使概率不因同请求中加入其他问题而改变计算形状。Python 调用方可以显式增大 `batch_size` 提高吞吐，但 CUDA BF16 下批量及 padding 变化可能改变概率。上述训练与离线评估仍采用 batch 4；服务逐题结果可能与其存在小幅数值差异，不应视为所有调用方式逐值相同。

## 适用范围与限制

适合本地实验中的有明确候选项的路由、分类、规则判断和等级评分。模型始终在给定候选集合内分配概率，漏掉正确选项时不能自动发现；none 的可靠性与训练覆盖相关。候选顺序、措辞、长度、领域变化及极端类别不平衡可能影响结果，应在实际分布上独立评估。

原生接口为非流式 `POST /v1/chat/completions`，结构化接口为 `POST /v1/systemone`。HTTP 多模态输入支持内嵌 PNG/JPEG/WebP 图片与显式采样视频帧，不直接读取本机路径、抓取远程 URL 或接受原始 MP4。资源预算、请求结构、confidence 定义与模型别名详见 [API 文档](API.md)。兼容别名 `jev-latest` 不会切换至 Jev 模型；Score confidence 为本项目近似约定，不声称复现 TypeSafe 未公开算法。

## 许可与来源

Qev 代码使用 Apache-2.0；官方 Qwen 基座、训练数据与依赖各自遵循其原有许可。**代码的 Apache-2.0 不代表所有数据或派生模型权利均已统一为 Apache-2.0。** 数据清单记录的是固定来源卡片中的许可声明，含未声明、unknown、other 及多种署名/相同方式共享许可，不构成额外授权。具体使用与再分发需核对对应原始条款。

<!-- BEGIN GENERATED LICENSES -->
| 原始数据来源 | 来源卡片声明的许可 | 固定 revision（前 12 位） |
| --- | --- | --- |
| [CogComp/trec](https://huggingface.co/datasets/CogComp/trec/blob/65752bf53af25bc935a0dce92fb5b6c930728450/README.md) | not_declared | `65752bf53af2` |
| [SetFit/amazon_reviews_multi_en](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en/blob/ec73b665e4be0f567b69d39425355401cfe0d29b/README.md) | apache-2.0 | `ec73b665e4be` |
| [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5/blob/e51bdcd8cd3a30da231967c1a249ba59361279a3/README.md) | not_declared | `e51bdcd8cd3a` |
| [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full/blob/c1f9ee939b7d05667af864ee1cb066393154bf85/README.md) | other | `c1f9ee939b7d` |
| [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news/blob/eb185aade064a813bc0b7f42de02595523103ca4/README.md) | unknown | `eb185aade064` |
| [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14/blob/9abd46cf7fc8b4c64290f26993c540b92aa145ac/README.md) | cc-by-sa-3.0 | `9abd46cf7fc8` |
| [google/boolq](https://huggingface.co/datasets/google/boolq/blob/35b264d03638db9f4ce671b711558bf7ff0f80d5/README.md) | cc-by-sa-3.0 | `35b264d03638` |
| [legacy-datasets/banking77](https://huggingface.co/datasets/legacy-datasets/banking77/blob/f54121560de48f2852f90be299010d1d6dc612ec/README.md) | cc-by-4.0 | `f54121560de4` |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli/blob/da70db2af9d09693783c3320c4249840212ee221/README.md) | cc-by-3.0, cc-by-sa-3.0, mit, other | `da70db2af9d0` |
| [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb/blob/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/README.md) | other | `e6281661ce1c` |
<!-- END GENERATED LICENSES -->

合成规则由 `qev/data.py` 生成，来源与代码摘要见数据清单。候选指针头、typed API 及数据转换/增强思路参考 Kev；MLX 原生实现和数值对照思路参考 Laya-MLX，没有使用 Laya 的 ModernBERT 权重。Jev / TypeSafe 是第三方名称。详见 [NOTICE](../NOTICE)、[架构说明](ARCHITECTURE.md) 和 [训练复现](TRAINING.md)。

更新本页时运行：

```bash
uv run python scripts/update_model_card.py --archive-results
```

脚本从 `models/qev-0.8b`、`models/qev-0.8b-mlx` 和 `docs/results` 读取报告，不启动推理或训练。`--archive-results` 将训练核心报告及 MLX parity 原样归档到 `docs/results`；实际服务、native 和性能报告也应归档到该目录后再次更新。缺失报告保留待验证状态。
