# Qev 命令行

[English](CLI.md) | **简体中文**

`qev` 可以在终端里直接让模型做结构化判断，也可以提供 HTTP API 和网页实验室，并负责模型管理。在 Apple Silicon 上它基于 MLX 运行，不需要 PyTorch。

## 安装

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull            # 下载并校验默认模型（1.0 GB），存放在 ~/.qev
qev doctor          # 检查芯片、MLX/Metal、模型文件和本机服务
```

要求：Apple Silicon，macOS 14 或更新。Homebrew 可能会先要求信任这个 tap（`brew trust loadchange/qev`）。在源码目录中，`uv sync --python 3.12 --extra mlx` 后用 `uv run qev …` 效果相同。

`qev pull --from 目录 [--link]` 可以导入已经下载好的检查点（例如用 `hf download twainsk/qev-450m-mlx` 下载的），导入前会按 qev 自带的发布清单逐个校验运行时文件。

注册表模型（用 `QEV_MODEL`、`--model` 或 `qev pull 名称` 选择）：

| 名称 | 底模 | 决策输入 | 下载体积 |
|---|---|---|---|
| `qev-450m`（默认） | LFM2.5-VL-450M | 文本、图片、视频帧 | 1.00 GB |
| `qev-230m` | LFM2.5-230M | 仅文本（多媒体会报错拒绝） | 0.53 GB |
| `qev-0.8b` | Qwen3.5-0.8B | 文本、图片、视频 | 3.48 GB |

LFM2.5 系列模型按 LFM Open License v1.0 分发：年营收达到或超过 1,000 万美元的实体不获商用许可。

## 判断

```bash
qev decide "结账时卡被拒两次" -i "哪个部门处理？" \
  --choice "billing=账单、付款和退款" "technical=故障和宕机" sales
qev decide "客户要求退还重复扣款" -i "是否要求退款？" --noul -q
cat ticket.txt | qev decide - -i "有多紧急？" --score low medium high
qev decide --image receipt.jpg -i "这是餐厅小票吗？" --noul
qev decide --frame f1.png --frame f2.png --fps 2 -i "门打开了吗？" --noul
qev decide --request request.json --json        # 完整的 /v1/systemone 请求，可包含多道题
```

| 选项 | 含义 |
|---|---|
| `STATE`、`-`、`--state-file`、`--state-json` | 要判断的内容：文本、标准输入、文本文件，或 JSON 对象/数组 |
| `--image 路径` | 附加图片（可重复）。大尺寸照片会按 EXIF 方向摆正并缩放到 API 限制以内 |
| `--frame 路径` + `--fps` | 以按顺序排列的帧附加视频 |
| `--choice 键[=描述]…` / `--noul` / `--score 等级…` | 题型；`-i` 设置问题，`--id` 设置题目名 |
| `--request 路径` | 直接发送完整请求 |
| 默认 / `--json` / `-q` | 易读的概率、完整 API 响应、或只输出答案 |
| `--exit-status` | 只有一道是/否题时，是返回 0，否返回 1 |

退出码：0 成功，1 表示 `--exit-status` 下的"否"，2 输入无效，3 模型未安装，4 服务错误。

## 让模型常驻

加载模型需要几秒钟。如果 `$QEV_SERVER`（默认 `http://127.0.0.1:8008`）上有服务在运行且使用所需的模型，`qev decide`、`qev chat` 和 `qev snake` 会直接把请求发给它：

```bash
brew services start qev      # launchd 服务：qev serve --no-request-log --decision-weights bf16
qev decide "…" --noul -q     # M4 上整条命令约 0.1–0.3 秒；在进程内加载模型约 5 秒
```

`--local` 总是在进程内加载模型；`--server URL` 要求使用指定的服务。

## HTTP API

```bash
qev serve [--port 8008] [--host 127.0.0.1] [--decision-weights bf16]
```

提供 `POST /v1/systemone`、`POST /v1/chat/completions`、`GET /v1/models`、`GET /health`，以及位于 `/` 的网页实验室，详见 [API](API.zh-CN.md)。手动运行的 `qev serve` 会把请求记录在 `$QEV_HOME/request-logs`，加 `--no-request-log` 可关闭；Homebrew 服务默认不记录。

## Apple Silicon 上的速度

`--decision-weights`（或 `$QEV_DECISION_WEIGHTS`）决定 MLX 运行时如何计算决策。原生生成始终使用未经改动的底模。默认模型 `qev-450m`（bfloat16 底模）在 M4 上：

| 模式 | 决策路径 | M4 上每步贪吃蛇决策 | MLX 峰值内存 |
|---|---|---|---|
| `adapter` | LoRA 单独计算（经过验证的参考路径） | 98 ms | 1.78 GiB |
| `merged` | 另存一份合并后的 float32 决策权重 | 96 ms | 2.94 GiB |
| `merged-bf16` | 另存一份合并后的 bfloat16 决策权重 | 70 ms | 1.91 GiB |
| `bf16` | 整个模型 bfloat16，决策权重合并 | 71 ms | 1.86 GiB |

在 1,380 道开发题上，四种模式准确率都是 86.2%（贪吃蛇 98.8%，通用 79.1%），且与 PyTorch 参考只差同样的 2 个决策（[结果](results/qev-450m_decision_weights.json)）。命令行默认使用 `adapter`，Homebrew 服务使用 `bf16`。纯文本的 `qev-230m` 在 `bf16` 下一步贪吃蛇决策 43 ms、峰值 1.18 GiB（[结果](results/qev-230m_decision_weights.json)）；此前 `qev-0.8b` 的测量（float32 底模，149–219 ms，最高 5.07 GB）保留在 [decision_weights.json](results/decision_weights.json)。

## 其他命令

| 命令 | 作用 |
|---|---|
| `qev chat "提示" [--image 路径]` | 关闭决策 adapter 的原生文本/图像生成 |
| `qev list`、`qev pull 名称`、`qev rm 名称` | 管理 `$QEV_HOME/models`（默认 `~/.qev`）中的注册表模型 |
| `qev doctor [--verify] [--json]` | 环境、模型和服务检查 |
| `qev snake` | 终端贪吃蛇；在源码目录外会使用运行中的服务或已安装的模型 |
| `qev predict --model 目录 --request 文件` | 运行一个请求文件并输出 JSON（为兼容保留） |

环境变量：`QEV_HOME`（模型和日志）、`QEV_MODEL`（默认模型）、`QEV_SERVER`（服务地址）、`QEV_DECISION_WEIGHTS`、`QEV_PROCESSOR`（`hf` 或 `numpy` 图像/视频处理器；`auto` 在装有 torch 时使用 Transformers 的版本）。
