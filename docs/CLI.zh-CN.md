# Qev 命令行

[English](CLI.md) | **简体中文**

`qev` 可以在终端里直接让模型做结构化判断，也可以提供 HTTP API 和网页实验室，并负责模型管理。在 Apple Silicon 上它基于 MLX 运行，不需要 PyTorch。

## 安装

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
qev pull            # 下载并校验默认模型（3.5 GB），存放在 ~/.qev
qev doctor          # 检查芯片、MLX/Metal、模型文件和本机服务
```

要求：Apple Silicon，macOS 14 或更新。Homebrew 可能会先要求信任这个 tap（`brew trust loadchange/qev`）。在源码目录中，`uv sync --python 3.12 --extra mlx` 后用 `uv run qev …` 效果相同。

`qev pull --from 目录 [--link]` 可以导入已经下载好的检查点（例如用 `hf download twainsk/qev-0.8b-mlx` 下载的），导入前会按 qev 自带的发布清单逐个校验运行时文件。

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

`--decision-weights`（或 `$QEV_DECISION_WEIGHTS`）决定 MLX 运行时如何计算决策。原生生成始终使用未经改动的底模。

| 模式 | 决策路径 | M4 上每步贪吃蛇决策 | 内存 |
|---|---|---|---|
| `adapter` | float32 底模，LoRA 单独计算（经过验证的参考路径） | 219 ms | 3.22 GB |
| `merged` | float32 底模加一份合并后的 float32 决策权重 | 185 ms | 5.07 GB |
| `merged-bf16` | float32 底模加一份合并后的 bfloat16 决策权重 | 171 ms | 4.15 GB |
| `bf16` | 整个模型 bfloat16，决策权重合并 | 149 ms | 2.54 GB |

在 1,380 道开发题上，四种模式准确率都是 87.9%（贪吃蛇 98.4%，通用 81.9%）；相对 `adapter`，`merged-bf16` 改变了 5 个决策，`bf16` 改变了 7 个，净准确率不变（[结果](results/decision_weights.json)）。`bf16` 模式下原生生成也使用 bfloat16，生成文本可能在几个 token 之后与 float32 不同。命令行默认使用 `adapter`，Homebrew 服务使用 `bf16`。

## 其他命令

| 命令 | 作用 |
|---|---|
| `qev chat "提示" [--image 路径]` | 关闭决策 adapter 的原生文本/图像生成 |
| `qev list`、`qev pull 名称`、`qev rm 名称` | 管理 `$QEV_HOME/models`（默认 `~/.qev`）中的注册表模型 |
| `qev doctor [--verify] [--json]` | 环境、模型和服务检查 |
| `qev snake` | 终端贪吃蛇；在源码目录外会使用运行中的服务或已安装的模型 |
| `qev predict --model 目录 --request 文件` | 运行一个请求文件并输出 JSON（为兼容保留） |

环境变量：`QEV_HOME`（模型和日志）、`QEV_MODEL`（默认模型）、`QEV_SERVER`（服务地址）、`QEV_DECISION_WEIGHTS`、`QEV_PROCESSOR`（`hf` 或 `numpy` 图像/视频处理器；`auto` 在装有 torch 时使用 Transformers 的版本）。
