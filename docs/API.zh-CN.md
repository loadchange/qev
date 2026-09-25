# Qev API

[English](API.md) | **简体中文**

Qev 提供两个入口：`/v1/systemone` 返回固定候选的概率；`/v1/chat/completions` 使用完整 Qwen3.5 的原生文本、图像和视频生成能力。原生生成会禁用 Qev 决策 LoRA，使用保留的视觉模块、语言模型和词表输出头。

当前决策适配器只使用文本训练。图像和视频可以进入决策前向计算，但**多模态决策准确率只有有限的零样本测量**：已发布的贪吃蛇检查点在 500 道 A-OKVQA 题上带图答对 69.4%（无图 32.6%，随机 25%），见[骨干对照实验](../experiments/diffusion/README.zh-CN.md)；文本任务拟合的温度不会应用到这些概率。API 兼容不表示与 Jev 的模型质量或置信度数值相同。

## 启动与模型名

```bash
uv run qev serve --model models/qev-snake-0.8b-mlx --backend mlx --port 8008
# 或指定完整 Torch checkpoint：
uv run qev serve --model models/qev-0.8b --backend torch --port 8008
```

默认只监听 `127.0.0.1`。本地接口不要求 API key。

| 接口 | 用途 |
|---|---|
| `GET /`、`GET /snake` | 内置模型实验室 / 自主贪吃蛇页面 |
| `GET /playground` | 接口测试页面 |
| `GET /health` | 服务与当前 backend |
| `GET /v1/models` | TypeSafe 风格模型列表 |
| `POST /v1/systemone` | Noul、Choice、Score 决策 |
| `POST /v1/chat/completions` | 非流式原生 Qwen 生成 |

页面与模型接口同源，无需单独构建前端。网页首次按浏览器语言选择英文或简体中文，并提供手动切换；选择保存在当前浏览器，刷新后仍生效。界面语言不会改写用户输入或 API 字段。贪吃蛇的创建、单步、导出等接口位于 `/api/snake/games`，状态与决策均由服务端维护，具体协议见 [实验室说明](DEMO.zh-CN.md)。这些演示接口不改变原有 Jev / TypeSafe 协议。

可用别名包括 `qev-latest`、`qev:0.8b`、`qev:0.8b-mlx`、`qev-0.8b`、`qev-0.8b-mlx` 和旧客户端的 `jev-latest`。生成入口默认 `qev-native`。这些名称指向**当前已加载的 checkpoint**，不会在请求内切换权重或 Torch/MLX 后端。

## 本地请求日志

`qev serve` 默认将 `POST /v1/systemone` 和 `POST /v1/chat/completions` 记录到 `$QEV_HOME/request-logs`（默认 `~/.qev/request-logs`）。可用 `--request-log-dir PATH` 更改目录，或用 `--no-request-log` 关闭。程序化调用 `create_app(agent)` 默认不写文件，需显式传入 `request_log` 才启用。GET 请求及贪吃蛇演示接口不在此日志范围内。

每次请求使用 UTC 时间加 UUID 命名的独立目录，包含原始 `request.json`（保留完整 data URL，可用于重放）、已完成响应的 `response.json`，以及记录请求 ID、时间、端点、状态、耗时和模型信息的 `metadata.json`。无效 JSON 也原样保留，并标记 `request_parse_error`。`media/` 提取保存原始 PNG/JPEG/WebP 字节，包括视频采样帧，不重新编码；每项媒体记录 `json_pointer`、`path`、`mime`、`bytes` 和 `sha256`，便于与请求核对。日志根目录的 `index.jsonl` 保存精简索引。不采集 HTTP 请求头及其中的认证凭据，这些文件也不通过 HTTP 提供。

已完成的记录响应带有 `X-Qev-Request-Id` 和 `X-Qev-Log-Status: saved` 或 `error`；日志写入失败不会改变模型响应。关闭日志时不添加这两个头。超过 12 MiB HTTP 上限的请求在记录前被拒绝，不保留请求体。未捕获的推理异常在日志写入成功时保留原请求和异常状态；最终错误响应由外层处理器生成，因此可能没有 `response.json` 或日志响应头。

## TypeSafe SDK

已使用官方 `typesafe-sdk==0.7.0` 检查请求序列化、模型列表及三种回答的 SDK 解析。`instructions` 可以省略；SDK 的默认 `None` 与空说明兼容。

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8008",
    model="qev:0.8b",
)
result = client.system_one(
    state="I was charged twice. Please refund the duplicate.",
    questions={
        "team": Choice(instructions="Who handles this?", criteria={
            "billing": "Payments and refunds", "technical": "Software problems",
        }),
        "refund": Noul(instructions="Does the customer request a refund?"),
        "urgency": Score(instructions="How urgent?", criteria=["Low", "Medium", "High"]),
    },
)
print(result.choices["team"].probabilities)
print(result.nouls["refund"].noul)
print(result.scores["urgency"].score)
```

`state` 仍接受字符串、对象、数组等 JSON。普通结构化 JSON 按字段名和顺序渲染，保持原文本接口。Choice 接受 1–255 个命名候选；Score 接受 2–255 个有序等级；每个请求最多 64 个问题。

回答包含：

- Noul：`noul = P(true)`。
- Choice：`choice`、候选概率及 `confidence`。
- Score：从 0 开始的期望等级、`legend`、等级概率及 `confidence`。

概率保留数值精度，不以生成 JSON 文本的方式产生。`confidence` 不是正确率保证；Score 的置信度是按离众数等级的期望距离计算的近似约定，TypeSafe 的精确公式未公开。`qev.question_temperatures` 按问题名报告实际温度：纯文本题沿用 checkpoint 校准，含图片或视频的题使用 `1.0`。所有题温度一致时，`qev.temperature` 返回该值，否则为 `null`；混合请求中的纯文本题仍保留文本校准。决策接口的 `usage.output_tokens` 恒为 0。

默认 HTTP 服务和 Python `Agent` 逐题推理，Torch 的默认 `batch_size=1` 与 MLX 的逐题行为一致，避免一个问题的计算形状随同请求内其他问题变化。Python 使用者可显式创建 `Agent(checkpoint, batch_size=4)` 以提高 Torch 吞吐；CUDA BF16 下批量与 padding 形状变化可能带来概率差异，因此该可选模式不保证与逐题结果相同。训练和离线评估仍保持各自的 batch 4 设置。

## 图像决策

接口页 <http://127.0.0.1:8008/playground> 提供纯文本、背景带图片、所有选项带图片三个 `choice` 场景。它们都请求 `/v1/systemone`，返回 `choice` 和 `probabilities`，不生成回答文字。纯文本仍可直接传字符串背景和候选说明；背景或候选带图片时，使用 OpenAI 风格 content array，或 `{"content": [...]}` 包装。图片只能作为 PNG、JPEG 或 WebP 的 base64 data URL 传入。

网页中的背景图片区和每个候选图片区都支持上传或粘贴。选择对应区域的粘贴入口后按 Ctrl/Cmd+V；浏览器支持剪贴板读取时，也可点击剪贴板按钮。上传和粘贴均在浏览器中转为相同的内联图片协议，并遵守下文的媒体预算；只有发送请求时才提交给服务。

背景图片示例：

```python
import base64
import httpx
from pathlib import Path

# 文件由客户端主动读取；服务端不会打开客户端指定的路径。
image = "data:image/png;base64," + base64.b64encode(Path("photo.png").read_bytes()).decode()
state = [
    {"type": "text", "text": "判断图片的主要内容。"},
    {"type": "image_url", "image_url": {"url": image}},
]
response = httpx.post("http://127.0.0.1:8008/v1/systemone", json={
    "model": "qev:0.8b",
    "state": state,
    "questions": {
        "scene": {"type": "choice", "instructions": "选择主要场景。",
                  "criteria": {"indoors": "室内", "outdoors": "室外"}},
    },
}, timeout=120)
response.raise_for_status()
print(response.json())
```

每个候选各带图片时，将图片放进对应的 `criteria[key].content`，保留稳定的候选 key。候选内容支持文字和图片，视频采样帧仍放在 `state`。下面的客户端代码沿用上面的导入，读取两张本地 PNG 后提交；服务只接收图片字节：

```python
def candidate(path, description):
    url = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    return {"content": [
        {"type": "text", "text": description},
        {"type": "image_url", "image_url": {"url": url}},
    ]}

response = httpx.post("http://127.0.0.1:8008/v1/systemone", json={
    "model": "qev:0.8b",
    "state": "从下面的候选图片中选择红色圆形。",
    "questions": {
        "shape": {
            "type": "choice",
            "instructions": "选择最符合背景要求的图片。",
            "criteria": {
                "a": candidate("candidate-a.png", "候选 A"),
                "b": candidate("candidate-b.png", "候选 B"),
            },
        },
    },
}, timeout=120)
response.raise_for_status()
answer = response.json()["answers"]["shape"]
print(answer["choice"], answer["probabilities"])
```

候选图片在各自的候选边界内编码，保留视觉 patch、图像网格和多模态位置编码，每个问题独立运行；同一问题也可以同时包含背景图和候选图。不会把 base64 当成文本，也不会用图片说明替代真实视觉输入。含媒体的问题使用 `T=1`，响应标明 `multimodal_decision_accuracy_validated: false`；当前训练监督仍为文本，这项输入支持不代表已验证视觉选择准确率。

## 原生多模态生成

```python
response = httpx.post("http://127.0.0.1:8008/v1/chat/completions", json={
    "model": "qev-native",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "请描述这张图片。"},
        {"type": "image_url", "image_url": {"url": image}},
    ]}],
    "max_tokens": 128,
    "temperature": 0,
}, timeout=120)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

返回 OpenAI 风格 `chat.completion`，包括 `choices`、`finish_reason` 及真实 `prompt_tokens/completion_tokens/total_tokens`。`qev.decision_adapter_enabled` 为 `false`。此入口调用原生生成头，不使用候选指针头；决策训练的准确率不用于描述原生生成质量。

支持 `system/user/assistant` 消息，`max_tokens` 或 `max_completion_tokens` 二选一，范围 1–2048，默认 256。`temperature=0` 使用贪心生成；大于 0 时支持 `top_p`。可传 `enable_thinking: true` 使用原生 thinking 模板。当前仅支持 `stream: false`，未实现工具调用或音频协议。

## 视频

视频采用 Qev 的显式采样帧扩展；先在客户端抽帧，再提交按时间排序、尺寸相同的图像 data URL：

```python
video_content = {
    "type": "video",
    "frames": [first_frame_data_url, second_frame_data_url],
    "fps": 2.0,
}
# 原生生成：messages=[{"role":"user","content":[
#   {"type":"text","text":"描述两个画面间的变化。"}, video_content]}]
# 候选决策：state={"content":[video_content]}
```

`fps` 表示**已提交帧序列的采样帧率**，不是原文件帧率；原生 processor 按此计算帧时间，并禁用再次抽帧。接口不接收本地视频路径、远程 `video_url` 或原始 MP4 字节；这是一项 API 输入约定，不是删除 Qwen3.5 的视频模块。

## 输入预算与错误

整个请求的 `state` 图片/视频帧与所有问题的 `criteria` 图片共享媒体预算：最多 8 张，单图解码后文件字节不超过 2 MiB，总计不超过 8 MiB；单图最多 400 万像素，总计最多 800 万像素；HTTP body 最多 12 MiB。帧率必须大于 0 且不超过 60。只接受单帧 PNG/JPEG/WebP，动画需拆成视频帧。

默认多模态决策展开后最多 8192 tokens；原生生成的 prompt 加生成预算最多 16384 tokens。超限明确报错，不静默丢弃图片、视频帧或候选项。原有纯文本决策仍采用 checkpoint 的文本长度预算，并在 `qev.truncated_questions` 报告状态截断。

无效媒体、超出内容预算或不支持的选项返回 422；HTTP body 超限返回 413；未知模型返回 404；不支持完整多模态能力的旧 checkpoint/backend 返回 501。服务端不会下载图片 URL、打开 `file://` 或执行客户端提供的本地路径。
