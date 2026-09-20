# Qev API

Qev 提供两个入口：`/v1/systemone` 返回固定候选的概率；`/v1/chat/completions` 使用完整 Qwen3.5 的原生文本、图像和视频生成能力。原生生成会禁用 Qev 决策 LoRA，使用保留的视觉模块、语言模型和词表输出头。

当前决策适配器只使用文本训练。图像和视频可以进入决策前向计算，但**尚未验证多模态决策准确率**；文本任务拟合的温度不会应用到这些概率。API 兼容不表示与 Jev 的模型质量或置信度数值相同。

## 启动与模型名

```bash
uv run qev serve --model models/qev-0.8b-mlx --port 8008
# 或指定完整 Torch checkpoint：
uv run qev serve --model models/qev-0.8b --backend torch --port 8008
```

默认只监听 `127.0.0.1`。本地接口不要求 API key。

| 接口 | 用途 |
|---|---|
| `GET /health` | 服务与当前 backend |
| `GET /v1/models` | TypeSafe 风格模型列表 |
| `POST /v1/systemone` | Noul、Choice、Score 决策 |
| `POST /v1/chat/completions` | 非流式原生 Qwen 生成 |

可用别名包括 `qev-latest`、`qev:0.8b`、`qev:0.8b-mlx`、`qev-0.8b`、`qev-0.8b-mlx` 和旧客户端的 `jev-latest`。生成入口默认 `qev-native`。这些名称指向**当前已加载的 checkpoint**，不会在请求内切换权重或 Torch/MLX 后端。

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

概率保留数值精度，不以生成 JSON 文本的方式产生。`confidence` 不是正确率保证；Score 的置信度是按离众数等级的期望距离计算的近似约定，TypeSafe 的精确公式未公开。文本请求的 `qev.temperature` 显示实际使用的校准温度；校准对未见任务的效果需要独立测量。决策接口的 `usage.output_tokens` 恒为 0。

默认 HTTP 服务和 Python `Agent` 逐题推理，Torch 的默认 `batch_size=1` 与 MLX 的逐题行为一致，避免一个问题的计算形状随同请求内其他问题变化。Python 使用者可显式创建 `Agent(checkpoint, batch_size=4)` 以提高 Torch 吞吐；CUDA BF16 下批量与 padding 形状变化可能带来概率差异，因此该可选模式不保证与逐题结果相同。训练和离线评估仍保持各自的 batch 4 设置。

## 图像决策

使用 OpenAI 风格 content array，或 `{"content": [...]}` 包装。图片只能作为 PNG、JPEG 或 WebP 的 base64 data URL 传入。

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

此路径会保留视觉 patch、图像网格和多模态位置编码，每个问题独立运行。不会把 base64 当成文本，也不会用图片说明替代真实视觉输入。响应会标明 `multimodal_decision_accuracy_validated: false`。

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

整个请求最多 8 张图片/视频帧，单图解码后文件字节不超过 2 MiB，总计不超过 8 MiB；单图最多 400 万像素，总计最多 800 万像素；HTTP body 最多 12 MiB。帧率必须大于 0 且不超过 60。只接受单帧 PNG/JPEG/WebP，动画需拆成视频帧。

默认多模态决策展开后最多 8192 tokens；原生生成的 prompt 加生成预算最多 16384 tokens。超限明确报错，不静默丢弃图片、视频帧或候选项。原有纯文本决策仍采用 checkpoint 的文本长度预算，并在 `qev.truncated_questions` 报告状态截断。

无效媒体、超出内容预算或不支持的选项返回 422；HTTP body 超限返回 413；未知模型返回 404；不支持完整多模态能力的旧 checkpoint/backend 返回 501。服务端不会下载图片 URL、打开 `file://` 或执行客户端提供的本地路径。
