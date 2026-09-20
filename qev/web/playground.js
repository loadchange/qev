const MIB = 1024 * 1024;
const ROUTES = {
  "/v1/systemone": { method: "POST", label: "结构化决策" },
  "/v1/chat/completions": { method: "POST", label: "原生生成" },
  "/health": { method: "GET", label: "服务状态" },
  "/v1/models": { method: "GET", label: "模型列表" },
};
const PRESETS = {
  decisions: {
    label: "Choice + Noul + Score",
    endpoint: "/v1/systemone",
    note: "一次请求测试分类、真假概率和有序评分；每个问题独立推理。",
    body: {
      model: "qev-latest",
      state: "用户说：昨天同一笔订单被扣了两次钱，请退还多扣的钱。",
      questions: {
        team: { type: "choice", instructions: "请选择处理这条请求的部门。", criteria: {
          billing: "账单与退款", technical: "软件与技术支持", sales: "售前咨询",
        } },
        refund: { type: "noul", instructions: "用户是否在请求退款？" },
        urgency: { type: "score", instructions: "判断请求的紧急程度。", criteria: ["低", "中", "高"] },
      },
    },
  },
  text: {
    label: "原生文字生成",
    endpoint: "/v1/chat/completions",
    note: "关闭决策适配器，使用完整 Qwen 原生生成路径；此页面使用非流式响应。",
    body: { model: "qev-native", messages: [{ role: "user", content: "用两句话解释什么是机器学习。" }], max_tokens: 128, temperature: 0, stream: false },
  },
  image: {
    label: "原生图片理解",
    endpoint: "/v1/chat/completions",
    note: "先添加图片，再发送。图片会以内嵌 data URL 放入当前用户消息。",
    body: { model: "qev-native", messages: [{ role: "user", content: [{ type: "text", text: "请简洁描述图片中最主要的内容。" }] }], max_tokens: 128, temperature: 0, stream: false },
  },
  video: {
    label: "原生视频帧理解",
    endpoint: "/v1/chat/completions",
    note: "添加多张图片作为采样帧，按文件名排序。帧尺寸须相同；fps 表示采样帧率。",
    body: { model: "qev-native", messages: [{ role: "user", content: [{ type: "text", text: "请按时间顺序描述这些视频帧中的变化。" }] }], max_tokens: 160, temperature: 0, stream: false },
  },
  health: { label: "GET · 服务状态", endpoint: "/health", note: "读取当前服务状态与后端。", body: null },
  models: { label: "GET · 模型列表", endpoint: "/v1/models", note: "读取服务公开的模型别名。别名不会切换已加载的模型。", body: null },
};
let mountCount = 0;

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") element.textContent = text;
  return element;
}

function parsePayload(text) {
  const value = JSON.parse(text, (_key, item) => {
    if (typeof item === "number" && !Number.isFinite(item)) throw new Error("JSON 数值超出有限范围。");
    return item;
  });
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("请求 JSON 的最外层须为对象。");
  return value;
}

function contentArrays(payload, endpoint) {
  if (endpoint === "/v1/chat/completions") {
    return Array.isArray(payload.messages) ? payload.messages.filter(Boolean).map((message) => message.content).filter(Array.isArray) : [];
  }
  const state = payload.state;
  if (Array.isArray(state) && state.length
      && state.every((item) => item && typeof item === "object" && !Array.isArray(item))
      && state.some((item) => ["text", "image_url", "video", "video_url", "audio_url", "input_audio"].includes(item.type))) return [state];
  if (state && typeof state === "object" && Object.keys(state).length === 1 && Array.isArray(state.content)) return [state.content];
  return [];
}

function mediaEntries(payload, endpoint) {
  const entries = [];
  let videos = 0;
  for (const content of contentArrays(payload, endpoint)) {
    for (const item of content) {
      if (item?.type === "image_url") entries.push({ url: item.image_url?.url, label: `图片 ${entries.length + 1}` });
      if (item?.type === "video" && Array.isArray(item.frames)) {
        videos += 1;
        item.frames.forEach((url, index) => entries.push({ url, label: `视频 ${videos} · 帧 ${index + 1}` }));
      }
    }
  }
  return entries;
}

function inlineImageBytes(url) {
  if (typeof url !== "string") return null;
  const match = /^data:image\/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})$/.exec(url);
  if (!match || !match[1] || match[1].length % 4) return null;
  return match[1].length * 3 / 4 - (match[1].endsWith("==") ? 2 : match[1].endsWith("=") ? 1 : 0);
}

function mediaTarget(payload, endpoint) {
  if (endpoint === "/v1/chat/completions") {
    if (!Array.isArray(payload.messages)) throw new Error("原生请求须包含 messages 数组。");
    let message = null;
    for (let index = payload.messages.length - 1; index >= 0; index -= 1) {
      if (payload.messages[index]?.role === "user") { message = payload.messages[index]; break; }
    }
    if (!message) { message = { role: "user", content: [] }; payload.messages.push(message); }
    if (typeof message.content === "string") message.content = [{ type: "text", text: message.content }];
    if (!Array.isArray(message.content)) throw new Error("用户消息的 content 须为文字或数组。");
    return message.content;
  }
  const existing = contentArrays(payload, endpoint)[0];
  if (existing) return existing;
  const text = typeof payload.state === "string" ? payload.state : JSON.stringify(payload.state ?? "", null, 2);
  payload.state = [{ type: "text", text }];
  return payload.state;
}

function compactPreview(value) {
  const text = JSON.stringify(value, (_key, item) => {
    if (typeof item !== "string") return item;
    if (item.startsWith("data:image/")) return `${item.slice(0, item.indexOf(",") + 1)}[媒体已折叠，${item.length.toLocaleString()} 字符]`;
    return item.length > 5000 ? `${item.slice(0, 5000)}…[长文本已折叠]` : item;
  }, 2);
  return text.length > 16000 ? `${text.slice(0, 16000)}\n…[预览已截短；展开或复制可获取完整响应]` : text;
}

function shellQuote(value) { return `'${value.replaceAll("'", "'\\''")}'`; }

function requestCode(snapshot, language, preview = false) {
  const url = `${window.location.origin}${snapshot.endpoint}`;
  const body = preview && snapshot.payload ? compactPreview(snapshot.payload) : snapshot.body;
  if (language === "python") {
    if (snapshot.method === "GET") return `import httpx\n\nresponse = httpx.get(${JSON.stringify(url)}, timeout=120)\nresponse.raise_for_status()\nprint(response.json())`;
    return `import json\nimport httpx\n\npayload = json.loads(${JSON.stringify(body)})\nresponse = httpx.post(\n    ${JSON.stringify(url)},\n    json=payload,\n    timeout=120,\n)\nresponse.raise_for_status()\nprint(response.json())`;
  }
  if (snapshot.method === "GET") return `curl ${shellQuote(url)}`;
  let delimiter = "QEV_REQUEST_JSON";
  const lines = new Set(body.split("\n"));
  while (lines.has(delimiter)) delimiter += "_END";
  return `curl --request POST ${shellQuote(url)} \\\n  --header 'Content-Type: application/json' \\\n  --data-binary @- <<'${delimiter}'\n${body}\n${delimiter}`;
}

function numeric(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", { maximumFractionDigits: digits }) : "—";
}

/** Mount a same-origin API workbench. All request history stays in this mount. */
export function mountPlayground(container, { notify = () => {} } = {}) {
  const id = `qev-pg-${++mountCount}`;
  const listeners = new AbortController();
  const readers = new Set();
  const probes = new Set();
  const mediaNames = new Map();
  const dimensions = new Map();
  const history = [];
  let active = null;
  let disposed = false;
  let uploading = false;
  let ticker = null;
  let editorTimer = null;
  let historyId = 0;
  let fullResponse = "";
  let language = "curl";

  const root = node("section", "playground-grid");
  root.setAttribute("aria-label", "API Playground");
  const requestPanel = node("section", "panel playground-request");
  const responsePanel = node("section", "panel playground-response");
  const requestHeading = node("div", "panel-header");
  const requestTitle = node("div");
  requestTitle.append(node("p", "eyebrow", "API WORKBENCH"), node("h2", "section-title", "请求"));
  requestHeading.append(requestTitle, node("span", "badge", "同源 · 真实接口"));
  requestPanel.append(requestHeading);

  function on(element, event, callback) { element.addEventListener(event, callback, { signal: listeners.signal }); }
  function button(label, className = "btn") { const result = node("button", className, label); result.type = "button"; return result; }
  function field(label, control, suffix) {
    const wrapper = node("div", "field");
    const labelNode = node("label", "", label);
    control.id = `${id}-${suffix}`;
    labelNode.htmlFor = control.id;
    wrapper.append(labelNode, control);
    return wrapper;
  }
  function tell(message, kind = "info") { if (!disposed) notify(message, kind); }

  const preset = node("select", "pg-select");
  preset.append(new Option("自定义请求", ""));
  for (const [key, value] of Object.entries(PRESETS)) preset.append(new Option(value.label, key));
  const endpoint = node("select", "pg-select code");
  for (const [path, route] of Object.entries(ROUTES)) endpoint.append(new Option(`${route.method} ${path}`, path));
  const pickers = node("div", "pg-pickers");
  pickers.append(field("示例", preset, "preset"), field("接口", endpoint, "endpoint"));
  const presetNote = node("p", "hint");
  const getTools = node("div", "toolbar pg-get-tools");
  const healthButton = button("检查服务", "btn small ghost");
  const modelsButton = button("查看模型", "btn small ghost");
  getTools.append(healthButton, modelsButton);
  requestPanel.append(pickers, presetNote, getTools);

  const mediaBox = node("details", "pg-media");
  mediaBox.setAttribute("aria-label", "媒体附件");
  mediaBox.append(node("summary", "", "图片与视频帧（可选）"));
  const mediaTools = node("div", "toolbar");
  const imageButton = button("添加图片", "btn small");
  const videoButton = button("添加视频帧", "btn small");
  const clearMediaButton = button("移除所有媒体", "btn small ghost");
  const fps = node("input", "pg-fps");
  Object.assign(fps, { type: "number", min: "0.001", max: "60", step: "any", value: "2" });
  const imageInput = node("input");
  const videoInput = node("input");
  for (const input of [imageInput, videoInput]) {
    Object.assign(input, { type: "file", accept: "image/png,image/jpeg,image/webp", multiple: true, hidden: true });
  }
  mediaTools.append(imageButton, videoButton, field("帧率 fps", fps, "fps"), clearMediaButton);
  const mediaHint = node("p", "hint", "PNG / JPEG / WebP；每张 ≤ 2 MiB，总计 ≤ 8 MiB，最多 8 张图片/帧。视频帧按文件名排序、尺寸须一致。上传只在发送请求时传给服务。");
  const mediaPreview = node("div", "pg-media-grid");
  mediaBox.append(mediaTools, imageInput, videoInput, mediaHint, mediaPreview);
  requestPanel.append(mediaBox);

  const editorDetails = node("details", "pg-editor");
  editorDetails.open = true;
  const editorSummary = node("summary", "", "请求 JSON");
  const editor = node("textarea", "code pg-json-editor");
  Object.assign(editor, { rows: 18, spellcheck: false });
  editor.setAttribute("autocomplete", "off");
  editor.setAttribute("autocapitalize", "off");
  editor.setAttribute("aria-label", "请求 JSON 编辑器");
  const editorHelp = node("p", "hint", "此处保留完整请求。含大量 base64 时可折叠编辑器，用上方缩略图检查附件。");
  editorHelp.id = `${id}-editor-help`;
  editor.setAttribute("aria-describedby", editorHelp.id);
  editorDetails.append(editorSummary, editor, editorHelp);
  const editorTools = node("div", "toolbar pg-editor-tools");
  const formatButton = button("格式化 JSON", "btn small ghost");
  const copyRequestButton = button("复制请求 JSON", "btn small ghost");
  editorTools.append(formatButton, copyRequestButton);
  const localError = node("p", "error pg-local-error");
  localError.setAttribute("role", "alert");
  localError.hidden = true;
  const sendTools = node("div", "toolbar pg-send-tools");
  const sendButton = button("发送请求", "btn primary");
  const cancelButton = button("停止等待", "btn ghost");
  cancelButton.disabled = true;
  sendTools.append(sendButton, cancelButton);
  requestPanel.insertBefore(sendTools, pickers);
  requestPanel.append(editorDetails, editorTools, localError,
    node("p", "hint", "停止等待只取消浏览器的等待，服务可能仍在推理。"));

  const examples = node("details", "pg-examples");
  examples.append(node("summary", "", "在终端或 Python 中调用"));
  const codeTools = node("div", "toolbar");
  const curlTab = button("curl", "btn small");
  const pythonTab = button("Python · httpx", "btn small ghost");
  const copyCodeButton = button("复制 curl", "btn small ghost");
  codeTools.append(curlTab, pythonTab, copyCodeButton);
  const codePreview = node("pre", "code pg-code-preview");
  codePreview.tabIndex = 0;
  examples.append(codeTools, codePreview,
    node("p", "hint", "代码预览会折叠媒体和超长文本；复制按钮保留完整请求。Python 示例需要 httpx。"));
  const historyBox = node("section", "pg-history-section");
  historyBox.append(node("h3", "section-title", "最近请求"), node("p", "hint", "保留最近 5 次请求，含附件；刷新或关闭页面后清空。恢复后需要重新发送。"));
  const historyList = node("ol", "pg-history");
  historyBox.append(historyList);
  requestPanel.append(examples, historyBox);

  const responseHeader = node("div", "panel-header");
  const responseTitle = node("div");
  responseTitle.append(node("p", "eyebrow", "LIVE RESPONSE"), node("h2", "section-title", "响应"));
  const responseBadge = node("span", "badge", "尚未发送");
  responseHeader.append(responseTitle, responseBadge);
  const metrics = node("dl", "pg-metrics");
  function metric(label) {
    const item = node("div", "metric");
    const value = node("dd", "", "—");
    item.append(node("dt", "", label), value);
    metrics.append(item);
    return value;
  }
  const statusMetric = metric("HTTP 状态");
  const timeMetric = metric("往返耗时");
  const tokenMetric = metric("输入 / 输出 token");
  const responseSummary = node("div", "pg-response-summary empty-state", "发送一个请求，查看模型的真实响应。");
  responseSummary.setAttribute("aria-live", "polite");
  const responseTools = node("div", "toolbar");
  const copyResponseButton = button("复制完整响应", "btn small ghost");
  copyResponseButton.disabled = true;
  responseTools.append(node("h3", "section-title", "响应预览"), copyResponseButton);
  const responsePreview = node("pre", "code pg-response-preview", "尚无响应。");
  responsePreview.tabIndex = 0;
  const rawDetails = node("details", "pg-raw-response");
  const rawSummary = node("summary", "", "展开完整原始响应");
  const rawPre = node("pre", "code pg-response-raw");
  rawPre.tabIndex = 0;
  rawDetails.append(rawSummary, rawPre);
  rawDetails.hidden = true;
  responsePanel.append(responseHeader, metrics, responseSummary, responseTools, responsePreview, rawDetails);
  root.append(requestPanel, responsePanel);
  container.replaceChildren(root);

  function isGet() { return ROUTES[endpoint.value].method === "GET"; }
  function error(message) { if (disposed) return; localError.textContent = message; localError.hidden = false; tell(message, "error"); }
  function clearError() { localError.hidden = true; localError.textContent = ""; }
  function setBusy() {
    const busy = Boolean(active) || uploading;
    for (const control of [preset, endpoint, fps, healthButton, modelsButton, sendButton]) control.disabled = busy;
    for (const control of [imageButton, videoButton, clearMediaButton, imageInput, videoInput, formatButton, copyRequestButton]) control.disabled = busy || isGet();
    editor.readOnly = busy;
    cancelButton.disabled = !active;
    copyCodeButton.disabled = uploading;
    copyResponseButton.disabled = !fullResponse;
    sendButton.textContent = active ? "请求中…" : uploading ? "正在读取附件…" : "发送请求";
    responsePanel.setAttribute("aria-busy", String(Boolean(active)));
    for (const item of historyList.querySelectorAll("button")) item.disabled = busy;
    mediaBox.hidden = isGet();
    editorDetails.hidden = isGet();
    editorTools.hidden = isGet();
  }

  function snapshot() {
    const route = ROUTES[endpoint.value];
    if (!route) throw new Error("请选择支持的同源接口。");
    if (route.method === "GET") return { endpoint: endpoint.value, method: "GET", body: "", payload: null };
    const payload = parsePayload(editor.value);
    if (new TextEncoder().encode(editor.value).byteLength > 12 * MIB) throw new Error("请求超过 12 MiB，请减少附件或文字。");
    return { endpoint: endpoint.value, method: "POST", body: editor.value, payload };
  }

  function refreshCode() {
    curlTab.setAttribute("aria-pressed", String(language === "curl"));
    pythonTab.setAttribute("aria-pressed", String(language === "python"));
    curlTab.className = language === "curl" ? "btn small" : "btn small ghost";
    pythonTab.className = language === "python" ? "btn small" : "btn small ghost";
    copyCodeButton.textContent = language === "curl" ? "复制 curl" : "复制 Python";
    if (!examples.open) return;
    try { codePreview.textContent = requestCode(snapshot(), language, true); }
    catch (problem) { codePreview.textContent = `请先完成有效的 JSON 请求：${problem.message}`; }
  }

  function refreshEditor() {
    const bytes = new TextEncoder().encode(editor.value).byteLength;
    editorSummary.textContent = `请求 JSON · ${numeric(bytes / 1024, 1)} KiB`;
    mediaPreview.replaceChildren();
    if (!isGet()) {
      try {
        const entries = mediaEntries(parsePayload(editor.value), endpoint.value);
        for (const entry of entries.slice(0, 8)) {
          const figure = node("figure", "pg-media-item");
          if (inlineImageBytes(entry.url) !== null) {
            const image = node("img", "pg-thumbnail");
            Object.assign(image, { src: entry.url, alt: entry.label, width: 72, height: 72 });
            figure.append(image);
          } else figure.append(node("span", "hint", "无法预览此附件"));
          const name = mediaNames.get(entry.url);
          figure.append(node("figcaption", "hint", `${entry.label}${name ? ` · ${name}` : ""}`));
          mediaPreview.append(figure);
        }
        if (entries.length > 8) mediaPreview.append(node("p", "error", `当前有 ${entries.length} 张图片/帧；服务最多接受 8 张。`));
      } catch { /* Incomplete JSON is normal while typing; send/format reports it. */ }
    }
    refreshCode();
  }

  function resetResponse(message = "请求已准备好，尚未发送。") {
    fullResponse = "";
    responseBadge.textContent = "尚未发送";
    responseBadge.className = "badge";
    statusMetric.textContent = timeMetric.textContent = tokenMetric.textContent = "—";
    responseSummary.className = "pg-response-summary empty-state";
    responseSummary.textContent = message;
    responsePreview.textContent = "尚无响应。";
    rawDetails.hidden = true;
    rawDetails.open = false;
    rawPre.textContent = "";
    setBusy();
  }

  function applyPreset(key) {
    if (active || uploading || !PRESETS[key]) return;
    const example = PRESETS[key];
    clearError();
    preset.value = key;
    endpoint.value = example.endpoint;
    presetNote.textContent = example.note;
    editor.value = example.body ? JSON.stringify(example.body, null, 2) : "";
    editorDetails.open = true;
    mediaBox.open = key === "image" || key === "video";
    resetResponse();
    refreshEditor();
  }

  function renderHistory() {
    historyList.replaceChildren();
    if (!history.length) { historyList.append(node("li", "hint", "还没有请求。")); return; }
    for (const entry of history) {
      const item = node("li", "pg-history-item");
      const restore = button("", "btn small ghost pg-restore");
      restore.dataset.historyId = String(entry.id);
      restore.textContent = `${entry.time} · ${entry.method} ${entry.endpoint} · ${entry.status}`;
      restore.title = "恢复此请求；不会自动发送";
      item.append(restore);
      historyList.append(item);
    }
    setBusy();
  }

  async function copy(text, label) {
    try {
      await navigator.clipboard.writeText(text);
      tell(`${label}已复制。`);
    } catch (problem) { error(`无法复制：${problem.message || "浏览器未开放剪贴板权限"}。可展开内容后手动选择复制。`); }
  }

  function summarize(data, ok, httpStatus) {
    responseSummary.className = "pg-response-summary";
    responseSummary.replaceChildren();
    if (!ok) {
      responseSummary.append(node("h3", "error", `请求失败 · HTTP ${httpStatus}`));
      const detail = data?.detail;
      if (Array.isArray(detail)) {
        const list = node("ul", "pg-errors");
        for (const item of detail.slice(0, 8)) list.append(node("li", "", `${Array.isArray(item?.loc) ? item.loc.join(".") : "请求"}：${item?.msg || "校验失败"}`));
        if (detail.length > 8) list.append(node("li", "hint", `另有 ${detail.length - 8} 项，见完整响应。`));
        responseSummary.append(list);
      } else responseSummary.append(node("p", "", typeof detail === "string" ? detail : "服务返回错误，详情见下方原始响应。"));
      return;
    }
    if (data?.answers && typeof data.answers === "object") {
      for (const [name, answer] of Object.entries(data.answers)) {
        const block = node("article", "pg-answer");
        block.append(node("h3", "", `${name} · ${answer?.type || "回答"}`));
        let value = "详见响应 JSON";
        if (answer?.type === "choice") value = String(answer.choice ?? "—");
        if (answer?.type === "noul") value = `P(true) = ${numeric(answer.noul * 100)}%`;
        if (answer?.type === "score") value = `期望等级 ${numeric(answer.score, 4)}`;
        block.append(node("p", "pg-answer-value", value));
        if (typeof answer?.confidence === "number") block.append(node("p", "hint", `confidence ${numeric(answer.confidence, 4)}`));
        if (answer?.probabilities && typeof answer.probabilities === "object") {
          const probabilities = Object.entries(answer.probabilities).sort((a, b) => Number(b[1]) - Number(a[1]));
          const list = node("ul", "pg-probabilities");
          for (const [key, probability] of probabilities.slice(0, 6)) list.append(node("li", "", `${key}：${numeric(probability * 100)}%`));
          if (probabilities.length > 6) list.append(node("li", "hint", `另有 ${probabilities.length - 6} 个候选，见完整 JSON。`));
          block.append(list);
        }
        responseSummary.append(block);
      }
    } else if (Array.isArray(data?.choices)) {
      for (const choice of data.choices) {
        const content = choice?.message?.content;
        responseSummary.append(node("p", "pg-native-text", typeof content === "string" ? content : "响应中没有文字 content，请查看原始 JSON。"));
        if (choice?.finish_reason) responseSummary.append(node("p", "hint", `结束原因：${choice.finish_reason}`));
      }
    } else if (Array.isArray(data?.models)) {
      const list = node("ul", "pg-models");
      for (const model of data.models) list.append(node("li", "", `${model?.name ?? model?.id ?? "未命名"}${model?.description ? ` — ${model.description}` : ""}`));
      responseSummary.append(list);
    } else if (data?.status) {
      responseSummary.append(node("p", "pg-answer-value", `服务状态：${data.status}`));
      if (data.backend) responseSummary.append(node("p", "hint", `当前后端：${data.backend}`));
    } else responseSummary.append(node("p", "", "已收到响应，内容见下方预览。"));
    if (data?.qev?.decision_adapter_enabled === false) responseSummary.append(node("p", "hint", "原生生成：决策适配器已关闭。"));
    if (Array.isArray(data?.qev?.input_modalities) && data.qev.input_modalities.some((item) => item !== "text") && data.qev.multimodal_decision_accuracy_validated === false) responseSummary.append(node("p", "hint", "当前版本尚未评估多模态决策准确率。"));
  }

  async function send() {
    if (active || uploading || disposed) return;
    clearError();
    let request;
    try {
      request = snapshot();
      const content = request.payload ? contentArrays(request.payload, request.endpoint).flat() : [];
      if (preset.value === "image" && !content.some((item) => item?.type === "image_url")) throw new Error("请先添加图片，再发送图片示例。");
      if (preset.value === "video" && !content.some((item) => item?.type === "video" && item.frames?.length)) throw new Error("请先添加视频帧，再发送视频示例。");
    } catch (problem) { resetResponse("当前请求未发送，请先修正输入。"); error(`未发送：${problem.message}`); return; }
    resetResponse("请求已发送，等待服务返回…");
    const run = { controller: new AbortController(), started: performance.now(), cancelled: false };
    active = run;
    const record = { ...request, payload: undefined, id: ++historyId, time: new Date().toLocaleTimeString("zh-CN", { hour12: false }), status: "等待响应" };
    history.unshift(record);
    history.splice(5);
    renderHistory();
    responseBadge.textContent = "等待响应";
    ticker = setInterval(() => { if (!disposed && active === run) timeMetric.textContent = `${numeric(performance.now() - run.started, 0)} ms`; }, 100);
    setBusy();
    try {
      const response = await fetch(request.endpoint, {
        method: request.method, credentials: "same-origin", signal: run.controller.signal,
        ...(request.method === "POST" ? { headers: { "Content-Type": "application/json" }, body: request.body } : {}),
      });
      statusMetric.textContent = String(response.status);
      const text = await response.text();
      if (disposed || active !== run) return;
      let data = null;
      try { data = JSON.parse(text); } catch { /* A real non-JSON response is shown unchanged. */ }
      fullResponse = text;
      rawDetails.hidden = !text;
      rawSummary.textContent = `展开完整原始响应 · ${numeric(new TextEncoder().encode(text).byteLength / 1024, 1)} KiB`;
      responsePreview.textContent = data === null ? (text.length > 16000 ? `${text.slice(0, 16000)}\n…[展开查看完整内容]` : text || "空响应体。") : compactPreview(data);
      responseBadge.textContent = response.ok ? `HTTP ${response.status}` : `HTTP ${response.status} · 失败`;
      responseBadge.className = response.ok ? "badge" : "badge error";
      record.status = `HTTP ${response.status}`;
      const usage = data?.usage;
      if (usage) tokenMetric.textContent = `${numeric(usage.input_tokens ?? usage.prompt_tokens, 0)} / ${numeric(usage.output_tokens ?? usage.completion_tokens, 0)}`;
      summarize(data, response.ok, response.status);
      if (!response.ok) tell(`请求失败：HTTP ${response.status}，错误已保留在响应中。`, "error");
    } catch (problem) {
      if (disposed || active !== run) return;
      const cancelled = run.cancelled || problem.name === "AbortError";
      const message = cancelled ? "已停止等待。服务可能仍在推理；这不代表服务端任务已取消。" : `网络请求失败：${problem.message || "未收到服务响应"}`;
      record.status = cancelled ? "停止等待" : "网络错误";
      responseBadge.textContent = record.status;
      responseBadge.className = cancelled ? "badge" : "badge error";
      responseSummary.className = "pg-response-summary";
      responseSummary.textContent = message;
      responsePreview.textContent = "未取得完整响应体。";
      tell(message, cancelled ? "info" : "error");
    } finally {
      clearInterval(ticker);
      ticker = null;
      if (!disposed && active === run) {
        timeMetric.textContent = `${numeric(performance.now() - run.started, 0)} ms`;
        active = null;
        renderHistory();
        setBusy();
      }
    }
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      readers.add(reader);
      const finish = () => readers.delete(reader);
      reader.onload = () => { finish(); resolve(reader.result); };
      reader.onerror = () => { finish(); reject(new Error(`无法读取 ${file.name}。`)); };
      reader.onabort = () => { finish(); reject(new Error("附件读取已停止。")); };
      reader.readAsDataURL(file);
    });
  }

  function imageDimensions(url) {
    if (dimensions.has(url)) return Promise.resolve(dimensions.get(url));
    return new Promise((resolve, reject) => {
      const image = new Image();
      probes.add(image);
      image.onload = () => {
        probes.delete(image);
        const size = { width: image.naturalWidth, height: image.naturalHeight };
        dimensions.set(url, size);
        if (dimensions.size > 16) dimensions.delete(dimensions.keys().next().value);
        resolve(size);
      };
      image.onerror = () => { probes.delete(image); reject(new Error("图片无法解码，请使用有效的 PNG、JPEG 或 WebP。")); };
      image.src = url;
    });
  }

  async function addFiles(files, asVideo) {
    if (!files.length || active || uploading || disposed) return;
    clearError();
    uploading = true;
    setBusy();
    try {
      const request = snapshot();
      if (request.method !== "POST") throw new Error("GET 接口不接受媒体，请先选择决策或原生生成接口。");
      const existing = mediaEntries(request.payload, request.endpoint);
      if (existing.length + files.length > 8) throw new Error("一个请求的图片与视频帧合计最多 8 张；请先移除部分媒体。");
      let totalBytes = 0;
      for (const item of existing) {
        const bytes = inlineImageBytes(item.url);
        if (bytes === null) throw new Error("现有附件不是有效的内嵌 PNG/JPEG/WebP data URL，请先修正 JSON。");
        if (bytes > 2 * MIB) throw new Error("现有图片超过每张 2 MiB 的限制。");
        totalBytes += bytes;
      }
      const rate = Number(fps.value);
      if (asVideo && (!Number.isFinite(rate) || rate <= 0 || rate > 60)) throw new Error("fps 须大于 0 且不超过 60。");
      const ordered = [...files];
      if (asVideo) ordered.sort((a, b) => a.name.localeCompare(b.name, "zh-CN", { numeric: true }));
      for (const file of ordered) {
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error(`${file.name}：只接受 PNG、JPEG 或 WebP 图片。`);
        if (!file.size || file.size > 2 * MIB) throw new Error(`${file.name}：图片须非空且不超过 2 MiB。`);
        totalBytes += file.size;
      }
      if (totalBytes > 8 * MIB) throw new Error("全部图片与帧的原始文件大小合计不能超过 8 MiB。");
      const urls = [];
      for (const file of ordered) {
        const url = await readFile(file);
        if (disposed) return;
        urls.push(url);
        mediaNames.set(url, file.name);
        if (mediaNames.size > 16) mediaNames.delete(mediaNames.keys().next().value);
      }
      let totalPixels = 0;
      const sizes = [];
      for (const url of [...existing.map((entry) => entry.url), ...urls]) {
        const size = await imageDimensions(url);
        if (disposed) return;
        const pixels = size.width * size.height;
        if (!pixels || pixels > 4000000) throw new Error("每张图片最多 400 万像素，请先缩小图片。");
        totalPixels += pixels;
        sizes.push(size);
      }
      if (totalPixels > 8000000) throw new Error("请求内所有图片与帧合计最多 800 万像素。");
      const newSizes = sizes.slice(existing.length);
      if (asVideo && newSizes.some((size) => size.width !== newSizes[0].width || size.height !== newSizes[0].height)) throw new Error("同一视频的所有采样帧须有相同宽度和高度。");
      const content = mediaTarget(request.payload, request.endpoint);
      if (asVideo) content.push({ type: "video", frames: urls, fps: rate });
      else content.push(...urls.map((url) => ({ type: "image_url", image_url: { url } })));
      const updated = JSON.stringify(request.payload, null, 2);
      if (new TextEncoder().encode(updated).byteLength > 12 * MIB) throw new Error("加入附件后请求超过 12 MiB，请减少图片或文字。");
      editor.value = updated;
      editorDetails.open = updated.length < 120000;
      resetResponse("附件已加入当前请求，尚未发送。");
      refreshEditor();
      tell(`已加入 ${urls.length} ${asVideo ? "帧" : "张图片"}；请检查预览后发送。`);
    } catch (problem) { if (!disposed) error(problem.message); }
    finally { uploading = false; if (!disposed) setBusy(); }
  }

  on(preset, "change", () => applyPreset(preset.value));
  on(endpoint, "change", () => applyPreset({ "/v1/systemone": "decisions", "/v1/chat/completions": "text", "/health": "health", "/v1/models": "models" }[endpoint.value]));
  on(healthButton, "click", () => { applyPreset("health"); void send(); });
  on(modelsButton, "click", () => { applyPreset("models"); void send(); });
  on(sendButton, "click", () => { void send(); });
  on(cancelButton, "click", () => { if (active) { active.cancelled = true; active.controller.abort(); } });
  on(editor, "input", () => {
    preset.value = "";
    presetNote.textContent = "自定义请求。发送前会检查 JSON，接口字段由服务端校验。";
    clearError();
    clearTimeout(editorTimer);
    editorTimer = setTimeout(refreshEditor, 250);
  });
  on(editor, "keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      if (!active && !uploading) void send();
    }
  });
  on(formatButton, "click", () => {
    try { editor.value = JSON.stringify(parsePayload(editor.value), null, 2); clearError(); refreshEditor(); }
    catch (problem) { error(`无法格式化：${problem.message}`); }
  });
  on(copyRequestButton, "click", () => { try { void copy(snapshot().body, "请求 JSON"); } catch (problem) { error(problem.message); } });
  on(copyResponseButton, "click", () => { if (fullResponse) void copy(fullResponse, "完整响应"); });
  on(copyCodeButton, "click", () => { try { void copy(requestCode(snapshot(), language), language === "curl" ? "curl" : "Python 示例"); } catch (problem) { error(problem.message); } });
  on(curlTab, "click", () => { language = "curl"; refreshCode(); });
  on(pythonTab, "click", () => { language = "python"; refreshCode(); });
  on(examples, "toggle", refreshCode);
  on(rawDetails, "toggle", () => { rawPre.textContent = rawDetails.open ? fullResponse : ""; });
  on(imageButton, "click", () => imageInput.click());
  on(videoButton, "click", () => videoInput.click());
  on(imageInput, "change", () => { const files = [...imageInput.files]; imageInput.value = ""; void addFiles(files, false); });
  on(videoInput, "change", () => { const files = [...videoInput.files]; videoInput.value = ""; void addFiles(files, true); });
  on(clearMediaButton, "click", () => {
    try {
      const request = snapshot();
      for (const content of contentArrays(request.payload, request.endpoint)) {
        for (let index = content.length - 1; index >= 0; index -= 1) {
          if (["image_url", "video"].includes(content[index]?.type)) content.splice(index, 1);
        }
      }
      editor.value = JSON.stringify(request.payload, null, 2);
      editorDetails.open = true;
      clearError();
      resetResponse("已移除请求中的媒体，尚未发送。");
      refreshEditor();
    } catch (problem) { error(problem.message); }
  });
  on(historyList, "click", (event) => {
    const target = event.target.closest("button[data-history-id]");
    if (!target || active || uploading) return;
    const entry = history.find((item) => String(item.id) === target.dataset.historyId);
    if (!entry) return;
    endpoint.value = entry.endpoint;
    preset.value = "";
    presetNote.textContent = "已恢复历史请求；只有再次发送才会调用服务。";
    editor.value = entry.body;
    editorDetails.open = entry.body.length < 120000;
    clearError();
    resetResponse("历史请求已恢复，尚未重新发送。");
    refreshEditor();
  });
  applyPreset("decisions");
  renderHistory();

  return () => {
    disposed = true;
    listeners.abort();
    active?.controller.abort();
    clearInterval(ticker);
    clearTimeout(editorTimer);
    for (const reader of readers) reader.abort();
    for (const image of probes) image.src = "";
    readers.clear();
    probes.clear();
    dimensions.clear();
    mediaNames.clear();
    history.length = 0;
    fullResponse = "";
    root.remove();
  };
}
