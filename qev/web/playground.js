import { t, m, getLocale, onLocaleChange, setText, setAttr } from "./i18n.js";
import { CHOICE_SCENES, choicePreset, editableChoice, mountChoiceForm, validateChoiceScene } from "./choice-form.js";

const MIB = 1024 * 1024;
const ROUTES = {
  "/v1/systemone": { method: "POST", label: m("pg.typed") },
  "/v1/chat/completions": { method: "POST", label: m("pg.native") },
  "/health": { method: "GET", label: m("pg.health") },
  "/v1/models": { method: "GET", label: m("pg.models") },
};
const PRESETS = {
  ...Object.fromEntries(Object.entries(CHOICE_SCENES).map(([key, value]) => [key, { ...value, endpoint: "/v1/systemone" }])),
  decisions: {
    label: "Choice + Noul + Score",
    endpoint: "/v1/systemone",
    note: m("pg.mixedNote"),
    body: {
      model: "qev-latest",
      state: m("pg.exampleState"),
      questions: {
        team: { type: "choice", instructions: m("pg.exampleTeam"), criteria: {
          billing: m("pg.billing"), technical: m("pg.technical"), sales: m("pg.sales"),
        } },
        refund: { type: "noul", instructions: m("pg.exampleRefund") },
        urgency: { type: "score", instructions: m("pg.exampleUrgency"), criteria: [m("pg.low"), m("pg.medium"), m("pg.high")] },
      },
    },
  },
  text: {
    label: m("pg.nativeText"),
    endpoint: "/v1/chat/completions",
    note: m("pg.nativeNote"),
    body: { model: "qev-native", messages: [{ role: "user", content: m("pg.exampleText") }], max_tokens: 128, temperature: 0, stream: false },
  },
  image: {
    label: m("pg.nativeImage"),
    endpoint: "/v1/chat/completions",
    note: m("pg.imageNote"),
    body: { model: "qev-native", messages: [{ role: "user", content: [{ type: "text", text: m("pg.exampleImage") }] }], max_tokens: 128, temperature: 0, stream: false },
  },
  video: {
    label: m("pg.nativeVideo"),
    endpoint: "/v1/chat/completions",
    note: m("pg.videoNote"),
    body: { model: "qev-native", messages: [{ role: "user", content: [{ type: "text", text: m("pg.exampleVideo") }] }], max_tokens: 160, temperature: 0, stream: false },
  },
  health: { label: m("pg.getHealth"), endpoint: "/health", note: m("pg.healthNote"), body: null },
  models: { label: m("pg.getModels"), endpoint: "/v1/models", note: m("pg.modelsNote"), body: null },
};
let mountCount = 0;

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") setText(element, text);
  return element;
}

function parsePayload(text) {
  const value = JSON.parse(text, (_key, item) => {
    if (typeof item === "number" && !Number.isFinite(item)) throw new Error(m("pg.finiteJSON"));
    return item;
  });
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(m("pg.objectJSON"));
  return value;
}

function stateContent(payload) {
  const state = payload.state;
  if (Array.isArray(state) && state.length
      && state.every((item) => item && typeof item === "object" && !Array.isArray(item))
      && state.some((item) => ["text", "image_url", "video", "video_url", "audio_url", "input_audio"].includes(item.type))) return state;
  return state && typeof state === "object" && Object.keys(state).length === 1 && Array.isArray(state.content) ? state.content : null;
}

function contentGroups(payload, endpoint) {
  if (endpoint === "/v1/chat/completions") {
    return Array.isArray(payload.messages) ? payload.messages.filter(Boolean).flatMap((message, index) => Array.isArray(message.content) ? [{ content: message.content, label: m("pg.messageLabel", { number: index + 1 }) }] : []) : [];
  }
  const state = stateContent(payload);
  const groups = state ? [{ content: state, label: m("pg.background") }] : [];
  for (const [question, value] of Object.entries(payload.questions || {})) {
    if (!value?.criteria || typeof value.criteria !== "object") continue;
    for (const [key, criterion] of Object.entries(value.criteria)) {
      const content = stateContent({ state: criterion });
      if (content) groups.push({ content, label: m("pg.optionLabel", { question, key }) });
    }
  }
  return groups;
}

function contentArrays(payload, endpoint) { return contentGroups(payload, endpoint).map((group) => group.content); }

function mediaEntries(payload, endpoint) {
  const entries = [];
  let videos = 0;
  for (const { content, label } of contentGroups(payload, endpoint)) {
    for (const item of content) {
      if (item?.type === "image_url") entries.push({ url: item.image_url?.url, label: m("pg.imageLabel", { label, number: entries.length + 1 }) });
      if (item?.type === "video" && Array.isArray(item.frames)) {
        videos += 1;
        item.frames.forEach((url, index) => entries.push({ url, label: m("pg.videoLabel", { label, video: videos, frame: index + 1 }) }));
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
    if (!Array.isArray(payload.messages)) throw new Error(m("pg.messagesRequired"));
    let message = null;
    for (let index = payload.messages.length - 1; index >= 0; index -= 1) {
      if (payload.messages[index]?.role === "user") { message = payload.messages[index]; break; }
    }
    if (!message) { message = { role: "user", content: [] }; payload.messages.push(message); }
    if (typeof message.content === "string") message.content = [{ type: "text", text: message.content }];
    if (!Array.isArray(message.content)) throw new Error(m("pg.contentRequired"));
    return message.content;
  }
  const existing = stateContent(payload);
  if (existing) return existing;
  const text = typeof payload.state === "string" ? payload.state : JSON.stringify(payload.state ?? "", null, 2);
  payload.state = [{ type: "text", text }];
  return payload.state;
}

function compactPreview(value) {
  const text = JSON.stringify(value, (_key, item) => {
    if (typeof item !== "string") return item;
    if (item.startsWith("data:image/")) return `${item.slice(0, item.indexOf(",") + 1)}${t("pg.foldedMedia", { count: item.length.toLocaleString(getLocale()) })}`;
    return item.length > 5000 ? `${item.slice(0, 5000)}${t("pg.foldedText")}` : item;
  }, 2);
  return text.length > 16000 ? `${text.slice(0, 16000)}\n${t("pg.shortPreview")}` : text;
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
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(getLocale(), { maximumFractionDigits: digits }) : "—";
}

function responseTimings(data) {
  const valid = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const server = [data?.latency_ms, data?.qev?.latency_ms].find(valid) ?? null;
  const labels = { decode: m("pg.decode"), queue: m("pg.queue"), encode: m("pg.encode"), inference: m("pg.inference"), total: m("pg.total") };
  const parts = Object.entries(labels).flatMap(([key, label]) => {
    const value = data?.qev?.timings_ms?.[key];
    return valid(value) ? [{ key, label, value }] : [];
  });
  return { server, parts };
}

function responseLog(headers) {
  const requestId = headers?.get?.("X-Qev-Request-Id")?.trim() || "";
  const rawStatus = headers?.get?.("X-Qev-Log-Status")?.trim() || "";
  const status = ["saved", "error"].includes(rawStatus) ? rawStatus : null;
  if (!requestId && !status) return null;
  return { requestId, status };
}

function logLabel(log) {
  return m(log?.status === "saved" ? "pg.logSaved" : log?.status === "error" ? "pg.logFailed" : "pg.requestId");
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
  let lastResponse = null;
  let lastLog = null;
  let language = "curl";
  let activeScene = "choice_text";
  let choiceForm = null;

  const root = node("section", "playground-grid");
  setAttr(root, "aria-label", m("app.playground"));
  const scenePicker = node("div", "pg-scene-picker");
  scenePicker.setAttribute("role", "group");
  setAttr(scenePicker, "aria-label", m("pg.scenePicker"));
  for (const [key, value] of Object.entries(CHOICE_SCENES)) {
    const card = node("button", "pg-scene-button");
    card.type = "button";
    card.dataset.scene = key;
    card.append(node("span", "pg-scene-icon", value.icon), node("strong", "", value.label), node("span", "pg-scene-description", value.description));
    scenePicker.append(card);
  }
  root.append(scenePicker);
  const requestPanel = node("section", "panel playground-request");
  const responsePanel = node("section", "panel playground-response");
  const requestHeading = node("div", "panel-header");
  const requestTitle = node("div");
  requestTitle.append(node("p", "eyebrow", "API WORKBENCH"), node("h2", "section-title", m("pg.request")));
  requestHeading.append(requestTitle, node("span", "badge", m("pg.sameOrigin")));
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
  preset.append(Object.assign(node("option", "", m("pg.custom")), { value: "" }));
  for (const [key, value] of Object.entries(PRESETS)) preset.append(Object.assign(node("option", "", value.label), { value: key }));
  const endpoint = node("select", "pg-select code");
  for (const [path, route] of Object.entries(ROUTES)) endpoint.append(new Option(`${route.method} ${path}`, path));
  const pickers = node("div", "pg-pickers");
  pickers.append(field(m("pg.moreExamples"), preset, "preset"), field(m("pg.endpoint"), endpoint, "endpoint"));
  const presetNote = node("p", "hint");
  const getTools = node("div", "toolbar pg-get-tools");
  const healthButton = button(m("pg.checkHealth"), "btn small ghost");
  const modelsButton = button(m("pg.viewModels"), "btn small ghost");
  getTools.append(healthButton, modelsButton);
  requestPanel.append(pickers, presetNote, getTools);
  const choiceFormContainer = node("div", "pg-choice-form");
  choiceFormContainer.dataset.choiceForm = "true";
  requestPanel.append(choiceFormContainer);

  const mediaBox = node("details", "pg-media");
  setAttr(mediaBox, "aria-label", m("pg.attachments"));
  mediaBox.append(node("summary", "", m("pg.moreMedia")));
  const mediaTools = node("div", "toolbar");
  const imageButton = button(m("pg.addImage"), "btn small");
  const videoButton = button(m("pg.addVideo"), "btn small");
  const clearMediaButton = button(m("pg.clearMedia"), "btn small ghost");
  const fps = node("input", "pg-fps");
  Object.assign(fps, { type: "number", min: "0.001", max: "60", step: "any", value: "2" });
  const imageInput = node("input");
  const videoInput = node("input");
  for (const input of [imageInput, videoInput]) {
    Object.assign(input, { type: "file", accept: "image/png,image/jpeg,image/webp", multiple: true, hidden: true });
  }
  mediaTools.append(imageButton, videoButton, field(m("pg.fps"), fps, "fps"), clearMediaButton);
  const mediaHint = node("p", "hint", m("pg.mediaHint"));
  const mediaPreview = node("div", "pg-media-grid");
  mediaBox.append(mediaTools, imageInput, videoInput, mediaHint, mediaPreview);
  requestPanel.append(mediaBox);

  const editorDetails = node("details", "pg-editor");
  editorDetails.open = true;
  const editorSummary = node("summary", "", m("pg.advanced"));
  const editor = node("textarea", "code pg-json-editor");
  Object.assign(editor, { rows: 18, spellcheck: false });
  editor.setAttribute("autocomplete", "off");
  editor.setAttribute("autocapitalize", "off");
  setAttr(editor, "aria-label", m("pg.jsonEditor"));
  const editorHelp = node("p", "hint", m("pg.editorHelp"));
  editorHelp.id = `${id}-editor-help`;
  editor.setAttribute("aria-describedby", editorHelp.id);
  editorDetails.append(editorSummary, editor, editorHelp);
  const editorTools = node("div", "toolbar pg-editor-tools");
  const formatButton = button(m("pg.formatJSON"), "btn small ghost");
  const copyRequestButton = button(m("pg.copyJSON"), "btn small ghost");
  editorTools.append(formatButton, copyRequestButton);
  const localError = node("p", "error pg-local-error");
  localError.setAttribute("role", "alert");
  localError.hidden = true;
  const sendTools = node("div", "toolbar pg-send-tools");
  const sendButton = button(m("pg.send"), "btn primary");
  const cancelButton = button(m("pg.cancel"), "btn ghost");
  cancelButton.disabled = true;
  sendTools.append(sendButton, cancelButton);
  requestPanel.insertBefore(sendTools, pickers);
  requestPanel.append(editorDetails, editorTools, localError,
    node("p", "hint", m("pg.cancelHint")));

  const examples = node("details", "pg-examples");
  examples.append(node("summary", "", m("pg.callCode")));
  const codeTools = node("div", "toolbar");
  const curlTab = button("curl", "btn small");
  const pythonTab = button("Python · httpx", "btn small ghost");
  const copyCodeButton = button(m("pg.copyCurl"), "btn small ghost");
  codeTools.append(curlTab, pythonTab, copyCodeButton);
  const codePreview = node("pre", "code pg-code-preview");
  codePreview.tabIndex = 0;
  examples.append(codeTools, codePreview,
    node("p", "hint", m("pg.codeHint")));
  const historyBox = node("section", "pg-history-section");
  historyBox.append(node("h3", "section-title", m("pg.recent")), node("p", "hint", m("pg.historyHint")));
  const historyList = node("ol", "pg-history");
  historyBox.append(historyList);
  requestPanel.append(examples, historyBox);

  const responseHeader = node("div", "panel-header");
  const responseTitle = node("div");
  responseTitle.append(node("p", "eyebrow", "LIVE RESPONSE"), node("h2", "section-title", m("pg.response")));
  const responseBadge = node("span", "badge", m("pg.notSent"));
  responseHeader.append(responseTitle, responseBadge);
  const metrics = node("dl", "pg-metrics");
  function metric(label) {
    const item = node("div", "metric");
    const value = node("dd", "", "—");
    item.append(node("dt", "", label), value);
    metrics.append(item);
    return value;
  }
  const statusMetric = metric(m("pg.httpStatus"));
  const timeMetric = metric(m("pg.roundtrip"));
  const serverMetric = metric(m("pg.server"));
  const tokenMetric = metric(m("pg.tokens"));
  const timingHint = node("p", "hint pg-timing-hint", m("pg.timingHint"));
  const timingDetails = node("details", "pg-timings");
  timingDetails.dataset.serverTimings = "true";
  timingDetails.hidden = true;
  const timingBreakdown = node("dl", "pg-timing-breakdown");
  timingDetails.append(node("summary", "", m("pg.timingDetails")), timingBreakdown,
    node("p", "hint", m("pg.inferenceHint")));
  const requestLog = node("p", "hint pg-request-log");
  requestLog.dataset.requestLog = "true";
  requestLog.setAttribute("aria-live", "polite");
  requestLog.hidden = true;
  const requestLogStatus = node("span");
  const requestLogId = node("code", "code");
  const copyLogIdButton = button(m("pg.copyRequestId"), "btn small ghost");
  requestLog.append(requestLogStatus, requestLogId, copyLogIdButton);
  const responseSummary = node("div", "pg-response-summary empty-state", m("pg.responseHint"));
  responseSummary.setAttribute("aria-live", "polite");
  const responseTools = node("div", "toolbar");
  const copyResponseButton = button(m("pg.copyResponse"), "btn small ghost");
  copyResponseButton.disabled = true;
  responseTools.append(node("h3", "section-title", m("pg.responsePreview")), copyResponseButton);
  const responsePreview = node("pre", "code pg-response-preview", m("pg.noResponse"));
  responsePreview.tabIndex = 0;
  const rawDetails = node("details", "pg-raw-response");
  const rawSummary = node("summary", "", m("pg.rawResponse"));
  const rawPre = node("pre", "code pg-response-raw");
  rawPre.tabIndex = 0;
  rawDetails.append(rawSummary, rawPre);
  rawDetails.hidden = true;
  responsePanel.append(responseHeader, metrics, timingHint, timingDetails, requestLog, responseSummary, responseTools, responsePreview, rawDetails);
  root.append(requestPanel, responsePanel);
  container.replaceChildren(root);

  function isGet() { return ROUTES[endpoint.value].method === "GET"; }
  function error(message) { if (disposed) return; setText(localError, message); localError.hidden = false; tell(message, "error"); }
  function clearError() { localError.hidden = true; setText(localError, ""); }
  function setBusy() {
    const busy = Boolean(active) || uploading;
    for (const control of [preset, endpoint, fps, healthButton, modelsButton, sendButton]) control.disabled = busy;
    for (const control of [imageButton, videoButton, clearMediaButton, imageInput, videoInput, formatButton, copyRequestButton]) control.disabled = busy || isGet();
    editor.readOnly = busy;
    cancelButton.disabled = !active;
    copyCodeButton.disabled = uploading;
    copyResponseButton.disabled = !fullResponse;
    setText(sendButton, active ? m("pg.sending") : uploading ? m("pg.reading") : m("pg.send"));
    responsePanel.setAttribute("aria-busy", String(Boolean(active)));
    for (const item of historyList.querySelectorAll("button")) item.disabled = busy;
    for (const item of scenePicker.querySelectorAll("button")) {
      item.disabled = busy;
      item.classList.toggle("is-selected", item.dataset.scene === activeScene);
      item.setAttribute("aria-pressed", String(item.dataset.scene === activeScene));
    }
    choiceForm?.setBusy(busy);
    mediaBox.hidden = isGet();
    editorDetails.hidden = isGet();
    editorTools.hidden = isGet();
  }

  function snapshot() {
    const route = ROUTES[endpoint.value];
    if (!route) throw new Error(m("pg.selectEndpoint"));
    if (route.method === "GET") return { endpoint: endpoint.value, method: "GET", body: "", payload: null };
    const payload = parsePayload(editor.value);
    if (new TextEncoder().encode(editor.value).byteLength > 12 * MIB) throw new Error(m("pg.requestSize"));
    return { endpoint: endpoint.value, method: "POST", body: editor.value, payload };
  }

  function refreshCode() {
    curlTab.setAttribute("aria-pressed", String(language === "curl"));
    pythonTab.setAttribute("aria-pressed", String(language === "python"));
    curlTab.className = language === "curl" ? "btn small" : "btn small ghost";
    pythonTab.className = language === "python" ? "btn small" : "btn small ghost";
    setText(copyCodeButton, language === "curl" ? m("pg.copyCurl") : m("pg.copyPython"));
    if (!examples.open) return;
    try { setText(codePreview, requestCode(snapshot(), language, true)); }
    catch (problem) { setText(codePreview, m("pg.invalidCodeJSON", { error: problem.message })); }
  }

  function refreshEditor({ syncForm = true } = {}) {
    const bytes = new TextEncoder().encode(editor.value).byteLength;
    setText(editorSummary, m("pg.editorBytes", { size: numeric(bytes / 1024, 1) }));
    mediaPreview.replaceChildren();
    if (!isGet()) {
      try {
        const payload = parsePayload(editor.value);
        if (syncForm) choiceForm?.sync(endpoint.value === "/v1/systemone" ? payload : null, activeScene);
        const entries = mediaEntries(payload, endpoint.value);
        for (const entry of entries.slice(0, 8)) {
          const figure = node("figure", "pg-media-item");
          if (inlineImageBytes(entry.url) !== null) {
            const image = node("img", "pg-thumbnail");
            Object.assign(image, { src: entry.url, alt: entry.label, width: 72, height: 72 });
            figure.append(image);
          } else figure.append(node("span", "hint", m("pg.noPreview")));
          const name = mediaNames.get(entry.url);
          figure.append(node("figcaption", "hint", `${entry.label}${name ? ` · ${name}` : ""}`));
          mediaPreview.append(figure);
        }
        if (entries.length > 8) mediaPreview.append(node("p", "error", m("pg.tooManyPreviews", { count: entries.length })));
      } catch { if (syncForm) choiceForm?.sync(null, activeScene); /* Incomplete JSON stays in the advanced editor. */ }
    } else if (syncForm) choiceForm?.sync(null, null);
    refreshCode();
  }

  function resetResponse(message = m("pg.ready")) {
    fullResponse = "";
    lastResponse = null;
    lastLog = null;
    renderLog();
    setText(responseBadge, m("pg.notSent"));
    responseBadge.className = "badge";
    setText(statusMetric, timeMetric.textContent = serverMetric.textContent = tokenMetric.textContent = "—");
    timingDetails.hidden = true;
    timingDetails.open = false;
    timingBreakdown.replaceChildren();
    responseSummary.className = "pg-response-summary empty-state";
    setText(responseSummary, message);
    setText(responsePreview, m("pg.noResponse"));
    rawDetails.hidden = true;
    rawDetails.open = false;
    setText(rawPre, "");
    setBusy();
  }

  function applyPreset(key) {
    if (active || uploading || !PRESETS[key]) return;
    const example = CHOICE_SCENES[key] ? { ...PRESETS[key], ...CHOICE_SCENES[key] } : PRESETS[key];
    clearError();
    preset.value = key;
    activeScene = Object.hasOwn(CHOICE_SCENES, key) ? key : null;
    endpoint.value = example.endpoint;
    setText(presetNote, example.note);
    const body = activeScene ? choicePreset(activeScene) : example.body;
    editor.value = body ? JSON.stringify(body, null, 2) : "";
    editorDetails.open = !activeScene;
    mediaBox.open = key === "image" || key === "video";
    resetResponse();
    refreshEditor();
  }

  function renderHistory() {
    historyList.replaceChildren();
    if (!history.length) { historyList.append(node("li", "hint", m("pg.noHistory"))); return; }
    for (const entry of history) {
      const item = node("li", "pg-history-item");
      const restore = button("", "btn small ghost pg-restore");
      restore.dataset.historyId = String(entry.id);
      setText(restore, `${entry.time} · ${entry.method} ${entry.endpoint} · ${entry.status}`);
      setAttr(restore, "title", entry.log?.requestId ? m("pg.restoreWithRequestId", { id: entry.log.requestId }) : m("pg.restore"));
      item.append(restore);
      historyList.append(item);
    }
    setBusy();
  }

  async function copy(text, label) {
    try {
      await navigator.clipboard.writeText(text);
      tell(m("pg.copied", { label }));
    } catch (problem) { error(m("pg.copyFailed", { error: problem.message || m("pg.clipboardDenied") })); }
  }

  function renderLog() {
    requestLog.hidden = !lastLog;
    requestLog.className = `hint pg-request-log${lastLog?.status === "error" ? " error" : ""}`;
    setText(requestLogStatus, lastLog ? logLabel(lastLog) : "");
    setText(requestLogId, lastLog?.requestId || "");
    requestLogId.hidden = !lastLog?.requestId;
    copyLogIdButton.hidden = !lastLog?.requestId;
  }
  on(copyLogIdButton, "click", () => {
    if (lastLog?.requestId) void copy(lastLog.requestId, m("pg.requestId"));
  });

  function summarize(data, ok, httpStatus) {
    responseSummary.className = "pg-response-summary";
    setText(responseSummary, "");
    if (!ok) {
      responseSummary.append(node("h3", "error", m("pg.requestFailed", { status: httpStatus })));
      const detail = data?.detail;
      if (Array.isArray(detail)) {
        const list = node("ul", "pg-errors");
        for (const item of detail.slice(0, 8)) list.append(node("li", "", `${Array.isArray(item?.loc) ? item.loc.join(".") : m("pg.request")}：${item?.msg || m("pg.validationFailed")}`));
        if (detail.length > 8) list.append(node("li", "hint", m("pg.moreErrors", { count: detail.length - 8 })));
        responseSummary.append(list);
      } else responseSummary.append(node("p", "", typeof detail === "string" ? detail : m("pg.serverError")));
      return;
    }
    if (data?.answers && typeof data.answers === "object") {
      for (const [name, answer] of Object.entries(data.answers)) {
        const block = node("article", "pg-answer");
        block.append(node("h3", "", `${name} · ${answer?.type || m("pg.answer")}`));
        let value = m("pg.seeJSON");
        if (answer?.type === "choice") value = String(answer.choice ?? "—");
        if (answer?.type === "noul") value = `P(true) = ${numeric(answer.noul * 100)}%`;
        if (answer?.type === "score") value = m("pg.expectedScore", { score: numeric(answer.score, 4) });
        block.append(node("p", "pg-answer-value", value));
        if (typeof answer?.confidence === "number") block.append(node("p", "hint", `confidence ${numeric(answer.confidence, 4)}`));
        if (answer?.probabilities && typeof answer.probabilities === "object") {
          const probabilities = Object.entries(answer.probabilities).sort((a, b) => Number(b[1]) - Number(a[1]));
          const list = node("ul", "pg-probabilities");
          for (const [key, probability] of probabilities.slice(0, 6)) list.append(node("li", "", `${key}：${numeric(probability * 100)}%`));
          if (probabilities.length > 6) list.append(node("li", "hint", m("pg.moreCandidates", { count: probabilities.length - 6 })));
          block.append(list);
        }
        responseSummary.append(block);
      }
    } else if (Array.isArray(data?.choices)) {
      for (const choice of data.choices) {
        const content = choice?.message?.content;
        responseSummary.append(node("p", "pg-native-text", typeof content === "string" ? content : m("pg.noTextContent")));
        if (choice?.finish_reason) responseSummary.append(node("p", "hint", m("pg.finishReason", { reason: choice.finish_reason })));
      }
    } else if (Array.isArray(data?.models)) {
      const list = node("ul", "pg-models");
      for (const model of data.models) list.append(node("li", "", `${model?.name ?? model?.id ?? m("pg.unnamed")}${model?.description ? ` — ${model.description}` : ""}`));
      responseSummary.append(list);
    } else if (data?.status) {
      responseSummary.append(node("p", "pg-answer-value", m("pg.serviceStatus", { status: data.status })));
      if (data.backend) responseSummary.append(node("p", "hint", m("pg.currentBackend", { backend: data.backend })));
    } else responseSummary.append(node("p", "", m("pg.received")));
    if (data?.qev?.decision_adapter_enabled === false) responseSummary.append(node("p", "hint", m("pg.adapterOff")));
    if (Array.isArray(data?.qev?.input_modalities) && data.qev.input_modalities.some((item) => item !== "text") && data.qev.multimodal_decision_accuracy_validated === false) responseSummary.append(node("p", "hint", m("pg.mediaAccuracy")));
  }

  async function send() {
    if (active || uploading || disposed) return;
    clearError();
    lastLog = null;
    renderLog();
    let request;
    try {
      request = snapshot();
      if (request.endpoint === "/v1/systemone") {
        // Ctrl+Enter can precede the editor debounce: the submitted JSON is
        // authoritative even when the previous scenario is still highlighted.
        if (activeScene && !editableChoice(request.payload)) { activeScene = null; setBusy(); }
        validateChoiceScene(request.payload, activeScene);
      }
      const content = request.payload ? contentArrays(request.payload, request.endpoint).flat() : [];
      if (preset.value === "image" && !content.some((item) => item?.type === "image_url")) throw new Error(m("pg.needImage"));
      if (preset.value === "video" && !content.some((item) => item?.type === "video" && item.frames?.length)) throw new Error(m("pg.needVideo"));
      uploading = true;
      setBusy();
      if (request.payload) await validateMedia(request.payload, request.endpoint);
    } catch (problem) { uploading = false; resetResponse(m("pg.fixInput")); error(m("pg.notSentError", { error: problem.message })); return; }
    uploading = false;
    if (disposed) return;
    resetResponse(m("pg.waitingService"));
    const run = { controller: new AbortController(), started: null, finished: null, cancelled: false };
    active = run;
    const record = { ...request, payload: undefined, scene: activeScene, id: ++historyId, time: new Date().toLocaleTimeString(getLocale(), { hour12: false }), status: m("pg.waiting") };
    history.unshift(record);
    history.splice(5);
    renderHistory();
    setText(responseBadge, m("pg.waiting"));
    ticker = setInterval(() => {
      if (!disposed && active === run && run.started !== null && run.finished === null) setText(timeMetric, `${numeric(performance.now() - run.started, 1)} ms`);
    }, 100);
    setBusy();
    try {
      run.started = performance.now();
      const response = await fetch(request.endpoint, {
        method: request.method, credentials: "same-origin", signal: run.controller.signal,
        ...(request.method === "POST" ? { headers: { "Content-Type": "application/json" }, body: request.body } : {}),
      });
      const text = await response.text();
      run.finished = performance.now();
      if (disposed || active !== run) return;
      if (run.cancelled) {
        const cancelled = new Error();
        cancelled.name = "AbortError";
        throw cancelled;
      }
      lastLog = responseLog(response.headers);
      record.log = lastLog;
      renderLog();
      setText(statusMetric, String(response.status));
      let data = null;
      try { data = JSON.parse(text); } catch { /* A real non-JSON response is shown unchanged. */ }
      fullResponse = text;
      rawDetails.hidden = !text;
      setText(rawSummary, m("pg.rawBytes", { size: numeric(new TextEncoder().encode(text).byteLength / 1024, 1) }));
      setText(responsePreview, data === null ? (text.length > 16000 ? `${text.slice(0, 16000)}\n${t("pg.expandFull")}` : text || m("pg.emptyBody")) : compactPreview(data));
      setText(responseBadge, response.ok ? `HTTP ${response.status}` : m("pg.failedStatus", { status: response.status }));
      responseBadge.className = response.ok ? "badge" : "badge error";
      record.status = `HTTP ${response.status}`;
      const usage = data?.usage;
      if (usage) setText(tokenMetric, `${numeric(usage.input_tokens ?? usage.prompt_tokens, 0)} / ${numeric(usage.output_tokens ?? usage.completion_tokens, 0)}`);
      const timings = responseTimings(data);
      setText(serverMetric, timings.server === null ? "—" : `${numeric(timings.server)} ms`);
      for (const { label, value } of timings.parts) {
        const part = node("div", "pg-timing-part");
        part.append(node("dt", "", label), node("dd", "", `${numeric(value)} ms`));
        timingBreakdown.append(part);
      }
      timingDetails.hidden = !timings.parts.length;
      lastResponse = { data, ok: response.ok, status: response.status };
      summarize(data, response.ok, response.status);
      if (!response.ok) tell(m("pg.httpError", { status: response.status }), "error");
    } catch (problem) {
      run.finished ??= performance.now();
      if (disposed || active !== run) return;
      lastLog = null;
      record.log = null;
      renderLog();
      const cancelled = run.cancelled || problem.name === "AbortError";
      const message = cancelled ? m("pg.cancelled") : m("pg.networkFailed", { error: problem.message || m("pg.noServiceResponse") });
      record.status = cancelled ? m("pg.cancel") : m("pg.networkError");
      setText(responseBadge, record.status);
      responseBadge.className = cancelled ? "badge" : "badge error";
      responseSummary.className = "pg-response-summary";
      setText(responseSummary, message);
      setText(responsePreview, m("pg.incompleteBody"));
      tell(message, cancelled ? "info" : "error");
    } finally {
      clearInterval(ticker);
      ticker = null;
      if (!disposed && active === run) {
        setText(timeMetric, run.started === null ? "—" : `${numeric((run.finished ?? performance.now()) - run.started, 1)} ms`);
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
      reader.onerror = () => { finish(); reject(new Error(m("pg.fileReadFailed", { file: file.name }))); };
      reader.onabort = () => { finish(); reject(new Error(m("pg.readAborted"))); };
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
      image.onerror = () => { probes.delete(image); reject(new Error(m("pg.imageDecode"))); };
      image.src = url;
    });
  }

  async function validateMedia(payload, path) {
    const entries = mediaEntries(payload, path);
    if (entries.length > 8) throw new Error(m("pg.maxImages"));
    let bytes = 0, pixels = 0;
    for (const entry of entries) {
      const amount = inlineImageBytes(entry.url);
      if (amount === null) throw new Error(m("pg.invalidImage", { label: entry.label }));
      if (amount > 2 * MIB) throw new Error(m("pg.largeImage", { label: entry.label }));
      bytes += amount;
      if (bytes > 8 * MIB) throw new Error(m("pg.maxImageBytes"));
      const size = await imageDimensions(entry.url);
      const count = size.width * size.height;
      if (!count || count > 4000000) throw new Error(m("pg.manyPixels", { label: entry.label }));
      pixels += count;
      if (pixels > 8000000) throw new Error(m("pg.maxImagePixels"));
    }
    for (const content of contentArrays(payload, path)) {
      for (const item of content) {
        if (item?.type !== "video" || !Array.isArray(item.frames)) continue;
        const rate = item.fps === undefined ? 1 : item.fps;
        if (!Number.isFinite(rate) || rate <= 0 || rate > 60) throw new Error(m("pg.videoFps"));
        const sizes = await Promise.all(item.frames.map(imageDimensions));
        if (sizes.some((size) => size.width !== sizes[0].width || size.height !== sizes[0].height)) throw new Error(m("pg.frameDimensions"));
      }
    }
  }

  async function uploadChoiceImage(file, mutate, expected) {
    if (active || uploading || disposed) return;
    uploading = true;
    clearError();
    setBusy();
    try {
      const request = snapshot();
      if (request.endpoint !== "/v1/systemone" || JSON.stringify(request.payload) !== expected) throw new Error(m("pg.staleImage"));
      if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error(m("pg.imageType"));
      if (!file.size || file.size > 2 * MIB) throw new Error(m("pg.imageSize"));
      const url = await readFile(file);
      if (disposed) return;
      mutate(request.payload, url);
      await validateMedia(request.payload, request.endpoint);
      if (disposed) return;
      const updated = JSON.stringify(request.payload, null, 2);
      if (new TextEncoder().encode(updated).byteLength > 12 * MIB) throw new Error(m("pg.requestSize"));
      mediaNames.set(url, file.name);
      if (mediaNames.size > 16) mediaNames.delete(mediaNames.keys().next().value);
      editor.value = updated;
      resetResponse(m("pg.imageUpdated"));
      refreshEditor();
    } catch (problem) { if (!disposed) error(problem.message); }
    finally { uploading = false; if (!disposed) setBusy(); }
  }

  async function addFiles(files, asVideo) {
    if (!files.length || active || uploading || disposed) return;
    clearError();
    uploading = true;
    setBusy();
    try {
      const request = snapshot();
      if (request.method !== "POST") throw new Error(m("pg.getMedia"));
      const existing = mediaEntries(request.payload, request.endpoint);
      if (existing.length + files.length > 8) throw new Error(m("pg.maxMedia"));
      let totalBytes = 0;
      for (const item of existing) {
        const bytes = inlineImageBytes(item.url);
        if (bytes === null) throw new Error(m("pg.invalidExisting"));
        if (bytes > 2 * MIB) throw new Error(m("pg.largeExisting"));
        totalBytes += bytes;
      }
      const rate = Number(fps.value);
      if (asVideo && (!Number.isFinite(rate) || rate <= 0 || rate > 60)) throw new Error(m("pg.validFps"));
      const ordered = [...files];
      if (asVideo) ordered.sort((a, b) => a.name.localeCompare(b.name, "zh-CN", { numeric: true }));
      for (const file of ordered) {
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error(m("pg.fileType", { file: file.name }));
        if (!file.size || file.size > 2 * MIB) throw new Error(m("pg.fileSize", { file: file.name }));
        totalBytes += file.size;
      }
      if (totalBytes > 8 * MIB) throw new Error(m("pg.allImageBytes"));
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
        if (!pixels || pixels > 4000000) throw new Error(m("pg.imagePixels"));
        totalPixels += pixels;
        sizes.push(size);
      }
      if (totalPixels > 8000000) throw new Error(m("pg.allPixels"));
      const newSizes = sizes.slice(existing.length);
      if (asVideo && newSizes.some((size) => size.width !== newSizes[0].width || size.height !== newSizes[0].height)) throw new Error(m("pg.frameDimensions"));
      const content = mediaTarget(request.payload, request.endpoint);
      if (asVideo) content.push({ type: "video", frames: urls, fps: rate });
      else content.push(...urls.map((url) => ({ type: "image_url", image_url: { url } })));
      const updated = JSON.stringify(request.payload, null, 2);
      if (new TextEncoder().encode(updated).byteLength > 12 * MIB) throw new Error(m("pg.largeMediaRequest"));
      editor.value = updated;
      editorDetails.open = updated.length < 120000;
      resetResponse(m("pg.mediaAdded"));
      refreshEditor();
      tell(m("pg.addedFiles", { count: urls.length, kind: asVideo ? m("pg.frames") : m("pg.images") }));
    } catch (problem) { if (!disposed) error(problem.message); }
    finally { uploading = false; if (!disposed) setBusy(); }
  }

  choiceForm = mountChoiceForm(choiceFormContainer, {
    read: () => snapshot().payload,
    write(payload, syncForm) {
      editor.value = JSON.stringify(payload, null, 2);
      clearError();
      resetResponse(m("pg.requestEdited"));
      refreshEditor({ syncForm });
    },
    upload: uploadChoiceImage,
    onError: error,
  });
  on(scenePicker, "click", (event) => { const card = event.target.closest("button[data-scene]"); if (card) applyPreset(card.dataset.scene); });
  on(preset, "change", () => applyPreset(preset.value));
  on(endpoint, "change", () => applyPreset({ "/v1/systemone": "choice_text", "/v1/chat/completions": "text", "/health": "health", "/v1/models": "models" }[endpoint.value]));
  on(healthButton, "click", () => { applyPreset("health"); void send(); });
  on(modelsButton, "click", () => { applyPreset("models"); void send(); });
  on(sendButton, "click", () => { void send(); });
  on(cancelButton, "click", () => { if (active) { active.cancelled = true; active.controller.abort(); } });
  on(editor, "input", () => {
    preset.value = "";
    setText(presetNote, m("pg.customNote"));
    clearError();
    clearTimeout(editorTimer);
    editorTimer = setTimeout(() => {
      try {
        if (activeScene && !editableChoice(parsePayload(editor.value))) { activeScene = null; setBusy(); }
      } catch { /* Keep the selected scenario while incomplete JSON is being edited. */ }
      refreshEditor();
    }, 250);
  });
  on(editor, "keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      if (!active && !uploading) void send();
    }
  });
  on(formatButton, "click", () => {
    try { editor.value = JSON.stringify(parsePayload(editor.value), null, 2); clearError(); refreshEditor(); }
    catch (problem) { error(m("pg.formatFailed", { error: problem.message })); }
  });
  on(copyRequestButton, "click", () => { try { void copy(snapshot().body, m("pg.requestJSON")); } catch (problem) { error(problem.message); } });
  on(copyResponseButton, "click", () => { if (fullResponse) void copy(fullResponse, m("pg.fullResponse")); });
  on(copyCodeButton, "click", () => { try { void copy(requestCode(snapshot(), language), language === "curl" ? "curl" : m("pg.pythonExample")); } catch (problem) { error(problem.message); } });
  on(curlTab, "click", () => { language = "curl"; refreshCode(); });
  on(pythonTab, "click", () => { language = "python"; refreshCode(); });
  on(examples, "toggle", refreshCode);
  on(rawDetails, "toggle", () => { setText(rawPre, rawDetails.open ? fullResponse : ""); });
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
      resetResponse(m("pg.mediaRemoved"));
      refreshEditor();
    } catch (problem) { error(problem.message); }
  });
  on(historyList, "click", (event) => {
    const target = event.target.closest("button[data-history-id]");
    if (!target || active || uploading) return;
    const entry = history.find((item) => String(item.id) === target.dataset.historyId);
    if (!entry) return;
    endpoint.value = entry.endpoint;
    activeScene = entry.scene || null;
    preset.value = "";
    setText(presetNote, m("pg.historyNote"));
    editor.value = entry.body;
    editorDetails.open = !activeScene && entry.body.length < 120000;
    clearError();
    resetResponse(m("pg.historyRestored"));
    refreshEditor();
  });
  const stopLocale = onLocaleChange(() => {
    for (const [key, value] of Object.entries(CHOICE_SCENES)) {
      const card = scenePicker.querySelector(`[data-scene="${key}"]`);
      setText(card.querySelector("strong"), value.label);
      setText(card.querySelector(".pg-scene-description"), value.description);
      setText(preset.querySelector(`option[value="${key}"]`), value.label);
      if (preset.value === key) setText(presetNote, value.note);
    }
    choiceForm?.refreshLocale();
    refreshEditor({ syncForm: false });
    renderHistory();
    renderLog();
    if (lastResponse) {
      summarize(lastResponse.data, lastResponse.ok, lastResponse.status);
      if (lastResponse.data !== null) setText(responsePreview, compactPreview(lastResponse.data));
    }
    setBusy();
  });
  applyPreset("choice_text");
  renderHistory();

  return () => {
    disposed = true;
    stopLocale();
    listeners.abort();
    choiceForm.destroy();
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
    lastResponse = null;
    lastLog = null;
    root.remove();
  };
}
