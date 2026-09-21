// Run with: node --test tests/test_playground.mjs
// These checks exercise request construction and routing without browser dependencies.
// Real canvas pixels, image decoding and layout are covered by the browser acceptance run.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import { CHOICE_SCENES, choicePreset, editableChoice, validateChoiceScene } from "../qev/web/choice-form.js";
import { t, m, getLocale, setLocale, setText } from "../qev/web/i18n.js";

setLocale('zh-CN', { persist: false });
const pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=";
globalThis.document = {
  createElement(tag) {
    assert.equal(tag, "canvas");
    return { getContext: () => new Proxy({}, { get: () => () => {} }), toDataURL: () => pixel };
  },
};

const playgroundSource = await readFile(new URL("../qev/web/playground.js", import.meta.url), "utf8");
const scope = { CHOICE_SCENES, t, m, getLocale };
// Mounting needs a real DOM; evaluate the unchanged pure helpers without mounting.
vm.runInNewContext(playgroundSource.replace(/^import .*;\n/gm, "")
  .replace("export function mountPlayground", "function mountPlayground")
  + "\nthis.helpers = { contentArrays, mediaEntries, mediaTarget, responseTimings, responseLog, logLabel, presets: PRESETS };", scope);
const { contentArrays, mediaEntries, mediaTarget, responseTimings, responseLog, logLabel, presets } = scope.helpers;

test("all three scenarios send one choice, with option images belonging to criteria", () => {
  for (const scene of Object.keys(CHOICE_SCENES)) {
    const payload = choicePreset(scene);
    validateChoiceScene(payload, scene);
    assert.equal(Object.keys(payload.questions).length, 1);
    assert.equal(Object.values(payload.questions)[0].type, "choice");
    assert.equal(presets[scene].endpoint, "/v1/systemone");
  }
  assert.equal(mediaEntries(choicePreset("choice_text"), "/v1/systemone").length, 0);
  assert.equal(mediaEntries(choicePreset("choice_context_image"), "/v1/systemone").length, 1);
  const options = choicePreset("choice_option_images");
  assert.equal(typeof options.state, "string");
  assert.equal(mediaEntries(options, "/v1/systemone").length, 3);
  for (const candidate of Object.values(options.questions.image.criteria)) {
    assert.equal(candidate.content.filter((item) => item.type === "image_url").length, 1);
  }
});

test("missing background or candidate images are rejected before sending", () => {
  const background = choicePreset("choice_context_image");
  background.state.pop();
  assert.throws(() => validateChoiceScene(background, "choice_context_image"), /背景缺少图片/);
  const options = choicePreset("choice_option_images");
  options.questions.image.criteria.B.content.pop();
  assert.throws(() => validateChoiceScene(options, "choice_option_images"), /选项 B 缺少图片/);
});

test("complex JSON is kept out of the simplified editor without coercing it", () => {
  const payload = choicePreset("choice_text");
  payload.questions.team.instructions = { custom: "instructions" };
  const original = JSON.stringify(payload);
  assert.equal(editableChoice(payload), null);
  assert.equal(JSON.stringify(payload), original);
  payload.questions.team.instructions = "Choose";
  payload.state = { content: [{ type: "image_url", image_url: { url: pixel } }], extra: true };
  assert.equal(editableChoice(payload), null);
  assert.equal(mediaEntries(payload, "/v1/systemone").length, 0);
});

test("adding background media never targets an existing option, and cleanup includes all options", () => {
  const payload = choicePreset("choice_option_images");
  mediaTarget(payload, "/v1/systemone").push({ type: "image_url", image_url: { url: pixel } });
  assert.equal(payload.questions.image.criteria.A.content.length, 2);
  assert.equal(mediaEntries(payload, "/v1/systemone").length, 4);
  for (const content of contentArrays(payload, "/v1/systemone")) {
    for (let index = content.length - 1; index >= 0; index -= 1) {
      if (["image_url", "video"].includes(content[index]?.type)) content.splice(index, 1);
    }
  }
  assert.equal(mediaEntries(payload, "/v1/systemone").length, 0);
  assert.throws(() => validateChoiceScene(payload, "choice_option_images"), /缺少图片/);
});

test("legacy mixed, native, video and read-only presets remain available", () => {
  assert.deepEqual(Array.from(Object.values(presets.decisions.body.questions), (question) => question.type), ["choice", "noul", "score"]);
  for (const key of ["text", "image", "video"]) assert.equal(presets[key].endpoint, "/v1/chat/completions");
  assert.equal(presets.health.endpoint, "/health");
  assert.equal(presets.models.endpoint, "/v1/models");
});

test("direct content arrays in choice, noul and score criteria participate in preview and cleanup", () => {
  const image = () => [{ type: "text", text: "Candidate" }, { type: "image_url", image_url: { url: pixel } }];
  const payload = {
    state: "Compare the candidates",
    questions: {
      image: { type: "choice", instructions: "Choose", criteria: { A: image(), B: "Text" } },
      match: { type: "noul", criteria: { true: image(), false: "Text" } },
      quality: { type: "score", criteria: ["Low", image()] },
    },
  };
  assert.equal(mediaEntries(payload, "/v1/systemone").length, 3);
  assert.equal(contentArrays(payload, "/v1/systemone")[0], payload.questions.image.criteria.A);
  assert.ok(editableChoice({ state: payload.state, questions: { image: payload.questions.image } }));
  for (const content of contentArrays(payload, "/v1/systemone")) {
    const index = content.findIndex((item) => item.type === "image_url");
    content.splice(index, 1);
  }
  assert.equal(mediaEntries(payload, "/v1/systemone").length, 0);
  assert.equal(payload.questions.quality.criteria[1].length, 1);
});

test("service timing supports both APIs and safely omits unavailable breakdowns", () => {
  assert.equal(responseTimings({ latency_ms: 0, qev: { latency_ms: 99 } }).server, 0);
  assert.equal(responseTimings({ qev: { latency_ms: 42.5 } }).server, 42.5);
  assert.equal(responseTimings({ status: "ok" }).server, null);
  assert.equal(responseTimings({ latency_ms: -1, qev: { latency_ms: "25" } }).server, null);
  assert.equal(responseTimings({ latency_ms: 10 }).parts.length, 0);
  const parts = responseTimings({ qev: { timings_ms: { decode: 1, queue: 0, encode: 2, inference: 40, total: 43 } } }).parts;
  assert.deepEqual(Array.from(parts, (part) => [part.key, part.value]), [["decode", 1], ["queue", 0], ["encode", 2], ["inference", 40], ["total", 43]]);
  assert.equal(responseTimings({ qev: { timings_ms: { decode: null, queue: -1, inference: Infinity } } }).parts.length, 0);
});

test("HTTP measurement includes response body reading but excludes preparation and rendering", async () => {
  let clock = 0, calls = 0;
  const field = () => ({ textContent: "", hidden: false, setAttribute() {}, removeAttribute() {} });
  const requestScope = {
    active: null, uploading: false, disposed: false, activeScene: null, preset: { value: "health" },
    clearError() {}, snapshot: () => ({ endpoint: "/health", method: "GET", payload: null, body: "" }),
    resetResponse: () => { clock += 30; }, setBusy: () => { clock += 13; },
    renderHistory: () => { clock += 100; }, historyId: 0, history: [], ticker: null,
    AbortController, TextEncoder, performance: { now: () => clock },
    setInterval: () => 1, clearInterval() {}, numeric: (value) => String(value),
    responseTimings, responseLog, renderLog() {}, lastLog: null,
    t, m, getLocale, setText, lastResponse: null, summarize: () => { clock += 90; },
    compactPreview: () => { clock += 80; return "{}"; },
    responseBadge: field(), timeMetric: field(), serverMetric: field(), statusMetric: field(),
    tokenMetric: field(), rawDetails: field(), rawSummary: field(), responsePreview: field(),
    responseSummary: field(), timingDetails: field(), timingBreakdown: { append() {} },
    fullResponse: "", tell() {}, error(message) { throw new Error(message); },
    async fetch() {
      calls += 1;
      clock += 50;
      return { status: 200, ok: true, async text() { clock += 25; return "{}"; } };
    },
  };
  const start = playgroundSource.indexOf("  async function send() {");
  const end = playgroundSource.indexOf("\n  function readFile", start);
  vm.runInNewContext(playgroundSource.slice(start, end) + "\nthis.sendRequest = send;", requestScope);
  await requestScope.sendRequest();
  assert.equal(calls, 1);
  assert.equal(requestScope.timeMetric.textContent, "75 ms");
  assert.equal(requestScope.serverMetric.textContent, "—");
  assert.ok(clock > 300, "the fake preparation and rendering delays were exercised");
});

function logHeaders(requestId, status) {
  const headers = new Headers();
  if (requestId !== undefined) headers.set('X-Qev-Request-Id', requestId);
  if (status !== undefined) headers.set('X-Qev-Log-Status', status);
  return headers;
}

function responseHarness() {
  const field = () => ({
    textContent: '', hidden: false, children: [], attributes: {}, dataset: {},
    setAttribute(name, value) { this.attributes[name] = String(value); },
    removeAttribute(name) { delete this.attributes[name]; },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = [...items]; },
  });
  const node = (_tag, className = '', label = '') => {
    const result = field(); result.className = className; setText(result, label); return result;
  };
  const requestScope = {
    active: null, uploading: false, disposed: false, activeScene: null, preset: { value: 'health' },
    clearError() {}, snapshot: () => ({ endpoint: '/health', method: 'GET', payload: null, body: '' }),
    setBusy() {}, historyId: 0, history: [], ticker: null,
    AbortController, TextEncoder, performance: { now: () => 50 },
    setInterval: () => 1, clearInterval() {}, numeric: (value) => String(value),
    responseTimings, responseLog, logLabel, t, m, getLocale, setText,
    setAttr(element, name, value) { element.setAttribute(name, String(value)); },
    node, button: (label) => node('button', '', label),
    lastResponse: null, lastLog: null, summarize() {}, compactPreview: JSON.stringify,
    responseBadge: field(), timeMetric: field(), serverMetric: field(), statusMetric: field(),
    tokenMetric: field(), rawDetails: field(), rawSummary: field(), rawPre: field(), responsePreview: field(),
    responseSummary: field(), timingDetails: field(), timingBreakdown: field(), historyList: field(),
    requestLog: field(), requestLogStatus: field(), requestLogId: field(), copyLogIdButton: field(),
    fullResponse: '', tell() {}, error(message) { throw new Error(message); },
    CHOICE_SCENES: {}, choiceForm: null, refreshEditor() {},
    onLocaleChange(callback) { requestScope.localeChanged = callback; return () => {}; },
  };
  const fragments = [
    ['  function resetResponse(', '\n  function applyPreset(', 'this.resetResponse = resetResponse;'],
    ['  function renderHistory(', '\n  async function copy(', 'this.renderHistory = renderHistory;'],
    ['  function renderLog(', '\n  on(copyLogIdButton,', 'this.renderLog = renderLog;'],
    ['  async function send() {', '\n  function readFile', 'this.sendRequest = send;'],
    ['  const stopLocale = onLocaleChange(', '\n  applyPreset("choice_text");', ''],
  ];
  const context = vm.createContext(requestScope);
  for (const [start, finish, expose] of fragments) {
    const from = playgroundSource.indexOf(start);
    vm.runInContext(playgroundSource.slice(from, playgroundSource.indexOf(finish, from)) + '\n' + expose, context);
  }
  requestScope.respond = (requestId, logStatus, status = 200) => {
    requestScope.fetch = async () => ({
      status, ok: status < 400, headers: logHeaders(requestId, logStatus),
      text: async () => JSON.stringify({ response: 'Keep this response' }),
    });
  };
  return requestScope;
}

test('log headers distinguish saved, failed, neutral and legacy responses', () => {
  assert.equal(responseLog(undefined), null);
  assert.equal(responseLog(new Headers()), null);
  assert.equal(responseLog(logHeaders('', 'pending')), null);
  const saved = responseLog(logHeaders('qev-one', 'saved'));
  assert.equal(saved.requestId, 'qev-one');
  assert.equal(saved.status, 'saved');
  assert.equal(responseLog(logHeaders('qev-two', 'error')).status, 'error');
  for (const status of [undefined, 'pending', 'SAVED']) {
    const neutral = responseLog(logHeaders('qev-neutral', status));
    assert.equal(neutral.requestId, 'qev-neutral');
    assert.equal(neutral.status, null);
    assert.equal(String(logLabel(neutral)), String(m('pg.requestId')));
  }
});

test('saved log ID survives locale changes with response and history intact', async () => {
  setLocale('en', { persist: false });
  const view = responseHarness();
  view.respond('qev-unique-id', 'saved');
  await view.sendRequest();
  const response = view.fullResponse, history = view.history[0];
  assert.equal(view.requestLog.hidden, false);
  assert.equal(view.requestLogStatus.textContent, 'Local log saved');
  assert.equal(view.requestLogId.textContent, 'qev-unique-id');
  assert.equal(view.copyLogIdButton.hidden, false);
  assert.match(view.historyList.children[0].children[0].attributes.title, /Request ID: qev-unique-id/);
  setLocale('zh-CN', { persist: false });
  view.localeChanged();
  assert.equal(view.requestLogStatus.textContent, '本地日志已记录');
  assert.equal(view.requestLogId.textContent, 'qev-unique-id');
  assert.equal(view.fullResponse, response);
  assert.equal(view.history[0], history);
  assert.match(view.historyList.children[0].children[0].attributes.title, /请求 ID：qev-unique-id/);
});

test('log write failure remains distinct from successful HTTP and translates without claiming saved', async () => {
  const view = responseHarness();
  view.respond('qev-write-failed', 'error');
  await view.sendRequest();
  assert.equal(view.statusMetric.textContent, '200');
  assert.equal(view.requestLogStatus.textContent, '本地日志写入失败');
  assert.match(view.requestLog.className, /error/);
  assert.equal(view.requestLogId.textContent, 'qev-write-failed');
  setLocale('en', { persist: false }); view.localeChanged();
  assert.equal(view.requestLogStatus.textContent, 'Local log write failed');
  assert.ok(!view.requestLogStatus.textContent.includes('saved'));
  assert.equal(view.history[0].log.status, 'error');
});

test('HTTP error responses can still have saved logs and old servers hide the row', async () => {
  const view = responseHarness();
  view.respond('qev-http-error', 'saved', 422);
  await view.sendRequest();
  assert.equal(view.statusMetric.textContent, '422');
  assert.equal(view.requestLogStatus.textContent, 'Local log saved');
  view.respond(undefined, undefined);
  await view.sendRequest();
  assert.equal(view.requestLog.hidden, true);
  assert.equal(view.requestLogId.textContent, '');
  assert.equal(view.copyLogIdButton.hidden, true);
  assert.equal(view.history[0].log, null);
  assert.equal(view.history[1].log.requestId, 'qev-http-error');
});

test('new requests clear the old log immediately and network/body failures leave no stale ID', async () => {
  const view = responseHarness();
  view.respond('qev-old', 'saved'); await view.sendRequest();
  let reject;
  view.fetch = () => new Promise((_resolve, fail) => { reject = fail; });
  const request = view.sendRequest();
  assert.equal(view.requestLog.hidden, true);
  assert.equal(view.requestLogId.textContent, '');
  reject(new Error('offline')); await request;
  assert.equal(view.requestLog.hidden, true);
  assert.equal(view.history[0].log, null);
  view.fetch = async () => ({ status: 200, ok: true, headers: logHeaders('partial-body', 'saved'), text: async () => { throw new Error('body interrupted'); } });
  await view.sendRequest();
  assert.equal(view.requestLogId.textContent, '');
  assert.equal(view.history[0].log, null);
});

test('cancelled response cannot publish its headers even when a late body resolves', async () => {
  const view = responseHarness();
  let finishBody, startBody;
  const bodyStarted = new Promise((done) => { startBody = done; });
  view.fetch = async () => ({ status: 200, ok: true, headers: logHeaders('cancelled-id', 'saved'), text: () => new Promise((done) => { finishBody = done; startBody(); }) });
  const request = view.sendRequest();
  await bodyStarted;
  view.active.cancelled = true;
  view.active.controller.abort();
  finishBody('{}'); await request;
  assert.equal(view.requestLog.hidden, true);
  assert.equal(view.requestLogId.textContent, '');
  assert.equal(view.history[0].log, null);
  assert.equal(String(view.history[0].status), String(m('pg.cancel')));
});
