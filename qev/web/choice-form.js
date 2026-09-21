/** Choice scenarios edit the same request object as the advanced JSON editor. */
import { t, getLocale } from './i18n.js';

function sceneDescription(key, icon) {
  return {
    icon,
    get label() { return t(`choice.scene.${key}.label`); },
    get description() { return t(`choice.scene.${key}.description`); },
    get note() { return t(`choice.scene.${key}.note`); },
  };
}
export const CHOICE_SCENES = {
  choice_text: sceneDescription("text", "Aa"),
  choice_context_image: sceneDescription("context", "◈"),
  choice_option_images: sceneDescription("options", "▦"),
};

const examples = new Map();
function exampleImage(shape) {
  if (examples.has(shape)) return examples.get(shape);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 96;
  const context = canvas.getContext("2d");
  context.fillStyle = "#f6f8fb";
  context.fillRect(0, 0, 96, 96);
  context.fillStyle = { square: "#e64452", circle: "#2774e8", triangle: "#edb623" }[shape];
  if (shape === "square") context.fillRect(21, 21, 54, 54);
  if (shape === "circle") { context.beginPath(); context.arc(48, 48, 29, 0, Math.PI * 2); context.fill(); }
  if (shape === "triangle") { context.beginPath(); context.moveTo(48, 17); context.lineTo(80, 76); context.lineTo(16, 76); context.closePath(); context.fill(); }
  const image = canvas.toDataURL("image/png");
  examples.set(shape, image);
  return image;
}

function picture(url) { return { type: "image_url", image_url: { url } }; }
function contentOf(value) { return Array.isArray(value) ? value : value && Object.keys(value).length === 1 && Array.isArray(value.content) ? value.content : null; }
function imagesOf(value) { return (contentOf(value) || []).filter((item) => item?.type === "image_url"); }
function textOf(value) { return typeof value === "string" ? value : (contentOf(value) || []).filter((item) => item?.type === "text").map((item) => item.text).join("\n"); }
function withContent(value, content, option) { return option || !Array.isArray(value) ? { ...(typeof value === "object" && !Array.isArray(value) ? value : {}), content } : content; }
function withText(value, text, option) {
  if (typeof value === "string") return text;
  return withContent(value, [{ type: "text", text }, ...(contentOf(value) || []).filter((item) => item?.type !== "text")], option);
}

export function choicePreset(scene) {
  if (scene === "choice_text") return {
    model: "qev-latest", state: t("choice.preset.text.state"),
    questions: { team: { type: "choice", instructions: t("choice.preset.text.question"), criteria: { billing: t("choice.preset.text.billing"), technical: t("choice.preset.text.technical"), sales: t("choice.preset.text.sales") } } },
  };
  if (scene === "choice_context_image") return {
    model: "qev-latest", state: [{ type: "text", text: t("choice.preset.context.state") }, picture(exampleImage("square"))],
    questions: { shape: { type: "choice", instructions: t("choice.preset.context.question"), criteria: { A: t("choice.preset.redSquare"), B: t("choice.preset.blueCircle"), C: t("choice.preset.yellowTriangle") } } },
  };
  return {
    model: "qev-latest", state: t("choice.preset.options.state"),
    questions: { image: { type: "choice", instructions: t("choice.preset.options.question"), criteria: Object.fromEntries(
      ["square", "circle", "triangle"].map((shape, index) => ["ABC"[index], { content: [{ type: "text", text: t("choice.preset.candidate", { key: "ABC"[index] }) }, picture(exampleImage(shape))] }]),
    ) } },
  };
}

export function editableChoice(payload) {
  const entries = Object.entries(payload?.questions || {});
  if (entries.length !== 1 || entries[0][1]?.type !== "choice") return null;
  const [key, question] = entries[0];
  if (!question.criteria || typeof question.criteria !== "object" || Array.isArray(question.criteria)) return null;
  if (question.instructions != null && typeof question.instructions !== "string") return null;
  if (Object.keys(question.criteria).some((candidate) => !candidate)) return null;
  const simple = (value) => typeof value === "string" || (contentOf(value) !== null && contentOf(value).every((item) => ["text", "image_url"].includes(item?.type)));
  if (!simple(payload.state) || !Object.values(question.criteria).every(simple)) return null;
  return { key, question };
}

export function validateChoiceScene(payload, scene) {
  if (!scene) return;
  const editable = editableChoice(payload);
  if (!editable) throw new Error(t("choice.validation.compatible"));
  const { question } = editable;
  const entries = Object.entries(question.criteria);
  if (entries.length < 2) throw new Error(t("choice.validation.minOptions"));
  if (!question.instructions?.trim()) throw new Error(t("choice.validation.question"));
  const stateImages = imagesOf(payload.state);
  const optionImages = entries.flatMap(([, value]) => imagesOf(value));
  if (scene === "choice_text" && (stateImages.length || optionImages.length)) throw new Error(t("choice.validation.textNoImages"));
  if (scene === "choice_context_image") {
    if (!stateImages.length) throw new Error(t("choice.validation.backgroundMissing"));
    if (optionImages.length) throw new Error(t("choice.validation.backgroundTextOptions"));
  }
  if (scene === "choice_option_images") {
    const missing = entries.filter(([, value]) => !imagesOf(value).length).map(([key]) => key);
    if (missing.length) throw new Error(t("choice.validation.optionMissing", { keys: missing.join(getLocale() === "zh-CN" ? "、" : ", ") }));
  }
}

export function mountChoiceForm(container, { read, write, upload, onError }) {
  const listeners = new AbortController();
  let scene = null;
  let busy = false;
  let clipboardPending = false;
  let disposed = false;
  let revision = 0;
  let renderedRequest;
  let serial = 0;
  function element(tag, className = "", text = "") { const item = document.createElement(tag); item.className = className; item.textContent = text; return item; }
  function button(text, action, target = "", index = "") { const item = element("button", "btn small ghost", text); item.type = "button"; Object.assign(item.dataset, { action, target, index }); return item; }
  function field(label, value, kind, target = "") {
    const wrapper = element("label", "field pg-choice-field");
    wrapper.append(element("span", "", label));
    const input = element("textarea", "pg-choice-input");
    input.rows = kind === "state" ? 3 : 2;
    input.value = value;
    input.dataset.kind = kind;
    input.dataset.target = target;
    input.setAttribute("aria-label", label);
    input.id = `choice-field-${++serial}`;
    wrapper.append(input);
    return wrapper;
  }
  function media(value, target, title) {
    const region = element("div", "pg-choice-images");
    const images = imagesOf(value);
    function pasteTarget(tile, index = "") {
      tile.classList.add("pg-choice-paste-target");
      tile.tabIndex = 0;
      tile.setAttribute("role", "group");
      tile.setAttribute("aria-label", t(index === "" ? "choice.pasteTarget" : "choice.replacePasteTarget", { target: title, number: Number(index) + 1 }));
      Object.assign(tile.dataset, { pasteTarget: "true", target, index });
      tile.append(element("span", "pg-choice-paste-hint", t("choice.pasteHint")));
      return tile;
    }
    for (const [index, image] of images.entries()) {
      const tile = element("div", "pg-choice-image");
      const preview = element("img");
      const url = image.image_url?.url;
      preview.alt = t("choice.imageAlt", { target: title, number: index + 1 });
      if (typeof url === "string" && /^data:image\/(png|jpeg|webp);base64,/.test(url)) preview.src = url;
      const actions = element("div", "pg-choice-image-actions");
      actions.append(button(t("choice.replaceImage"), "upload", target, String(index)), button(t("choice.pasteImage"), "paste_image", target, String(index)), button(t("choice.removeImage"), "remove_image", target, String(index)));
      tile.append(preview, actions);
      region.append(pasteTarget(tile, String(index)));
    }
    if (!images.length) {
      const empty = element("div", "pg-choice-image-empty");
      const actions = element("div", "pg-choice-image-actions");
      actions.append(button(t("choice.addImage"), "upload", target), button(t("choice.pasteImage"), "paste_image", target));
      empty.append(element("span", "", t("choice.noImage")), actions);
      region.append(pasteTarget(empty));
    }
    return region;
  }
  function sync(payload, nextScene) {
    if (disposed) return;
    const serialized = JSON.stringify(payload);
    if (scene !== nextScene || renderedRequest !== serialized) revision += 1;
    renderedRequest = serialized;
    scene = nextScene;
    const editable = editableChoice(payload);
    container.replaceChildren();
    container.hidden = !editable;
    if (!editable) return;
    const background = element("section", "pg-choice-section");
    background.append(element("h3", "pg-choice-title", t("choice.backgroundTitle")), field(t("choice.backgroundLabel"), textOf(payload.state), "state"));
    if (scene === "choice_context_image" || imagesOf(payload.state).length) background.append(media(payload.state, "", t("choice.backgroundTarget")));
    const question = element("section", "pg-choice-section");
    question.append(element("h3", "pg-choice-title", t("choice.questionTitle")), field(t("choice.questionLabel"), editable.question.instructions || "", "question"));
    const options = element("section", "pg-choice-section");
    const heading = element("div", "pg-choice-options-heading");
    heading.append(element("h3", "pg-choice-title", t("choice.optionsTitle")), button(t("choice.addOption"), "add_option"));
    options.append(heading);
    const cards = element("div", "pg-choice-options");
    const entries = Object.entries(editable.question.criteria);
    for (const [key, value] of entries) {
      const hasPictures = scene === "choice_option_images" || imagesOf(value).length > 0;
      const card = element("article", `pg-choice-option${hasPictures ? " has-media" : ""}`);
      const header = element("div", "pg-choice-option-header");
      header.append(element("span", "pg-choice-option-key", key));
      if (entries.length > 2) header.append(button(t("choice.removeOption"), "remove_option", key));
      card.append(header, field(t("choice.optionLabel", { key }), textOf(value), "option", key));
      if (hasPictures) card.append(media(value, key, t("choice.optionTarget", { key })));
      cards.append(card);
    }
    options.append(cards);
    container.append(background, question, options);
    setBusy(busy);
  }
  function target(payload, key) {
    const editable = editableChoice(payload);
    if (!editable) throw new Error(t("choice.fixJson"));
    if (key && !Object.hasOwn(editable.question.criteria, key)) throw new Error(t("choice.paste.changed"));
    return key ? [editable.question.criteria, key, true] : [payload, "state", false];
  }
  function updateImage(payload, key, index, url) {
    const [owner, name, option] = target(payload, key);
    const value = owner[name];
    const content = [...(contentOf(value) || [{ type: "text", text: textOf(value) }])];
    const positions = content.flatMap((item, position) => item?.type === "image_url" ? [position] : []);
    if (index !== "" && positions[Number(index)] !== undefined) {
      if (url) content[positions[Number(index)]] = picture(url);
      else content.splice(positions[Number(index)], 1);
    } else if (url) content.push(picture(url));
    owner[name] = withContent(value, content, option);
  }
  function expectedRequest() { return { serialized: JSON.stringify(read()), revision, scene }; }
  function checkExpected(expected) {
    if (disposed || expected.revision !== revision || expected.scene !== scene || expected.serialized !== JSON.stringify(read())) {
      throw new Error(t("choice.paste.changed"));
    }
  }
  async function attachImage(file, key, index, expected) {
    checkExpected(expected);
    if (busy) throw new Error(t("choice.paste.busy"));
    await upload(file, (payload, url) => {
      checkExpected(expected);
      updateImage(payload, key, index, url);
    }, expected.serialized);
  }
  function clipboardFile(clipboard) {
    for (const item of clipboard?.items || []) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) return file;
      }
    }
    return Array.from(clipboard?.files || []).find((file) => file.type.startsWith("image/"));
  }
  async function pasteImage(key, index, file) {
    if (busy || clipboardPending) { onError(t("choice.paste.busy")); return; }
    clipboardPending = true;
    updateBusy();
    try {
      const expected = expectedRequest();
      if (!file) {
        if (globalThis.isSecureContext === false || typeof globalThis.navigator?.clipboard?.read !== "function") {
          throw new Error(t("choice.paste.unsupported"));
        }
        let items;
        try { items = await navigator.clipboard.read(); }
        catch (problem) { throw new Error(t(["NotAllowedError", "SecurityError"].includes(problem?.name) ? "choice.paste.denied" : "choice.paste.failed")); }
        checkExpected(expected);
        for (const item of items) {
          const type = ["image/png", "image/jpeg", "image/webp"].find((mime) => item.types.includes(mime)) || item.types.find((mime) => mime.startsWith("image/"));
          if (!type) continue;
          try { file = await item.getType(type); }
          catch (problem) { throw new Error(t(["NotAllowedError", "SecurityError"].includes(problem?.name) ? "choice.paste.denied" : "choice.paste.failed")); }
          break;
        }
        if (!file) throw new Error(t("choice.paste.empty"));
      }
      await attachImage(new File([file], "clipboard.png", { type: file.type }), key, index, expected);
    } catch (problem) { if (!disposed) onError(problem.message || t("choice.paste.failed")); }
    finally { clipboardPending = false; if (!disposed) updateBusy(); }
  }
  container.addEventListener("paste", async (event) => {
    const area = event.target.closest?.("[data-paste-target]");
    if (!area || !container.contains(area)) return;
    const file = clipboardFile(event.clipboardData);
    if (!file) return; // Preserve native text paste, including into the text fields.
    event.preventDefault();
    await pasteImage(area.dataset.target, area.dataset.index, file);
  }, { signal: listeners.signal });
  container.addEventListener("input", (event) => {
    if (busy || clipboardPending || !event.target.dataset.kind) return;
    try {
      const payload = read();
      const { kind, target: key } = event.target.dataset;
      if (kind === "question") editableChoice(payload).question.instructions = event.target.value;
      else {
        const [owner, name, option] = target(payload, kind === "state" ? "" : key);
        owner[name] = withText(owner[name], event.target.value, option);
      }
      revision += 1;
      renderedRequest = JSON.stringify(payload);
      write(payload, false);
    } catch (problem) { onError(problem.message); }
  }, { signal: listeners.signal });
  container.addEventListener("click", async (event) => {
    const control = event.target.closest?.("button[data-action]");
    if (!control || busy || clipboardPending) return;
    const { action, target: key, index } = control.dataset;
    try {
      if (action === "upload") {
        const expected = expectedRequest();
        const input = document.createElement("input");
        input.type = "file"; input.accept = "image/png,image/jpeg,image/webp";
        input.addEventListener("change", async () => {
          if (!input.files?.[0] || disposed) return;
          try { await attachImage(input.files[0], key, index, expected); }
          catch (problem) { if (!disposed) onError(problem.message); }
        }, { once: true });
        input.click();
        return;
      }
      if (action === "paste_image") { await pasteImage(key, index); return; }
      const payload = read();
      const criteria = editableChoice(payload).question.criteria;
      if (action === "remove_image") updateImage(payload, key, index, null);
      if (action === "remove_option") delete criteria[key];
      if (action === "add_option") {
        if (Object.keys(criteria).length >= 8) throw new Error(t("choice.maxOptions"));
        let number = 1;
        while (Object.hasOwn(criteria, `option_${number}`)) number += 1;
        criteria[`option_${number}`] = scene === "choice_option_images" ? { content: [{ type: "text", text: t("choice.newOption") }] } : t("choice.newOption");
      }
      revision += 1;
      write(payload, true);
    } catch (problem) { onError(problem.message); }
  }, { signal: listeners.signal });
  function updateBusy() {
    const disabled = busy || clipboardPending;
    for (const control of container.querySelectorAll("button,textarea")) control.disabled = disabled;
    for (const area of container.querySelectorAll("[data-paste-target]")) {
      area.setAttribute("aria-disabled", String(disabled));
      area.setAttribute("aria-busy", String(disabled));
      area.tabIndex = disabled ? -1 : 0;
    }
  }
  function setBusy(value) { busy = value; updateBusy(); }
  function refreshLocale() {
    const focused = document.activeElement;
    const saved = container.contains(focused) ? {
      kind: focused.dataset.kind, action: focused.dataset.action, pasteTarget: focused.dataset.pasteTarget,
      target: focused.dataset.target, index: focused.dataset.index,
      start: focused.selectionStart, end: focused.selectionEnd, direction: focused.selectionDirection,
    } : null;
    // Invalid advanced JSON remains owned by the editor; a language change must
    // never replace it with a preset or the last valid request.
    try { sync(read(), scene); } catch { return; }
    if (!saved) return;
    const replacement = Array.from(container.querySelectorAll("textarea,button,[data-paste-target]")).find((item) =>
      ["kind", "action", "pasteTarget", "target", "index"].every((key) => item.dataset[key] === saved[key]));
    replacement?.focus({ preventScroll: true });
    if (saved.kind && replacement) replacement.setSelectionRange(saved.start, saved.end, saved.direction);
  }
  return { sync, setBusy, refreshLocale, destroy() { disposed = true; listeners.abort(); container.replaceChildren(); } };
}
