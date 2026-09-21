// Run with: node --test tests/test_choice_form.mjs
// Exercises clipboard routing and asynchronous request ownership without a model.
import assert from 'node:assert/strict';
import { File } from 'node:buffer';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../qev/web/choice-form.js', import.meta.url), 'utf8');
const messageSource = await readFile(new URL('../qev/web/choice-messages.js', import.meta.url), 'utf8');
const { messages } = await import(`data:text/javascript;base64,${Buffer.from(messageSource).toString('base64')}`);
let locale = 'en';
globalThis.choiceTestI18n = {
  getLocale: () => locale,
  t(key, params = {}) {
    assert.ok(messages[locale][key], `Missing ${locale} translation: ${key}`);
    return messages[locale][key].replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(params, name) ? String(params[name]) : match);
  },
};
const { mountChoiceForm, choicePreset, CHOICE_SCENES, validateChoiceScene } = await import(
  `data:text/javascript;base64,${Buffer.from(source.replace(/^import .*i18n.js';$/m, 'const { t, getLocale } = globalThis.choiceTestI18n;')).toString('base64')}`,
);
globalThis.File = File;
const pixel = 'data:image/png;base64,old';
const newPixel = 'data:image/png;base64,new';
const picture = (url = pixel) => ({ type: 'image_url', image_url: { url } });
const text = (value) => ({ type: 'text', text: value });
const dataKey = (name) => name.replace(/^data-/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.attributes = {};
    this.handlers = {}; this.className = ''; this.textContent = ''; this.value = '';
    this.classList = { add: (value) => { this.className += ` ${value}`; } };
  }
  append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
  replaceChildren(...children) { for (const child of this.children) child.parent = null; this.children = []; this.append(...children); }
  setAttribute(name, value) { this.attributes[name] = String(value); if (name.startsWith('data-')) this.dataset[dataKey(name)] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  matches(selector) {
    return selector.split(',').some((part) => {
      const [, tag, attribute] = part.trim().match(/^(\w+)?(?:\[([^\]]+)\])?$/) || [];
      return (tag || attribute) && (!tag || this.tagName === tag.toUpperCase()) && (!attribute || (attribute.startsWith('data-') ? Object.hasOwn(this.dataset, dataKey(attribute)) : Object.hasOwn(this.attributes, attribute)));
    });
  }
  querySelectorAll(selector) { return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector); }
  contains(item) { return item === this || this.children.some((child) => child.contains(item)); }
  focus() { document.activeElement = this; }
  setSelectionRange(start, end, direction) { this.selectionStart = start; this.selectionEnd = end; this.selectionDirection = direction; }
  addEventListener(name, handler, options = {}) {
    this.handlers[name] = handler;
    options.signal?.addEventListener('abort', () => { delete this.handlers[name]; }, { once: true });
  }
  async emit(name, event) { await this.handlers[name]?.(event); }
}

function setup(initialScene = 'choice_option_images') {
  locale = 'en';
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { clipboard: {} } });
  globalThis.isSecureContext = true;
  globalThis.document = {
    activeElement: null,
    createElement(tag) {
      if (tag === 'canvas') return { getContext: () => new Proxy({}, { get: () => () => {} }), toDataURL: () => pixel };
      return new Element(tag);
    },
  };
  let payload = choicePreset(initialScene), scene = initialScene, writes = 0;
  const errors = [], uploads = [], container = new Element('div');
  const harness = { container, errors, uploads, validate: null };
  const form = mountChoiceForm(container, {
    read: () => structuredClone(payload),
    write(next, sync) { payload = structuredClone(next); writes++; if (sync) form.sync(payload, scene); },
    async upload(file, mutate, expected) {
      assert.equal(expected, JSON.stringify(payload));
      uploads.push(file);
      const next = structuredClone(payload);
      mutate(next, newPixel);
      if (harness.validate) await harness.validate(next, file);
      payload = next;
      form.sync(payload, scene);
    },
    onError(message) { errors.push(message); },
  });
  form.sync(payload, scene);
  return Object.assign(harness, {
    form,
    getPayload: () => payload, writes: () => writes,
    setPayload(next) { payload = next; form.sync(payload, scene); },
    switchScene(next) { scene = next; payload = choicePreset(next); form.sync(payload, scene); },
    area(key = '', index = '0') { return container.querySelectorAll('[data-paste-target]').find((item) => item.dataset.target === key && item.dataset.index === index); },
    button(action, key = '', index = '0') { return container.querySelectorAll('button').find((item) => item.dataset.action === action && item.dataset.target === key && item.dataset.index === index); },
    click(button) { return container.emit('click', { target: button }); },
    async paste(target, { image = true, mixed = false, filesOnly = false } = {}) {
      const event = {
        target, prevented: false, preventDefault() { this.prevented = true; },
        clipboardData: {
          items: [...(image && !filesOnly ? [{ kind: 'file', type: 'image/png', getAsFile: () => new File(['image'], '', { type: 'image/png' }) }] : []), ...(mixed || !image ? [{ kind: 'string', type: 'text/plain' }] : [])],
          files: image && filesOnly ? [new File(['image'], '', { type: 'image/png' })] : [],
        },
      };
      await container.emit('paste', event);
      return event;
    },
  });
}

function deferred() { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; }
function clipboardItems() { return [{ types: ['text/plain', 'image/png'], getType: async () => new Blob(['image'], { type: 'image/png' }) }]; }

test('language updates form and scenes while preserving edited JSON and selection', () => {
  const h = setup('choice_text');
  const input = h.container.querySelectorAll('textarea')[0];
  input.value = 'My edited state';
  h.container.handlers.input({ target: input });
  input.focus(); input.setSelectionRange(3, 9, 'forward');
  const before = JSON.stringify(h.getPayload()), writes = h.writes();
  locale = 'zh-CN';
  h.form.refreshLocale();
  assert.equal(JSON.stringify(h.getPayload()), before);
  assert.equal(h.writes(), writes);
  assert.equal(document.activeElement.value, 'My edited state');
  assert.equal(document.activeElement.selectionStart, 3);
  assert.equal(document.activeElement.getAttribute('aria-label'), '背景文字');
  assert.equal(CHOICE_SCENES.choice_text.label, '纯文本选择');
  assert.match(choicePreset('choice_text').state, /用户说/);
  assert.deepEqual(Object.keys(messages.en).sort(), Object.keys(messages['zh-CN']).sort());
});

test('mixed image and text paste replaces exactly the second option image', async () => {
  const h = setup();
  h.getPayload().questions.image.criteria.B.content = [text('Keep this'), picture('first'), text('Between'), picture('second')];
  h.setPayload(h.getPayload());
  const firstOption = structuredClone(h.getPayload().questions.image.criteria.A);
  const event = await h.paste(h.area('B', '1'), { mixed: true });
  assert.equal(event.prevented, true);
  assert.equal(h.uploads.length, 1);
  assert.equal(h.uploads[0].name, 'clipboard.png');
  assert.deepEqual(h.getPayload().questions.image.criteria.B.content, [text('Keep this'), picture('first'), text('Between'), picture(newPixel)]);
  assert.deepEqual(h.getPayload().questions.image.criteria.A, firstOption);
  assert.equal(typeof h.getPayload().state, 'string');
});

test('empty context paste target and file-list fallback use the shared upload path', async () => {
  const h = setup('choice_context_image');
  h.getPayload().state = 'Keep this context'; h.setPayload(h.getPayload());
  const area = h.area('', '');
  assert.equal(area.tabIndex, 0);
  assert.match(area.getAttribute('aria-label'), /Context/);
  const event = await h.paste(area, { filesOnly: true });
  assert.equal(event.prevented, true);
  assert.deepEqual(h.getPayload().state.content, [text('Keep this context'), picture(newPixel)]);
  assert.equal(h.uploads.length, 1);
});

test('direct candidate arrays remain editable and keep text while pasting an image', async () => {
  const h = setup();
  h.getPayload().questions.image.criteria.A = [text('Direct candidate'), picture('original')];
  h.setPayload(h.getPayload());
  await h.paste(h.area('A'));
  const candidate = h.getPayload().questions.image.criteria.A;
  assert.deepEqual(Array.isArray(candidate) ? candidate : candidate.content, [text('Direct candidate'), picture(newPixel)]);
  assert.equal(h.errors.length, 0);
});

test('text paste and image paste into a text field are not intercepted', async () => {
  const h = setup();
  assert.equal((await h.paste(h.area('A'), { image: false })).prevented, false);
  assert.equal((await h.paste(h.container.querySelectorAll('textarea')[0])).prevented, false);
  assert.equal(h.uploads.length, 0);
});

test('busy state blocks paste and restores keyboard access when idle', async () => {
  const h = setup();
  h.form.setBusy(true);
  assert.equal(h.area('A').tabIndex, -1);
  assert.equal((await h.paste(h.area('A'))).prevented, true);
  assert.equal(h.uploads.length, 0);
  assert.match(h.errors[0], /Please wait/);
  h.form.setBusy(false);
  assert.equal(h.area('A').tabIndex, 0);
});

test('clipboard button handles unsupported, denied and empty clipboard in the selected language', async () => {
  const h = setup();
  await h.click(h.button('paste_image', 'A'));
  assert.match(h.errors.at(-1), /Ctrl\/Cmd\+V/);
  navigator.clipboard.read = async () => { throw Object.assign(new Error(), { name: 'NotAllowedError' }); };
  locale = 'zh-CN'; h.form.refreshLocale();
  await h.click(h.button('paste_image', 'A'));
  assert.match(h.errors.at(-1), /未获得剪贴板读取权限/);
  navigator.clipboard.read = async () => [{ types: ['text/plain'] }];
  await h.click(h.button('paste_image', 'A'));
  assert.match(h.errors.at(-1), /剪贴板中没有图片/);
  assert.equal(h.uploads.length, 0);
  assert.equal(h.button('paste_image', 'A').disabled, false);
});

test('clipboard button reads image data and reuses upload validation without committing rejected data', async () => {
  const h = setup();
  navigator.clipboard.read = async () => clipboardItems();
  const before = JSON.stringify(h.getPayload());
  h.validate = async () => { throw new Error('Image exceeds request budget'); };
  await h.click(h.button('paste_image', 'C'));
  assert.equal(h.uploads.length, 1);
  assert.equal(h.uploads[0].name, 'clipboard.png');
  assert.equal(JSON.stringify(h.getPayload()), before);
  assert.deepEqual(h.errors, ['Image exceeds request budget']);
});

test('clipboard completion cannot attach to a scene that switched away and back', async () => {
  const h = setup(), pending = deferred();
  navigator.clipboard.read = () => pending.promise;
  const operation = h.click(h.button('paste_image', 'A'));
  assert.equal(h.button('paste_image', 'A').disabled, true);
  h.switchScene('choice_text'); h.switchScene('choice_option_images');
  pending.resolve(clipboardItems()); await operation;
  assert.equal(h.uploads.length, 0);
  assert.match(h.errors.at(-1), /target changed/);
});

test('clipboard blob completion cannot attach after the request was edited', async () => {
  const h = setup(), pending = deferred();
  navigator.clipboard.read = async () => [{ types: ['image/png'], getType: () => pending.promise }];
  const operation = h.click(h.button('paste_image', 'A'));
  await Promise.resolve();
  h.getPayload().questions.image.instructions = 'A different question';
  h.setPayload(h.getPayload());
  pending.resolve(new Blob(['image'], { type: 'image/png' })); await operation;
  assert.equal(h.uploads.length, 0);
  assert.match(h.errors.at(-1), /target changed/);
});

test('locale refresh does not invalidate a pending clipboard operation; destroy does', async () => {
  const h = setup(), pending = deferred();
  navigator.clipboard.read = () => pending.promise;
  const operation = h.click(h.button('paste_image', 'B'));
  locale = 'zh-CN'; h.form.refreshLocale();
  pending.resolve(clipboardItems()); await operation;
  assert.equal(h.uploads.length, 1);
  const pending2 = deferred(); navigator.clipboard.read = () => pending2.promise;
  const operation2 = h.click(h.button('paste_image', 'B'));
  h.form.destroy(); pending2.resolve(clipboardItems()); await operation2;
  assert.equal(h.uploads.length, 1);
  assert.equal(h.errors.length, 0);
});

test('scene validation errors follow the active language', () => {
  const h = setup();
  h.getPayload().questions.image.criteria.B.content.pop();
  assert.throws(() => validateChoiceScene(h.getPayload(), 'choice_option_images'), /Options B are missing images/);
  locale = 'zh-CN';
  assert.throws(() => validateChoiceScene(h.getPayload(), 'choice_option_images'), /选项 B 缺少图片/);
});
