import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { messages as ui } from '../qev/web/messages.js';
import { messages as choice } from '../qev/web/choice-messages.js';
import { detectLocale, getLocale, setLocale, t, m, setText, setAttr, translateTree, onLocaleChange } from '../qev/web/i18n.js';

test('browser preference order recognizes English and Chinese variants, with English fallback', () => {
  for (const variant of ['zh-CN', 'zh-TW', 'zh-HK', 'zh-Hans', 'ZH']) assert.equal(detectLocale([variant]), 'zh-CN');
  assert.equal(detectLocale(['en-US', 'zh-CN']), 'en');
  assert.equal(detectLocale(['fr-FR', 'zh-TW', 'en']), 'zh-CN');
  assert.equal(detectLocale(['de-DE']), 'en');
  assert.equal(detectLocale([]), 'en');
});

test('both dictionaries cover the same keys and interpolation arguments', async () => {
  const en = { ...ui.en, ...choice.en }, zh = { ...ui['zh-CN'], ...choice['zh-CN'] };
  assert.deepEqual(Object.keys(en).sort(), Object.keys(zh).sort());
  for (const key of Object.keys(en)) {
    assert.ok(en[key].trim() && zh[key].trim(), key);
    const args = (text) => Array.from(text.matchAll(/\{(\w+)\}/g), (match) => match[1]).sort();
    assert.deepEqual(args(en[key]), args(zh[key]), key);
  }
  for (const file of ['app.js', 'playground.js', 'choice-form.js', 'snake.js', 'index.html']) {
    const source = await readFile(new URL(`../qev/web/${file}`, import.meta.url), 'utf8');
    for (const [, key] of source.matchAll(/(?:\b[mt]\(\s*|data-i18n-[\w-]+=)["']((?:app|pg|snake|choice)\.[\w.]+)["']/g)) assert.ok(en[key], `${file}: ${key}`);
  }
});

test('saved preference overrides browser language and denied storage does not break switching', async () => {
  const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  try {
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { languages: ['zh-TW'] } });
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => 'en', setItem() { throw new Error('disabled'); } } });
    const saved = await import(`../qev/web/i18n.js?saved=${Date.now()}`);
    assert.equal(saved.getLocale(), 'en');
    saved.setLocale('zh-CN');
    assert.equal(saved.getLocale(), 'zh-CN');
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, get() { throw new Error('disabled'); } });
    const blocked = await import(`../qev/web/i18n.js?blocked=${Date.now()}`);
    assert.equal(blocked.getLocale(), 'zh-CN');
    blocked.setLocale('en');
    assert.equal(blocked.getLocale(), 'en');
  } finally {
    if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage); else delete globalThis.localStorage;
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator); else delete globalThis.navigator;
  }
});

test('bindings translate UI labels and attributes while leaving raw values untouched', () => {
  const attributes = new Map();
  const element = {
    textContent: '', dataset: {},
    setAttribute(name, value) { attributes.set(name, String(value)); this.dataset[name.replace(/^data-/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value; },
    removeAttribute(name) { attributes.delete(name); },
    hasAttribute: (name) => attributes.has(name), getAttribute: (name) => attributes.get(name),
    matches: () => true, querySelectorAll: () => [],
  };
  setLocale('en', { persist: false });
  setText(element, m('pg.send'));
  setAttr(element, 'aria-label', m('pg.request'));
  assert.equal(element.textContent, 'Send request');
  setLocale('zh-CN', { persist: false });
  translateTree(element);
  assert.equal(element.textContent, '发送请求');
  assert.equal(element.getAttribute('aria-label'), '请求');
  setText(element, 'User text and 模型原文');
  setLocale('en', { persist: false });
  translateTree(element);
  assert.equal(element.textContent, 'User text and 模型原文');
});

test('language changes notify listeners once and message parameters stay literal', () => {
  setLocale('en', { persist: false });
  let calls = 0;
  const stop = onLocaleChange(() => calls++);
  setLocale('zh-CN', { persist: false });
  setLocale('zh-CN', { persist: false });
  setLocale('not-supported', { persist: false });
  assert.equal(calls, 1);
  assert.equal(getLocale(), 'zh-CN');
  stop();
  setLocale('en', { persist: false });
  assert.equal(calls, 1);
  assert.equal(t('pg.notSentError', { error: '<img onerror=bad>{key}' }), 'Not sent: <img onerror=bad>{key}');
  assert.equal(JSON.stringify({ instruction: m('pg.exampleRefund') }), '{"instruction":"Is the customer requesting a refund?"}');
});
