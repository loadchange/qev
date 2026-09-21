import { messages as choiceMessages } from './choice-messages.js';
import { messages as uiMessages } from './messages.js';

export const LOCALE_STORAGE_KEY = 'qev.locale';
const dictionaries = Object.fromEntries(['en', 'zh-CN'].map((locale) => [locale, {
  ...uiMessages[locale], ...choiceMessages[locale],
}]));
const listeners = new Set();
const bindings = new WeakMap();

export function detectLocale(languages = []) {
  for (const language of languages) {
    if (/^zh(?:-|$)/i.test(language)) return 'zh-CN';
    if (/^en(?:-|$)/i.test(language)) return 'en';
  }
  return 'en';
}

function initialLocale() {
  try {
    const saved = globalThis.localStorage?.getItem(LOCALE_STORAGE_KEY);
    if (saved === 'en' || saved === 'zh-CN') return saved;
  } catch { /* Private browsing can disable storage; language selection still works. */ }
  return detectLocale(globalThis.navigator?.languages?.length
    ? globalThis.navigator.languages : [globalThis.navigator?.language || 'en']);
}

let locale = initialLocale();
export function getLocale() { return locale; }
export function t(key, params = {}) {
  const template = dictionaries[locale][key] ?? dictionaries.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(params, name) ? String(params[name]) : match);
}

// A message retains its key until rendered, so labels update without rebuilding
// inputs, requests or running games. JSON serialization is only used for presets.
export function m(key, params = {}) {
  return { key, params, qevMessage: true, toString() { return t(key, params); }, toJSON() { return t(key, params); } };
}

export function setText(element, value) {
  const bound = bindings.get(element) || new Map();
  if (value?.qevMessage) { bound.set('textContent', value); element.setAttribute('data-i18n-text', value.key); }
  else { bound.delete('textContent'); element.removeAttribute('data-i18n-text'); }
  bindings.set(element, bound);
  element.textContent = String(value ?? '');
  return element;
}

export function setAttr(element, name, value) {
  element.removeAttribute(`data-i18n-${name}`);
  const bound = bindings.get(element) || new Map();
  if (value?.qevMessage) { bound.set(name, value); element.setAttribute('data-i18n-bound', ''); }
  else bound.delete(name);
  bindings.set(element, bound);
  element.setAttribute(name, String(value));
}

export function translateTree(root = globalThis.document) {
  if (!root?.querySelectorAll) return;
  const selector = '[data-i18n-text], [data-i18n-bound], [data-i18n-aria-label], [data-i18n-content]';
  const elements = [...(root.matches?.(selector) ? [root] : []), ...root.querySelectorAll(selector)];
  for (const element of elements) {
    const bound = bindings.get(element);
    if (element.hasAttribute('data-i18n-text')) element.textContent = String(bound?.get('textContent') ?? t(element.dataset.i18nText));
    for (const [name, message] of bound || []) if (name !== 'textContent') element.setAttribute(name, String(message));
    for (const name of ['aria-label', 'content']) {
      const key = element.getAttribute(`data-i18n-${name}`);
      if (key) element.setAttribute(name, t(key));
    }
  }
}

export function setLocale(next, { persist = true } = {}) {
  if (next !== 'en' && next !== 'zh-CN') return;
  const changed = next !== locale;
  locale = next;
  if (persist) {
    try { globalThis.localStorage?.setItem(LOCALE_STORAGE_KEY, locale); }
    catch { /* Keep the in-memory preference even when storage is unavailable. */ }
  }
  if (globalThis.document?.documentElement) document.documentElement.lang = locale;
  translateTree();
  if (changed) for (const listener of listeners) listener(locale);
}

export function onLocaleChange(listener) { listeners.add(listener); return () => listeners.delete(listener); }
