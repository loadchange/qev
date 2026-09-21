import { mountSnake } from './snake.js';
import { mountPlayground } from './playground.js';
import { getLocale, setLocale, onLocaleChange, t, m, setText, translateTree } from './i18n.js';

setLocale(getLocale(), { persist: false });
const localeSwitch = document.querySelector('#locale-switch');
localeSwitch.value = getLocale();
localeSwitch.addEventListener('change', () => setLocale(localeSwitch.value));

const toast = document.querySelector('#toast');
let toastTimer;
function notify(message) {
  setText(toast, message);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4500);
}
let snake = null;
const cleanupPlayground = mountPlayground(document.querySelector('#playground-view'), { notify });
const playgroundHeading = document.createElement('div');
playgroundHeading.className = 'page-heading';
playgroundHeading.innerHTML = '<div><div class="eyebrow">02 / API PLAYGROUND</div><h1 data-i18n-text="app.heading"></h1><p data-i18n-text="app.subtitle"></p></div><span class="badge green" data-i18n-text="app.local"></span>';
translateTree(playgroundHeading);
document.querySelector('#playground-view').prepend(playgroundHeading);
function route() {
  const page = location.hash === '#playground' || (!location.hash && location.pathname === '/playground') ? 'playground' : 'snake';
  for (const view of ['snake', 'playground']) document.querySelector(`#${view}-view`).hidden = page !== view;
  for (const link of document.querySelectorAll('[data-tab]')) {
    if (link.dataset.tab === page) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  if (page === 'snake' && !snake) snake = mountSnake(document.querySelector('#snake-view'), { notify });
  if (page !== 'snake') snake?.pause();
  document.title = `${t(page === 'snake' ? 'app.snake' : 'app.playground')} · Qev`;
}
window.addEventListener('hashchange', route);
document.addEventListener('visibilitychange', () => { if (document.hidden) snake?.pause(); });
route();
const stopLocale = onLocaleChange(() => { localeSwitch.value = getLocale(); route(); });
async function health() {
  const connection = document.querySelector('#connection');
  try {
    const response = await fetch('/health', { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error('offline');
    const result = await response.json();
    connection.replaceChildren(document.createElement('i'), setText(document.createElement('span'), m('app.online')));
    connection.classList.add('online');
    setText(document.querySelector('#backend-label'), m('app.backend', { backend: result.backend.toUpperCase() }));
  } catch {
    connection.replaceChildren(document.createElement('i'), setText(document.createElement('span'), m('app.offline')));
    connection.classList.remove('online');
    setText(document.querySelector('#backend-label'), m('app.checkService'));
  }
}
health();
const healthTimer = setInterval(() => { if (!document.hidden) health(); }, 30000);
window.addEventListener('pagehide', () => { snake?.pause(); });
window.addEventListener('beforeunload', () => { stopLocale(); clearInterval(healthTimer); clearTimeout(toastTimer); cleanupPlayground(); snake?.destroy(); });
