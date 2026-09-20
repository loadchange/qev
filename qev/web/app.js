import { mountSnake } from './snake.js';
import { mountPlayground } from './playground.js';

const toast = document.querySelector('#toast');
let toastTimer;
function notify(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4500);
}
const snake = mountSnake(document.querySelector('#snake-view'), { notify });
const cleanupPlayground = mountPlayground(document.querySelector('#playground-view'), { notify });
const playgroundHeading = document.createElement('div');
playgroundHeading.className = 'page-heading';
playgroundHeading.innerHTML = '<div><div class="eyebrow">02 / API PLAYGROUND</div><h1>从请求，到真实响应。</h1><p>在浏览器里验证结构化决策与多模态生成。</p></div><span class="badge green">LOCAL · 同源接口</span>';
document.querySelector('#playground-view').prepend(playgroundHeading);
function route() {
  const page = location.hash === '#playground' || (!location.hash && location.pathname === '/playground') ? 'playground' : 'snake';
  for (const view of ['snake', 'playground']) document.querySelector(`#${view}-view`).hidden = page !== view;
  for (const link of document.querySelectorAll('[data-tab]')) {
    if (link.dataset.tab === page) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  if (page !== 'snake') snake.pause();
  document.title = `${page === 'snake' ? '自主贪吃蛇' : '接口测试'} · Qev`;
}
window.addEventListener('hashchange', route);
document.addEventListener('visibilitychange', () => { if (document.hidden) snake.pause(); });
route();
async function health() {
  const connection = document.querySelector('#connection');
  try {
    const response = await fetch('/health', { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error('offline');
    const result = await response.json();
    connection.replaceChildren(Object.assign(document.createElement('i'), {}), document.createTextNode('模型在线'));
    connection.classList.add('online');
    document.querySelector('#backend-label').textContent = `${result.backend.toUpperCase()} 后端 · 同源连接`;
  } catch {
    connection.replaceChildren(document.createElement('i'), document.createTextNode('服务未连接'));
    connection.classList.remove('online');
    document.querySelector('#backend-label').textContent = '请检查 Qev 服务';
  }
}
health();
const healthTimer = setInterval(() => { if (!document.hidden) health(); }, 30000);
window.addEventListener('pagehide', () => { snake.pause(); });
window.addEventListener('beforeunload', () => { clearInterval(healthTimer); clearTimeout(toastTimer); cleanupPlayground(); snake.destroy(); });
