const DIRECTIONS = { UP: ['↑', '向上'], DOWN: ['↓', '向下'], LEFT: ['←', '向左'], RIGHT: ['→', '向右'] };
const REASONS = { wall: '撞到边界', body: '碰到自己的身体', starvation: '太久没有吃到食物', max_steps: '达到本局步数上限', model_error: '模型请求出错' };

export function mountSnake(container, { notify }) {
  container.innerHTML = `
    <div class="page-heading"><div><div class="eyebrow">01 / AUTONOMOUS PLAY</div><h1>让模型，走下一步。</h1><p>把每一次方向选择，变成看得见的模型决策。</p></div><span class="badge green">CHOICE · 真实模型推理</span></div>
    <div class="snake-layout">
      <section class="panel arena-panel" aria-label="游戏棋盘与控制">
        <div class="arena-top"><h2>贪吃蛇 <span class="badge">LIVE</span></h2><div class="arena-meta"><span id="game-size">12 × 12</span><span> / </span><span id="game-seed">SEED 7</span></div></div>
        <div class="arena-wrap"><canvas id="board" width="720" height="720" role="img" aria-label="贪吃蛇棋盘，等待创建游戏"></canvas><div class="arena-status"><span class="status-label" id="game-status" role="status">正在准备棋盘…</span><span id="game-limit">每局最多 500 步</span></div></div>
        <div class="stats-grid"><div class="metric"><div class="metric-label">吃到食物</div><div class="metric-value" id="score">0</div></div><div class="metric"><div class="metric-label">完成步数</div><div class="metric-value" id="steps">0</div></div><div class="metric"><div class="metric-label">蛇身长度</div><div class="metric-value" id="length">—</div></div><div class="metric"><div class="metric-label">模型耗时</div><div class="metric-value"><span id="latency">—</span><small>ms</small></div></div></div>
        <div class="arena-controls"><div class="toolbar"><button class="btn primary" id="play" disabled>▶ 开始自主运行</button><button class="btn" id="step" disabled>→ 单步决策</button><button class="btn ghost" id="reset" disabled>↻ 重开本局</button></div><div class="arena-settings"><label class="field">随机种子<input id="seed" type="number" min="0" max="2147483647" step="1" value="7"></label><label class="field">棋盘尺寸<select id="size"><option value="8">8 × 8</option><option value="12" selected>12 × 12</option><option value="16">16 × 16</option></select></label><label class="field">运行速度<select id="speed"><option value="1000">每秒至多 1 步</option><option value="250" selected>每秒至多 4 步</option><option value="125">每秒至多 8 步</option><option value="0">随推理速度运行</option></select></label></div><p class="hint" style="margin-top:12px">种子与尺寸在重开后生效。每一步都等待模型返回；切换页面或隐藏标签页会暂停。</p><div id="game-error" class="error" role="alert" hidden></div></div>
      </section>
      <aside class="snake-sidebar">
        <section class="panel"><div class="panel-header"><h2>模型的选择</h2><span class="badge" id="decision-badge">等待首步</span></div><div class="decision-label">LAST DECISION</div><div class="decision-value"><span class="direction-icon" id="direction-icon">·</span><div><strong id="direction-label">尚未推理</strong><br><small id="direction-step">点击开始，让 Qev 接手</small></div></div><div id="probabilities" role="group" aria-label="候选选择概率"></div><div class="decision-foot"><span>候选选择概率，非存活率</span><span id="input-tokens">— tokens</span></div></section>
        <section class="panel"><div class="panel-header"><h2>这一步如何决定</h2><span class="badge">透明运行</span></div><div class="evidence-list"><div class="evidence-item"><span>控制方式</span><b>模型直接选择</b></div><div class="evidence-item"><span>模型输入</span><b>文字环境特征</b></div><div class="evidence-item"><span>可选动作</span><b>3 个非反向方向</b></div><div class="evidence-item"><span>安全接管</span><b>关闭</b></div><div class="evidence-item"><span>实际执行</span><b id="executed">等待首步</b></div></div><p class="policy-note">引擎提供<strong>碰撞、静态 BFS 路径与空间特征</strong>，Qev 直接选择下一步。没有教师答案注入或动作替换，选错也会结束游戏。<br>游戏能力取决于所加载权重；棋盘仅作可视化，不作为图像输入。</p></section>
      </aside>
    </div>
    <section class="panel log-panel"><div class="panel-header"><div><h2>决策记录</h2><p class="hint">最近 30 步 · 请求、概率与实际执行均来自本局</p></div><button class="btn small ghost" id="export" disabled>↓ 导出本局 JSON</button></div><div id="game-log" class="log-list"><p class="empty-state">还没有决策记录。模型的第一步会出现在这里。</p></div><details id="decision-details"><summary>查看最近一步的原始请求与响应</summary><pre id="decision-json">尚未发送请求。</pre></details></section>`;
  const $ = (id) => container.querySelector(`#${id}`);
  const actionbar = container.querySelector('.arena-controls > .toolbar');
  actionbar.classList.add('arena-actionbar');
  container.querySelector('.arena-top').after(actionbar);
  const canvas = $('board');
  const ctx = canvas.getContext('2d');
  let game = null, running = false, pending = false, disposed = false, timer = null;
  let log = [], seen = new Set(), lastDecision = null;
  const controllers = new Set();

  for (const [direction, [arrow, label]] of Object.entries(DIRECTIONS)) {
    const row = document.createElement('div');
    row.className = 'probability-row';
    row.dataset.direction = direction;
    row.innerHTML = `<span>${arrow} ${label}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="probability-number">—</span>`;
    $('probabilities').append(row);
  }

  function controls() {
    $('play').disabled = !game || game.status !== 'running' || (pending && !running);
    $('play').textContent = running ? 'Ⅱ 暂停运行' : '▶ 开始自主运行';
    $('step').disabled = pending || running || !game || game.status !== 'running';
    $('reset').disabled = pending;
    $('export').disabled = pending || !game;
    if (!game) return;
    let status = running ? (pending ? '模型正在决策…' : '自主运行中') : (pending ? '等待本步结束…' : '已暂停 · 等待下一步');
    if (game.status === 'dead') status = `本局结束 · ${REASONS[game.terminal_reason] || game.terminal_reason}`;
    if (game.status === 'won') status = '已填满棋盘 · 本局完成';
    if (game.status === 'finished') status = `本局结束 · ${REASONS[game.terminal_reason] || game.terminal_reason}`;
    if (game.step === 0 && !pending && !running) status = '准备就绪 · 等待模型接手';
    $('game-status').textContent = status;
  }
  function pause() { running = false; clearTimeout(timer); controls(); }
  function error(message) { $('game-error').textContent = message; $('game-error').hidden = !message; }
  async function request(path, options = {}) {
    const controller = new AbortController();
    controllers.add(controller);
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers }, signal: controller.signal });
      if (response.status === 204) return null;
      const body = await response.json();
      if (!response.ok) {
        const fault = new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body));
        fault.status = response.status;
        throw fault;
      }
      return body;
    } finally { clearTimeout(timeout); controllers.delete(controller); }
  }
  function rectangle(x, y, width, height, radius, color) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(x, y, width, height, radius); ctx.fill();
  }
  function draw() {
    const n = game?.size || Number($('size').value), length = canvas.width, unit = length / n;
    ctx.clearRect(0, 0, length, length);
    ctx.fillStyle = '#111812'; ctx.fillRect(0, 0, length, length);
    ctx.strokeStyle = '#253021'; ctx.lineWidth = 1;
    for (let i = 1; i < n; i++) { ctx.beginPath(); ctx.moveTo(i * unit, 0); ctx.lineTo(i * unit, length); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * unit); ctx.lineTo(length, i * unit); ctx.stroke(); }
    if (!game) return;
    const pad = Math.max(3, unit * .07);
    const body = game.body;
    [...body].reverse().forEach(([x, y], index) => {
      const head = index === body.length - 1;
      rectangle(x * unit + pad, y * unit + pad, unit - pad * 2, unit - pad * 2, unit * .17, head ? '#c9f68d' : `hsl(94 40% ${34 + index / Math.max(1, body.length - 1) * 20}%)`);
    });
    if (body.length) {
      const [x, y] = body[0], cx = (x + .5) * unit, cy = (y + .5) * unit;
      const offsets = { UP: [[-.15, -.17], [.15, -.17]], DOWN: [[-.15, .17], [.15, .17]], LEFT: [[-.17, -.15], [-.17, .15]], RIGHT: [[.17, -.15], [.17, .15]] };
      ctx.fillStyle = '#18311c';
      for (const [dx, dy] of offsets[game.direction] || offsets.RIGHT) { ctx.beginPath(); ctx.arc(cx + dx * unit, cy + dy * unit, unit * .055, 0, Math.PI * 2); ctx.fill(); }
    }
    if (game.food) {
      const [x, y] = game.food, cx = (x + .5) * unit, cy = (y + .53) * unit;
      ctx.fillStyle = '#f5a572'; ctx.beginPath(); ctx.arc(cx, cy, unit * .24, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#bddc8c'; ctx.beginPath(); ctx.ellipse(cx + unit * .09, cy - unit * .29, unit * .13, unit * .065, -.6, 0, Math.PI * 2); ctx.fill();
    }
    if (game.status !== 'running') { ctx.fillStyle = '#0a120939'; ctx.fillRect(0, 0, length, length); }
    canvas.setAttribute('aria-label', `${n} 行 ${n} 列贪吃蛇棋盘。得分 ${game.score}，已走 ${game.step} 步，长度 ${game.length}。蛇头坐标 ${body[0]?.join(', ')}，食物坐标 ${game.food?.join(', ') || '无'}。`);
  }
  function paintDecision() {
    const decision = lastDecision;
    const proposed = decision?.proposed?.toUpperCase();
    const direction = DIRECTIONS[proposed];
    $('direction-icon').textContent = direction?.[0] || '·';
    $('direction-label').textContent = direction?.[1] || (decision?.error ? '决策失败' : '尚未推理');
    $('direction-step').textContent = decision ? `第 ${decision.step_after ?? game.step} 步的模型输出` : '点击开始，让 Qev 接手';
    $('decision-badge').textContent = decision ? '原始候选分布' : '等待首步';
    $('executed').textContent = DIRECTIONS[decision?.executed?.toUpperCase()]?.join(' ') || '未执行';
    $('latency').textContent = Number.isFinite(decision?.inference_ms) ? Math.round(decision.inference_ms) : '—';
    const tokens = decision?.response?.usage?.input_tokens;
    $('input-tokens').textContent = Number.isFinite(tokens) ? `${tokens} input tokens` : '— tokens';
    for (const row of $('probabilities').children) {
      const probabilities = decision?.probabilities || {};
      const value = probabilities[row.dataset.direction] ?? probabilities[row.dataset.direction.toLowerCase()];
      row.classList.toggle('selected', row.dataset.direction === proposed);
      row.classList.toggle('disabled', decision != null && value == null);
      row.querySelector('.bar-fill').style.width = `${Number.isFinite(value) ? Math.max(0, Math.min(100, value * 100)) : 0}%`;
      row.querySelector('.probability-number').textContent = Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : (decision ? '不可选' : '—');
    }
    $('decision-json').textContent = decision ? JSON.stringify({ request: decision.request, response: decision.response, proposed: decision.proposed, executed: decision.executed, error: decision.error }, null, 2) : '尚未发送请求。';
  }
  function renderLog() {
    $('game-log').replaceChildren();
    if (!log.length) { const empty = document.createElement('p'); empty.className = 'empty-state'; empty.textContent = '还没有决策记录。模型的第一步会出现在这里。'; $('game-log').append(empty); return; }
    for (const entry of [...log].reverse()) {
      const row = document.createElement('div'); row.className = 'log-row';
      const move = DIRECTIONS[entry.proposed?.toUpperCase()]?.join(' ') || '未执行';
      const values = [`#${String(entry.step).padStart(3, '0')}`, move + (entry.eaten ? ' · 吃到食物' : ''), `${Math.round(entry.inference_ms || 0)} ms`, entry.status];
      for (const value of values) { const span = document.createElement('span'); span.textContent = value; row.append(span); }
      if (entry.eaten) row.classList.add('eaten');
      $('game-log').append(row);
    }
  }
  function update(next, before = null) {
    game = next; lastDecision = game.last_decision;
    $('score').textContent = game.score; $('steps').textContent = game.step; $('length').textContent = game.length;
    $('game-size').textContent = `${game.size} × ${game.size}`; $('game-seed').textContent = `SEED ${game.seed}`;
    $('game-limit').textContent = `每局最多 ${game.max_steps} 步`;
    if (lastDecision) {
      const key = `${game.id}:${lastDecision.step_before ?? game.step}`;
      if (!seen.has(key)) {
        seen.add(key); log.push({ step: lastDecision.step_after ?? game.step, proposed: lastDecision.proposed, inference_ms: lastDecision.inference_ms, eaten: before != null && game.score > before.score, status: game.status === 'running' ? '已执行' : (REASONS[game.terminal_reason] || '完成') });
        log = log.slice(-30);
      }
      if (lastDecision.error) error(`模型未执行动作：${lastDecision.error.message || JSON.stringify(lastDecision.error)}`);
    }
    if (game.status !== 'running') running = false;
    draw(); paintDecision(); renderLog(); controls();
  }
  async function create() {
    if (pending) return;
    const seed = Number($('seed').value);
    const size = Number($('size').value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) { error('随机种子须为 0–2147483647 的整数。'); return; }
    if (![8, 12, 16].includes(size)) { error('棋盘尺寸须为 8、12 或 16。'); return; }
    pause(); pending = true; error(''); controls();
    try {
      if (game) { await request(`/api/snake/games/${game.id}`, { method: 'DELETE' }).catch((fault) => { if (fault.status !== 404) throw fault; }); game = null; }
      const next = await request('/api/snake/games', { method: 'POST', body: JSON.stringify({ seed, size, max_steps: 500 }) });
      if (disposed) { fetch(`/api/snake/games/${next.id}`, { method: 'DELETE', keepalive: true }); return; }
      log = []; seen = new Set(); update(next);
    } catch (fault) { if (!disposed) error(`无法创建游戏：${fault.message}`); }
    finally { pending = false; if (!disposed) controls(); }
  }
  async function step() {
    if (pending || !game || game.status !== 'running' || disposed) return;
    pending = true; error(''); controls(); const started = performance.now(), before = game;
    try {
      const next = await request(`/api/snake/games/${game.id}/step`, { method: 'POST', body: JSON.stringify({ expected_step: game.step }) });
      if (!disposed) update(next, before);
    } catch (fault) {
      pause();
      if (!disposed) {
        error(fault.name === 'AbortError' ? '等待超时，正在核对服务器上的本局状态；没有自动重试动作。' : `请求失败：${fault.message}`);
        try { update(await request(`/api/snake/games/${before.id}`), before); } catch { error('无法读取本局状态，请检查服务后重开。'); }
      }
    } finally {
      pending = false;
      if (!disposed) { controls(); if (running && game?.status === 'running') timer = setTimeout(step, Math.max(0, Number($('speed').value) - (performance.now() - started))); }
    }
  }
  function toggle() { if (running) pause(); else if (!pending && game?.status === 'running') { running = true; step(); } }
  $('play').addEventListener('click', toggle);
  $('step').addEventListener('click', step);
  $('reset').addEventListener('click', create);
  $('export').addEventListener('click', async () => {
    if (!game) return;
    pause(); $('export').disabled = true;
    try {
      const data = await request(`/api/snake/games/${game.id}?history=true`);
      const url = URL.createObjectURL(new Blob([JSON.stringify({ format: 'qev-snake-v1', exported_at: new Date().toISOString(), game: data }, null, 2)], { type: 'application/json' }));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `qev-snake-seed-${data.seed}.json`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      notify('已导出本局真实决策记录。');
    } catch (fault) { notify(`导出失败：${fault.message}`); }
    finally { if (!disposed) controls(); }
  });
  const keydown = (event) => {
    if (container.hidden || /INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.code === 'Space') { event.preventDefault(); toggle(); }
    if (event.code === 'ArrowRight' && !running) { event.preventDefault(); step(); }
  };
  window.addEventListener('keydown', keydown);
  draw(); create();
  return { pause, destroy() { disposed = true; running = false; clearTimeout(timer); window.removeEventListener('keydown', keydown); for (const controller of controllers) controller.abort(); if (game) fetch(`/api/snake/games/${game.id}`, { method: 'DELETE', keepalive: true }).catch(() => {}); } };
}
