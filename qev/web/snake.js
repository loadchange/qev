import { m, onLocaleChange, setText, setAttr, translateTree } from "./i18n.js";

const DIRECTIONS = { UP: ['↑', m("snake.up")], DOWN: ['↓', m("snake.down")], LEFT: ['←', m("snake.left")], RIGHT: ['→', m("snake.right")] };
const REASONS = { wall: m("snake.wall"), body: m("snake.body"), starvation: m("snake.starvation"), max_steps: m("snake.maxSteps"), model_error: m("snake.modelError") };

export function mountSnake(container, { notify }) {
  container.innerHTML = `
    <div class="page-heading"><div><div class="eyebrow">01 / AUTONOMOUS PLAY</div><h1 data-i18n-text="snake.heading">Let the model make the next move.</h1><p data-i18n-text="snake.subtitle">See the model's decision behind every change of direction.</p></div><span class="badge green" data-i18n-text="snake.live">CHOICE · REAL MODEL INFERENCE</span></div>
    <div class="snake-layout">
      <section class="panel arena-panel" aria-label="Game board and controls" data-i18n-aria-label="snake.arena">
        <div class="arena-top"><h2><span data-i18n-text="snake.name">Snake</span><span class="badge">LIVE</span></h2><div class="arena-meta"><span id="game-size">12 × 12</span><span> / </span><span id="game-seed">SEED 7</span></div></div>
        <div class="arena-wrap"><canvas id="board" width="720" height="720" role="img" aria-label="Snake board, waiting for a game" data-i18n-aria-label="snake.boardWaiting"></canvas><div class="arena-status"><span class="status-label" id="game-status" role="status" data-i18n-text="snake.preparing">Preparing the board…</span><span id="game-limit" data-i18n-text="snake.defaultLimit">Up to 500 steps per episode</span></div></div>
        <div class="stats-grid"><div class="metric"><div class="metric-label" data-i18n-text="snake.food">Food eaten</div><div class="metric-value" id="score">0</div></div><div class="metric"><div class="metric-label" data-i18n-text="snake.steps">Steps taken</div><div class="metric-value" id="steps">0</div></div><div class="metric"><div class="metric-label" data-i18n-text="snake.length">Snake length</div><div class="metric-value" id="length">—</div></div><div class="metric"><div class="metric-label" data-i18n-text="snake.latency">Model latency</div><div class="metric-value"><span id="latency">—</span><small>ms</small></div></div></div>
        <div class="arena-controls"><div class="toolbar"><button class="btn primary" id="play" disabled data-i18n-text="snake.play">▶ Start autonomous play</button><button class="btn" id="step" disabled data-i18n-text="snake.step">→ Single step</button><button class="btn ghost" id="reset" disabled data-i18n-text="snake.reset">↻ New episode</button></div><div class="arena-settings"><label class="field"><span data-i18n-text="snake.seed">Random seed</span><input id="seed" type="number" min="0" max="2147483647" step="1" value="7"></label><label class="field"><span data-i18n-text="snake.size">Board size</span><select id="size"><option value="8">8 × 8</option><option value="12" selected>12 × 12</option><option value="16">16 × 16</option></select></label><label class="field"><span data-i18n-text="snake.speed">Play speed</span><select id="speed"><option value="1000" data-i18n-text="snake.speed1">Up to 1 step/sec</option><option value="250" selected data-i18n-text="snake.speed4">Up to 4 steps/sec</option><option value="125" data-i18n-text="snake.speed8">Up to 8 steps/sec</option><option value="0" data-i18n-text="snake.speedMax">As fast as inference</option></select></label></div><p class="hint" style="margin-top:12px" data-i18n-text="snake.settingsHint">Seed and size apply to the next episode. Each step waits for the model; switching views or hiding the tab pauses play.</p><div id="game-error" class="error" role="alert" hidden></div></div>
      </section>
      <aside class="snake-sidebar">
        <section class="panel"><div class="panel-header"><h2 data-i18n-text="snake.choice">The model's choice</h2><span class="badge" id="decision-badge" data-i18n-text="snake.firstStep">Waiting for the first step</span></div><div class="decision-label" data-i18n-text="snake.lastDecision">LAST DECISION</div><div class="decision-value"><span class="direction-icon" id="direction-icon">·</span><div><strong id="direction-label" data-i18n-text="snake.notRun">No inference yet</strong><br><small id="direction-step" data-i18n-text="snake.startHint">Press Start to let Qev take over</small></div></div><div id="probabilities" role="group" aria-label="Candidate choice probabilities" data-i18n-aria-label="snake.probabilities"></div><div class="decision-foot"><span data-i18n-text="snake.probabilityHint">Choice probabilities, not survival odds</span><span id="input-tokens">— tokens</span></div></section>
        <section class="panel"><div class="panel-header"><h2 data-i18n-text="snake.how">How this move was chosen</h2><span class="badge" data-i18n-text="snake.transparent">Transparent execution</span></div><div class="evidence-list"><div class="evidence-item"><span data-i18n-text="snake.control">Control</span><b data-i18n-text="snake.directChoice">Direct model choice</b></div><div class="evidence-item"><span data-i18n-text="snake.input">Model input</span><b data-i18n-text="snake.textFeatures">Text environment features</b></div><div class="evidence-item"><span data-i18n-text="snake.actions">Available moves</span><b data-i18n-text="snake.threeDirections">3 non-reversing directions</b></div><div class="evidence-item"><span data-i18n-text="snake.safety">Safety override</span><b data-i18n-text="snake.off">Off</b></div><div class="evidence-item"><span data-i18n-text="snake.executed">Executed move</span><b id="executed" data-i18n-text="snake.firstStep">Waiting for the first step</b></div></div><p class="policy-note"><span data-i18n-text="snake.policyPrefix">The engine provides </span><strong data-i18n-text="snake.policyFeatures">collision, static BFS path and space features</strong><span data-i18n-text="snake.policySuffix">. Qev chooses the next move directly. No teacher answers are injected and no moves are replaced; a bad choice can end the game.</span><br><span data-i18n-text="snake.policyEnd">Game performance depends on the loaded weights. The board is a visualization, not an image input.</span></p></section>
      </aside>
    </div>
    <section class="panel log-panel"><div class="panel-header"><div><h2 data-i18n-text="snake.log">Decision history</h2><p class="hint" data-i18n-text="snake.logHint">Last 30 steps · Requests, probabilities and executed moves from this episode</p></div><button class="btn small ghost" id="export" disabled data-i18n-text="snake.export">↓ Export episode JSON</button></div><div id="game-log" class="log-list"><p class="empty-state" data-i18n-text="snake.emptyLog">No decisions yet. The model's first move will appear here.</p></div><details id="decision-details"><summary data-i18n-text="snake.raw">Show the latest raw request and response</summary><pre id="decision-json" data-i18n-text="snake.notSent">No request sent yet.</pre></details></section>`;
  translateTree(container);
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
    row.innerHTML = `<span></span><div class="bar-track"><div class="bar-fill"></div></div><span class="probability-number">—</span>`;
    setText(row.firstElementChild, m('snake.directionLabel', { arrow, label }));
    $('probabilities').append(row);
  }

  function controls() {
    $('play').disabled = !game || game.status !== 'running' || (pending && !running);
    setText($('play'), running ? m("snake.pause") : m("snake.play"));
    $('step').disabled = pending || running || !game || game.status !== 'running';
    $('reset').disabled = pending;
    $('export').disabled = pending || !game;
    if (!game) return;
    let status = running ? (pending ? m("snake.thinking") : m("snake.running")) : (pending ? m("snake.waitStep") : m("snake.paused"));
    if (game.status === 'dead') status = m("snake.ended", { reason: REASONS[game.terminal_reason] || game.terminal_reason });
    if (game.status === 'won') status = m("snake.won");
    if (game.status === 'finished') status = m("snake.ended", { reason: REASONS[game.terminal_reason] || game.terminal_reason });
    if (game.step === 0 && !pending && !running) status = m("snake.ready");
    setText($('game-status'), status);
  }
  function pause() { running = false; clearTimeout(timer); controls(); }
  function error(message) { setText($('game-error'), message); $('game-error').hidden = !message; }
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
    setAttr(canvas, 'aria-label', m("snake.boardDescription", { size: n, score: game.score, step: game.step, length: game.length, head: body[0]?.join(', '), food: game.food?.join(', ') || m('snake.none') }));
  }
  function paintDecision() {
    const decision = lastDecision;
    const proposed = decision?.proposed?.toUpperCase();
    const direction = DIRECTIONS[proposed];
    setText($('direction-icon'), direction?.[0] || '·');
    setText($('direction-label'), direction?.[1] || (decision?.error ? m("snake.failed") : m("snake.notRun")));
    setText($('direction-step'), decision ? m("snake.stepOutput", { step: decision.step_after ?? game.step }) : m("snake.startHint"));
    setText($('decision-badge'), decision ? m("snake.rawDistribution") : m("snake.firstStep"));
    setText($('executed'), DIRECTIONS[decision?.executed?.toUpperCase()]?.join(' ') || m("snake.notExecuted"));
    setText($('latency'), Number.isFinite(decision?.inference_ms) ? Math.round(decision.inference_ms) : '—');
    const tokens = decision?.response?.usage?.input_tokens;
    setText($('input-tokens'), Number.isFinite(tokens) ? `${tokens} input tokens` : '— tokens');
    for (const row of $('probabilities').children) {
      const probabilities = decision?.probabilities || {};
      const value = probabilities[row.dataset.direction] ?? probabilities[row.dataset.direction.toLowerCase()];
      row.classList.toggle('selected', row.dataset.direction === proposed);
      row.classList.toggle('disabled', decision != null && value == null);
      row.querySelector('.bar-fill').style.width = `${Number.isFinite(value) ? Math.max(0, Math.min(100, value * 100)) : 0}%`;
      setText(row.querySelector('.probability-number'), Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : (decision ? m("snake.unavailable") : '—'));
    }
    setText($('decision-json'), decision ? JSON.stringify({ request: decision.request, response: decision.response, proposed: decision.proposed, executed: decision.executed, error: decision.error }, null, 2) : m("snake.notSent"));
  }
  function renderLog() {
    $('game-log').replaceChildren();
    if (!log.length) { const empty = document.createElement('p'); empty.className = 'empty-state'; setText(empty, m("snake.emptyLog")); $('game-log').append(empty); return; }
    for (const entry of [...log].reverse()) {
      const row = document.createElement('div'); row.className = 'log-row';
      const move = DIRECTIONS[entry.proposed?.toUpperCase()]?.join(' ') || m("snake.notExecuted");
      const values = [`#${String(entry.step).padStart(3, '0')}`, move + (entry.eaten ? m("snake.ate") : ''), `${Math.round(entry.inference_ms || 0)} ms`, entry.status];
      for (const value of values) { const span = document.createElement('span'); setText(span, value); row.append(span); }
      if (entry.eaten) row.classList.add('eaten');
      $('game-log').append(row);
    }
  }
  function update(next, before = null) {
    game = next; lastDecision = game.last_decision;
    setText($('score'), game.score); setText($('steps'), game.step); setText($('length'), game.length);
    setText($('game-size'), `${game.size} × ${game.size}`); setText($('game-seed'), `SEED ${game.seed}`);
    setText($('game-limit'), m("snake.stepLimit", { count: game.max_steps }));
    if (lastDecision) {
      const key = `${game.id}:${lastDecision.step_before ?? game.step}`;
      if (!seen.has(key)) {
        seen.add(key); log.push({ step: lastDecision.step_after ?? game.step, proposed: lastDecision.proposed, inference_ms: lastDecision.inference_ms, eaten: before != null && game.score > before.score, status: game.status === 'running' ? m("snake.executedStatus") : (REASONS[game.terminal_reason] || m("snake.done")) });
        log = log.slice(-30);
      }
      if (lastDecision.error) error(m("snake.noAction", { error: lastDecision.error.message || JSON.stringify(lastDecision.error) }));
    }
    if (game.status !== 'running') running = false;
    draw(); paintDecision(); renderLog(); controls();
  }
  async function create() {
    if (pending) return;
    const seed = Number($('seed').value);
    const size = Number($('size').value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) { error(m("snake.seedInvalid")); return; }
    if (![8, 12, 16].includes(size)) { error(m("snake.sizeInvalid")); return; }
    pause(); pending = true; error(''); controls();
    try {
      if (game) { await request(`/api/snake/games/${game.id}`, { method: 'DELETE' }).catch((fault) => { if (fault.status !== 404) throw fault; }); game = null; }
      const next = await request('/api/snake/games', { method: 'POST', body: JSON.stringify({ seed, size, max_steps: 500 }) });
      if (disposed) { fetch(`/api/snake/games/${next.id}`, { method: 'DELETE', keepalive: true }); return; }
      log = []; seen = new Set(); update(next);
    } catch (fault) { if (!disposed) error(m("snake.createFailed", { error: fault.message })); }
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
        error(fault.name === 'AbortError' ? m("snake.timeout") : m("snake.requestFailed", { error: fault.message }));
        try { update(await request(`/api/snake/games/${before.id}`), before); } catch { error(m("snake.unreadable")); }
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
      notify(m("snake.exported"));
    } catch (fault) { notify(m("snake.exportFailed", { error: fault.message })); }
    finally { if (!disposed) controls(); }
  });
  const keydown = (event) => {
    if (container.hidden || /INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.code === 'Space') { event.preventDefault(); toggle(); }
    if (event.code === 'ArrowRight' && !running) { event.preventDefault(); step(); }
  };
  window.addEventListener('keydown', keydown);
  const stopLocale = onLocaleChange(() => {
    if (disposed) return;
    controls(); paintDecision(); renderLog(); draw();
  });
  draw(); create();
  return { pause, destroy() { disposed = true; stopLocale(); running = false; clearTimeout(timer); window.removeEventListener('keydown', keydown); for (const controller of controllers) controller.abort(); if (game) fetch(`/api/snake/games/${game.id}`, { method: 'DELETE', keepalive: true }).catch(() => {}); } };
}
