// Barah Goti web client. The server is stateless: we keep the action list and send it with every request.
const SVG = "http://www.w3.org/2000/svg";
const STEP = 100;
const board = document.getElementById("board");
const statusEl = document.getElementById("status");
const infoEl = document.getElementById("info");
const historyEl = document.getElementById("history");
const evalFill = document.getElementById("eval-fill");
const COLOURS = ["carnelian", "turquoise"];

let actions = [], state = null, selected = null, busy = false, recorded = false, lastMove = null;

const rc = sq => [Math.floor(sq / 5), sq % 5];
const xy = sq => { const [r, c] = rc(sq); return [c * STEP, r * STEP]; };
const strong = sq => { const [r, c] = rc(sq); return (r + c) % 2 === 0; };

async function api(path, body) {
  const res = await fetch(path, { method: body ? "POST" : "GET", headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

function el(name, attrs, parent = board) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent.appendChild(e);
  return e;
}

function drawLines() {
  for (let sq = 0; sq < 25; sq++) {
    const [r, c] = rc(sq);
    const nbrs = [[0, 1], [1, 0]];
    if (strong(sq)) nbrs.push([1, 1], [1, -1]);
    for (const [dr, dc] of nbrs) {
      const r2 = r + dr, c2 = c + dc;
      if (r2 > 4 || c2 < 0 || c2 > 4) continue;
      const [x1, y1] = xy(sq), [x2, y2] = xy(r2 * 5 + c2);
      el("line", { x1, y1, x2, y2, class: "line" });
    }
  }
  for (let c = 0; c < 5; c++) el("text", { x: c * STEP, y: 450, class: "label" }).textContent = "abcde"[c];
  for (let r = 0; r < 5; r++) el("text", { x: -38, y: r * STEP + 6, class: "label" }).textContent = 5 - r;
}

function render() {
  board.innerHTML = "";
  drawLines();
  for (let sq = 0; sq < 25; sq++) { const [x, y] = xy(sq); el("circle", { cx: x, cy: y, r: 6, class: "point" }); }
  if (lastMove) {
    const [x1, y1] = xy(lastMove.src), [x2, y2] = xy(lastMove.dst);
    el("line", { x1, y1, x2, y2, class: "last" });
  }
  const human = COLOURS.indexOf(document.getElementById("colour").value);
  const myTurn = state && !state.result && COLOURS.indexOf(state.turn) === human && !busy;
  const movable = new Set(myTurn ? state.legal.map(m => m.src) : []);
  state.board.forEach((p, sq) => {
    if (p < 0) return;
    const [x, y] = xy(sq);
    const b = el("circle", { cx: x, cy: y, r: 30, class: `bead c${p}` + (sq === selected ? " selected" : "") });
    if (movable.has(sq)) b.addEventListener("click", () => { selected = sq; render(); });
  });
  if (myTurn && selected !== null) {
    for (const m of state.legal.filter(m => m.src === selected)) {
      const [x, y] = xy(m.dst);
      const t = el("circle", { cx: x, cy: y, r: 20, class: "target" + (m.captured !== null ? " capture" : "") });
      t.addEventListener("click", () => play(m));
    }
  }
  historyEl.innerHTML = "";
  for (let i = 0; i < state.history.length; i++) {
    const li = document.createElement("li");
    li.textContent = `${state.history[i].player}: ${state.history[i].notation}`;
    historyEl.appendChild(li);
  }
  historyEl.scrollTop = historyEl.scrollHeight;
}

async function refresh() {
  state = await api("/api/state", { actions });
  if (state.chain < 0) selected = null; else selected = state.chain;
  const analysis = await api("/api/analyse", { actions });
  evalFill.style.height = `${50 + 50 * analysis.carnelian_value}%`;
  const human = COLOURS.indexOf(document.getElementById("colour").value);
  if (state.result) {
    const w = state.result.winner;
    statusEl.textContent = w === null ? `Draw (${state.result.reason}).` :
      (COLOURS.indexOf(w) === human ? `You win (${state.result.reason})!` : `The model wins (${state.result.reason}).`);
    if (!recorded) {
      recorded = true;
      api("/api/games", { actions, human_colour: COLOURS[human], mode: document.getElementById("mode").value }).catch(() => {});
    }
  } else if (COLOURS.indexOf(state.turn) === human) {
    statusEl.textContent = state.chain >= 0 ? "Keep jumping with the same bead." :
      (state.legal.some(m => m.captured !== null) ? "Your move. A capture is available, so it is mandatory." : "Your move.");
  } else {
    statusEl.textContent = "The model is thinking…";
  }
  render();
  if (!state.result && COLOURS.indexOf(state.turn) !== human) await modelMove();
}

async function play(m) {
  lastMove = m;
  actions.push(m.action);
  selected = null;
  await refresh();
}

async function modelMove() {
  busy = true;
  try {
    const r = await api("/api/move", { actions, mode: document.getElementById("mode").value, simulations: 150 });
    const legal = state.legal.find(m => m.action === r.action);
    lastMove = legal;
    actions.push(r.action);
    let info = `Model v${r.model_version} · ${r.mode} · ${r.latency_ms} ms`;
    if (r.shadow) info += ` · shadow v${r.shadow.version} ${r.shadow.agrees ? "agrees" : "disagrees"}`;
    infoEl.textContent = info;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    busy = false;
    return;
  }
  busy = false;
  await refresh();
}

async function newGame() {
  actions = []; selected = null; recorded = false; lastMove = null;
  try {
    const m = await api("/api/model");
    infoEl.textContent = `Champion v${m.champion.version}` + (m.shadow ? ` · v${m.shadow.version} running in shadow mode` : "");
  } catch (e) { infoEl.textContent = ""; }
  await refresh();
}

document.getElementById("new").addEventListener("click", newGame);
newGame().catch(e => { statusEl.textContent = `Error: ${e.message}`; });
