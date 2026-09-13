// Widget: click the scalar loss landscape E(a,b) = 1/2 (s - ab)^2 to launch gradient-flow trajectories.
(function () {
  const { C, landColor, setupCanvas, Axes } = window.SaxeCommon;
  const root = document.getElementById("saddle-widget");
  if (!root) return;
  const LIM = 2, SIZE = 430, DT = 0.002, T_MAX = 14, DOT_DT = 0.4, SPEED = 1.6; // time units per second
  const state = { s: 1.0, shrink: 0, trajs: [], raf: null, last: null };

  root.innerHTML = `
    <div class="sw-controls">
      <label>target strength <i>s</i> <input type="range" id="sw-s" min="0.25" max="3" step="0.05" value="1"> <span id="sw-s-v">1.00</span></label>
      <label>shrink clicked init by <input type="range" id="sw-k" min="0" max="4" step="1" value="0"> <span id="sw-k-v">×1</span></label>
      <button id="sw-clear">clear</button>
      <button id="sw-demo">demo: same direction, 1× … 10⁻⁴×</button>
    </div>
    <div class="sw-row">
      <div class="sw-land"><canvas id="sw-canvas"></canvas>
        <div class="sw-hint">click anywhere · the saddle is the circle in the middle</div></div>
      <div class="sw-side"><canvas id="sw-plot"></canvas><div id="sw-table" class="sw-table"></div></div>
    </div>`;

  const cv = root.querySelector("#sw-canvas");
  const ctx = setupCanvas(cv, SIZE, SIZE);
  const pv = root.querySelector("#sw-plot");
  const PW = Math.min(520, Math.max(320, (root.clientWidth || 900) - SIZE - 60));
  const pctx = setupCanvas(pv, PW, 300);
  const table = root.querySelector("#sw-table");

  const toPx = (a, b) => [(a + LIM) / (2 * LIM) * SIZE, (LIM - b) / (2 * LIM) * SIZE];
  const toAB = (x, y) => [x / SIZE * 2 * LIM - LIM, LIM - y / SIZE * 2 * LIM];

  let bg = null;
  function renderBackground() {
    const off = document.createElement("canvas");
    const n = 300; off.width = n; off.height = n;
    const o = off.getContext("2d"), img = o.createImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const a = (i + 0.5) / n * 2 * LIM - LIM, b = LIM - (j + 0.5) / n * 2 * LIM;
      const E = 0.5 * (state.s - a * b) ** 2;
      const x = (Math.log10(E + 1e-3) + 3) / 4.2;
      const c = landColor(x), k = 4 * (j * n + i);
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
    }
    o.putImageData(img, 0, 0);
    bg = off;
  }

  function drawLandscape() {
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bg, 0, 0, SIZE, SIZE);
    // conserved hyperbolas a^2 - b^2 = c
    ctx.strokeStyle = "rgba(40,48,80,0.18)"; ctx.lineWidth = 1;
    for (let c = -3; c <= 3.01; c += 0.5) {
      for (const sa of [1, -1]) {
        ctx.beginPath();
        for (let k = 0; k <= 80; k++) {
          const v = -LIM + 2 * LIM * k / 80;
          let a, b;
          if (c >= 0) { b = v; a = sa * Math.sqrt(c + v * v); } else { a = v; b = sa * Math.sqrt(-c + v * v); }
          const [x, y] = toPx(a, b);
          k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }
    // gradient-flow arrows
    ctx.strokeStyle = "rgba(30,36,60,0.45)"; ctx.lineWidth = 1;
    for (let i = 0; i < 17; i++) for (let j = 0; j < 17; j++) {
      const a = -LIM + (i + 0.5) * 2 * LIM / 17, b = -LIM + (j + 0.5) * 2 * LIM / 17;
      const e = state.s - a * b; let da = b * e, db = a * e;
      const m = Math.hypot(da, db); if (m < 1e-6) continue;
      const L = 7 * Math.min(1, 0.35 + 0.65 * Math.tanh(m));
      da /= m; db /= m;
      const [x, y] = toPx(a, b), x2 = x + da * L, y2 = y - db * L;
      ctx.beginPath(); ctx.moveTo(x - da * L, y + db * L); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - da * 3 + db * 2, y2 + db * 3 + da * 2); ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - da * 3 - db * 2, y2 + db * 3 - da * 2); ctx.stroke();
    }
    // minima
    ctx.strokeStyle = C.accent; ctx.lineWidth = 3; ctx.lineCap = "round";
    for (const sg of [1, -1]) {
      ctx.beginPath();
      for (let k = 0; k <= 200; k++) {
        const a = sg * (state.s / LIM + (LIM - state.s / LIM) * k / 200);
        const [x, y] = toPx(a, state.s / a);
        k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    const [ox, oy] = toPx(0, 0);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.arc(ox, oy, 7, 0, 2 * Math.PI); ctx.stroke();
    // trajectories
    state.trajs.forEach(tr => {
      ctx.strokeStyle = tr.color; ctx.lineWidth = 2.4; ctx.lineJoin = "round";
      ctx.beginPath();
      tr.pts.forEach((p, k) => { const [x, y] = toPx(p[0], p[1]); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      tr.dots.forEach(p => {
        const [x, y] = toPx(p[0], p[1]);
        ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 2 * Math.PI);
        ctx.fillStyle = tr.color; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
      });
      const [hx, hy] = toPx(tr.a, tr.b);
      ctx.beginPath(); ctx.arc(hx, hy, 5.5, 0, 2 * Math.PI); ctx.fillStyle = tr.color; ctx.fill();
      ctx.strokeStyle = C.paper; ctx.lineWidth = 2; ctx.stroke();
    });
    ctx.fillStyle = C.ink; ctx.font = "12px Inter, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("a  (first layer) →", SIZE - 118, SIZE - 8);
    ctx.save(); ctx.translate(12, 118); ctx.rotate(-Math.PI / 2); ctx.fillText("b  (second layer) →", 0, 0); ctx.restore();
  }

  function predicted(tr, t) {
    const u0 = (tr.a0 + tr.b0) ** 2 / 4, s = state.s;
    const e = Math.exp(Math.min(700, 2 * s * t));
    return s * e / (e - 1 + s / u0);
  }

  function drawPlot() {
    pctx.clearRect(0, 0, PW, 300);
    const ax = Axes(pctx, { x: 50, y: 14, w: PW - 64, h: 236 }, [0, T_MAX], [-0.05 * state.s, 1.15 * state.s]);
    ax.draw([0, 2, 4, 6, 8, 10, 12, 14], [0, state.s / 2, state.s], "training time t", "network map  a·b");
    const ts = [];
    for (let t = 0; t <= T_MAX; t += 0.05) ts.push(t);
    state.trajs.forEach(tr => {
      ax.line(ts, ts.map(t => predicted(tr, t)), tr.color, 1.2, 0.8, [4, 4]);
      ax.line(tr.ts, tr.ab, tr.color, 2.4);
    });
    pctx.fillStyle = C.muted; pctx.font = "11px Inter, 'Helvetica Neue', Arial, sans-serif";
    pctx.fillText("solid: gradient flow · dashed: Saxe sigmoid with u₀ = ((a₀+b₀)/2)²", 56, 12);
  }

  function drawTable() {
    if (!state.trajs.length) { table.innerHTML = "<p class='sw-empty'>No trajectories yet. Click the landscape.</p>"; return; }
    const rows = state.trajs.map(tr => {
      const th = tr.tHalf == null ? "…" : tr.tHalf.toFixed(2);
      const u0 = (tr.a0 + tr.b0) ** 2 / 4;
      const pred = u0 < state.s / 2 ? (Math.log(state.s / u0 - 1) / (2 * state.s)).toFixed(2) : "—";
      return `<tr><td><span class="sw-sw" style="background:${tr.color}"></span></td>
        <td>(${tr.a0.toPrecision(2)}, ${tr.b0.toPrecision(2)})</td>
        <td>${(tr.a * tr.a - tr.b * tr.b).toFixed(4)}</td><td>${th}</td><td>${pred}</td></tr>`;
    }).join("");
    table.innerHTML = `<table><thead><tr><th></th><th>start (a₀, b₀)</th><th>a² − b² now</th><th>t½ measured</th><th>t½ predicted</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function f(a, b) { const e = state.s - a * b; return [b * e, a * e]; }
  function advance(tr, dtTotal) {
    let n = Math.ceil(dtTotal / DT);
    while (n-- > 0 && tr.t < T_MAX) {
      const a = tr.a, b = tr.b;
      const k1 = f(a, b), k2 = f(a + DT / 2 * k1[0], b + DT / 2 * k1[1]);
      const k3 = f(a + DT / 2 * k2[0], b + DT / 2 * k2[1]), k4 = f(a + DT * k3[0], b + DT * k3[1]);
      tr.a += DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
      tr.b += DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      const prevAb = a * b;
      tr.t += DT;
      const ab = tr.a * tr.b;
      if (tr.tHalf == null && prevAb < state.s / 2 && ab >= state.s / 2) tr.tHalf = tr.t - DT * (ab - state.s / 2) / (ab - prevAb);
      if (tr.t >= tr.nextRec) { tr.pts.push([tr.a, tr.b]); tr.ts.push(tr.t); tr.ab.push(ab); tr.nextRec += 0.02; }
      if (tr.t >= tr.nextDot) { tr.dots.push([tr.a, tr.b]); tr.nextDot += DOT_DT; }
    }
  }

  function launch(a0, b0) {
    const k = Math.pow(10, -state.shrink);
    a0 *= k; b0 *= k;
    if (state.trajs.length >= 5) state.trajs.shift();
    const tr = { a0, b0, a: a0, b: b0, t: 0, pts: [[a0, b0]], ts: [0], ab: [a0 * b0], dots: [[a0, b0]], nextRec: 0.02,
                 nextDot: DOT_DT, tHalf: a0 * b0 >= state.s / 2 ? 0 : null, color: C.modes[state.trajs.length] };
    state.trajs.push(tr);
    state.trajs.forEach((x, i) => x.color = C.modes[i]);
    start();
    return tr;
  }

  function frame(now) {
    const dtw = state.last == null ? 0.016 : Math.min(0.05, (now - state.last) / 1000);
    state.last = now;
    let running = false;
    state.trajs.forEach(tr => { if (tr.t < T_MAX) { advance(tr, dtw * SPEED); running = true; } });
    drawLandscape(); drawPlot(); drawTable();
    if (running) state.raf = requestAnimationFrame(frame); else { state.raf = null; state.last = null; }
  }
  function start() { if (!state.raf) state.raf = requestAnimationFrame(frame); }

  cv.addEventListener("click", ev => {
    const r = cv.getBoundingClientRect();
    const [a, b] = toAB((ev.clientX - r.left) * SIZE / r.width, (ev.clientY - r.top) * SIZE / r.height);
    launch(a, b);
  });
  root.querySelector("#sw-s").addEventListener("input", ev => {
    state.s = +ev.target.value; root.querySelector("#sw-s-v").textContent = state.s.toFixed(2);
    state.trajs = []; renderBackground(); drawLandscape(); drawPlot(); drawTable();
  });
  root.querySelector("#sw-k").addEventListener("input", ev => {
    state.shrink = +ev.target.value;
    root.querySelector("#sw-k-v").textContent = state.shrink ? `×10⁻${"¹²³⁴"[state.shrink - 1]}` : "×1";
  });
  root.querySelector("#sw-clear").addEventListener("click", () => { state.trajs = []; drawLandscape(); drawPlot(); drawTable(); });
  root.querySelector("#sw-demo").addEventListener("click", () => {
    state.trajs = [];
    const keep = state.shrink;
    [0, 1, 2, 3, 4].forEach(k => { state.shrink = k; launch(0.9, -0.55); });
    state.shrink = keep;
  });

  renderBackground(); drawLandscape(); drawPlot(); drawTable();

  // Preload a demo so the static page (and screenshots) show something meaningful.
  [0, 2, 4].forEach(k => { state.shrink = k; launch(0.9, -0.55); });
  state.shrink = 0;
  state.trajs.forEach(tr => advance(tr, T_MAX));
  drawLandscape(); drawPlot(); drawTable();

  if (location.hash.indexOf("selftest") >= 0) {
    try {
      state.trajs = []; state.s = 1.5; renderBackground();
      const tr = launch(1e-3, 1e-3); advance(tr, T_MAX);
      const pred = Math.log(1.5 / 1e-6 - 1) / 3;
      const err = Math.abs(tr.tHalf - pred) / pred;
      const tr2 = launch(0.7, -0.3); advance(tr2, T_MAX);
      const cons = Math.abs((tr2.a ** 2 - tr2.b ** 2) - (0.49 - 0.09));
      if (err > 0.01 || cons > 1e-6) console.error("saddle selftest FAILED", err, cons);
      else console.log("saddle selftest ok: t_half rel err " + err.toExponential(2) + ", conservation err " + cons.toExponential(2));
    } catch (e) { console.error("saddle selftest exception", e); }
  }
})();
