// Widget: optimizer geometry.  Five optimizers train logistic regression (no bias) on the same separable 2D data,
// live, in float64.  Each one's direction drifts toward the max-margin solution of *its own* norm.
(function () {
  const { C, canvas, scale, line, dot, text } = window.IB;
  const root = document.getElementById("geometry-widget");
  if (!root || !window.GEOM_DATA) return;
  const D = window.GEOM_DATA;
  const Z = D.X.map((x, i) => [x[0] * D.y[i], x[1] * D.y[i]]);
  const N = Z.length;
  const YEL = "#c98500";
  const OPTS = [
    { key: "gd", name: "gradient descent", short: "GD", color: C.blue },
    { key: "ngd", name: "normalized GD", short: "NGD", color: C.violet },
    { key: "sign", name: "sign GD", short: "sign", color: C.orange },
    { key: "adam", name: "Adam", short: "Adam", color: YEL },
    { key: "cd", name: "coordinate descent", short: "CD", color: C.aqua },
  ];
  const deg = r => r * 180 / Math.PI;

  // ------------------------------------------------------------------ reference max-margin directions (theta scan in JS)
  function normP(u, p) { return p === "l2" ? Math.hypot(u[0], u[1]) : p === "linf" ? Math.max(Math.abs(u[0]), Math.abs(u[1])) : Math.abs(u[0]) + Math.abs(u[1]); }
  function nmargin(u, p) { let m = Infinity; for (const z of Z) m = Math.min(m, z[0] * u[0] + z[1] * u[1]); return m / normP(u, p); }
  function bestDir(p) {
    let best = -Infinity, bt = 0;
    for (let i = 0; i < 36000; i++) { const t = 2 * Math.PI * i / 36000, v = nmargin([Math.cos(t), Math.sin(t)], p); if (v > best) { best = v; bt = t; } }
    return { theta: bt, gamma: best };
  }
  const REF = { l2: bestDir("l2"), linf: bestDir("linf"), l1: bestDir("l1") };

  // ------------------------------------------------------------------ optimizers
  let sigmaMax2 = (() => { // largest eigenvalue of Z^T Z
    let a = 0, b = 0, c = 0; for (const z of Z) { a += z[0] * z[0]; b += z[0] * z[1]; c += z[1] * z[1]; }
    return 0.5 * (a + c) + Math.sqrt(0.25 * (a - c) * (a - c) + b * b);
  })();
  function logaddexp0(m) { return m > 0 ? m + Math.log1p(Math.exp(-m)) : Math.log1p(Math.exp(m)); }
  // direction-only gradient: g = e^{Cn} * gh, with gh computed from log-domain weights (never underflows)
  function gradLog(w) {
    const lw = new Float64Array(N); let Cn = -Infinity;
    for (let n = 0; n < N; n++) { lw[n] = -logaddexp0(Z[n][0] * w[0] + Z[n][1] * w[1]); if (lw[n] > Cn) Cn = lw[n]; }
    let g0 = 0, g1 = 0;
    for (let n = 0; n < N; n++) { const p = Math.exp(lw[n] - Cn); g0 -= p * Z[n][0]; g1 -= p * Z[n][1]; }
    return { gh: [g0, g1], Cn };
  }
  function makeState(cfg) {
    return { w: [0, 0], M: [0, 0], V: [0, 0], C: 0, t: 0, cfg };
  }
  function step(key, s) {
    s.t++;
    const { gh, Cn } = gradLog(s.w);
    if (key === "gd") {
      const lr = 1 / (0.25 * sigmaMax2), e = Math.exp(Cn);
      s.w[0] -= lr * e * gh[0]; s.w[1] -= lr * e * gh[1];
    } else if (key === "ngd") {
      const nrm = Math.hypot(gh[0], gh[1]); s.w[0] -= 0.05 * gh[0] / nrm; s.w[1] -= 0.05 * gh[1] / nrm;
    } else if (key === "sign") {
      s.w[0] -= 0.02 * Math.sign(gh[0]); s.w[1] -= 0.02 * Math.sign(gh[1]);
    } else if (key === "cd") {
      const i = Math.abs(gh[0]) >= Math.abs(gh[1]) ? 0 : 1; s.w[i] -= 0.02 * Math.sign(gh[i]);
    } else if (key === "adam") {
      const { lr, b1, b2, eps } = s.cfg, fac = Math.exp(s.C - Cn);
      let epsH = 0;
      if (eps > 0) epsH = Math.exp(Math.min(Math.log(eps) - Cn, 700));
      for (let i = 0; i < 2; i++) {
        s.M[i] = b1 * s.M[i] * fac + (1 - b1) * gh[i];
        s.V[i] = b2 * s.V[i] * fac * fac + (1 - b2) * gh[i] * gh[i];
        const mh = s.M[i] / (1 - Math.pow(b1, s.t)), vh = s.V[i] / (1 - Math.pow(b2, s.t));
        s.w[i] -= lr * mh / (Math.sqrt(vh) + epsH);
      }
      s.C = Cn;
    }
  }

  // ------------------------------------------------------------------ DOM
  root.innerHTML = `
    <div class="ib-controls">
      <button id="ge-run">▸ run</button><button id="ge-reset">reset</button>
      <label>show boundary of <select id="ge-show">${OPTS.map((o, i) => `<option value="${o.key}" ${i === 2 ? "selected" : ""}>${o.name}</option>`).join("")}</select></label>
      <span id="ge-t" class="ib-read"></span>
    </div>
    <div class="ib-controls">
      <span class="ib-sub">Adam:</span>
      <label>ε <select id="ge-eps"><option value="0">0</option><option value="1e-16">1e-16</option><option value="1e-12">1e-12</option><option value="1e-8" selected>1e-8</option><option value="1e-4">1e-4</option></select></label>
      <label>learning rate <select id="ge-lr"><option>0.001</option><option selected>0.01</option><option>0.1</option></select></label>
      <label>β₂ <select id="ge-b2"><option>0.9</option><option>0.99</option><option selected>0.999</option></select></label>
    </div>
    <div class="ib-row"><div id="ge-data"></div><div id="ge-w"></div></div>
    <div id="ge-angle"></div>
    <pre id="ge-test" class="ib-test"></pre>`;
  const el = id => root.querySelector(id);
  const cData = canvas(el("#ge-data"), 440, 420), cW = canvas(el("#ge-w"), 440, 420), cA = canvas(el("#ge-angle"), 960, 270);

  let states, traces, running = false, raf = null;
  function adamCfg() { return { lr: +el("#ge-lr").value, b1: 0.9, b2: +el("#ge-b2").value, eps: +el("#ge-eps").value }; }
  function reset() {
    states = {}; traces = {};
    for (const o of OPTS) { states[o.key] = makeState(o.key === "adam" ? adamCfg() : null); traces[o.key] = []; }
    drawAll();
  }
  let T = 0;
  function advance(budgetMs) {
    const t0 = performance.now();
    const target = Math.min(1e7, Math.max(T + 1, Math.floor(T * Math.pow(10, 0.012))));
    while (T < target && performance.now() - t0 < budgetMs) {
      for (const o of OPTS) step(o.key, states[o.key]);
      T++;
    }
    for (const o of OPTS) { const w = states[o.key].w; if (w[0] || w[1]) traces[o.key].push([Math.log10(T), deg(Math.atan2(w[1], w[0]))]); }
  }

  // --- panel A: data space
  const AX = scale(-2.3, 2.3, 20, 420), AY = scale(-2.8, 3.0, 400, 20);
  function boundary(ctx, theta, color, lw, dash, X_, Y_) {
    const u = [-Math.sin(theta), Math.cos(theta)], s = 8;
    line(ctx, [[X_(-s * u[0]), Y_(-s * u[1])], [X_(s * u[0]), Y_(s * u[1])]], color, lw, dash);
  }
  function drawData() {
    const ctx = cData.ctx; ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cData.w, cData.h);
    ctx.save(); ctx.beginPath(); ctx.rect(10, 10, 420, 400); ctx.clip();
    const show = el("#ge-show").value, w = states[show].w;
    if (w[0] || w[1]) { // soft class tint of the selected model
      const th = Math.atan2(w[1], w[0]), u = [Math.cos(th), Math.sin(th)], s = 10;
      ctx.fillStyle = "rgba(235,104,52,0.05)"; ctx.beginPath();
      ctx.moveTo(AX(-s * u[1]), AY(s * u[0])); ctx.lineTo(AX(s * u[1]), AY(-s * u[0])); ctx.lineTo(AX(s * u[1] + s * u[0]), AY(-s * u[0] + s * u[1])); ctx.lineTo(AX(-s * u[1] + s * u[0]), AY(s * u[0] + s * u[1])); ctx.fill();
    }
    boundary(ctx, REF.l2.theta, C.blue, 1.4, [6, 4], AX, AY);
    boundary(ctx, REF.linf.theta, C.orange, 1.4, [6, 4], AX, AY);
    boundary(ctx, REF.l1.theta, C.aqua, 1.4, [6, 4], AX, AY);
    if (w[0] || w[1]) boundary(ctx, Math.atan2(w[1], w[0]), C.ink, 2.4, null, AX, AY);
    D.X.forEach((x, i) => {
      if (D.y[i] > 0) dot(ctx, AX(x[0]), AY(x[1]), 4.5, C.ink, C.paper);
      else { ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(AX(x[0]), AY(x[1]), 4, 0, 7); ctx.fillStyle = C.paper; ctx.fill(); ctx.stroke(); ctx.restore(); }
    });
    ctx.restore();
    ctx.fillStyle = "rgba(252,251,248,0.93)"; ctx.fillRect(14, 8, 330, 40);
    text(ctx, "data space", 20, 14, { size: 13, color: C.ink, weight: 500, base: "top" });
    text(ctx, "● / ○ the two classes;  dashed: max-margin boundaries", 20, 32, { size: 11, base: "top" });
    const lg = [["L₂", C.blue], ["L∞", C.orange], ["L₁", C.aqua]];
    lg.forEach(([s, c], i) => { line(ctx, [[300, 392 - 16 * (2 - i)], [322, 392 - 16 * (2 - i)]], c, 1.6, [6, 4]); text(ctx, s + " max-margin", 328, 392 - 16 * (2 - i), { size: 11 }); });
    line(ctx, [[300, 344], [322, 344]], C.ink, 2.4); text(ctx, OPTS.find(o => o.key === show).name, 328, 344, { size: 11 });
  }

  // --- panel B: weight space. Feasible set {w : z_n . w >= 1} and the smallest ball of each norm that touches it
  const BX = scale(-0.2, 1.5, 40, 430), BY = scale(-0.2, 1.5, 410, 20);
  function feasiblePoly() {
    let poly = [[-5, -5], [8, -5], [8, 8], [-5, 8]];
    for (const z of Z) { // clip by z.w >= 1
      const out = [];
      for (let i = 0; i < poly.length; i++) {
        const P = poly[i], Q = poly[(i + 1) % poly.length], fp = z[0] * P[0] + z[1] * P[1] - 1, fq = z[0] * Q[0] + z[1] * Q[1] - 1;
        if (fp >= 0) out.push(P);
        if ((fp >= 0) !== (fq >= 0)) { const t = fp / (fp - fq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
      }
      poly = out;
    }
    return poly;
  }
  const POLY = feasiblePoly();
  function drawW() {
    const ctx = cW.ctx; ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cW.w, cW.h);
    ctx.save(); ctx.beginPath(); ctx.rect(10, 0, 430, 410); ctx.clip();
    ctx.fillStyle = "#efece4"; ctx.beginPath(); POLY.forEach((p, i) => i ? ctx.lineTo(BX(p[0]), BY(p[1])) : ctx.moveTo(BX(p[0]), BY(p[1]))); ctx.fill();
    line(ctx, POLY.concat([POLY[0]]).map(p => [BX(p[0]), BY(p[1])]), C.axis, 1);
    line(ctx, [[BX(-0.2), BY(0)], [BX(1.5), BY(0)]], C.axis, 1); line(ctx, [[BX(0), BY(-0.2)], [BX(0), BY(1.5)]], C.axis, 1);
    const r2 = 1 / REF.l2.gamma, rinf = 1 / REF.linf.gamma, r1 = 1 / REF.l1.gamma;
    const circ = []; for (let i = 0; i <= 160; i++) { const t = 2 * Math.PI * i / 160; circ.push([BX(r2 * Math.cos(t)), BY(r2 * Math.sin(t))]); }
    line(ctx, circ, C.blue, 1.6);
    line(ctx, [[rinf, rinf], [-rinf, rinf], [-rinf, -rinf], [rinf, -rinf], [rinf, rinf]].map(p => [BX(p[0]), BY(p[1])]), C.orange, 1.6);
    line(ctx, [[r1, 0], [0, r1], [-r1, 0], [0, -r1], [r1, 0]].map(p => [BX(p[0]), BY(p[1])]), C.aqua, 1.6);
    // rays of each optimizer's current direction
    for (const o of OPTS) {
      const w = states[o.key].w; if (!(w[0] || w[1])) continue;
      const th = Math.atan2(w[1], w[0]);
      line(ctx, [[BX(0), BY(0)], [BX(2.4 * Math.cos(th)), BY(2.4 * Math.sin(th))]], o.color, 1.8);
    }
    for (const [p, c] of [["l2", C.blue], ["linf", C.orange], ["l1", C.aqua]]) {
      const t = REF[p].theta, r = 1 / REF[p].gamma / normP([Math.cos(t), Math.sin(t)], p);
      dot(ctx, BX(r * Math.cos(t)), BY(r * Math.sin(t)), 5, c, C.paper);
    }
    ctx.restore();
    ctx.fillStyle = "rgba(252,251,248,0.93)"; ctx.fillRect(14, 8, 372, 58);
    text(ctx, "weight space", 20, 14, { size: 13, color: C.ink, weight: 500, base: "top" });
    text(ctx, "shaded: all w with margin ≥ 1.  Each norm's ball grows until it", 20, 32, { size: 11, base: "top" });
    text(ctx, "touches the shaded set; lines: current direction of each optimizer", 20, 47, { size: 11, base: "top" });
  }

  // --- panel C: direction vs log time
  const CX = scale(0, 7, 60, 780), CY = scale(30, 100, 238, 20);
  function drawAngle() {
    const ctx = cA.ctx; ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cA.w, cA.h);
    for (let v = 30; v <= 100; v += 10) { line(ctx, [[CX(0), CY(v)], [CX(7), CY(v)]], C.grid, 1); text(ctx, v + "°", CX(0) - 8, CY(v), { align: "right", size: 10, color: C.muted }); }
    for (let e = 0; e <= 7; e++) text(ctx, e === 0 ? "1" : "10" + "⁰¹²³⁴⁵⁶⁷"[e], CX(e), 254, { align: "center", size: 10.5, color: C.muted });
    text(ctx, "direction of w (angle from the w₁ axis)  vs  training steps t (log scale)", 60, 8, { size: 12.5, color: C.ink, weight: 500, base: "top" });
    [["l2", C.blue, "L₂ max-margin"], ["linf", C.orange, "L∞ max-margin"], ["l1", C.aqua, "L₁ max-margin"]].forEach(([p, c, s]) => {
      const v = deg(REF[p].theta); line(ctx, [[CX(0), CY(v)], [CX(7), CY(v)]], c, 1.2, [5, 4]);
    });
    const lastY = [];
    for (const o of OPTS) {
      const tr = traces[o.key]; if (tr.length < 2) continue;
      line(ctx, tr.map(p => [CX(p[0]), CY(Math.max(30, Math.min(100, p[1])))]), o.color, 2);
      const q = tr[tr.length - 1]; dot(ctx, CX(q[0]), CY(Math.max(30, Math.min(100, q[1]))), 4, o.color, C.paper);
      lastY.push([CY(Math.max(30, Math.min(100, q[1]))), o]);
    }
    // right-edge labels with simple de-collision
    lastY.sort((a, b) => a[0] - b[0]);
    let prev = -1e9;
    for (const [y, o] of lastY) { const yy = Math.max(y, prev + 14); prev = yy; dot(ctx, 800, yy, 3.5, o.color); text(ctx, o.name, 810, yy, { size: 11.5 }); }
    [["l2", "L₂"], ["linf", "L∞"], ["l1", "L₁"]].forEach(([p, s]) => text(ctx, s, CX(0) + 4, CY(deg(REF[p].theta)) - 8, { size: 10.5, color: C.ink2 }));
  }
  function drawAll() {
    drawData(); drawW(); drawAngle();
    el("#ge-t").textContent = `t = ${T.toLocaleString()} steps`;
  }
  function loop() { advance(14); drawAll(); if (running && T < 1e7) raf = requestAnimationFrame(loop); else { running = false; el("#ge-run").textContent = "▸ run"; } }
  el("#ge-run").addEventListener("click", () => { running = !running; el("#ge-run").textContent = running ? "❚❚ pause" : "▸ run"; if (running) loop(); });
  el("#ge-reset").addEventListener("click", () => { T = 0; reset(); });
  ["#ge-eps", "#ge-lr", "#ge-b2"].forEach(id => el(id).addEventListener("change", () => { T = 0; reset(); }));
  el("#ge-show").addEventListener("change", drawAll);
  reset();
  // pre-roll a little so the static page is not empty
  while (T < 2000) advance(1e9);
  drawAll();

  if (location.hash.includes("selftest")) {
    const out = []; let ok = true;
    const chk = (name, cond, msg) => { ok = ok && cond; out.push(`${name}: ${msg} ${cond ? "ok" : "FAIL"}`); };
    for (const [p, v] of [["l2", D.ref.l2], ["linf", D.ref.linf], ["l1", D.ref.l1]]) {
      const d = Math.abs(deg(REF[p].theta) - deg(Math.atan2(v[1], v[0]))); chk(`JS ${p} max-margin direction vs cvxpy`, d < 0.02, `${d.toFixed(4)}°`);
    }
    T = 0; reset();
    while (T < 200000) advance(1e9);
    const ang = k => deg(Math.atan2(states[k].w[1], states[k].w[0]));
    chk("sign GD -> Linf", Math.abs(ang("sign") - deg(REF.linf.theta)) < 0.5, `${ang("sign").toFixed(3)}°`);
    chk("coordinate descent -> L1", Math.abs(ang("cd") - deg(REF.l1.theta)) < 0.5, `${ang("cd").toFixed(3)}°`);
    chk("normalized GD -> L2", Math.abs(ang("ngd") - deg(REF.l2.theta)) < 0.5, `${ang("ngd").toFixed(3)}°`);
    chk("GD closer to L2 than to Linf/L1", Math.abs(ang("gd") - deg(REF.l2.theta)) < 5, `${ang("gd").toFixed(3)}°`);
    chk("Adam finite", isFinite(ang("adam")), `${ang("adam").toFixed(3)}°`);
    el("#ge-test").textContent = (ok ? "SELFTEST PASS\n" : "SELFTEST FAIL\n") + out.join("\n");
    el("#ge-test").style.display = "block";
    console.log("geometry selftest", ok ? "PASS" : "FAIL");
    drawAll();
  }
})();
