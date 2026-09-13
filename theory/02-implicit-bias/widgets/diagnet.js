// Widget: the d=2, n=1 diagonal linear network  w = u^D - v^D,  u = v = alpha at init,  one data point x = (1, x2), y = 1.
// Everything is live: the closed-form gradient-flow path (Woodworth et al. 2020), the Q_alpha level set that the
// landing point minimizes, and optionally a real discrete-GD run on (u, v) that you can watch follow the path.
(function () {
  const { C, ramp, canvas, scale, line, dot, text, bisect } = window.IB;
  const root = document.getElementById("diagnet-widget");
  if (!root) return;

  // ------------------------------------------------------------------ math
  // path: w(k) = 2 a^2 sinh(k x) for D = 2;  w(k) = a^D h_D(k x) for D >= 3 with h_D(z) = (1-z)^(-D/(D-2)) - (1+z)^(-D/(D-2))
  function hD(z, D) { const p = -D / (D - 2); return Math.pow(1 - z, p) - Math.pow(1 + z, p); }
  function wOf(k, x, a, D) {
    return D === 2 ? x.map(xi => 2 * a * a * Math.sinh(k * xi)) : x.map(xi => Math.pow(a, D) * hD(k * xi, D));
  }
  function kStar(x, a, D) {
    const f = k => wOf(k, x, a, D).reduce((s, wi, i) => s + wi * x[i], 0) - 1;
    let hi;
    if (D === 2) { hi = 1; while (f(hi) < 0) hi *= 2; }
    else { hi = (1 - 1e-13) / Math.max(...x.map(Math.abs)); }
    return bisect(f, 0, hi, 300);
  }
  function path(x, a, D, n) {
    const ks = kStar(x, a, D), P = [];
    // sample densely in k, then resample evenly by arclength
    const raw = [];
    for (let i = 0; i <= 3000; i++) { const t = i / 3000; raw.push(wOf(ks * (D === 2 ? t : 1 - Math.pow(1 - t, 1)), x, a, D)); }
    raw[raw.length - 1] = wOf(ks, x, a, D);
    const sl = [0];
    for (let i = 1; i < raw.length; i++) sl.push(sl[i - 1] + Math.hypot(raw[i][0] - raw[i - 1][0], raw[i][1] - raw[i - 1][1]));
    let j = 0;
    for (let i = 0; i < n; i++) {
      const s = sl[sl.length - 1] * i / (n - 1);
      while (j < sl.length - 2 && sl[j + 1] < s) j++;
      const f = (s - sl[j]) / Math.max(1e-300, sl[j + 1] - sl[j]);
      P.push([raw[j][0] + f * (raw[j + 1][0] - raw[j][0]), raw[j][1] + f * (raw[j + 1][1] - raw[j][1])]);
    }
    return { P, end: wOf(ks, x, a, D), k: ks };
  }
  // q_D(z) = int_0^|z| h_D^{-1}(t) dt ; D = 2 has the closed form 2 - sqrt(4+z^2) + z asinh(z/2) (with h_2 = 2 sinh)
  const qTables = {};
  function hInv(t, D) { return bisect(z => hD(z, D) - t, 0, 1 - 1e-15, 80); }
  function qTable(D) {
    if (qTables[D]) return qTables[D];
    const L0 = -10, L1 = 14, N = 1500, us = [], qs = [], hs = [];
    let acc = 0, prevU = null, prevF = null;
    // below e^L0 use the small-z expansion h^{-1}(t) ~ t (D-2)/(2D)
    const c = (D - 2) / (2 * D);
    for (let i = 0; i <= N; i++) {
      const u = L0 * Math.log(10) + (L1 - L0) * Math.log(10) * i / N, t = Math.exp(u);
      const hi = hInv(t, D), F = hi * t; // integrand in log space
      if (i === 0) acc = 0.5 * c * t * t; else acc += 0.5 * (F + prevF) * (u - prevU);
      us.push(u); qs.push(acc); hs.push(hi); prevU = u; prevF = F;
    }
    return (qTables[D] = { us, qs, hs, L0: us[0], du: us[1] - us[0], c });
  }
  function qD(z, D) {
    z = Math.abs(z);
    if (D === 2) return 2 - Math.sqrt(4 + z * z) + z * Math.asinh(z / 2);
    const T = qTable(D);
    if (z === 0) return 0;
    const u = Math.log(z);
    if (u <= T.L0) return 0.5 * T.c * z * z;
    const f = (u - T.L0) / T.du, i = Math.min(Math.floor(f), T.us.length - 2), r = f - i;
    return T.qs[i] + r * (T.qs[i + 1] - T.qs[i]);
  }
  function Q(w, a, D) { const s = D === 2 ? a * a : Math.pow(a, D); return s * (qD(w[0] / s, D) + qD(w[1] / s, D)); }
  function gradQdir(w, a, D) { // h^{-1}(w/s): proportional to grad Q
    const s = D === 2 ? a * a : Math.pow(a, D);
    return w.map(wi => D === 2 ? Math.asinh(wi / s / 2) : Math.sign(wi) * hInv(Math.abs(wi / s), D));
  }
  function levelSet(level, a, D, n) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const th = 2 * Math.PI * i / n, u = [Math.cos(th), Math.sin(th)];
      let hi = 0.5; while (Q([hi * u[0], hi * u[1]], a, D) < level && hi < 1e3) hi *= 2;
      const r = bisect(r => Q([r * u[0], r * u[1]], a, D) - level, 0, hi, 60);
      pts.push([r * u[0], r * u[1]]);
    }
    return pts;
  }
  // real discrete GD on (u, v) for L = 1/2 (x.w - 1)^2, curvature-adaptive step (a time reparameterization)
  function gdRun(x, a, D, c, maxSteps, every) {
    let u = [a, a], v = [a, a];
    const tr = [[0, 0]];
    for (let t = 0; t < maxSteps; t++) {
      const w = [Math.pow(u[0], D) - Math.pow(v[0], D), Math.pow(u[1], D) - Math.pow(v[1], D)];
      const r = w[0] * x[0] + w[1] * x[1] - 1;
      if (Math.abs(r) < 1e-10) { tr.push(w); break; }
      const g = [r * x[0], r * x[1]];
      const big = Math.max(u[0], u[1], v[0], v[1]);
      const curv = 2 * D * D * (x[0] * x[0] + x[1] * x[1]) * Math.pow(big, 2 * D - 2) + D * (D - 1) * Math.pow(big, Math.max(D - 2, 0)) * Math.max(Math.abs(g[0]), Math.abs(g[1]));
      const eta = c / curv;
      for (let i = 0; i < 2; i++) {
        const gu = D * Math.pow(u[i], D - 1) * g[i], gv = -D * Math.pow(v[i], D - 1) * g[i];
        u[i] -= eta * gu; v[i] -= eta * gv;
      }
      if (t % every === 0) tr.push(w);
    }
    const w = [Math.pow(u[0], D) - Math.pow(v[0], D), Math.pow(u[1], D) - Math.pow(v[1], D)];
    tr.push(w);
    return tr;
  }

  // ------------------------------------------------------------------ DOM
  root.innerHTML = `
    <div class="ib-controls">
      <label>init scale α <input type="range" id="dn-a" min="-3" max="1" step="0.01" value="-1"><span id="dn-a-val" class="ib-val"></span></label>
      <label>depth D <select id="dn-D"><option value="2" selected>2</option><option value="3">3</option><option value="4">4</option></select></label>
      <label>data point x = (1, <input type="range" id="dn-x2" min="-0.6" max="0.9" step="0.01" value="0.4"><span id="dn-x2-val" class="ib-val"></span>)</label>
    </div>
    <div class="ib-controls">
      <label><input type="checkbox" id="dn-fan" checked> show paths for all α</label>
      <button id="dn-run">run real gradient descent ▸</button>
      <span id="dn-read" class="ib-read"></span>
    </div>
    <div class="ib-row"><div id="dn-main"></div><div id="dn-side"></div></div>
    <pre id="dn-test" class="ib-test"></pre>`;
  const main = canvas(root.querySelector("#dn-main"), 560, 520);
  const side = canvas(root.querySelector("#dn-side"), 420, 520);
  const el = id => root.querySelector(id);
  let gdTrace = null, gdAnim = null;

  const XL = [-0.2, 1.25], YL = [-0.62, 0.83];
  const X = scale(XL[0], XL[1], 30, 550), Y = scale(YL[0], YL[1], 510, -10);
  function state() { return { a: Math.pow(10, +el("#dn-a").value), D: +el("#dn-D").value, x: [1, +el("#dn-x2").value] }; }

  function drawMain() {
    const { a, D, x } = state(), ctx = main.ctx;
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, main.w, main.h);
    // axes
    line(ctx, [[X(XL[0]), Y(0)], [X(XL[1]), Y(0)]], C.axis, 1);
    line(ctx, [[X(0), Y(YL[0])], [X(0), Y(YL[1])]], C.axis, 1);
    text(ctx, "w₁", X(1.23), Y(0) - 12, { align: "right", size: 15 });
    text(ctx, "w₂", X(0) + 10, Y(0.76), { size: 15 });
    [0.5, 1].forEach(v => { text(ctx, v, X(v), Y(0) + 13, { align: "center", size: 11, color: C.muted }); });
    // references: min-L2 point + its circle, min-L1 point + its diamond
    const n2 = x[0] * x[0] + x[1] * x[1], l2 = [x[0] / n2, x[1] / n2], r2 = Math.hypot(...l2);
    const circ = []; for (let i = 0; i <= 120; i++) { const t = 2 * Math.PI * i / 120; circ.push([X(r2 * Math.cos(t)), Y(r2 * Math.sin(t))]); }
    line(ctx, circ, "rgba(42,120,214,0.55)", 1.2);
    const im = Math.abs(x[0]) >= Math.abs(x[1]) ? 0 : 1, l1 = [0, 0]; l1[im] = 1 / x[im];
    const R1 = Math.abs(l1[0]) + Math.abs(l1[1]);
    line(ctx, [[X(R1), Y(0)], [X(0), Y(R1)], [X(-R1), Y(0)], [X(0), Y(-R1)], [X(R1), Y(0)]], "rgba(27,175,122,0.8)", 1.2);
    // the solution line
    const seg = [];
    if (Math.abs(x[1]) > 1e-6) { XL.forEach(w1 => seg.push([X(w1), Y((1 - x[0] * w1) / x[1])])); }
    else seg.push([X(1), Y(YL[0])], [X(1), Y(YL[1])]);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, main.w, main.h); ctx.clip();
    line(ctx, seg, C.ink, 1.6);
    // fan
    if (el("#dn-fan").checked) {
      for (let la = -3; la <= 1.001; la += 0.25) { const p = path(x, Math.pow(10, la), D, 160).P; line(ctx, p.map(q => [X(q[0]), Y(q[1])]), ramp((la + 3) / 4), 1, null); }
      ctx.globalAlpha = 1;
    }
    // Q level set through the landing point
    const pa = path(x, a, D, 400);
    const lvl = Q(pa.end, a, D);
    line(ctx, levelSet(lvl, a, D, 240).map(q => [X(q[0]), Y(q[1])]), C.ink, 1.3, [4, 4]);
    // current path
    const col = ramp((Math.log10(a) + 3) / 4);
    ctx.globalAlpha = 0.18; line(ctx, pa.P.map(q => [X(q[0]), Y(q[1])]), col, 9); ctx.globalAlpha = 1;
    line(ctx, pa.P.map(q => [X(q[0]), Y(q[1])]), col, 3);
    if (gdTrace) {
      const n = gdAnim === null ? gdTrace.length : Math.min(gdTrace.length, gdAnim);
      for (let i = 0; i < n; i += 1) dot(ctx, X(gdTrace[i][0]), Y(gdTrace[i][1]), 1.8, C.orange);
      if (n > 0) dot(ctx, X(gdTrace[n - 1][0]), Y(gdTrace[n - 1][1]), 5, C.orange, C.paper);
    }
    ctx.restore();
    dot(ctx, X(l2[0]), Y(l2[1]), 4.5, C.paper); ctx.strokeStyle = C.blue; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(X(l2[0]), Y(l2[1]), 4.5, 0, 7); ctx.stroke();
    dot(ctx, X(l1[0]), Y(l1[1]), 4.5, C.paper); ctx.strokeStyle = C.aqua; ctx.beginPath(); ctx.arc(X(l1[0]), Y(l1[1]), 4.5, 0, 7); ctx.stroke();
    dot(ctx, X(pa.end[0]), Y(pa.end[1]), 6.5, col, C.paper);
    text(ctx, "min-L₂", X(l2[0]) - 10, Y(l2[1]) - 12, { align: "right", size: 13.5, color: C.ink });
    text(ctx, "min-L₁", X(l1[0]) + 10, Y(l1[1]) + 15, { size: 13.5, color: C.ink });
    // legend
    const lx = 300, ly = 400;
    ctx.fillStyle = "rgba(252,251,248,0.92)"; ctx.fillRect(lx - 8, ly - (gdTrace ? 30 : 12), 262, gdTrace ? 60 : 42);
    line(ctx, [[lx, ly], [lx + 22, ly]], col, 3); text(ctx, "gradient-flow path (closed form)", lx + 28, ly, { size: 11.5 });
    line(ctx, [[lx, ly + 18], [lx + 22, ly + 18]], C.ink, 1.3, [4, 4]); text(ctx, "level set of Q_α through the landing point", lx + 28, ly + 18, { size: 11.5 });
    if (gdTrace) { dot(ctx, lx + 11, ly - 18, 3.5, C.orange); text(ctx, "discrete GD on (u, v), running live", lx + 28, ly - 18, { size: 11.5 }); }
    el("#dn-read").textContent = `lands at w = (${pa.end[0].toFixed(3)}, ${pa.end[1].toFixed(3)})   ‖w‖₁ = ${(Math.abs(pa.end[0]) + Math.abs(pa.end[1])).toFixed(3)}   ‖w‖₂ = ${Math.hypot(...pa.end).toFixed(3)}`;
    return pa;
  }

  const sideCache = {};
  function drawSide() {
    const { a, D, x } = state(), ctx = side.ctx;
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, side.w, side.h);
    const key = x[1].toFixed(3);
    if (!sideCache[key]) {
      sideCache[key] = {};
      [2, 3, 4].forEach(d => { const arr = []; for (let la = -3; la <= 1.0001; la += 0.04) arr.push([la, path(x, Math.pow(10, la), d, 8).end]); sideCache[key][d] = arr; });
    }
    const n2 = x[0] * x[0] + x[1] * x[1], l2y = x[1] / n2;
    const lo = Math.min(0, l2y), hi = Math.max(0, l2y);
    const pad = 0.12 * Math.max(0.1, hi - lo);
    const SX = scale(-3, 1, 60, 400), SY = scale(lo - pad, hi + pad, 440, 90);
    text(ctx, "Where it lands: w₂ of the final solution", 20, 28, { size: 14, color: C.ink, weight: 500 });
    text(ctx, "one curve per depth; deeper nets reach the sparse answer at larger α", 20, 50, { size: 11.5 });
    [[l2y, "min-L₂", C.blue], [0, "min-L₁ (sparse)", C.aqua]].forEach(([v, s, c]) => {
      line(ctx, [[SX(-3), SY(v)], [SX(1), SY(v)]], c, 1, [3, 3]);
      text(ctx, s, SX(v === 0 ? 1 : -3), SY(v) + (v >= l2y && v !== 0 ? -10 : 10) * (l2y >= 0 ? 1 : -1), { align: v === 0 ? "right" : "left", size: 11 });
    });
    [2, 3, 4].forEach(d => {
      const arr = sideCache[key][d];
      line(ctx, arr.map(([la, w]) => [SX(la), SY(w[1])]), d === D ? C.violet : "rgba(74,58,167,0.28)", d === D ? 2.5 : 1.4);
      // label where the curve crosses half-way between the two answers
      const half = 0.5 * l2y;
      let lab = arr[0];
      for (const q of arr) { if ((q[1][1] - half) * Math.sign(l2y || 1) >= 0) { lab = q; break; } }
      text(ctx, `D = ${d}`, SX(lab[0]) + 8, SY(lab[1][1]), { size: 11.5, color: d === D ? C.ink : C.muted });
    });
    const end = path(x, a, D, 8).end;
    dot(ctx, SX(Math.log10(a)), SY(end[1]), 6, ramp((Math.log10(a) + 3) / 4), C.paper);
    [-3, -2, -1, 0, 1].forEach(la => text(ctx, ["0.001", "0.01", "0.1", "1", "10"][la + 3], SX(la), 462, { align: "center", size: 10.5, color: C.muted }));
    text(ctx, "init scale α (log scale)", SX(-1), 488, { align: "center", size: 11.5 });
  }

  function redraw() {
    el("#dn-a-val").textContent = Math.pow(10, +el("#dn-a").value).toPrecision(2);
    el("#dn-x2-val").textContent = (+el("#dn-x2").value).toFixed(2);
    drawMain(); drawSide();
  }
  ["#dn-a", "#dn-D", "#dn-x2", "#dn-fan"].forEach(id => el(id).addEventListener("input", () => { gdTrace = null; gdAnim = null; redraw(); }));
  el("#dn-run").addEventListener("click", () => {
    const { a, D, x } = state();
    gdTrace = gdRun(x, a, D, 0.05, 400000, 40);
    // resample the trace to ~150 points evenly in arclength so the animation has a steady speed
    gdAnim = 0;
    const step = () => { gdAnim += Math.max(1, Math.round(gdTrace.length / 150)); drawMain(); if (gdAnim < gdTrace.length) requestAnimationFrame(step); else gdAnim = null; };
    requestAnimationFrame(step);
  });
  redraw();

  // ------------------------------------------------------------------ self-test (#selftest)
  if (location.hash.includes("selftest")) {
    const out = [];
    let ok = true;
    for (const D of [2, 3, 4]) for (const a of [1e-3, 0.03, 1, 10]) for (const x2 of [0.4, -0.5]) {
      const x = [1, x2], pa = path(x, a, D, 50), w = pa.end;
      const res = Math.abs(w[0] * x[0] + w[1] * x[1] - 1);
      const g = gradQdir(w, a, D), cos = Math.abs(g[0] * x[0] + g[1] * x[1]) / Math.hypot(...g) / Math.hypot(...x);
      const pass = res < 1e-8 && cos > 1 - 1e-6;
      ok = ok && pass;
      out.push(`D=${D} a=${a} x2=${x2}: residual ${res.toExponential(1)}, grad-Q parallel to x: 1-cos=${(1 - cos).toExponential(1)} ${pass ? "ok" : "FAIL"}`);
    }
    for (const D of [2, 3]) {
      const x = [1, 0.4], a = 0.1, tr = gdRun(x, a, D, 0.02, 2000000, 1000), w = tr[tr.length - 1], e = path(x, a, D, 5).end;
      const dist = Math.hypot(w[0] - e[0], w[1] - e[1]), pass = dist < 5e-3;
      ok = ok && pass;
      out.push(`live GD (D=${D}, a=0.1) lands ${dist.toExponential(2)} from the closed form ${pass ? "ok" : "FAIL"}`);
    }
    // level set passes through the landing point
    { const x = [1, 0.4], a = 0.05, D = 3, e = path(x, a, D, 5).end, L = levelSet(Q(e, a, D), a, D, 3600);
      const dmin = Math.min(...L.map(p => Math.hypot(p[0] - e[0], p[1] - e[1]))), pass = dmin < 3e-3; ok = ok && pass;
      out.push(`Q level set passes through landing point: ${dmin.toExponential(1)} ${pass ? "ok" : "FAIL"}`); }
    el("#dn-test").textContent = (ok ? "SELFTEST PASS\n" : "SELFTEST FAIL\n") + out.join("\n");
    el("#dn-test").style.display = "block";
    console.log("diagnet selftest", ok ? "PASS" : "FAIL");
  }
})();
