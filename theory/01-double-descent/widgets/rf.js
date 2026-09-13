// Widget: live 1-D random-features regression. Min-norm least squares (or ridge) solved in the browser with a
// one-sided Jacobi SVD. Default features = the exact draw used in the hero animation (exported from python).
(function () {
  const root = document.getElementById("rf-widget");
  if (!root || !window.RF_DATA) return;
  const { C, FONT, heat, setupCanvas, Axes, fmtNum, rng, jacobiSVD } = window.DD;
  const D = window.RF_DATA;
  const n = D.x.length, PMAX = 400;
  const target = t => 0.8 * Math.sin(2 * Math.PI * t) + 0.3 * t;
  const XS = Array.from({ length: 301 }, (_, i) => -1 + 2 * i / 300);
  const FS = XS.map(target);

  // slider positions -> P values: every integer up to 60, then log-spaced to 400
  const PV = []; for (let p = 1; p <= 60; p++) PV.push(p);
  for (let i = 1; i <= 40; i++) { const p = Math.round(60 * Math.pow(400 / 60, i / 40)); if (p > PV[PV.length - 1]) PV.push(p); }

  root.innerHTML = `
    <div class="dd-controls">
      <label>features p <input type="range" id="rf-p" min="0" max="${PV.length - 1}" step="1" value="${PV.indexOf(12)}"><span id="rf-pv" class="dd-val"></span></label>
      <label>ridge λ <input type="range" id="rf-lam" min="-10.2" max="0" step="0.05" value="-10.2"><span id="rf-lamv" class="dd-val"></span></label>
      <button id="rf-play">▶ sweep p</button>
      <button id="rf-feat">new random features</button>
      <button id="rf-noise">new noise</button>
      <button id="rf-reset">reset</button>
    </div>
    <canvas id="rf-c1" class="dd-dark"></canvas>
    <canvas id="rf-c2" class="dd-dark"></canvas>
    <div class="dd-readout" id="rf-read"></div>
    <div class="dd-hint">Drag any data point up or down. The fit is recomputed exactly each time: the minimum-norm solution when λ = 0, ridge otherwise (penalty λ‖a‖²). Bottom: test error against p for <em>this</em> draw of features and data (bright), and the median over 200 feature draws for the original data at λ = 0 (faint band).</div>`;
  const $ = id => root.querySelector("#" + id);
  const Wd = Math.max(520, Math.min(1040, (root.clientWidth || 1000) - 34));
  const H1 = 330, H2 = 190;
  const ctx1 = setupCanvas($("rf-c1"), Wd, H1), ctx2 = setupCanvas($("rf-c2"), Wd, H2);

  let W = D.W.slice(), B = D.B.slice(), Y = D.y.slice(), X = D.x.slice();
  let seed = 1;

  function phiRow(t, P) { const s = Math.sqrt(2 / P), r = new Float64Array(P); for (let j = 0; j < P; j++) r[j] = s * Math.cos(W[j] * t + B[j]); return r; }

  // returns {a, smin}
  function solve(P, lam, xs, ys) {
    const rows = xs.map(t => phiRow(t, P));                // n rows of length P
    let U, S, V, a = new Float64Array(P);
    if (P <= n) {                                         // Phi = U S V^T, columns of Phi
      const cols = []; for (let j = 0; j < P; j++) cols.push(Float64Array.from(rows, r => r[j]));
      ({ U, S, V } = jacobiSVD(cols));
      const smax = Math.max(...S), tol = smax * 2.220446049250313e-16 * Math.max(n, P);
      for (let k = 0; k < P; k++) {
        if (S[k] <= tol) continue;
        let uy = 0; for (let i = 0; i < n; i++) uy += U[k][i] * ys[i];
        const f = lam > 0 ? S[k] / (S[k] * S[k] + lam) : 1 / S[k];
        for (let j = 0; j < P; j++) a[j] += V[k][j] * f * uy;
      }
    } else {                                              // Phi^T = U S V^T  =>  Phi = V S U^T
      ({ U, S, V } = jacobiSVD(rows));
      const smax = Math.max(...S), tol = smax * 2.220446049250313e-16 * Math.max(n, P);
      for (let k = 0; k < n; k++) {
        if (S[k] <= tol) continue;
        let vy = 0; for (let i = 0; i < n; i++) vy += V[k][i] * ys[i];
        const f = lam > 0 ? S[k] / (S[k] * S[k] + lam) : 1 / S[k];
        for (let j = 0; j < P; j++) a[j] += U[k][j] * f * vy;
      }
    }
    return { a, smin: Math.min(...S) };
  }
  function predict(a, P, xs) { return xs.map(t => { const r = phiRow(t, P); let s = 0; for (let j = 0; j < P; j++) s += a[j] * r[j]; return s; }); }
  function testRisk(a, P) { const f = predict(a, P, XS); let s = 0; for (let i = 0; i < XS.length; i++) s += (f[i] - FS[i]) ** 2; return s / XS.length; }

  const GRID = Array.from(new Set([...Array.from({ length: 34 }, (_, i) => Math.round(Math.pow(400, i / 33))), 16, 17, 18, 19, 20, 21, 22, 23, 25, 28])).sort((a, b) => a - b);
  let curve = null, curveKey = "";

  function state() {
    const P = PV[+$("rf-p").value];
    const lr = +$("rf-lam").value;
    return { P, lam: lr <= -10.1 ? 0 : Math.pow(10, lr) };
  }

  function computeCurve(lam) {
    const key = lam + "|" + seed + "|" + Y.join(",");
    if (key === curveKey) return;
    curve = GRID.map(P => { const s = solve(P, lam, X, Y); return testRisk(s.a, P); });
    curveKey = key;
  }

  const norm = r => (Math.log10(r) - Math.log10(0.006)) / (Math.log10(20) - Math.log10(0.006));
  let dragging = -1, fitAx = null;

  function draw() {
    const st = state();
    $("rf-pv").textContent = st.P;
    $("rf-lamv").textContent = st.lam === 0 ? "0" : fmtNum(st.lam, 0);
    const sol = solve(st.P, st.lam, X, Y);
    const r = testRisk(sol.a, st.P);
    const col = heat(norm(r));
    const aNorm = Math.sqrt(sol.a.reduce((s, v) => s + v * v, 0));

    // ---- fit panel ----
    ctx1.fillStyle = C.night; ctx1.fillRect(0, 0, Wd, H1);
    const ax = Axes(ctx1, { x: 20, y: 16, w: Wd - 40, h: H1 - 32 }, [-1.05, 1.05], [-2.6, 2.6], { dark: true });
    fitAx = ax;
    ax.line(XS, FS, C.nightMuted, 1.2, 0.9, [2, 4]);
    const fx = Array.from({ length: 601 }, (_, i) => -1.05 + 2.1 * i / 600);
    const f = predict(sol.a, st.P, fx);
    [[12, 0.06], [7, 0.12], [4, 0.25], [2.2, 1]].forEach(([lw, al]) => ax.line(fx, f, col, lw, al));
    for (let i = 0; i < n; i++) {
      ax.dot(X[i], Y[i], 11, "rgba(255,246,232,0.10)");
      ax.dot(X[i], Y[i], i === dragging ? 6.5 : 5, C.data, C.night);
    }
    ctx1.save(); ctx1.font = `600 22px ${FONT}`; ctx1.fillStyle = C.nightInk; ctx1.textAlign = "right"; ctx1.textBaseline = "top";
    ctx1.fillText(`p = ${st.P}`, Wd - 18, 12);
    ctx1.font = `13px ${FONT}`; ctx1.fillStyle = C.nightMuted;
    ctx1.fillText(st.P < n ? "p < n: least squares" : st.P === n ? "p = n: the unique exact fit" : (st.lam === 0 ? "p > n: smallest-norm exact fit" : "p > n: ridge"), Wd - 18, 40);
    ctx1.restore();

    // ---- risk vs p panel ----
    computeCurve(st.lam);
    ctx2.fillStyle = C.night; ctx2.fillRect(0, 0, Wd, H2);
    const ax2 = Axes(ctx2, { x: 56, y: 26, w: Wd - 80, h: H2 - 70 }, [1, 400], [3e-3, 300], { xlog: true, ylog: true, dark: true });
    ax2.frame([1, 10, 20, 100, 400], [0.01, 0.1, 1, 10, 100], "number of features p", "test error");
    const px = ax2.X(n);
    ctx2.save(); ctx2.strokeStyle = C.goldD; ctx2.globalAlpha = 0.8; ctx2.beginPath(); ctx2.moveTo(px, ax2.box.y); ctx2.lineTo(px, ax2.box.y + ax2.box.h); ctx2.stroke(); ctx2.restore();
    ax2.band(D.Ps, D.q1, D.q3, "#6b5b9a", 0.35);
    ax2.line(D.Ps, D.med, "#b9aee0", 1.2, 0.8);
    // live curve coloured segment by segment
    for (let i = 0; i + 1 < GRID.length; i++) ax2.line([GRID[i], GRID[i + 1]], [curve[i], curve[i + 1]], heat(norm(Math.sqrt(curve[i] * curve[i + 1]))), 2.2, 1);
    ax2.dot(st.P, Math.max(3e-3, Math.min(300, r)), 9, "rgba(255,246,232,0.2)");
    ax2.dot(st.P, Math.max(3e-3, Math.min(300, r)), 4.5, C.data);
    ctx2.save(); ctx2.font = `12px ${FONT}`; ctx2.fillStyle = C.goldD; ctx2.fillText("p = n", px + 5, ax2.box.y + 10);
    ctx2.fillStyle = C.nightInk; ctx2.fillText("this draw", ax2.box.x + ax2.box.w - 120, ax2.box.y + 4);
    ctx2.fillStyle = C.nightMuted; ctx2.fillText("median of 200 draws (λ = 0)", ax2.box.x + ax2.box.w - 170, ax2.box.y + 20);
    ctx2.restore();

    $("rf-read").innerHTML = `test error <b>${fmtNum(r, 1)}</b> · smallest singular value of Φ <b>${fmtNum(sol.smin, 1)}</b> · coefficient norm ‖a‖ <b>${fmtNum(aNorm, 1)}</b>`;
    return { r, smin: sol.smin };
  }

  // interactions
  root.querySelectorAll("input").forEach(el => el.addEventListener("input", draw));
  $("rf-feat").onclick = () => { seed += 1; const g = rng(seed * 9973); for (let j = 0; j < PMAX; j++) { W[j] = g.g() * D.scale; B[j] = g.u() * 2 * Math.PI; } curveKey = ""; draw(); };
  $("rf-noise").onclick = () => { const g = rng(Date.now() % 100000); Y = X.map(t => target(t) + D.noise * g.g()); curveKey = ""; draw(); };
  $("rf-reset").onclick = () => { W = D.W.slice(); B = D.B.slice(); Y = D.y.slice(); seed = 1; $("rf-lam").value = -10.2; $("rf-p").value = PV.indexOf(12); curveKey = ""; draw(); };
  let playing = null;
  $("rf-play").onclick = () => {
    if (playing) { clearInterval(playing); playing = null; $("rf-play").textContent = "▶ sweep p"; return; }
    $("rf-p").value = 0; $("rf-play").textContent = "❚❚ pause";
    playing = setInterval(() => {
      const el = $("rf-p"); const v = +el.value + 1;
      if (v > +el.max) { clearInterval(playing); playing = null; $("rf-play").textContent = "▶ sweep p"; return; }
      el.value = v; draw();
    }, 140);
  };
  const cv = $("rf-c1");
  const toData = e => { const b = cv.getBoundingClientRect(); return [fitAx.Xinv(e.clientX - b.left), fitAx.Yinv(e.clientY - b.top), e.clientX - b.left, e.clientY - b.top]; };
  cv.addEventListener("pointerdown", e => {
    const [, , px, py] = toData(e);
    let best = -1, bd = 14;
    for (let i = 0; i < n; i++) { const d = Math.hypot(fitAx.X(X[i]) - px, fitAx.Y(Y[i]) - py); if (d < bd) { bd = d; best = i; } }
    if (best >= 0) { dragging = best; cv.setPointerCapture(e.pointerId); }
  });
  cv.addEventListener("pointermove", e => { if (dragging < 0) return; const [, yv] = toData(e); Y[dragging] = Math.max(-2.5, Math.min(2.5, yv)); draw(); });
  cv.addEventListener("pointerup", () => { if (dragging >= 0) { dragging = -1; curveKey = ""; draw(); } });
  draw();

  if (location.hash.indexOf("selftest") >= 0) {
    try {
      let worstR = 0, worstS = 0; const rep = [];
      D.selftest.forEach(q => {
        const s = solve(q.P, q.lam, X, Y), rr = testRisk(s.a, q.P);
        const er = Math.abs(rr - q.risk) / q.risk, es = Math.abs(s.smin - q.smin) / q.smin;
        worstR = Math.max(worstR, er); worstS = Math.max(worstS, es); rep.push(`P=${q.P},λ=${q.lam}:${er.toExponential(1)}`);
      });
      // min-norm near threshold is ill-conditioned (cond ~ 1e9), so allow 1e-4 there
      if (!(worstR < 1e-4) || !(worstS < 1e-5)) console.error("rf selftest FAILED: risk rel err " + worstR + ", smin rel err " + worstS + " | " + rep.join(" "));
      else console.log("rf selftest ok: " + D.selftest.length + " fits vs numpy, worst risk rel err " + worstR.toExponential(2) + ", smin rel err " + worstS.toExponential(2));
      $("rf-feat").onclick(); $("rf-noise").onclick(); $("rf-p").value = $("rf-p").max; draw(); $("rf-reset").onclick();
      console.log("rf selftest ok: new features / noise / reset paths ran");
    } catch (e) { console.error("rf selftest exception " + e + " " + e.stack); }
  }
})();
