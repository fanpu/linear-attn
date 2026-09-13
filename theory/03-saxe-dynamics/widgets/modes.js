// Widget: a real deep linear network trained in the browser (gradient flow via RK4, whitened inputs) next to
// Saxe's reduced mode equations  du/dt = L u^{2-2/L} (s - u).
(function () {
  const { C, setupCanvas, Axes, rng } = window.SaxeCommon;
  const root = document.getElementById("modes-widget");
  if (!root) return;
  const N = 6;

  root.innerHTML = `
    <div class="mw-controls">
      <div class="mw-group">${[0, 1, 2, 3].map(i => `
        <label><span class="mw-key" style="background:${C.modes[i]}"></span><i>s</i><sub>${i + 1}</sub>
          <input type="range" id="mw-s${i}" min="-1" max="0.7" step="0.01"> <span id="mw-s${i}-v"></span></label>`).join("")}
      </div>
      <div class="mw-group">
        <label>init scale <input type="range" id="mw-sig" min="-6" max="-0.3" step="0.1" value="-3"> <span id="mw-sig-v"></span></label>
        <label>depth <select id="mw-L"><option value="2">2 layers</option><option value="3">3 layers</option><option value="4">4 layers</option></select></label>
        <label>init <select id="mw-init"><option value="decoupled">decoupled (theory's assumption)</option><option value="random">small random Gaussian</option></select></label>
        <button id="mw-play">replay ▶</button>
      </div>
    </div>
    <div class="mw-row">
      <canvas id="mw-modes"></canvas>
      <div class="mw-right"><canvas id="mw-loss"></canvas>
        <div class="mw-matwrap"><canvas id="mw-mat"></canvas><div class="mw-matcap">network map in the target's SVD basis, <i>U</i><sup>T</sup><i>W</i>(t)<i>V</i>: each entry scaled by its target strength; the diagonal fills in mode by mode, off-diagonal colour = modes interfering</div></div>
      </div>
    </div>
    <div class="mw-note" id="mw-note"></div>`;

  const W = Math.max(640, Math.min(1060, root.clientWidth || 1000));
  const MW = Math.round(W * 0.55), RW = W - MW - 30;
  const mctx = setupCanvas(root.querySelector("#mw-modes"), MW, 360);
  const lctx = setupCanvas(root.querySelector("#mw-loss"), RW, 200);
  const xctx = setupCanvas(root.querySelector("#mw-mat"), 132, 132);
  const note = root.querySelector("#mw-note");

  const defaults = [5, 2, 1, 0.5];
  const state = { s: defaults.slice(), sig: -3, L: 2, init: "decoupled", sim: null, play: 1, raf: null };
  [0, 1, 2, 3].forEach(i => { root.querySelector(`#mw-s${i}`).value = Math.log10(defaults[i]); });

  // ---------- linear algebra on flat N x N arrays
  const mul = (A, B) => { const R = new Float64Array(N * N); for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) { const a = A[i * N + k]; if (a === 0) continue; for (let j = 0; j < N; j++) R[i * N + j] += a * B[k * N + j]; } return R; };
  const T = A => { const R = new Float64Array(N * N); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) R[j * N + i] = A[i * N + j]; return R; };
  const eye = () => { const R = new Float64Array(N * N); for (let i = 0; i < N; i++) R[i * N + i] = 1; return R; };
  function orth(r) { // Gram-Schmidt on a random matrix (columns)
    const M = new Float64Array(N * N); for (let i = 0; i < N * N; i++) M[i] = r.g();
    for (let j = 0; j < N; j++) {
      for (let k = 0; k < j; k++) { let d = 0; for (let i = 0; i < N; i++) d += M[i * N + j] * M[i * N + k]; for (let i = 0; i < N; i++) M[i * N + j] -= d * M[i * N + k]; }
      let n = 0; for (let i = 0; i < N; i++) n += M[i * N + j] ** 2; n = Math.sqrt(n); for (let i = 0; i < N; i++) M[i * N + j] /= n;
    }
    return M;
  }
  const R0 = rng(7), U = orth(R0), V = orth(R0);

  function theoryMode(ts, s, u0, L) {
    const out = new Float64Array(ts.length);
    let z = Math.log(u0), t = 0;
    const f = z => { const u = Math.exp(z); return L * Math.pow(u, 1 - 2 / L) * (s - u); };
    for (let i = 0; i < ts.length; i++) {
      const n = Math.max(1, Math.ceil((ts[i] - t) * 30 * Math.max(1, s) * L));
      const h = (ts[i] - t) / n;
      for (let k = 0; k < n; k++) { const k1 = f(z), k2 = f(z + h / 2 * k1), k3 = f(z + h / 2 * k2), k4 = f(z + h * k3); z += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); }
      t = ts[i]; out[i] = Math.exp(z);
    }
    return out;
  }

  function gradsOf(Ws, Syx, L) {
    const right = [eye()]; for (let l = 0; l < L; l++) right.push(mul(Ws[l], right[l]));
    const G = new Float64Array(N * N); for (let i = 0; i < N * N; i++) G[i] = right[L][i] - Syx[i];
    let left = eye(); const grads = new Array(L);
    for (let l = L - 1; l >= 0; l--) { grads[l] = mul(mul(T(left), G), T(right[l])); left = mul(left, Ws[l]); }
    return grads;
  }
  const addScaled = (Ws, Gs, c) => Ws.map((w, l) => w.map((v, i) => v + c * Gs[l][i]));

  function simulate() {
    const s = state.s, L = state.L, sig = Math.pow(10, state.sig);
    const smax = Math.max(...s), smin = Math.min(...s);
    const u0 = state.init === "decoupled" ? Math.pow(sig, L) : Math.pow(sig * sig * N / 2, L / 2);
    // horizon: when the weakest theory mode reaches 97%
    let tEnd = 1;
    { const ts = []; for (let k = 1; k <= 400; k++) ts.push(k * 0.5); let found = false;
      for (let grow = 0; grow < 8 && !found; grow++) {
        const scale = Math.pow(4, grow); const tt = ts.map(x => x * scale / smax);
        const u = theoryMode(tt, smin, Math.min(u0, smin * 0.5), L);
        for (let k = 0; k < tt.length; k++) if (u[k] > 0.97 * smin) { tEnd = tt[k]; found = true; break; }
      }
      if (!found) tEnd = 200 / smax; }
    tEnd *= 1.08;
    const dt = Math.min(0.05, 0.12 / (L * Math.pow(smax, 2 - 2 / L)));
    const maxSteps = 40000;
    const nSteps = Math.min(maxSteps, Math.ceil(tEnd / dt));
    const truncated = nSteps === maxSteps;
    tEnd = nSteps * dt;
    const S = new Float64Array(N * N); s.forEach((v, i) => S[i * N + i] = v);
    const Syx = mul(mul(U, S), T(V));
    const r = rng(11 + L);
    let Ws = [];
    if (state.init === "decoupled") {
      for (let l = 0; l < L; l++) {
        let M = eye().map(v => v * sig);
        if (l === 0) M = mul(M, T(V));
        if (l === L - 1) M = mul(U, M);
        Ws.push(M);
      }
    } else {
      for (let l = 0; l < L; l++) Ws.push(new Float64Array(N * N).map(() => r.g() * sig));
    }
    const nRec = 360, every = Math.max(1, Math.floor(nSteps / nRec));
    const rec = { t: [], modes: [[], [], [], []], loss: [], mats: [] };
    const UT = T(U);
    for (let step = 0; step <= nSteps; step++) {
      if (step % every === 0 || step === nSteps) {
        let Wn = eye(); for (let l = 0; l < L; l++) Wn = mul(Ws[l], Wn);
        const M = mul(mul(UT, Wn), V);
        rec.t.push(step * dt);
        for (let a = 0; a < 4; a++) rec.modes[a].push(M[a * N + a]);
        let lo = 0; for (let i = 0; i < N * N; i++) lo += (Wn[i] - Syx[i]) ** 2; rec.loss.push(0.5 * lo);
        rec.mats.push(M);
      }
      if (step === nSteps) break;
      // one RK4 step of gradient flow on all layers
      const k1 = gradsOf(Ws, Syx, L);
      const k2 = gradsOf(addScaled(Ws, k1, -dt / 2), Syx, L);
      const k3 = gradsOf(addScaled(Ws, k2, -dt / 2), Syx, L);
      const k4 = gradsOf(addScaled(Ws, k3, -dt), Syx, L);
      for (let l = 0; l < L; l++) for (let i = 0; i < N * N; i++)
        Ws[l][i] -= dt / 6 * (k1[l][i] + 2 * k2[l][i] + 2 * k3[l][i] + k4[l][i]);
    }
    const th = s.map(sv => theoryMode(rec.t, sv, u0, L));
    const thLoss = rec.t.map((_, k) => 0.5 * s.reduce((acc, sv, a) => acc + (sv - th[a][k]) ** 2, 0));
    return { rec, th, thLoss, u0, tEnd, truncated, L, s: s.slice(), init: state.init };
  }

  function draw(frac) {
    const sim = state.sim; if (!sim) return;
    const { rec, th, thLoss } = sim;
    const kmax = Math.max(1, Math.round(frac * (rec.t.length - 1)));
    mctx.clearRect(0, 0, MW, 360);
    const tEnd = rec.t[rec.t.length - 1];
    const ax = Axes(mctx, { x: 52, y: 26, w: MW - 70, h: 290 }, [0, tEnd], [-0.05, 1.12]);
    const ticks = niceTicks(tEnd);
    ax.draw(ticks, [0, 0.5, 1], "training time t", "mode strength  u / s");
    for (let a = 0; a < 4; a++) {
      const sv = sim.s[a];
      ax.line(rec.t.slice(0, kmax + 1), rec.modes[a].slice(0, kmax + 1).map(v => v / sv), C.modes[a], 4.5, 0.55);
      ax.line(rec.t, Array.from(th[a]).map(v => v / sv), C.ink, 1.2, 0.95);
      const x = ax.X(rec.t[kmax]), y = ax.Y(rec.modes[a][kmax] / sv);
      mctx.beginPath(); mctx.arc(x, Math.max(20, Math.min(330, y)), 4.5, 0, 2 * Math.PI);
      mctx.fillStyle = C.modes[a]; mctx.fill(); mctx.strokeStyle = C.paper; mctx.lineWidth = 2; mctx.stroke();
    }
    mctx.fillStyle = C.muted; mctx.font = "11.5px Inter, 'Helvetica Neue', Arial, sans-serif";
    mctx.fillText("thick: a 6-unit network trained (gradient flow) in your browser · thin black: reduced ODE (Saxe et al.)", 54, 14);

    lctx.clearRect(0, 0, RW, 200);
    const lmax = Math.max(rec.loss[0], thLoss[0]) * 1.5;
    const lmin = Math.max(Math.min(...rec.loss, ...thLoss) * 0.5, lmax * 1e-6);
    const bx = Axes(lctx, { x: 52, y: 14, w: RW - 64, h: 146 }, [0, tEnd], [lmin, lmax], { ylog: true });
    const yt = []; for (let e = Math.ceil(Math.log10(lmin)); e <= Math.floor(Math.log10(lmax)); e++) yt.push(Math.pow(10, e));
    bx.draw(ticks, yt, "training time t", "loss (log)");
    bx.line(rec.t, thLoss, C.ink, 1.1, 0.9);
    bx.line(rec.t.slice(0, kmax + 1), rec.loss.slice(0, kmax + 1), C.accent, 2.6);

    // matrix heatmap
    const M = rec.mats[kmax], cell = 132 / N, sc = i => Math.max(i < 4 ? sim.s[i] : 0, 0.25);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = Math.max(-1, Math.min(1, M[i * N + j] / Math.sqrt(sc(i) * sc(j))));
      xctx.fillStyle = diverging(v);
      xctx.fillRect(j * cell + 1, i * cell + 1, cell - 2, cell - 2);
    }
    const warn = sim.truncated ? " Stopped early (too many steps): try a larger init or fewer layers." : "";
    note.innerHTML = `theory uses u₀ = ${sim.u0.toExponential(1)} (${sim.init === "decoupled" ? "exact for this init" : "balanced-equivalent guess (σ²N/2)<sup>L/2</sup>; not exact"}).` + warn;
  }

  function diverging(v) { // blue <- gray -> red
    const neg = [42, 120, 214], pos = [181, 69, 43], mid = [240, 239, 236];
    const c = v < 0 ? neg : pos, f = Math.pow(Math.abs(v), 0.7);
    return `rgb(${mid.map((m, k) => Math.round(m + f * (c[k] - m))).join(",")})`;
  }
  function niceTicks(tEnd) {
    const raw = tEnd / 5, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    const out = []; for (let t = 0; t <= tEnd + 1e-9; t += step) out.push(+t.toPrecision(6)); return out;
  }

  function labels() {
    [0, 1, 2, 3].forEach(i => root.querySelector(`#mw-s${i}-v`).textContent = state.s[i].toFixed(2));
    root.querySelector("#mw-sig-v").textContent = "10^" + state.sig.toFixed(1);
  }

  function rerun() {
    labels();
    state.sim = simulate();
    state.play = 0;
    if (!state.raf) { const t0 = performance.now(); const loop = now => { state.play = Math.min(1, (now - t0) / 3500); draw(state.play); state.raf = state.play < 1 ? requestAnimationFrame(loop) : null; }; state.raf = requestAnimationFrame(loop); }
  }
  let timer = null;
  const schedule = () => { clearTimeout(timer); timer = setTimeout(() => { if (state.raf) { cancelAnimationFrame(state.raf); state.raf = null; } rerun(); }, 60); };
  [0, 1, 2, 3].forEach(i => root.querySelector(`#mw-s${i}`).addEventListener("input", ev => { state.s[i] = Math.pow(10, +ev.target.value); labels(); schedule(); }));
  root.querySelector("#mw-sig").addEventListener("input", ev => { state.sig = +ev.target.value; labels(); schedule(); });
  root.querySelector("#mw-L").addEventListener("change", ev => { state.L = +ev.target.value; schedule(); });
  root.querySelector("#mw-init").addEventListener("change", ev => { state.init = ev.target.value; schedule(); });
  root.querySelector("#mw-play").addEventListener("click", () => { if (state.raf) { cancelAnimationFrame(state.raf); state.raf = null; } rerun(); });

  labels();
  state.sim = simulate();
  draw(1);

  if (location.hash.indexOf("selftest") >= 0) {
    try {
      const report = [];
      for (const [L, init, tol] of [[2, "decoupled", 0.02], [3, "decoupled", 0.02], [2, "random", 0.25]]) {
        state.L = L; state.init = init; state.sig = L === 2 ? -3 : -1;
        const sim = simulate();
        let err = 0;
        for (let a = 0; a < 4; a++) for (let k = 0; k < sim.rec.t.length; k++) err = Math.max(err, Math.abs(sim.rec.modes[a][k] - sim.th[a][k]) / sim.s[a]);
        draw(1);
        report.push(`L=${L} ${init}: max|sim-theory|/s=${err.toFixed(4)}`);
        if (err > tol) console.error("modes selftest FAILED", L, init, err);
      }
      console.log("modes selftest: " + report.join("; "));
      state.L = 2; state.init = "decoupled"; state.sig = -3; state.sim = simulate(); draw(1);
    } catch (e) { console.error("modes selftest exception", e); }
  }
})();
