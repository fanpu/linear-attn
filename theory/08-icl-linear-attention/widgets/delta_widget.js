// DeltaNet vs linear attention as fast-weight learners, live.
// Stream of (key, value) tokens with v = W k + noise. Two recurrent states read the same stream:
//   linear attention (Hebbian):  S <- S + v k^T              (shown scaled by d/t)
//   (gated) DeltaNet:            S <- a S (I - b k k^T) + b v k^T   == one SGD step on 1/2|S k - v|^2 after decay a
// All computed here in JS; nothing precomputed.
(function () {
  const root = document.getElementById("delta-widget");
  if (!root) return;
  const DK = 12, DV = 12;
  const PAPER = "#fcfbf8", INK = "#1d1d1f", MUTED = "#6b6b70", RULE = "#e6e2da";
  const COL = { linear: "#2a78d6", delta: "#1baf7a", target: "#1d1d1f" };

  // ------------------------------------------------------------------ utilities
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  let rand = rng(1);
  function gauss() { let u = 0, v = 0; while (u === 0) u = rand(); v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const lerp = (a, b, t) => a + (b - a) * t;
  function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
  const DIV = ["#184f95", "#3987e5", "#cde2fb", "#f3f1ec", "#fbd3c4", "#e34948", "#9c2224"].map(hex);
  function divColor(x) { // x in [-1, 1]
    const u = Math.max(0, Math.min(1, (x + 1) / 2)) * (DIV.length - 1);
    const i = Math.min(DIV.length - 2, Math.floor(u)), f = u - i;
    const c = [0, 1, 2].map(j => Math.round(lerp(DIV[i][j], DIV[i + 1][j], f)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // glyph targets: a Delta and a Nabla, softly anti-aliased, plus mean removal so the maps are signed
  function glyph(up) {
    const W = new Float64Array(DV * DK);
    const cx = (DK - 1) / 2;
    for (let i = 0; i < DV; i++) for (let j = 0; j < DK; j++) {
      const r = up ? i : DV - 1 - i;                 // row measured from the apex
      const half = 0.5 + r * 0.5;                    // half-width grows with depth
      const dx = Math.abs(j - cx);
      const edge = Math.abs(dx - half), base = Math.abs(r - (DV - 1.5));
      let v = Math.exp(-(edge * edge) / 0.8);
      if (dx <= half + 0.5) v = Math.max(v, Math.exp(-(base * base) / 0.8));
      if (dx > half + 1.6) v *= 0.15;
      W[i * DK + j] = v;
    }
    let m = 0; for (const v of W) m += v; m /= W.length;
    for (let k = 0; k < W.length; k++) W[k] = 1.6 * (W[k] - m);
    return W;
  }
  const TARGETS = [glyph(true), glyph(false)];

  // ------------------------------------------------------------------ state
  const st = { t: 0, target: 0, Sd: new Float64Array(DV * DK), Sl: new Float64Array(DV * DK), errD: [], errL: [], switches: [], k: new Float64Array(DK), v: new Float64Array(DV), playing: false };
  const ctl = {};

  function reset() {
    rand = rng(1 + Math.floor(Math.random() * 1e6));
    st.t = 0; st.target = 0; st.Sd.fill(0); st.Sl.fill(0); st.errD = []; st.errL = []; st.switches = [];
    draw();
  }

  function relErr(S, W) { let n = 0, d = 0; for (let i = 0; i < S.length; i++) { n += (S[i] - W[i]) ** 2; d += W[i] ** 2; } return Math.sqrt(n / d); }

  function step() {
    const beta = +ctl.beta.value, rho = +ctl.rho.value, alpha = +ctl.alpha.value, noise = +ctl.noise.value;
    const W = TARGETS[st.target];
    // AR(1)-correlated key, L2-normalised (as in DeltaNet)
    let prev = gauss(); st.k[0] = prev;
    for (let j = 1; j < DK; j++) { prev = rho * prev + Math.sqrt(1 - rho * rho) * gauss(); st.k[j] = prev; }
    let nk = 0; for (let j = 0; j < DK; j++) nk += st.k[j] ** 2; nk = Math.sqrt(nk);
    for (let j = 0; j < DK; j++) st.k[j] /= nk;
    for (let i = 0; i < DV; i++) { let s = 0; for (let j = 0; j < DK; j++) s += W[i * DK + j] * st.k[j]; st.v[i] = s + noise * gauss() / Math.sqrt(DK); }
    // gated delta rule: S <- alpha S ; pred = S k ; S <- S + beta (v - pred) k^T
    for (let i = 0; i < st.Sd.length; i++) st.Sd[i] *= alpha;
    for (let i = 0; i < DV; i++) {
      let pred = 0; for (let j = 0; j < DK; j++) pred += st.Sd[i * DK + j] * st.k[j];
      const e = beta * (st.v[i] - pred);
      for (let j = 0; j < DK; j++) st.Sd[i * DK + j] += e * st.k[j];
    }
    // Hebbian linear attention: S <- S + v k^T
    for (let i = 0; i < DV; i++) for (let j = 0; j < DK; j++) st.Sl[i * DK + j] += st.v[i] * st.k[j];
    st.t++;
    const scale = DK / st.t, Sls = st.Sl.map(x => x * scale);
    st.errD.push(relErr(st.Sd, W)); st.errL.push(relErr(Sls, W));
  }

  // ------------------------------------------------------------------ drawing
  function heat(cv, S, scale, label, sub, color) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = "#ffffff"; g.fillRect(0, 0, w, h);
    const top = 38, size = Math.min(w, h - top - 6), cell = size / DK, x0 = (w - size) / 2;
    for (let i = 0; i < DV; i++) for (let j = 0; j < DK; j++) {
      g.fillStyle = divColor((S[i * DK + j] * scale) / 1.6);
      g.fillRect(x0 + j * cell + 0.5, top + i * cell + 0.5, cell - 1, cell - 1);
    }
    g.font = "600 13px Inter, Helvetica Neue, Arial, sans-serif"; g.fillStyle = color; g.textAlign = "left";
    g.fillText(label, x0, 15);
    g.font = "12px Inter, Helvetica Neue, Arial, sans-serif"; g.fillStyle = MUTED;
    g.fillText(sub, x0, 31);
  }

  function plot(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = "#ffffff"; g.fillRect(0, 0, w, h);
    const L = 46, R = 150, T = 14, B = 30, pw = w - L - R, ph = h - T - B;
    const tmax = Math.max(100, st.t);
    const ylo = Math.log10(0.005), yhi = Math.log10(3);
    const X = t => L + (t / tmax) * pw, Y = e => T + ph * (1 - (Math.log10(Math.max(0.005, Math.min(3, e))) - ylo) / (yhi - ylo));
    g.strokeStyle = RULE; g.lineWidth = 1; g.font = "11px Inter, Helvetica Neue, Arial, sans-serif"; g.fillStyle = MUTED; g.textAlign = "right";
    for (const e of [0.01, 0.03, 0.1, 0.3, 1, 3]) { const y = Y(e); g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke(); g.fillText(e, L - 6, y + 4); }
    g.textAlign = "center";
    for (let t = 0; t <= tmax; t += tmax > 400 ? 200 : 50) g.fillText(t, X(t), T + ph + 16);
    g.fillText("tokens seen", L + pw / 2, h - 2);
    g.save(); g.translate(12, T + ph / 2); g.rotate(-Math.PI / 2); g.fillText("‖S − W‖ / ‖W‖", 0, 0); g.restore();
    for (const s of st.switches) { g.strokeStyle = "#c9c4ba"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(s), T); g.lineTo(X(s), T + ph); g.stroke(); g.setLineDash([]); }
    const line = (arr, col) => {
      if (!arr.length) return;
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
      arr.forEach((e, i) => { const x = X(i + 1), y = Y(e); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
      const e = arr[arr.length - 1];
      g.fillStyle = col; g.textAlign = "left"; g.font = "600 12px Inter, Helvetica Neue, Arial, sans-serif";
      return Y(e);
    };
    const yl = line(st.errL, COL.linear), yd = line(st.errD, COL.delta);
    if (st.t) {
      let a = yl, b = yd; if (Math.abs(a - b) < 16) { if (a < b) { a -= 8; b += 8; } else { a += 8; b -= 8; } }
      g.fillStyle = COL.linear; g.fillText(`linear attn  ${st.errL[st.t - 1].toFixed(2)}`, L + pw + 8, a + 4);
      g.fillStyle = COL.delta; g.fillText(`DeltaNet  ${st.errD[st.t - 1].toFixed(2)}`, L + pw + 8, b + 4);
    }
  }

  function draw() {
    const W = TARGETS[st.target];
    heat(ctl.cvW, W, 1, "hidden map W", `target ${st.target ? "∇" : "Δ"} · v = W k + noise`, INK);
    heat(ctl.cvL, st.Sl, st.t ? DK / st.t : 0, "linear attention state", "S += v kᵀ   (shown × d/t)", COL.linear);
    heat(ctl.cvD, st.Sd, 1, "DeltaNet state", "S += β (v − S k) kᵀ", COL.delta);
    plot(ctl.cvE);
    ctl.tlabel.textContent = `t = ${st.t}`;
  }

  // ------------------------------------------------------------------ DOM
  root.innerHTML = `
  <div class="dw-head">
    <button data-a="play">▶ play</button><button data-a="step">step</button><button data-a="switch">switch target</button><button data-a="reset">reset</button>
    <span class="dw-t"></span>
  </div>
  <div class="dw-sliders">
    <label>write strength β <input type="range" min="0.02" max="2.2" step="0.01" value="0.5" data-s="beta"><output></output></label>
    <label>key correlation ρ <input type="range" min="0" max="0.95" step="0.01" value="0.6" data-s="rho"><output></output></label>
    <label>decay α (gate) <input type="range" min="0.9" max="1" step="0.001" value="1" data-s="alpha"><output></output></label>
    <label>value noise <input type="range" min="0" max="1" step="0.01" value="0.05" data-s="noise"><output></output></label>
  </div>
  <div class="dw-maps"><canvas class="dw-W"></canvas><canvas class="dw-L"></canvas><canvas class="dw-D"></canvas></div>
  <canvas class="dw-E"></canvas>`;
  ctl.cvW = root.querySelector(".dw-W"); ctl.cvL = root.querySelector(".dw-L"); ctl.cvD = root.querySelector(".dw-D"); ctl.cvE = root.querySelector(".dw-E");
  ctl.tlabel = root.querySelector(".dw-t");
  root.querySelectorAll("input[data-s]").forEach(inp => {
    ctl[inp.dataset.s] = inp;
    const out = inp.nextElementSibling, upd = () => { out.textContent = (+inp.value).toFixed(inp.dataset.s === "alpha" ? 3 : 2); };
    inp.addEventListener("input", upd); upd();
  });
  const playBtn = root.querySelector('[data-a="play"]');
  let last = 0;
  function loop(ts) {
    if (!st.playing) return;
    if (ts - last > 40) { for (let i = 0; i < 2; i++) step(); draw(); last = ts; }
    if (st.t >= 1200) { st.playing = false; playBtn.textContent = "▶ play"; return; }
    requestAnimationFrame(loop);
  }
  root.addEventListener("click", e => {
    const a = e.target.dataset && e.target.dataset.a; if (!a) return;
    if (a === "play") { st.playing = !st.playing; playBtn.textContent = st.playing ? "❚❚ pause" : "▶ play"; if (st.playing) requestAnimationFrame(loop); }
    if (a === "step") { step(); draw(); }
    if (a === "switch") { st.target = 1 - st.target; st.switches.push(st.t); draw(); }
    if (a === "reset") reset();
  });
  window.addEventListener("resize", draw);
  reset();

  if (location.hash === "#selftest") {
    ctl.beta.value = 0.5; ctl.rho.value = 0.6; ctl.alpha.value = 1; ctl.noise.value = 0.05;
    for (let i = 0; i < 400; i++) step();
    const d = st.errD[st.t - 1], l = st.errL[st.t - 1];
    console.log(`SELFTEST delta_widget t=${st.t} deltaErr=${d.toFixed(3)} linearErr=${l.toFixed(3)} ${d < 0.2 && l > 0.5 ? "PASS" : "FAIL"}`);
    st.target = 1; st.switches.push(st.t); ctl.alpha.value = 0.98;
    for (let i = 0; i < 300; i++) step();
    console.log(`SELFTEST delta_widget after switch (alpha=.98) deltaErr=${st.errD[st.t - 1].toFixed(3)} linearErr=${st.errL[st.t - 1].toFixed(3)}`);
    draw();
  }
})();
