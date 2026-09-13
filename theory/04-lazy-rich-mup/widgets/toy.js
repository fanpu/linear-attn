// Widget: slide the output scale alpha and scrub through training of the 2-D toy network.
// Data: widgets/data_toy.js (window.TOY_DATA), exported by export_widgets.py from cache/toy/widget_*.npz.
(function () {
  const D = window.TOY_DATA, U = window.LR, C = U.C;
  const root = document.getElementById("toy-widget");
  if (!D || !root) return;
  const m = D.m, R = D.R, F = D.steps.length, NA = D.alphas.length;
  const X = U.f32(D.x), Y = U.f32(D.y), n = Y.length;
  const runs = D.runs.map(r => ({ ...r, p: U.i16(r.p) }));
  // ramp from rich (small alpha, ember) to lazy (large alpha, blue) through a neutral grey
  const RAMP = ["#e0561f", "#c9794f", "#9b8f86", "#7d8ea6", "#5a7fb8", "#2f6db5"];

  root.innerHTML = `
    <div class="tw-controls">
      <label>output scale α <input type="range" id="tw-alpha" min="0" max="${NA - 1}" step="1" value="0"> <b id="tw-alpha-val" class="tw-val"></b></label>
      <button id="tw-play">❚❚ pause</button>
      <label>step <input type="range" id="tw-step" min="0" max="${F - 1}" step="1" value="0" style="width:170px"> <span id="tw-step-val" class="tw-val"></span></label>
      <label><input type="checkbox" id="tw-ghost"> show starting lines</label>
    </div>
    <div class="tw-row">
      <canvas id="tw-plane"></canvas>
      <div class="tw-side">
        <canvas id="tw-move"></canvas>
        <canvas id="tw-loss"></canvas>
        <div class="tw-read" id="tw-read"></div>
      </div>
    </div>`;
  const $ = id => root.querySelector("#" + id);
  const S = 440;
  const pctx = U.setupCanvas($("tw-plane"), S, S);
  const mctx = U.setupCanvas($("tw-move"), 330, 205);
  const lctx = U.setupCanvas($("tw-loss"), 330, 175);
  const state = { a: 0, f: 0, playing: true, ghost: false };

  function params(ai, fi) {
    const r = runs[ai], off = fi * m * 4, s = r.scale[fi];
    const w1 = new Float32Array(m), w2 = new Float32Array(m), b = new Float32Array(m), a = new Float32Array(m);
    for (let j = 0; j < m; j++) {
      w1[j] = r.p[off + 4 * j] * s[0]; w2[j] = r.p[off + 4 * j + 1] * s[0];
      b[j] = r.p[off + 4 * j + 2] * s[1]; a[j] = r.p[off + 4 * j + 3] * s[2];
    }
    return { w1, w2, b, a };
  }

  function evalF(P, alpha, x1, x2) {
    let s = 0;
    for (let j = 0; j < m; j++) { const z = P.w1[j] * x1 + P.w2[j] * x2 + P.b[j]; if (z > 0) s += P.a[j] * z; }
    return alpha * s / m;
  }

  const G = 60, grid = new Float32Array(G * G), off = document.createElement("canvas");
  off.width = G; off.height = G;
  const octx = off.getContext("2d"), img = octx.createImageData(G, G);
  const toPx = v => (v + R) / (2 * R) * S;

  function drawLines(P, alpha, strength) {
    let mean = 0;
    const wt = new Float32Array(m);
    for (let j = 0; j < m; j++) { wt[j] = Math.abs(P.a[j]) * Math.hypot(P.w1[j], P.w2[j]); mean += wt[j] / m; }
    pctx.save();
    pctx.globalCompositeOperation = "lighter";
    pctx.lineWidth = 1;
    for (let j = 0; j < m; j++) {
      const nw = Math.hypot(P.w1[j], P.w2[j]) + 1e-12, ux = P.w1[j] / nw, uy = P.w2[j] / nw, c = -P.b[j] / nw;
      if (Math.abs(c) > 1.8 * R) continue;
      const L = 2 * R;
      const x0 = c * ux - L * uy, y0 = c * uy + L * ux, x1 = c * ux + L * uy, y1 = c * uy - L * ux;
      pctx.globalAlpha = Math.min(0.75, strength * wt[j] / mean);
      pctx.strokeStyle = P.a[j] > 0 ? C.pos : C.neg;
      pctx.beginPath(); pctx.moveTo(toPx(x0), S - toPx(y0)); pctx.lineTo(toPx(x1), S - toPx(y1)); pctx.stroke();
    }
    pctx.restore();
  }

  function contour() {
    // marching squares on the G x G grid, level 0, unconnected segments
    pctx.save();
    pctx.strokeStyle = "rgba(255,255,255,0.95)"; pctx.lineWidth = 2; pctx.lineCap = "round";
    pctx.shadowColor = "rgba(255,255,255,0.6)"; pctx.shadowBlur = 6;
    pctx.beginPath();
    const px = i => i / (G - 1) * S;
    for (let i = 0; i < G - 1; i++) for (let k = 0; k < G - 1; k++) {
      const v = [grid[k * G + i], grid[k * G + i + 1], grid[(k + 1) * G + i + 1], grid[(k + 1) * G + i]];
      const P = [[i, k], [i + 1, k], [i + 1, k + 1], [i, k + 1]];
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const a = v[e], b = v[(e + 1) % 4];
        if ((a > 0) !== (b > 0)) {
          const t = a / (a - b), pa = P[e], pb = P[(e + 1) % 4];
          pts.push([px(pa[0] + t * (pb[0] - pa[0])), S - px(pa[1] + t * (pb[1] - pa[1]))]);
        }
      }
      for (let q = 0; q + 1 < pts.length; q += 2) { pctx.moveTo(pts[q][0], pts[q][1]); pctx.lineTo(pts[q + 1][0], pts[q + 1][1]); }
    }
    pctx.stroke(); pctx.restore();
  }

  function drawPlane() {
    const r = runs[state.a], alpha = D.alphas[state.a];
    const P = params(state.a, state.f);
    pctx.fillStyle = C.night; pctx.fillRect(0, 0, S, S);
    // decision-region wash
    for (let k = 0; k < G; k++) for (let i = 0; i < G; i++) {
      const x1 = -R + 2 * R * i / (G - 1), x2 = -R + 2 * R * k / (G - 1);
      const f = evalF(P, alpha, x1, x2); grid[k * G + i] = f;
      const t = Math.tanh(2.5 * f), idx = ((G - 1 - k) * G + i) * 4;
      const pc = t > 0 ? [255, 180, 84] : [88, 196, 245], s = 0.13 * Math.abs(t);
      img.data[idx] = 7 + s * pc[0]; img.data[idx + 1] = 9 + s * pc[1]; img.data[idx + 2] = 18 + s * pc[2]; img.data[idx + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    pctx.save(); pctx.imageSmoothingEnabled = true; pctx.drawImage(off, 0, 0, S, S); pctx.restore();
    if (state.ghost) {
      pctx.save(); pctx.globalAlpha = 1; drawLinesGhost(params(state.a, 0)); pctx.restore();
    }
    drawLines(P, alpha, 0.075);
    contour();
    const rg = pctx.createRadialGradient(S / 2, S / 2, S * 0.40, S / 2, S / 2, S * 0.5 * 1.02);
    rg.addColorStop(0, "rgba(7,9,18,0)"); rg.addColorStop(1, "rgba(7,9,18,1)");
    pctx.fillStyle = rg; pctx.fillRect(0, 0, S, S);
    pctx.save();
    for (let i = 0; i < n; i++) {
      pctx.fillStyle = Y[i] > 0 ? "rgba(255,196,120,0.9)" : "rgba(140,214,250,0.9)";
      pctx.beginPath(); pctx.arc(toPx(X[2 * i]), S - toPx(X[2 * i + 1]), 1.6, 0, 2 * Math.PI); pctx.fill();
    }
    pctx.restore();
    pctx.save();
    pctx.font = "600 13px " + C.font; pctx.fillStyle = C.nightInk;
    const regime = alpha <= 2 ? "rich" : alpha >= 128 ? "lazy" : "in between";
    pctx.fillText(`α = ${alpha}  ·  ${regime}`, 12, 22);
    pctx.font = "12px " + C.font; pctx.fillStyle = C.nightMuted;
    pctx.fillText(`step ${D.steps[state.f].toLocaleString()}`, 12, 40);
    pctx.restore();
  }

  function drawLinesGhost(P0) {
    pctx.save(); pctx.lineWidth = 1; pctx.strokeStyle = "rgba(220,220,235,0.07)";
    pctx.beginPath();
    for (let j = 0; j < m; j++) {
      const nw = Math.hypot(P0.w1[j], P0.w2[j]) + 1e-12, ux = P0.w1[j] / nw, uy = P0.w2[j] / nw, c = -P0.b[j] / nw, L = 2 * R;
      pctx.moveTo(toPx(c * ux - L * uy), S - toPx(c * uy + L * ux)); pctx.lineTo(toPx(c * ux + L * uy), S - toPx(c * uy - L * ux));
    }
    pctx.stroke(); pctx.restore();
  }

  function drawCharts() {
    const st = D.steps.map(s => Math.max(s, 1));
    for (const [ctx, key, H, yr, ylog, title, yt, yfmt] of [
      [mctx, "move", 205, [1e-4, 100], true, "weights moved  ‖θₜ − θ₀‖ / ‖θ₀‖", [1e-3, 1e-1, 10], v => (v >= 1 ? (100 * v).toLocaleString() : +(100 * v).toPrecision(2)) + "%"],
      [lctx, "loss", 175, [0, 0.52], false, "train loss", [0, 0.25, 0.5], null]]) {
      ctx.clearRect(0, 0, 330, H);
      const ax = U.Axes(ctx, { x: 58, y: 24, w: 258, h: H - 64 }, [1, 40000], yr, { xlog: true, ylog });
      ax.draw([1, 100, 10000], yt, key === "loss" ? "gradient step" : "", "", v => U.fmt(v), yfmt);
      ctx.save(); ctx.font = "600 12px " + C.font; ctx.fillStyle = C.ink; ctx.fillText(title, 8, 13); ctx.restore();
      runs.forEach((r, ai) => { if (ai !== state.a) ax.line(st, r[key], RAMP[ai], 1.2, 0.45); });
      ax.line(st, runs[state.a][key], RAMP[state.a], 2.6, 1);
      const xs = st[state.f];
      ctx.save(); ctx.strokeStyle = C.muted; ctx.globalAlpha = .5; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(ax.X(xs), 24); ctx.lineTo(ax.X(xs), H - 40); ctx.stroke(); ctx.restore();
      ax.dot(xs, Math.max(runs[state.a][key][state.f], ylog ? 1e-4 : 0), 4, RAMP[state.a], "#fff");
    }
    // alpha legend under the loss chart
    const r = runs[state.a];
    $("tw-read").innerHTML = D.alphas.map((a, i) =>
      `<span class="tw-key" style="background:${RAMP[i]};opacity:${i === state.a ? 1 : .45}"></span>${a}`).join(" &nbsp;") +
      `<br>at this step: weights moved <b>${(100 * r.move[state.f]).toPrecision(3)}%</b>, train accuracy <b>${(100 * r.acc[state.f]).toFixed(1)}%</b>`;
  }

  function draw() {
    $("tw-alpha-val").textContent = D.alphas[state.a];
    $("tw-step-val").textContent = D.steps[state.f].toLocaleString();
    $("tw-step").value = state.f;
    drawPlane(); drawCharts();
  }

  $("tw-alpha").addEventListener("input", e => { state.a = +e.target.value; draw(); });
  $("tw-step").addEventListener("input", e => { state.f = +e.target.value; state.playing = false; $("tw-play").textContent = "▶ play"; draw(); });
  $("tw-ghost").addEventListener("change", e => { state.ghost = e.target.checked; draw(); });
  $("tw-play").addEventListener("click", () => {
    state.playing = !state.playing; $("tw-play").textContent = state.playing ? "❚❚ pause" : "▶ play";
    if (state.playing && state.f === F - 1) state.f = 0;
  });
  let last = 0, hold = 0;
  function tick(ts) {
    if (state.playing && ts - last > 70) {
      last = ts;
      if (state.f === F - 1) { if (++hold > 25) { state.f = 0; hold = 0; } }
      else state.f++;
      draw();
    }
    requestAnimationFrame(tick);
  }
  draw();
  requestAnimationFrame(tick);

  if (location.hash.indexOf("selftest") >= 0) {
    try {
      const rep = [];
      runs.forEach((r, ai) => {
        const P = params(ai, F - 1);
        let correct = 0;
        for (let i = 0; i < n; i++) if ((evalF(P, D.alphas[ai], X[2 * i], X[2 * i + 1]) > 0) === (Y[i] > 0)) correct++;
        const err = Math.abs(correct / n - r.acc[F - 1]);
        rep.push(`α=${D.alphas[ai]} acc js ${(correct / n).toFixed(3)} vs py ${r.acc[F - 1].toFixed(3)}`);
        if (err > 0.01) console.error("toy selftest FAILED", D.alphas[ai], err);
      });
      const P0 = params(0, 0), P1 = params(NA - 1, 0);
      let d = 0; for (let j = 0; j < m; j++) d = Math.max(d, Math.abs(P0.b[j] - P1.b[j]));
      if (d > 1e-3) console.error("toy selftest FAILED: inits differ", d);
      state.a = NA - 1; state.f = F - 1; state.ghost = true; draw(); state.a = 0; state.f = 0; state.ghost = false; draw();
      console.log("toy selftest ok: " + rep.join("; ") + `; init max diff ${d.toExponential(1)}`);
    } catch (e) { console.error("toy selftest exception", e); }
  }
})();
