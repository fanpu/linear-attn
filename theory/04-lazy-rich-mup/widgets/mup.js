// Widget: muTransfer explorer. Pick a sweep (width or depth) and a parameterization; hover a curve or a chip to see
// that model's training curves at every learning rate. Data: widgets/data_mup.js (window.MUP_DATA).
(function () {
  const D = window.MUP_DATA, U = window.LR, C = U.C;
  const root = document.getElementById("mup-widget");
  if (!D || !root) return;
  const SWEEPS = {
    width: { label: "width", params: [["sp", "SP"], ["mup", "μP"]], ramp: C.widths, unit: v => `d = ${v.toLocaleString()}` },
    depth: { label: "depth", params: [["plain", "μP, no depth scaling"], ["dmup", "μP + depth-μP"]], ramp: C.depths, unit: v => `L = ${v}` },
  };
  const LRRAMP = ["#fbd3bd", "#f5ab84", "#ec8250", "#e0561f", "#b8410f", "#8c2f08", "#5f1f04", "#3d1402"];
  const state = { sweep: "width", param: "sp", hi: null };

  root.innerHTML = `
    <div class="mx-controls">
      <span>sweep:</span><span id="mx-sweep"></span>
      <span style="margin-left:.8rem">parameterization:</span><span id="mx-param"></span>
    </div>
    <div class="mx-row">
      <div><canvas id="mx-main"></canvas><div class="mx-read" id="mx-chips" style="max-width:520px"></div></div>
      <div><canvas id="mx-train"></canvas><div class="mx-read" id="mx-read"></div></div>
    </div>`;
  const $ = id => root.querySelector("#" + id);
  const MW = 520, MH = 340, TW = 340, TH = 340;
  const mctx = U.setupCanvas($("mx-main"), MW, MH);
  const tctx = U.setupCanvas($("mx-train"), TW, TH);
  const box = { x: 56, y: 30, w: MW - 76, h: MH - 78 };
  const YR = D.ylim;
  let axMain = null;

  function buttons() {
    $("mx-sweep").innerHTML = Object.keys(SWEEPS).map(k => `<button data-s="${k}" class="${k === state.sweep ? "on" : ""}">${SWEEPS[k].label}</button>`).join(" ");
    $("mx-param").innerHTML = SWEEPS[state.sweep].params.map(([k, l]) => `<button data-p="${k}" class="${k === state.param ? "on" : ""}">${l}</button>`).join(" ");
    root.querySelectorAll("[data-s]").forEach(b => b.onclick = () => { state.sweep = b.dataset.s; state.param = SWEEPS[state.sweep].params[1][0]; state.hi = null; draw(); });
    root.querySelectorAll("[data-p]").forEach(b => b.onclick = () => { state.param = b.dataset.p; draw(); });
  }

  const curves = () => D[state.sweep][state.param];

  function drawMain() {
    const cs = curves(), ramp = SWEEPS[state.sweep].ramp;
    mctx.clearRect(0, 0, MW, MH);
    const ax = U.Axes(mctx, box, [D.log2lr[0] - 0.4, D.log2lr[1] + 0.4], YR, {});
    axMain = ax;
    const xt = []; for (let k = Math.ceil(D.log2lr[0]); k <= D.log2lr[1]; k++) xt.push(k);
    ax.draw(xt, D.yticks, "log₂ of the base Adam learning rate η", "validation loss", v => String(v), v => v.toFixed(1));
    cs.forEach((c, i) => {
      const dim = state.hi != null && state.hi !== i;
      const xs = c.lrs.map(Math.log2), ys = c.val.map(v => v == null ? NaN : Math.min(v, YR[1] + 0.2));
      ax.line(xs, ys, ramp[i], dim ? 1.4 : 2.6, dim ? 0.3 : 1);
      xs.forEach((x, k) => {
        if (c.val[k] == null) {  // diverged
          mctx.save(); mctx.strokeStyle = ramp[i]; mctx.globalAlpha = dim ? .3 : 1; mctx.lineWidth = 2;
          const px = ax.X(x), py = box.y + 4; mctx.beginPath(); mctx.moveTo(px - 4, py - 4); mctx.lineTo(px + 4, py + 4); mctx.moveTo(px + 4, py - 4); mctx.lineTo(px - 4, py + 4); mctx.stroke(); mctx.restore();
        } else if (c.val[k] <= YR[1]) {
          mctx.save(); mctx.globalAlpha = dim ? .3 : 1; ax.dot(x, c.val[k], 3.2, ramp[i], C.paper); mctx.restore();
        }
      });
      if (c.opt != null) {
        mctx.save(); mctx.globalAlpha = dim ? .3 : 1;
        star(ax.X(c.opt), ax.Y(c.optval), 8, ramp[i]);
        mctx.setLineDash([2, 3]); mctx.strokeStyle = ramp[i]; mctx.lineWidth = 1.2;
        mctx.beginPath(); mctx.moveTo(ax.X(c.opt), ax.Y(c.optval) + 8); mctx.lineTo(ax.X(c.opt), box.y + box.h); mctx.stroke();
        mctx.restore();
      }
    });
    const lab = SWEEPS[state.sweep].params.find(p => p[0] === state.param)[1];
    mctx.save(); mctx.font = "600 13px " + C.font; mctx.fillStyle = C.ink; mctx.fillText(lab, box.x, 18); mctx.restore();
    // drift readout
    const ok = cs.filter(c => c.opt != null);
    if (ok.length >= 2) {
      const drift = Math.pow(2, ok[ok.length - 1].opt - ok[0].opt);
      mctx.save(); mctx.font = "12px " + C.font; mctx.fillStyle = C.muted; mctx.textAlign = "right";
      mctx.fillText(`best η moves ×${drift < 1 ? (1 / drift).toFixed(1) + " smaller" : drift.toFixed(1) + " larger"} from ${SWEEPS[state.sweep].unit(ok[0].size)} to ${SWEEPS[state.sweep].unit(ok[ok.length - 1].size)}`, box.x + box.w, 18);
      mctx.restore();
    }
    $("mx-chips").innerHTML = cs.map((c, i) => `<span class="mx-chip" data-i="${i}" style="cursor:pointer;margin-right:.7rem;${state.hi === i ? "font-weight:600" : ""}"><span class="mx-key" style="background:${ramp[i]}"></span>${SWEEPS[state.sweep].unit(c.size)}</span>`).join("") + '<br><span style="color:#8b8984">hover a curve or a label · ★ = parabola-fit optimum · × = diverged</span>';
    root.querySelectorAll(".mx-chip").forEach(el => { el.onmouseenter = () => { state.hi = +el.dataset.i; draw(); }; });
  }

  function star(x, y, r, fill) {
    mctx.beginPath();
    for (let k = 0; k < 10; k++) { const rr = k % 2 ? r * 0.45 : r, t = -Math.PI / 2 + k * Math.PI / 5; mctx.lineTo(x + rr * Math.cos(t), y + rr * Math.sin(t)); }
    mctx.closePath(); mctx.fillStyle = fill; mctx.fill(); mctx.strokeStyle = C.ink; mctx.lineWidth = 0.8; mctx.stroke();
  }

  function drawTrain() {
    tctx.clearRect(0, 0, TW, TH);
    const cs = curves(), i = state.hi == null ? cs.length - 1 : state.hi, c = cs[i];
    const ax = U.Axes(tctx, { x: 48, y: 30, w: TW - 64, h: TH - 78 }, [0, D.steps], [2.4, 7], {});
    ax.draw([0, D.steps / 2, D.steps], [3, 4, 5, 6, 7], "training step", "train loss", v => v.toLocaleString(), v => String(v));
    tctx.save(); tctx.font = "600 13px " + C.font; tctx.fillStyle = C.ink;
    tctx.fillText(`${SWEEPS[state.sweep].unit(c.size)}: training curves by η`, 8, 18); tctx.restore();
    c.hist.forEach((h, k) => {
      if (!h) return;
      const best = c.opt != null && Math.abs(Math.log2(c.lrs[k]) - c.opt) <= 0.5;
      ax.line(h[0], h[1], LRRAMP[k % LRRAMP.length], best ? 2.6 : 1.4, best ? 1 : .8);
      const last = h[1][h[1].length - 1];
      tctx.save(); tctx.font = "10px " + C.font; tctx.fillStyle = C.muted;
      if (k === 0 || k === c.hist.length - 1) tctx.fillText(`2^${Math.log2(c.lrs[k])}`, ax.X(h[0][h[0].length - 1]) - 26, ax.Y(Math.min(last, 6.9)) - 4);
      tctx.restore();
    });
    const div = c.val.map((v, k) => v == null ? `2^${Math.log2(c.lrs[k])}` : null).filter(Boolean);
    $("mx-read").innerHTML = `light → dark: η = 2^${Math.log2(c.lrs[0])} … 2^${Math.log2(c.lrs[c.lrs.length - 1])}` +
      (c.opt != null ? `<br>best η ≈ 2^${c.opt.toFixed(2)} = ${Math.pow(2, c.opt).toExponential(1)}, loss ${c.optval.toFixed(3)}` : "") +
      (div.length ? `<br>diverged at η = ${div.join(", ")}` : "");
  }

  function draw() { buttons(); drawMain(); drawTrain(); }

  $("mx-main").addEventListener("mousemove", e => {
    if (!axMain) return;
    const rect = e.target.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bd = 18;
    curves().forEach((c, i) => c.lrs.forEach((lr, k) => {
      if (c.val[k] == null) return;
      const dd = Math.hypot(axMain.X(Math.log2(lr)) - mx, axMain.Y(Math.min(c.val[k], YR[1])) - my);
      if (dd < bd) { bd = dd; best = i; }
    }));
    if (best != null && best !== state.hi) { state.hi = best; draw(); }
  });
  draw();

  if (location.hash.indexOf("selftest") >= 0) {
    try {
      const rep = [];
      for (const sw of Object.keys(SWEEPS)) for (const [p] of SWEEPS[sw].params) {
        state.sweep = sw; state.param = p; D[sw][p].forEach((c, i) => { state.hi = i; draw(); });
        D[sw][p].forEach(c => {
          // recompute the parabola optimum from the grid and compare with the exported one
          const xs = c.lrs.map(Math.log2), ys = c.val;
          let bi = -1; ys.forEach((v, k) => { if (v != null && (bi < 0 || v < ys[bi])) bi = k; });
          if (bi > 0 && bi < ys.length - 1 && ys[bi - 1] != null && ys[bi + 1] != null && c.opt != null) {
            const [x0, x1, x2] = [xs[bi - 1], xs[bi], xs[bi + 1]], [y0, y1, y2] = [ys[bi - 1], ys[bi], ys[bi + 1]];
            const den = (x0 - x1) * (x0 - x2) * (x1 - x2);
            const A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den, B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den;
            const xm = A > 0 ? Math.min(x2, Math.max(x0, -B / (2 * A))) : x1;
            if (Math.abs(xm - c.opt) > 1e-3) console.error("mup selftest FAILED", sw, p, c.size, xm, c.opt);
          }
        });
        rep.push(`${sw}/${p}: ${D[sw][p].length} curves`);
      }
      state.sweep = "width"; state.param = "sp"; state.hi = null; draw();
      console.log("mup selftest ok: " + rep.join("; "));
    } catch (e) { console.error("mup selftest exception", e); }
  }
})();
