// Widget: the closed-form risk of ridge / ridgeless regression, live (isotropic features, r^2 = 1, sigma^2 = 1/SNR).
(function () {
  const root = document.getElementById("risk-widget");
  if (!root) return;
  const { C, FONT, setupCanvas, Axes, fmtNum, risk, mpDensity } = window.DD;

  root.innerHTML = `
    <div class="dd-controls">
      <label>γ = p/n <input type="range" id="rw-g" min="-1" max="1" step="0.002" value="-0.05"><span id="rw-gv" class="dd-val"></span></label>
      <label>SNR <input type="range" id="rw-snr" min="-0.6" max="1.4" step="0.01" value="0.6"><span id="rw-snrv" class="dd-val"></span></label>
      <label>ridge λ <input type="range" id="rw-lam" min="-4.2" max="1" step="0.01" value="-4.2"><span id="rw-lamv" class="dd-val"></span></label>
      <label class="dd-check"><input type="checkbox" id="rw-opt"> use optimal λ* = γ / SNR</label>
    </div>
    <div class="dd-row">
      <canvas id="rw-c1"></canvas>
      <div>
        <canvas id="rw-c2"></canvas>
        <div class="dd-readout" id="rw-read"></div>
      </div>
    </div>
    <div class="dd-hint">Drag γ through 1 with λ = 0 and watch the variance (red) explode. Then raise λ, or tick "optimal". The right panel is the Marchenko–Pastur eigenvalue density at the current γ; eigenvalues left of the dashed λ line are the ones ridge "fills in".</div>`;

  const $ = id => root.querySelector("#" + id);
  const W1 = Math.min(640, (root.clientWidth || 900) - 320), H1 = 330, W2 = 290, H2 = 200;
  const ctx1 = setupCanvas($("rw-c1"), Math.max(420, W1), H1);
  const ctx2 = setupCanvas($("rw-c2"), W2, H2);

  function state() {
    const g = Math.pow(10, +$("rw-g").value);
    const snr = Math.pow(10, +$("rw-snr").value);
    const opt = $("rw-opt").checked;
    const lr = +$("rw-lam").value;
    let lam = lr <= -4.19 ? 0 : Math.pow(10, lr);
    return { g, snr, opt, lam, s2: 1 / snr };
  }
  const lamOf = (st, g) => (st.opt ? g / st.snr : st.lam);

  function draw() {
    const st = state();
    $("rw-gv").textContent = st.g.toFixed(2);
    $("rw-snrv").textContent = fmtNum(st.snr, 1);
    $("rw-lamv").textContent = st.opt ? `${fmtNum(st.g / st.snr, 1)} (opt)` : (st.lam === 0 ? "0" : fmtNum(st.lam, 1));
    $("rw-lam").disabled = st.opt;

    // ---- risk vs gamma ----
    const w = parseFloat($("rw-c1").style.width);
    ctx1.clearRect(0, 0, w, H1);
    const ax = Axes(ctx1, { x: 52, y: 14, w: w - 70, h: H1 - 58 }, [0.1, 10], [0, 2], { xlog: true });
    ax.frame([0.1, 0.3, 1, 3, 10], [0, 0.5, 1, 1.5, 2], "γ = p / n", "test risk  (null risk = 1)");
    ctx1.save(); ctx1.strokeStyle = C.gold; ctx1.lineWidth = 1.2; ctx1.beginPath(); ctx1.moveTo(ax.X(1), ax.box.y); ctx1.lineTo(ax.X(1), ax.box.y + ax.box.h); ctx1.stroke(); ctx1.restore();
    const gs = [], R0 = [], R = [], B = [], V = [];
    for (let i = 0; i <= 400; i++) {
      let g = Math.pow(10, -1 + 2 * i / 400);
      if (Math.abs(g - 1) < 1e-3) g = i < 200 ? 0.999 : 1.001;
      gs.push(g);
      R0.push(risk(g, 0, 1, st.s2)[0]);
      const r = risk(g, lamOf(st, g), 1, st.s2);
      R.push(r[0]); B.push(r[1]); V.push(r[2]);
    }
    // split lines at gamma = 1 where they diverge
    const seg = (arr, col, lw, al, dash) => {
      const lo = gs.map((g, i) => (g < 1 ? arr[i] : NaN)), hi = gs.map((g, i) => (g > 1 ? arr[i] : NaN));
      ax.line(gs, lo, col, lw, al, dash); ax.line(gs, hi, col, lw, al, dash);
    };
    ctx1.save(); ctx1.strokeStyle = C.muted; ctx1.setLineDash([1, 3]); ctx1.beginPath(); ctx1.moveTo(ax.box.x, ax.Y(1)); ctx1.lineTo(ax.box.x + ax.box.w, ax.Y(1)); ctx1.stroke(); ctx1.restore();
    if (st.opt || st.lam > 0) seg(R0, C.muted, 1.2, 0.8, [4, 3]);
    seg(B, C.bias, 2.2, 1); seg(V, C.vari, 2.2, 1); seg(R, C.ink, 2.4, 1);
    const cur = risk(st.g, lamOf(st, st.g), 1, st.s2);
    const clampY = v => Math.min(2.05, v);
    ax.dot(st.g, clampY(cur[1]), 5, C.bias, C.paper);
    ax.dot(st.g, clampY(cur[2]), 5, C.vari, C.paper);
    ax.dot(st.g, clampY(cur[0]), 6, C.ink, C.paper);
    ctx1.save(); ctx1.font = `12px ${FONT}`; ctx1.textBaseline = "middle";
    const key = [["total", C.ink], ["bias²", C.bias], ["variance", C.vari]];
    key.forEach(([t, col], i) => {
      const x0 = ax.box.x + 12, y0 = ax.box.y + 12 + i * 18;
      ctx1.strokeStyle = col; ctx1.lineWidth = 2.4; ctx1.beginPath(); ctx1.moveTo(x0, y0); ctx1.lineTo(x0 + 18, y0); ctx1.stroke();
      ctx1.fillStyle = C.ink; ctx1.fillText(t, x0 + 24, y0);
    });
    if (st.opt || st.lam > 0) { ctx1.fillStyle = C.muted; ctx1.fillText("dashed: ridgeless", ax.box.x + 12, ax.box.y + 12 + 3 * 18); }
    ctx1.restore();

    // ---- MP density at current gamma, log eigenvalue axis ----
    ctx2.clearRect(0, 0, W2, H2);
    const g = st.g;
    const ax2 = Axes(ctx2, { x: 12, y: 26, w: W2 - 24, h: H2 - 66 }, [1e-4, 30], [0, 1], { xlog: true });
    const ss = [], dd = [];
    let dmax = 0;
    for (let i = 0; i <= 500; i++) {
      const s = Math.pow(10, -4 + (Math.log10(30) + 4) * i / 500);
      const d = mpDensity(s, g) * s * Math.LN10 * (g > 1 ? g : 1);
      ss.push(s); dd.push(d); dmax = Math.max(dmax, d);
    }
    const dn = dd.map(v => v / (dmax * 1.08));
    ax2.band(ss, dn.map(() => 0), dn, C.indigo, 0.35);
    ax2.line(ss, dn, C.indigo, 1.6, 1);
    ax2.frame([1e-4, 1e-2, 1, 10], [], "eigenvalue of XᵀX/n (log)", null, v => (v >= 1 ? String(v) : "1e" + Math.round(Math.log10(v))));
    const lamc = lamOf(st, g);
    if (lamc > 0) {
      const px = ax2.X(Math.max(1e-4, Math.min(30, lamc)));
      ctx2.save(); ctx2.strokeStyle = C.teal; ctx2.setLineDash([4, 3]); ctx2.lineWidth = 1.5;
      ctx2.beginPath(); ctx2.moveTo(px, ax2.box.y); ctx2.lineTo(px, ax2.box.y + ax2.box.h); ctx2.stroke();
      ctx2.fillStyle = C.ink; ctx2.font = `11px ${FONT}`; ctx2.textAlign = px > W2 / 2 ? "right" : "left";
      ctx2.fillText("λ", px + (px > W2 / 2 ? -4 : 4), ax2.box.y + 10); ctx2.restore();
    }
    const edge = (1 - Math.sqrt(g)) ** 2;
    ctx2.save(); ctx2.font = `12px ${FONT}`; ctx2.fillStyle = C.ink; ctx2.textBaseline = "top";
    ctx2.fillText(`lower edge (1−√γ)² = ${fmtNum(edge, 1)}`, 12, 4);
    ctx2.restore();

    const zeros = g > 1 ? `, plus ${(100 * (1 - 1 / g)).toFixed(0)}% of eigenvalues exactly 0` : "";
    $("rw-read").innerHTML =
      `<b>total ${fmtNum(cur[0], 1)}</b> = <span style="color:${C.bias}">bias² ${fmtNum(cur[1], 1)}</span> + <span style="color:${C.vari}">variance ${fmtNum(cur[2], 1)}</span><br>` +
      `λ used: ${lamOf(st, g) === 0 ? "0 (min-norm)" : fmtNum(lamOf(st, g), 1)}${zeros}`;
  }

  root.querySelectorAll("input").forEach(el => el.addEventListener("input", draw));
  draw();

  if (location.hash.indexOf("selftest") >= 0) {
    try {
      let worst = 0;
      (window.RISK_REF || []).forEach(q => {
        const r = risk(q.g, q.lam, q.snr, 1.0);
        [[r[0], q.R], [r[1], q.B], [r[2], q.V]].forEach(([a, b]) => { worst = Math.max(worst, Math.abs(a - b) / Math.max(1e-12, Math.abs(b))); });
      });
      if (!(worst < 1e-9) || !(window.RISK_REF || []).length) console.error("risk selftest FAILED: worst rel err " + worst);
      else console.log("risk selftest ok: " + window.RISK_REF.length + " cases vs python, worst rel err " + worst.toExponential(2));
      ["rw-g", "rw-snr", "rw-lam"].forEach(id => { const el = $(id); el.value = el.max; el.dispatchEvent(new Event("input")); el.value = el.min; el.dispatchEvent(new Event("input")); });
      $("rw-opt").checked = true; $("rw-opt").dispatchEvent(new Event("input")); draw();
      console.log("risk selftest ok: slider extremes + optimal-λ path drew without exceptions");
    } catch (e) { console.error("risk selftest exception " + e); }
  }
})();
