// Shared helpers for the double-descent widgets (plain JS, no dependencies).
(function () {
  const C = {
    paper: "#fcfbf8", ink: "#1f1d24", muted: "#6e6a75", rule: "#e7e2dc",
    bias: "#2f6db5", vari: "#d1495b", gold: "#e09f3e", indigo: "#3b2f8f", teal: "#146c6c",
    night: "#0e0b12", nightInk: "#f2ede6", nightMuted: "#8f8898", nightRule: "#2a2431",
    biasD: "#6aa8f0", varD: "#ff6b7d", goldD: "#ffc45e", data: "#fff6e8",
    heat: ["#5b6cf0", "#9d5cf0", "#ff5d8f", "#ff9a5a", "#ffe6a8"],
  };
  const FONT = "Inter, 'Helvetica Neue', Arial, sans-serif";

  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  const HEAT = C.heat.map(hexToRgb);
  function heat(t) {                       // t in [0,1] -> css colour
    t = Math.max(0, Math.min(1, t)) * (HEAT.length - 1);
    const i = Math.min(HEAT.length - 2, Math.floor(t)), f = t - i;
    const c = HEAT[i].map((v, k) => Math.round(v + f * (HEAT[i + 1][k] - v)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // data -> pixel mapping with optional log axes
  function Axes(ctx, box, xr, yr, opts) {
    opts = opts || {};
    const tx = v => opts.xlog ? Math.log10(Math.max(v, 1e-300)) : v;
    const ty = v => opts.ylog ? Math.log10(Math.max(v, 1e-300)) : v;
    const X = x => box.x + (tx(x) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0])) * box.w;
    const Y = y => box.y + box.h - (ty(y) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0])) * box.h;
    const Xinv = px => { const u = tx(xr[0]) + (px - box.x) / box.w * (tx(xr[1]) - tx(xr[0])); return opts.xlog ? Math.pow(10, u) : u; };
    const Yinv = py => { const u = ty(yr[0]) + (box.y + box.h - py) / box.h * (ty(yr[1]) - ty(yr[0])); return opts.ylog ? Math.pow(10, u) : u; };
    const th = opts.dark ? { ink: C.nightInk, muted: C.nightMuted, rule: C.nightRule } : { ink: C.ink, muted: C.muted, rule: C.rule };
    function frame(xticks, yticks, xlabel, ylabel, fmtx, fmty) {
      fmtx = fmtx || fmt; fmty = fmty || fmt;
      ctx.save();
      ctx.font = `11px ${FONT}`; ctx.fillStyle = th.muted; ctx.strokeStyle = th.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(box.x, box.y + box.h + 0.5); ctx.lineTo(box.x + box.w, box.y + box.h + 0.5); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      (xticks || []).forEach(t => ctx.fillText(fmtx(t), X(t), box.y + box.h + 5));
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      (yticks || []).forEach(t => {
        ctx.fillText(fmty(t), box.x - 6, Y(t));
        ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(box.x, Math.round(Y(t)) + 0.5); ctx.lineTo(box.x + box.w, Math.round(Y(t)) + 0.5); ctx.stroke(); ctx.globalAlpha = 1;
      });
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      if (xlabel) ctx.fillText(xlabel, box.x + box.w / 2, box.y + box.h + 21);
      if (ylabel) { ctx.save(); ctx.translate(box.x - 42, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore(); }
      ctx.restore();
    }
    function line(xs, ys, color, width, alpha, dash) {
      ctx.save(); ctx.beginPath(); ctx.rect(box.x - 1, box.y - 6, box.w + 2, box.h + 7); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.globalAlpha = alpha == null ? 1 : alpha;
      ctx.lineJoin = "round"; ctx.lineCap = "round"; if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); let on = false;
      for (let i = 0; i < xs.length; i++) {
        const yv = ys[i];
        if (!isFinite(yv)) { on = false; continue; }
        const px = X(xs[i]), py = Math.max(box.y - 20, Math.min(box.y + box.h + 20, Y(yv)));
        if (!on) { ctx.moveTo(px, py); on = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.restore();
    }
    function band(xs, lo, hi, color, alpha) {
      ctx.save(); ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
      ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.beginPath();
      xs.forEach((x, i) => { const px = X(x), py = Y(hi[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Y(lo[i]));
      ctx.closePath(); ctx.fill(); ctx.restore();
    }
    function dot(x, y, r, color, ring) {
      ctx.save(); ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, 2 * Math.PI);
      ctx.fillStyle = color; ctx.fill(); if (ring) { ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke(); }
      ctx.restore();
    }
    return { X, Y, Xinv, Yinv, frame, line, band, dot, box };
  }

  function fmt(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e4 || a < 1e-3) { const e = Math.round(Math.log10(a)); return Math.abs(v / Math.pow(10, e) - 1) < 1e-9 ? "1e" + e : v.toExponential(0); }
    return String(+v.toPrecision(3));
  }
  function fmtNum(v, d) {
    if (!isFinite(v)) return "∞";
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(d == null ? 1 : d).replace("e+", "e");
    return v.toPrecision(d == null ? 3 : d + 2).replace(/\.?0+$/, m => (m.includes(".") ? "" : m));
  }

  function rng(seed) {
    let s = seed >>> 0;
    const u = () => { s += 0x6D2B79F5; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const g = () => { let a = 0; while (a === 0) a = u(); const b = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); };
    return { u, g };
  }

  // ---- Marchenko-Pastur / isotropic ridge closed forms (mirror dd_core.py) ----
  function mpStieltjesNeg(lam, g) {
    const b = 1 - g + lam;
    const m = (-b + Math.sqrt(b * b + 4 * g * lam)) / (2 * g * lam);
    const z = -lam;
    const dm = -(g * m * m + m) / (2 * g * z * m + z - 1 + g);
    return [m, dm];
  }
  function risk(g, lam, r2, s2) {          // returns [total, bias, variance]
    if (lam <= 0) {
      if (g === 1) return [Infinity, 0, Infinity];
      const B = g < 1 ? 0 : r2 * (1 - 1 / g);
      const V = g < 1 ? s2 * g / (1 - g) : s2 / (g - 1);
      return [B + V, B, V];
    }
    const [m, dm] = mpStieltjesNeg(lam, g);
    const B = r2 * lam * lam * dm, V = s2 * g * (m - lam * dm);
    return [B + V, B, V];
  }
  function mpDensity(s, g) {
    const a = (1 - Math.sqrt(g)) ** 2, b = (1 + Math.sqrt(g)) ** 2;
    if (s <= a || s >= b) return 0;
    return Math.sqrt((b - s) * (s - a)) / (2 * Math.PI * g * s);
  }

  // ---- one-sided Jacobi SVD: cols = array of k Float64Array(m). Returns {U (cols), S, V (k x k, column-major arrays)} ----
  function jacobiSVD(cols) {
    const k = cols.length, m = cols[0].length;
    const A = cols.map(c => Float64Array.from(c));
    const V = []; for (let i = 0; i < k; i++) { const v = new Float64Array(k); v[i] = 1; V.push(v); }
    for (let sweep = 0; sweep < 60; sweep++) {
      let rotated = false;
      for (let i = 0; i < k - 1; i++) for (let j = i + 1; j < k; j++) {
        const ai = A[i], aj = A[j];
        let al = 0, be = 0, ga = 0;
        for (let r = 0; r < m; r++) { al += ai[r] * ai[r]; be += aj[r] * aj[r]; ga += ai[r] * aj[r]; }
        if (ga === 0 || Math.abs(ga) <= 1e-15 * Math.sqrt(al * be)) continue;
        rotated = true;
        const zeta = (be - al) / (2 * ga);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let r = 0; r < m; r++) { const x = ai[r], y = aj[r]; ai[r] = c * x - s * y; aj[r] = s * x + c * y; }
        const vi = V[i], vj = V[j];
        for (let r = 0; r < k; r++) { const x = vi[r], y = vj[r]; vi[r] = c * x - s * y; vj[r] = s * x + c * y; }
      }
      if (!rotated) break;
    }
    const S = A.map(c => Math.sqrt(c.reduce((acc, v) => acc + v * v, 0)));
    const U = A.map((c, i) => S[i] > 0 ? c.map(v => v / S[i]) : c);
    return { U, S, V };
  }

  window.DD = { C, FONT, heat, setupCanvas, Axes, fmt, fmtNum, rng, mpStieltjesNeg, risk, mpDensity, jacobiSVD };
})();
