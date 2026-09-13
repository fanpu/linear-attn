// Shared helpers for the implicit-bias widgets: palette, HiDPI canvases, tiny plotting utilities.
(function () {
  const C = {
    paper: "#fcfbf8", ink: "#1d1d1f", ink2: "#52514e", muted: "#8b8984", grid: "#ebe8e1", axis: "#c3c2b7",
    blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", violet: "#4a3aa7", red: "#e34948",
  };
  // alpha ramp (rich -> kernel), same stops as style.ALPHA_CMAP
  const RAMP = ["#0d4f3c", "#17866a", "#2a8fb0", "#5a9ee6", "#a9c8f2"];
  function hex2rgb(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function ramp(t) {
    t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    const i = Math.min(Math.floor(t), RAMP.length - 2), f = t - i;
    const a = hex2rgb(RAMP[i]), b = hex2rgb(RAMP[i + 1]);
    return `rgb(${a.map((v, k) => Math.round(v + f * (b[k] - v))).join(",")})`;
  }
  function canvas(el, w, h) {
    const c = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    c.style.width = "100%"; c.style.maxWidth = w + "px"; c.style.height = "auto"; c.style.display = "block";
    const ctx = c.getContext("2d");
    ctx.scale(dpr, dpr);
    el.appendChild(c);
    return { c, ctx, w, h };
  }
  // linear map helper
  function scale(d0, d1, r0, r1) { const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0); f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0); return f; }
  function line(ctx, pts, color, lw, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw || 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke(); ctx.restore();
  }
  function dot(ctx, x, y, r, fill, ring) {
    ctx.save(); ctx.beginPath(); ctx.arc(x, y, r + (ring ? 2 : 0), 0, 2 * Math.PI); ctx.fillStyle = ring || fill; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); ctx.restore();
  }
  function text(ctx, s, x, y, opt) {
    opt = opt || {};
    ctx.save(); ctx.fillStyle = opt.color || C.ink2; ctx.font = `${opt.weight || 400} ${opt.size || 12}px Inter, "Helvetica Neue", Arial, sans-serif`;
    ctx.textAlign = opt.align || "left"; ctx.textBaseline = opt.base || "middle"; ctx.fillText(s, x, y); ctx.restore();
  }
  function bisect(f, lo, hi, it) { let flo = f(lo); for (let k = 0; k < (it || 200); k++) { const m = 0.5 * (lo + hi), fm = f(m); if ((fm > 0) === (flo > 0)) { lo = m; flo = fm; } else hi = m; } return 0.5 * (lo + hi); }
  window.IB = { C, ramp, canvas, scale, line, dot, text, bisect };
})();
