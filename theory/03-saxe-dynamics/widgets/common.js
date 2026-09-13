// Shared helpers for the Saxe-dynamics widgets (plain JS, no dependencies).
(function () {
  const C = {
    paper: "#fcfbf8", ink: "#1d1d1f", muted: "#6b6b70", rule: "#e6e2da", accent: "#b5452b",
    modes: ["#e0952a", "#d2553a", "#a23f84", "#4a45a8", "#1a7fa6"],
    // light slice of cmcrameri lapaz_r (low loss = light)
    landscape: ["#fef2f3", "#fae3da", "#efd3c0", "#d9c0a7", "#bfb199", "#a7a895", "#92a298", "#7e9c9d",
                "#6b94a1", "#5a8ba3", "#4a7fa3", "#3e72a1"],
  };

  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const LAND = C.landscape.map(hexToRgb);

  function landColor(x) { // x in [0,1]
    x = Math.max(0, Math.min(1, x)) * (LAND.length - 1);
    const i = Math.min(LAND.length - 2, Math.floor(x)), f = x - i;
    return LAND[i].map((v, k) => v + f * (LAND[i + 1][k] - v));
  }

  // HiDPI canvas: returns 2d context scaled so drawing uses CSS pixels.
  function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // Minimal axes: maps data -> pixels and draws ticks/labels.
  function Axes(ctx, box, xr, yr, opts) {
    opts = opts || {};
    const ylog = !!opts.ylog;
    const ty = v => ylog ? Math.log10(Math.max(v, 1e-12)) : v;
    const X = x => box.x + (x - xr[0]) / (xr[1] - xr[0]) * box.w;
    const Y = y => box.y + box.h - (ty(y) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0])) * box.h;
    function draw(xticks, yticks, xlabel, ylabel) {
      ctx.save();
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(box.x, box.y + box.h + 0.5); ctx.lineTo(box.x + box.w, box.y + box.h + 0.5); ctx.stroke();
      ctx.fillStyle = C.muted; ctx.font = "11px Inter, 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      xticks.forEach(t => { ctx.fillText(fmt(t), X(t), box.y + box.h + 5); });
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      yticks.forEach(t => {
        ctx.fillText(fmt(t), box.x - 6, Y(t));
        ctx.strokeStyle = C.rule; ctx.globalAlpha = 0.6;
        ctx.beginPath(); ctx.moveTo(box.x, Y(t) + 0.5); ctx.lineTo(box.x + box.w, Y(t) + 0.5); ctx.stroke();
        ctx.globalAlpha = 1;
      });
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      if (xlabel) ctx.fillText(xlabel, box.x + box.w / 2, box.y + box.h + 22);
      if (ylabel) {
        ctx.save(); ctx.translate(box.x - 38, box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore();
      }
      ctx.restore();
    }
    function line(xs, ys, color, width, alpha, dash) {
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.globalAlpha = alpha == null ? 1 : alpha;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < xs.length; i++) {
        if (!isFinite(ys[i])) { started = false; continue; }
        const px = X(xs[i]), py = Math.max(box.y - 4, Math.min(box.y + box.h + 4, Y(ys[i])));
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.restore();
    }
    return { X, Y, draw, line, box };
  }

  function fmt(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1000 || a < 0.01) return v.toExponential(0).replace("e", "e");
    return String(+v.toFixed(2));
  }

  // tiny seeded RNG (mulberry32) + gaussian
  function rng(seed) {
    let s = seed >>> 0;
    const u = () => { s += 0x6D2B79F5; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const g = () => { let a = 0, b = 0; while (a === 0) a = u(); b = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); };
    return { u, g };
  }

  window.SaxeCommon = { C, landColor, setupCanvas, Axes, fmt, rng };
})();
