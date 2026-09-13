// Shared helpers for the lazy/rich + muP widgets (plain JS, no dependencies).
(function () {
  const C = {
    paper: "#fcfbf8", ink: "#1d1d1f", muted: "#6b6b70", rule: "#e6e2da",
    lazy: "#2f6db5", rich: "#e0561f",
    night: "#070912", nightInk: "#ecebe6", nightMuted: "#8a8d9c", nightRule: "#22263a",
    pos: "#ffb454", neg: "#58c4f5",
    widths: ["#ad92e0", "#8561cf", "#5e37b0", "#361a7d"],
    depths: ["#6fb8a7", "#3f9e8a", "#1d7564", "#0a4a3f"],
    font: "'Ubuntu Sans', Ubuntu, Inter, 'Helvetica Neue', Arial, sans-serif",
  };

  function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function b64ToBytes(s) {
    const bin = atob(s), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const i16 = s => { const b = b64ToBytes(s); return new Int16Array(b.buffer, b.byteOffset, b.byteLength / 2); };
  const f32 = s => { const b = b64ToBytes(s); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

  function fmt(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1000 || a < 1e-2) {
      const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
      return (Math.abs(m - 1) < 1e-9 ? "" : (+m.toFixed(1)) + "·") + "10" + sup(e);
    }
    return String(+v.toPrecision(3));
  }
  function sup(e) {
    const map = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
    return String(e).split("").map(c => map[c]).join("");
  }

  // Axes with optional log scales. box = {x, y, w, h} in CSS px.
  function Axes(ctx, box, xr, yr, opts) {
    opts = opts || {};
    const tx = v => opts.xlog ? Math.log10(v) : v, ty = v => opts.ylog ? Math.log10(v) : v;
    const X = x => box.x + (tx(x) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0])) * box.w;
    const Y = y => box.y + box.h - (ty(y) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0])) * box.h;
    const theme = opts.dark ? { rule: C.nightRule, muted: C.nightMuted } : { rule: C.rule, muted: C.muted };
    function draw(xticks, yticks, xlabel, ylabel, xfmt, yfmt) {
      ctx.save();
      ctx.font = "11px " + C.font; ctx.fillStyle = theme.muted;
      ctx.strokeStyle = theme.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(box.x, box.y + box.h + .5); ctx.lineTo(box.x + box.w, box.y + box.h + .5); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      xticks.forEach(t => ctx.fillText((xfmt || fmt)(t), X(t), box.y + box.h + 5));
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      yticks.forEach(t => {
        ctx.fillText((yfmt || fmt)(t), box.x - 6, Y(t));
        ctx.globalAlpha = .55; ctx.beginPath(); ctx.moveTo(box.x, Math.round(Y(t)) + .5); ctx.lineTo(box.x + box.w, Math.round(Y(t)) + .5); ctx.stroke(); ctx.globalAlpha = 1;
      });
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      if (xlabel) ctx.fillText(xlabel, box.x + box.w / 2, box.y + box.h + 21);
      if (ylabel) { ctx.save(); ctx.translate(box.x - 44, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore(); }
      ctx.restore();
    }
    function line(xs, ys, color, width, alpha, dash) {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.globalAlpha = alpha == null ? 1 : alpha;
      ctx.lineJoin = "round"; ctx.lineCap = "round"; if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); let on = false;
      for (let i = 0; i < xs.length; i++) {
        const y = ys[i];
        if (y == null || !isFinite(y) || (opts.ylog && y <= 0)) { on = false; continue; }
        const px = X(xs[i]), py = Math.max(box.y - 3, Math.min(box.y + box.h + 3, Y(y)));
        if (!on) { ctx.moveTo(px, py); on = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.restore();
    }
    function dot(x, y, r, fill, stroke) {
      ctx.save(); ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, 2 * Math.PI);
      ctx.fillStyle = fill; ctx.fill(); if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
      ctx.restore();
    }
    return { X, Y, draw, line, dot, box };
  }

  window.LR = { C, setupCanvas, i16, f32, fmt, Axes };
})();
