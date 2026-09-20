'use strict';
// The board: 113 nails at measured positions (data.js), one red thread a -> a+1, paper tags on 30% of the
// nails, and the wind that takes them. Everything here is a pure function of time. Load after winder.js.
const Board = (() => {
  const G = GROK, NP = G.P, U = 190, CXB = 650, CYB = 445, HALF = 400, FLOOR = 962;
  const RED = Winder.RED, GRAPHITE = Winder.GRAPHITE, PAPER = Winder.PAPER, TAGFILL = '#fbf7ee';
  const sstep = (a, b, x) => { const u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); };

  // Measured nail positions at a fractional training step (linear between saved checkpoints).
  function nailsAt(step) {
    const st = G.steps; let hi = 1; while (hi < st.length - 1 && st[hi] < step) hi++;
    const lo = hi - 1, u = clamp((step - st[lo]) / (st[hi] - st[lo]), 0, 1), a = G.ring[lo], b = G.ring[hi], out = new Array(NP);
    for (let k = 0; k < NP; k++) out[k] = [CXB + U * lerp(a[2 * k], b[2 * k], u), CYB + U * lerp(a[2 * k + 1], b[2 * k + 1], u)];
    return out;
  }
  const Rat = step => { const st = G.steps; let hi = 1; while (hi < st.length - 1 && st[hi] < step) hi++; return lerp(G.R[hi - 1], G.R[hi], clamp((step - st[hi - 1]) / (st[hi] - st[hi - 1]), 0, 1)); };

  // The board leans back from the wind about the middle of its foot.
  const swayOf = g => -.011 * g;
  function toWorld(p, sway) { const x = p[0] - CXB, y = p[1] - FLOOR, c = Math.cos(sway), s = Math.sin(sway); return [CXB + x * c - y * s, FLOOR + x * s + y * c]; }

  // Static panel and easel, compiled once as a pencil cel.
  let panelSprite = null, panelKey = '';
  function panel() {
    const key = String(S); if (panelSprite && panelKey === key) return panelSprite;
    const strokes = [], add = (id, points, width = 1.6, opacity = .8, extra = {}) => strokes.push({ id, points, width, opacity, ...extra });
    const x0 = CXB - HALF, x1 = CXB + HALF, y0 = CYB - HALF, y1 = CYB + HALF;
    add('edge/top', [[x0 - 3, y0], [CXB, y0 - 2], [x1 + 2, y0 + 1]], 2.2, .9); add('edge/right', [[x1, y0 - 2], [x1 + 1.5, CYB], [x1, y1 + 3]], 2.2, .9);
    add('edge/bottom', [[x1 + 3, y1], [CXB, y1 + 2], [x0 - 2, y1]], 2.4, .95); add('edge/left', [[x0, y1 + 2], [x0 - 1.5, CYB], [x0, y0 - 3]], 2.2, .9);
    add('thick/bottom', [[x0 + 5, y1 + 11], [CXB, y1 + 13], [x1 + 8, y1 + 11]], 1.5, .7); add('thick/right', [[x1 + 9, y0 + 9], [x1 + 10, CYB], [x1 + 9, y1 + 11]], 1.5, .7);
    add('thick/c1', [[x1, y0], [x1 + 9, y0 + 9]], 1.3, .7); add('thick/c2', [[x1, y1], [x1 + 9, y1 + 11]], 1.3, .7); add('thick/c3', [[x0, y1], [x0 + 5, y1 + 11]], 1.3, .7);
    for (let i = 0; i < 13; i++) {                                 // faint grain, quiet enough for the thread
      const y = y0 + 34 + i * 58 + (hash(i, 3) - .5) * 22, a = x0 + 14 + hash(i, 4) * 200, b = a + 180 + hash(i, 5) * 380;
      add('grain/' + i, [[a, y], [lerp(a, b, .35), y + (hash(i, 6) - .5) * 9], [lerp(a, b, .7), y + (hash(i, 7) - .5) * 9], [Math.min(b, x1 - 12), y + 2]], .9, .2);
    }
    add('ledge', [[x0 - 26, y1 + 22], [CXB, y1 + 25], [x1 + 30, y1 + 22]], 2.6, .9); add('ledge/under', [[x0 - 22, y1 + 33], [CXB, y1 + 36], [x1 + 26, y1 + 33]], 1.4, .7);
    const leg = (id, xa, xb) => { add(id + '/a', [[xa, y1 + 34], [lerp(xa, xb, .5) - 1, lerp(y1 + 34, FLOOR, .5)], [xb, FLOOR]], 2.2, .9); add(id + '/b', [[xa + 15, y1 + 34], [lerp(xa, xb, .5) + 15, lerp(y1 + 34, FLOOR, .5)], [xb + 17, FLOOR]], 1.8, .85); };
    leg('leg/l', x0 + 120, x0 + 84); leg('leg/r', x1 - 130, x1 - 96);
    add('leg/back/a', [[CXB + 8, y1 + 36], [CXB + 40, FLOOR - 26]], 1.5, .5); add('leg/back/b', [[CXB + 22, y1 + 36], [CXB + 56, FLOOR - 26]], 1.3, .45);
    const cel = compileCel({ strokes }, { id: 'board/panel' });
    const cv = document.createElement('canvas'); cv.width = Math.ceil(W * S); cv.height = Math.ceil(H * S);
    const g = cv.getContext('2d'); g.scale(S, S); g.fillStyle = '#f8f4ea'; g.fillRect(x0, y0, HALF * 2, HALF * 2);
    drawCel(g, cel, { material: 'pencil', color: GRAPHITE });
    panelSprite = cv; panelKey = key; return cv;
  }
  let floorSprite = null, floorKey = '';
  function floor() {
    const key = String(S); if (floorSprite && floorKey === key) return floorSprite;
    const strokes = [{ id: 'floor', points: [[-20, FLOOR + 1], [300, FLOOR - 1], [700, FLOOR + 2], [1100, FLOOR]], width: 2, opacity: .75 }];
    for (let i = 0; i < 26; i++) { const x = 230 + i * 33 + hash(i, 11) * 14, l = 30 + hash(i, 12) * 40; strokes.push({ id: 'shade/' + i, points: [[x, FLOOR + 10 + hash(i, 13) * 26], [x + l, FLOOR + 8 + hash(i, 13) * 26]], width: 1, opacity: .22 + hash(i, 14) * .16 }); }
    const cv = document.createElement('canvas'); cv.width = Math.ceil(W * S); cv.height = Math.ceil(H * S);
    const g = cv.getContext('2d'); g.scale(S, S); drawCel(g, compileCel({ strokes }, { id: 'board/floor' }), { material: 'pencil', color: GRAPHITE });
    floorSprite = cv; floorKey = key; return cv;
  }

  // One thread through every nail in counting order. Slack hangs; wind bellies it to the left.
  // upTo = how many segments exist yet (fractional while winding); returns the thread's free end.
  function drawThread(c, nails, { slack = 0, gust = 0, t = 0, upTo = NP } = {}) {
    let head = nails[0];
    const pathOf = off => { c.beginPath();
      for (let a = 0; a < Math.min(NP, Math.ceil(upTo)); a++) { const p = nails[a], q0 = nails[(a + 1) % NP], f = clamp(upTo - a, 0, 1), q = [lerp(p[0], q0[0], f), lerp(p[1], q0[1], f)], len = Math.hypot(q[0] - p[0], q[1] - p[1]);
        const sag = slack * Math.min(len * .085, 34) * (.55 + hash(a, 21) * .9), belly = gust * slack * Math.min(len * .05, 16) * (1 + .5 * Math.sin(t * 17 + a));
        c.moveTo(p[0] + off, p[1] + off); c.quadraticCurveTo((p[0] + q[0]) / 2 - belly + off, (p[1] + q[1]) / 2 + sag + off, q[0] + off, q[1] + off); head = q; } };
    c.save(); c.lineCap = 'round'; c.strokeStyle = RED;
    c.globalAlpha = .62; c.lineWidth = 1.5; pathOf(0); c.stroke();
    c.globalAlpha = .3; c.lineWidth = .8; pathOf(.9); c.stroke();
    c.restore(); return head;
  }
  // the feed line from the spool to the thread's free end
  function drawFeed(c, from, to, sag = 26) {
    c.save(); c.strokeStyle = RED; c.lineCap = 'round'; c.globalAlpha = .7; c.lineWidth = 1.5; c.beginPath(); c.moveTo(from[0], from[1]);
    c.quadraticCurveTo((from[0] + to[0]) / 2, (from[1] + to[1]) / 2 + sag, to[0], to[1]); c.stroke(); c.restore();
  }
  function drawSpool(c, x, y, fill = 1) {
    c.save(); c.translate(x, y); c.rotate(-.25); c.fillStyle = Winder.RED_FILL; c.strokeStyle = RED; c.lineWidth = 1.3;
    const h = 5 + 5 * fill; c.beginPath(); c.rect(-9, -h, 18, h * 2); c.fill(); c.globalAlpha = .8; c.stroke();
    c.beginPath(); for (let k = -2; k <= 2; k++) { c.moveTo(-9, k * h / 2.6 - 1.5); c.lineTo(9, k * h / 2.6 + 1.5); } c.globalAlpha = .5; c.stroke();
    c.globalAlpha = .9; c.strokeStyle = GRAPHITE; c.fillStyle = PAPER; c.lineWidth = 1.5;
    for (const sx of [-1, 1]) { c.beginPath(); c.rect(sx * 11 - 2, -13, 4, 26); c.fill(); c.stroke(); }
    c.restore();
  }
  // nail numbers: rendered once each at double resolution so the closing push-in stays sharp
  const labelSprites = new Map();
  function drawLabels(c, nails, alphaOf) {
    const k = S * 2.4;
    for (let a = 0; a < NP; a++) { const al = alphaOf(a); if (al <= .01) continue;
      const key = a + '|' + S; if (!labelSprites.has(key)) { const cv = document.createElement('canvas'); cv.width = Math.ceil(34 * k); cv.height = Math.ceil(18 * k); const g = cv.getContext('2d'); g.scale(k, k);
        Lettering.draw(g, String(a), 17, 14, 9.5, { align: 'center', weight: 1.25 }); labelSprites.set(key, cv); }
      const p = nails[a], dx = p[0] - CXB, dy = p[1] - CYB, d = Math.hypot(dx, dy) || 1, sp = labelSprites.get(key);
      c.globalAlpha = al; c.drawImage(sp, p[0] + dx / d * 15 - 17, p[1] + dy / d * 13 - 9, 34, 18); }
    c.globalAlpha = 1;
  }

  // The pinned chart: measured train and test loss (log scale), drawn only up to the current step.
  const CH = { x: 30, y: 78, w: 196, h: 138 };
  let chartSprite = null, chartKey = '';
  function drawChart(c, step, gust = 0, t = 0) {
    const key = String(S);
    if (chartKey !== key) { chartKey = key; const cv = document.createElement('canvas'); cv.width = Math.ceil(260 * S * 2); cv.height = Math.ceil(210 * S * 2); const g = cv.getContext('2d'); g.scale(S * 2, S * 2); g.translate(-CH.x + 20, -CH.y + 30);
      g.fillStyle = TAGFILL; g.fillRect(CH.x - 14, CH.y - 22, CH.w + 30, CH.h + 48);
      const st = [], L = (id, pts, w = 1.4, o = .8) => st.push({ id, points: pts, width: w, opacity: o });
      L('p/t', [[CH.x - 14, CH.y - 22], [CH.x + CH.w / 2, CH.y - 23], [CH.x + CH.w + 16, CH.y - 22]], 1.6); L('p/r', [[CH.x + CH.w + 16, CH.y - 22], [CH.x + CH.w + 17, CH.y + CH.h + 26]], 1.6);
      L('p/b', [[CH.x + CH.w + 16, CH.y + CH.h + 26], [CH.x + CH.w / 2, CH.y + CH.h + 27], [CH.x - 14, CH.y + CH.h + 26]], 1.8); L('p/l', [[CH.x - 14, CH.y + CH.h + 26], [CH.x - 15, CH.y - 22]], 1.6);
      L('ax/y', [[CH.x, CH.y - 4], [CH.x, CH.y + CH.h]], 1.5, .85); L('ax/x', [[CH.x, CH.y + CH.h], [CH.x + CH.w + 4, CH.y + CH.h]], 1.5, .85);
      for (let i = 1; i <= 4; i++) L('tick/' + i, [[CH.x + CH.w * i / 4, CH.y + CH.h], [CH.x + CH.w * i / 4, CH.y + CH.h + 5]], 1, .6);
      drawCel(g, compileCel({ strokes: st }, { id: 'chart/frame' }), { material: 'pencil', color: GRAPHITE });
      Lettering.draw(g, 'test', CH.x + 58, CH.y + 38, 11, { color: RED, weight: 1.3 }); Lettering.draw(g, 'train', CH.x + 14, CH.y + CH.h - 30, 11, { weight: 1.3 });
      Lettering.draw(g, 'step', CH.x + CH.w - 26, CH.y + CH.h + 20, 10, { weight: 1.2, opacity: .8 });
      chartSprite = cv; }
    const X = s => CH.x + CH.w * s / 40000, Y = v => CH.y + CH.h * (1 - (Math.log10(Math.max(v, 1e-8)) + 8) / 10);
    c.save(); c.translate(CH.x + CH.w / 2, CH.y - 22); c.rotate(-.025 + gust * .1 + gust * .03 * Math.sin(t * 15)); c.translate(-(CH.x + CH.w / 2), -(CH.y - 22));
    c.drawImage(chartSprite, CH.x - 20, CH.y - 30, 260, 210);
    const cs = G.curve_steps, curve = (vals, color, al) => { c.beginPath(); let last = null;
      for (let i = 0; i < cs.length && cs[i] <= step; i++) { const x = X(cs[i]), y = Y(vals[i]); i ? c.lineTo(x, y) : c.moveTo(x, y); last = [x, y]; }
      c.strokeStyle = color; c.lineJoin = 'round'; c.globalAlpha = al; c.lineWidth = 1.7; c.stroke(); c.globalAlpha = al * .4; c.lineWidth = .8; c.translate(.7, .7); c.stroke(); c.translate(-.7, -.7); return last; };
    curve(G.train_loss, GRAPHITE, .8); const tip = curve(G.test_loss, RED, .85);
    if (tip) { c.globalAlpha = .9; c.fillStyle = RED; c.beginPath(); c.arc(tip[0], tip[1], 2.6, 0, TAU); c.fill(); }
    c.globalAlpha = .9; c.fillStyle = GRAPHITE; c.beginPath(); c.arc(CH.x + CH.w / 2, CH.y - 15, 3, 0, TAU); c.fill();
    c.restore();
  }

  function drawNails(c, nails) {
    c.save(); c.fillStyle = GRAPHITE; c.globalAlpha = .9; c.beginPath();
    for (const p of nails) { c.moveTo(p[0] + 2.7, p[1]); c.arc(p[0], p[1], 2.7, 0, TAU); } c.fill();
    c.strokeStyle = GRAPHITE; c.globalAlpha = .35; c.lineWidth = 1; c.beginPath();
    for (const p of nails) { c.moveTo(p[0] + 2, p[1] + 2.5); c.lineTo(p[0] + 6, p[1] + 6.5); } c.stroke(); c.restore();
  }

  // Paper tags: memorised answers, one nail each. Drawn in tag-local space, pivot at the string's top.
  function drawTag(c, x, y, ang, { flip = 1, seed = 0, alpha = 1 } = {}) {
    c.save(); c.translate(x, y); c.rotate(ang); c.scale(flip * 1.2, 1.2); c.globalAlpha = alpha;
    c.strokeStyle = GRAPHITE; c.lineWidth = .9; c.beginPath(); c.moveTo(0, 0); c.lineTo(0, 9); c.stroke();
    c.beginPath(); c.moveTo(-5, 9); c.lineTo(5, 9); c.lineTo(10, 15); c.lineTo(10, 40); c.lineTo(-10, 40); c.lineTo(-10, 15); c.closePath();
    c.fillStyle = TAGFILL; c.fill(); c.lineWidth = 1.3; c.globalAlpha = alpha * .85; c.stroke();
    c.lineWidth = .9; c.globalAlpha = alpha * .55; c.beginPath();
    for (let r = 0; r < 3; r++) { const w = 4 + hash(seed, 40 + r) * 9; c.moveTo(-6, 21 + r * 6); c.lineTo(-6 + w, 21 + r * 6 + (hash(seed, 50 + r) - .5) * 1.5); } c.stroke();
    c.restore();
  }

  function makeTags({ count = 34, seed = 7, t0 = 3.7, spread = 3.4, early = [1.72, 2.05], last = 9.7, finalNails } = {}) {
    const r = rng(seed), idx = Array.from({ length: NP }, (_, a) => a);
    for (let i = NP - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const chosen = idx.slice(0, count);
    // the stubborn tag hangs lowest on the finished ring, so its fall is short and in front of the board
    const stubborn = chosen.reduce((m, a) => finalNails[a][1] > finalNails[m][1] ? a : m, chosen[0]);
    let e = 0;
    return chosen.map((a, k) => {
      const right = clamp((finalNails[a][0] - (CXB - U)) / (2 * U), 0, 1);
      const release = a === stubborn ? last : (e < early.length && right > .45 && hash(a, 3) > .5) ? early[e++] : t0 + (1 - right) * spread * .55 + hash(a, 60) * spread * .45;
      return { a, k, release, stubborn: a === stubborn, base: (hash(a, 61) - .5) * .24, vx: -(240 + hash(a, 62) * 260), vy: -(30 + hash(a, 63) * 170), spin: (hash(a, 64) > .5 ? 1 : -1) * (4 + hash(a, 65) * 6), ph: hash(a, 66) * TAU, front: hash(a, 67) > .55 };
    });
  }
  // Attached: a pendulum pushed by the gust. Free: thrown left, tumbling. The stubborn one see-saws down.
  function tagState(tag, t, gustAt, nailWorld) {
    if (t < tag.release) { const g = gustAt(t); return { attached: true, x: nailWorld(t)[0], y: nailWorld(t)[1], ang: tag.base + g * 1.2 + g * .38 * Math.sin(t * 21 + tag.ph) + .04 * Math.sin(t * 2.2 + tag.ph), flip: 1 }; }
    const tau = t - tag.release, p0 = nailWorld(tag.release);
    if (tag.stubborn) {                                            // calm air: a slow see-saw that ends flat on the floor
      const D = FLOOR - 8 - p0[1], u = Math.min(tau, D / 118), f = u * 118 / D, lie = sstep(.72, 1, f);
      return { attached: false, x: p0[0] - 30 * u + 44 * Math.sin(u * 2.7) * (1 - lie), y: p0[1] + u * 118, ang: .75 * Math.cos(u * 2.7 + .4) * (1 - lie) + 1.5 * lie, flip: 1 };
    }
    const g0 = gustAt(tag.release), A = 520 * Math.max(g0, .25);
    return { attached: false, x: p0[0] + tag.vx * tau - .5 * A * tau * tau, y: p0[1] + tag.vy * tau + 120 * tau * tau - 36 * Math.sin(tau * 6 + tag.ph), ang: tag.base + g0 * 1.2 + tag.spin * tau, flip: Math.max(.18, Math.abs(Math.cos(tau * 5.5 + tag.ph))) };
  }

  // Wind marks: short tapered pencil gestures with a curl, only while the gust exists.
  function drawStreaks(c, t, g) {
    if (g < .06) return; c.save(); c.strokeStyle = GRAPHITE; c.lineCap = 'round';
    for (let k = 0; k < 9; k++) {
      const speed = 1300 + hash(k, 70) * 600, len = 110 + hash(k, 71) * 150, span = W + len * 2 + 700 + hash(k, 72) * 1100, s = t * speed + hash(k, 73) * span;
      const cyc = Math.floor(s / span), x = W + len - (s % span), y = 70 + hash(k * 31 + cyc, 74) * (FLOOR - 160), bow = (hash(k * 17 + cyc, 75) - .5) * 30, a = Math.min(.42, g * .42) * (.55 + hash(k + cyc, 76) * .45);
      for (let m = 0; m < 2; m++) {                                 // a pair of lines; each drawn as tapering segments
        const yy = y + m * (9 + hash(k, 78) * 6), L = len * (m ? .62 : 1), n = 12;
        for (let q = 0; q < n; q++) { const u0 = q / n, u1 = (q + 1) / n, P = u => [x - L * u - (u > .8 ? 0 : 0), yy + bow * Math.sin(u * Math.PI) - (u > .78 ? (u - .78) * (u - .78) * 900 * (m ? .6 : 1) : 0)];
          const p0 = P(u0), p1 = P(u1); c.globalAlpha = a * Math.sin(Math.PI * Math.min(1, (u0 + .08))) ; c.lineWidth = (.7 + hash(k, 77) * .7) * (.4 + Math.sin(Math.PI * u0) * .8);
          c.beginPath(); c.moveTo(p0[0], p0[1]); c.lineTo(p1[0], p1[1]); c.stroke(); }
      }
    }
    c.restore();
  }
  return { nailsAt, Rat, swayOf, toWorld, panel, floor, drawThread, drawFeed, drawSpool, drawLabels, drawChart, drawNails, drawTag, makeTags, tagState, drawStreaks, sstep, U, CXB, CYB, HALF, FLOOR, NP };
})();
