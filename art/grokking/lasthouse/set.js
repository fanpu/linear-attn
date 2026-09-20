'use strict';
// The set: a pond in three-quarter view ringed by thirteen houses, through autumn, winter and spring,
// by day and by night. Also the corner constellation: the measured embedding, drawn straight onto the sky.
// Everything is a pure function of its arguments. Load after cast.js and data.js.
const Stage = (() => {
  const INK = Cast.INK, PC = [900, 675], RX = 640, RY = 215, NH = 11, HOME = 8, MOLE = 2, HORIZON = 330;
  const sstep = (a, b, x) => { const u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); };
  const depth = y => lerp(.62, 1.25, clamp((y - (PC[1] - RY - 70)) / (2 * RY + 140), 0, 1));
  const H = Array.from({ length: NH }, (_, k) => { const a = -Math.PI / 2 + (k + .5) / NH * TAU, x = PC[0] + Math.cos(a) * (RX + 120), y = PC[1] + Math.sin(a) * (RY + 70); return { k, a, x, y, s: depth(y) * 1.6, near: Math.sin(a) > .6 }; });
  // where a visitor stands: on the shore in front of the door
  const spot = k => { const h = H[k]; return [h.x + (PC[0] - h.x) * .17 + 14 * h.s, h.y + 14 * h.s]; };
  const dims = k => ({ w: 58 + hash(k, 1) * 26, hh: 44 + hash(k, 2) * 22, rf: 30 + hash(k, 3) * 22 });
  const ROOFS = ['#b9796a', '#8f9a86', '#a08a72', '#7f93ad'], cels = new Map();
  const cel = (id, make) => { if (!cels.has(id)) cels.set(id, compileCel({ strokes: make() }, { id })); return cels.get(id); };
  const mixc = (a, b, u) => mix(a, b, u);

  function house(c, k, { winter = 0, lit = 0, flag = -1, t = 0, doorLetter = 0, lampOnStep = 0 } = {}) {
    const h = H[k], { w, hh, rf } = dims(k); c.save(); c.translate(h.x, h.y); c.scale(h.s, h.s);
    c.fillStyle = h.near ? '#cdc4b6' : '#efe7d8'; c.fillRect(-w / 2, -hh, w, hh);
    c.fillStyle = mixc(ROOFS[k % 4], '#f4f1ea', winter * .85); c.beginPath(); c.moveTo(-w / 2 - 8, -hh); c.lineTo(0, -hh - rf); c.lineTo(w / 2 + 8, -hh); c.closePath(); c.fill();
    c.fillStyle = '#d9d0c0'; c.fillRect(w * .18, -hh - rf * .95, 10, rf * .6);
    if (!h.near) { c.fillStyle = mixc('#bfb8aa', Cast.LAMP, lit); c.fillRect(-w * .32, -hh * .72, 16, 16); c.fillStyle = k === 0 ? '#b5523f' : '#8a6f55'; c.fillRect(w * .1, -hh * .62, 14, hh * .62); }
    drawCel(c, cel('house/' + k, () => { const st = [], L = (id, pts, wd = 1.8, o = .9, ex = {}) => st.push({ id, points: pts, width: wd, opacity: o, ...ex });
      L('l', [[-w / 2, 0], [-w / 2, -hh]]); L('r', [[w / 2, 0], [w / 2, -hh]]); L('b', [[-w / 2 - 4, 0], [w / 2 + 4, 0]], 1.6, .7); L('ra', [[-w / 2 - 8, -hh], [0, -hh - rf]], 2); L('rb', [[0, -hh - rf], [w / 2 + 8, -hh]], 2); L('e', [[-w / 2 - 8, -hh], [w / 2 + 8, -hh]], 1.4, .8);
      L('ch', [[w * .18, -hh - rf * .4], [w * .18, -hh - rf * .95], [w * .18 + 10, -hh - rf * .95], [w * .18 + 10, -hh - rf * .55]], 1.2, .8, { corner: 1 });
      if (!h.near) { const wx = -w * .32, wy = -hh * .72; L('win', [[wx, wy], [wx + 16, wy], [wx + 16, wy + 16], [wx, wy + 16], [wx, wy]], 1.2, .9, { corner: 1 }); L('wc', [[wx + 8, wy], [wx + 8, wy + 16]], .8, .6); L('door', [[w * .1, 0], [w * .1, -hh * .62], [w * .1 + 14, -hh * .62], [w * .1 + 14, 0]], 1.3, .9, { corner: 1 }); L('knob', [[w * .1 + 10, -hh * .3], [w * .1 + 11, -hh * .28]], 1.8, .9);
        L('post', [[w / 2 + 16, 4], [w / 2 + 16, -22]], 1.6, .9); L('box', [[w / 2 + 9, -22], [w / 2 + 9, -32], [w / 2 + 25, -32], [w / 2 + 25, -22], [w / 2 + 9, -22]], 1.3, .9, { corner: 1 }); }
      if (k === MOLE) { L('step', [[-w / 2 - 30, 6], [w / 2 - 6, 6]], 1.8, .85); L('awn', [[-w / 2 - 26, -hh * .8], [w * .1 + 20, -hh * .95]], 1.6, .8); L('awnpost', [[-w / 2 - 24, -hh * .8], [-w / 2 - 24, 6]], 1.4, .8); }
      return st; }), { material: 'pencil', color: INK });
    if (!h.near && flag >= 0) { const up = flag > t ? 0 : 1 + .5 * Math.exp(-7 * (t - flag)) * Math.cos(16 * (t - flag)) - Math.exp(-9 * (t - flag));   // the mailbox flag pops up when a letter lands
      c.save(); c.translate(w / 2 + 25, -24); c.rotate(-up * 1.45); c.fillStyle = '#b5523f'; c.fillRect(0, -3, 12, 6); c.strokeStyle = INK; c.lineWidth = 1; c.strokeRect(0, -3, 12, 6); c.restore(); }
    if (lampOnStep) { c.save(); c.translate(-w / 2 - 12, 4); c.fillStyle = '#d8d1c2'; c.strokeStyle = INK; c.lineWidth = 1.2; c.beginPath(); c.moveTo(-5, -14); c.lineTo(5, -14); c.lineTo(6, 0); c.lineTo(-6, 0); c.closePath(); c.fill(); c.stroke(); c.beginPath(); c.moveTo(-3, -14); c.lineTo(0, -19); c.lineTo(3, -14); c.stroke(); c.restore(); }
    if (doorLetter) { c.save(); c.translate(w * .1 + 5, -2); c.rotate(-.22); c.globalAlpha = doorLetter; c.fillStyle = '#fbf7ee'; c.strokeStyle = INK; c.lineWidth = 1.1; c.fillRect(-8, -11, 16, 11); c.strokeRect(-8, -11, 16, 11); c.beginPath(); c.moveTo(-8, -11); c.lineTo(0, -5); c.lineTo(8, -11); c.stroke(); c.restore(); }
    if (winter > .05) { c.save(); c.globalAlpha = winter * .9; c.fillStyle = '#fbfaf6'; c.beginPath(); c.moveTo(-w / 2 - 9, -hh - 1); c.lineTo(0, -hh - rf - 3); c.lineTo(w / 2 + 9, -hh - 1); c.lineTo(w / 2 + 2, -hh - 6); c.lineTo(0, -hh - rf + 7); c.lineTo(-w / 2 - 2, -hh - 6); c.closePath(); c.fill(); c.restore(); }
    c.restore();
  }
  function tree(c, x, y, s, seed, { winter = 0, spring = 0, willow = false } = {}) {
    c.save(); c.translate(x, y); c.scale(s, s);
    const leaf = mixc(mixc('#d59a55', '#cfc8ba', winter), '#9db77f', spring), r = 26 + hash(seed, 1) * 12;
    if (winter < .7 || spring > .3) { c.globalAlpha = (1 - winter) * .9 + spring * .9 > 1 ? 1 : Math.max((1 - winter) * .9, spring * .9); c.fillStyle = leaf; c.beginPath(); c.ellipse(0, -62, r * (willow ? 1.5 : 1), r * (willow ? 1.25 : 1.05), 0, 0, TAU); c.fill(); c.globalAlpha = 1; }
    drawCel(c, cel('tree/' + seed, () => { const st = [{ id: 'trunk', points: [[-2, 0], [0, -30], [1, -58]], width: 2.6, opacity: .9 }, { id: 'b1', points: [[0, -36], [-14, -58], [-18, -74]], width: 1.6, opacity: .8 }, { id: 'b2', points: [[0, -44], [12, -62], [20, -80]], width: 1.5, opacity: .8 }, { id: 'b3', points: [[0, -56], [-3, -76], [2, -90]], width: 1.3, opacity: .75 }];
      if (willow) for (let i = 0; i < 7; i++) st.push({ id: 'w/' + i, points: [[-34 + i * 11, -70], [-38 + i * 11, -40], [-36 + i * 11, -14]], width: 1, opacity: .5 }); return st; }), { material: 'pencil', color: INK });
    c.restore();
  }
  const TREES = [[250, 400, 1.0, 1], [520, 345, .85, 2], [1320, 350, .85, 3], [1640, 420, 1.0, 4], [40, 600, 1.4, 5], [1870, 620, 1.45, 6], [1060, 335, .8, 7]];
  const WILLOW = [760, 345, 1.0, 8];

  function glow(c, x, y, r, a, col = '255,206,110') { if (a <= .003) return; const g = c.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, `rgba(${col},${a})`); g.addColorStop(.35, `rgba(${col},${a * .35})`); g.addColorStop(1, `rgba(${col},0)`); c.save(); c.globalCompositeOperation = 'screen'; c.fillStyle = g; c.fillRect(x - r, y - r, 2 * r, 2 * r); c.restore(); }

  // Layers are drawn by the film in this order: ground() -> far houses -> pond -> actors -> near houses -> night() -> weather -> constellation
  function ground(c, { winter = 0, spring = 0 }) {
    const sky = mixc(mixc('#f3e9d8', '#e9ecee', winter), '#eef1e4', spring), land = mixc(mixc('#e6d9bd', '#f1efe9', winter), '#d9e2bf', spring);
    c.fillStyle = sky; c.fillRect(-400, -400, W + 800, HORIZON + 400); c.fillStyle = land; c.fillRect(-400, HORIZON, W + 800, 1400);
    c.save(); c.fillStyle = mixc(land, '#8a8375', .22); c.beginPath(); c.moveTo(-400, HORIZON); for (let x = -400; x <= W + 400; x += 80) c.lineTo(x, HORIZON - 26 - 22 * Math.sin(x * .004 + 1) - 12 * Math.sin(x * .011)); c.lineTo(W + 400, HORIZON); c.closePath(); c.fill(); c.restore();
    drawCel(c, cel('horizon', () => [{ id: 'h', points: [[-60, HORIZON], [500, HORIZON - 2], [1300, HORIZON + 1], [1990, HORIZON]], width: 1.4, opacity: .5 }]), { material: 'pencil', color: INK });
    for (const [x, y, s, seed] of TREES) tree(c, x, y, s, seed, { winter, spring }); tree(c, ...WILLOW, { winter, spring, willow: true });
  }
  function pond(c, { winter = 0, spring = 0 }) {
    c.save(); c.fillStyle = mixc(mixc('#b9cbc9', '#dfe7ec', winter), '#b5d0d6', spring); c.beginPath(); c.ellipse(PC[0], PC[1], RX, RY, 0, 0, TAU); c.fill();
    c.strokeStyle = INK; c.globalAlpha = .5; c.lineWidth = 2; c.stroke(); c.globalAlpha = .22; c.lineWidth = 1;
    for (let i = 0; i < 18; i++) { const a = hash(i, 5) * TAU, r = Math.sqrt(hash(i, 6)) * .82, x = PC[0] + Math.cos(a) * RX * r, y = PC[1] + Math.sin(a) * RY * r; c.beginPath(); c.moveTo(x, y); c.lineTo(x + 40 + hash(i, 7) * 90, y + 2); c.stroke(); } c.restore();
  }
  function night(c, n, lights, { moleLamp = 0, moleLampAt = null } = {}) {
    if (n <= .003) return; const T = c.getTransform();
    c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.globalCompositeOperation = 'multiply'; c.globalAlpha = n; const gr = c.createLinearGradient(0, 0, 0, c.canvas.height); gr.addColorStop(0, '#2f3763'); gr.addColorStop(.45, '#46518a'); gr.addColorStop(1, '#3a4478'); c.fillStyle = gr; c.fillRect(0, 0, c.canvas.width, c.canvas.height); c.restore(); c.setTransform(T);
    glow(c, PC[0], PC[1], RX * .9, .16 * n, '190,205,255');
    for (const h of H) { const l = (lights[h.k] ?? 0) * n; if (h.near) glow(c, h.x, h.y - 50 * h.s, 80 * h.s, .4 * l); else { const { w, hh } = dims(h.k); glow(c, h.x + (-w * .32 + 8) * h.s, h.y + (-hh * .72 + 8) * h.s, 70 * h.s, 1 * l); glow(c, h.x - 10 * h.s, h.y + 22 * h.s, 90 * h.s, .3 * l); } }
    if (moleLampAt) glow(c, moleLampAt[0], moleLampAt[1], 86, .95 * moleLamp * Math.max(n, .25));
  }
  function snow(c, t, amount, wind = 0) {
    if (amount <= .01) return; c.save(); c.fillStyle = '#fff'; const N = Math.round(260 * amount);
    for (let i = 0; i < N; i++) { const sp = 60 + hash(i, 21) * 80, x = ((hash(i, 22) * (W + 400) - t * (20 + wind * 900) * (.6 + hash(i, 23)) + 40 * Math.sin(t * 1.3 + i)) % (W + 400) + W + 400) % (W + 400) - 200, y = (hash(i, 24) * H + t * sp * (1 + wind)) % H;
      c.globalAlpha = .35 + hash(i, 25) * .5; c.beginPath(); if (wind > .3) { c.ellipse(x, y, (1 + hash(i, 26) * 1.8) * (1 + wind * 3), 1 + hash(i, 26), -.25, 0, TAU); } else c.arc(x, y, 1 + hash(i, 26) * 1.8, 0, TAU); c.fill(); } c.restore();
  }
  function leaves(c, t, amount) {
    if (amount <= .01) return; c.save(); const N = Math.round(26 * amount);
    for (let i = 0; i < N; i++) { const x = (hash(i, 31) * W + 60 * Math.sin(t * 1.1 + i) + t * 30) % W, y = (hash(i, 32) * H + t * (50 + hash(i, 33) * 40)) % H; c.save(); c.translate(x, y); c.rotate(t * 2 + i); c.globalAlpha = .75; c.fillStyle = ['#d59a55', '#c9784a', '#d9b15f'][i % 3]; c.beginPath(); c.ellipse(0, 0, 6, 3, 0, 0, TAU); c.fill(); c.restore(); } c.restore();
  }
  function streaks(c, t, g) {
    if (g < .06) return; c.save(); c.strokeStyle = '#ffffff'; c.lineCap = 'round';
    for (let k = 0; k < 12; k++) { const speed = 1700 + hash(k, 70) * 800, len = 160 + hash(k, 71) * 200, span = W + len * 2 + 500 + hash(k, 72) * 900, s = t * speed + hash(k, 73) * span, cyc = Math.floor(s / span), x = W + len - (s % span), y = 120 + hash(k * 31 + cyc, 74) * 860, bow = (hash(k * 17 + cyc, 75) - .5) * 30;
      c.globalAlpha = Math.min(.4, g * .4); c.lineWidth = 1 + hash(k, 77) * 1.2; c.beginPath(); c.moveTo(x, y); c.quadraticCurveTo(x - len * .5, y + bow, x - len, y + bow * .3); c.stroke(); } c.restore();
  }

  // The constellation: measured nail positions, thread a -> a+1, no card. Pencil by day, starlight by night.
  function nailsAt(step) { const G = GROK, st = G.steps; let hi = 1; while (hi < st.length - 1 && st[hi] < step) hi++; const u = clamp((step - st[hi - 1]) / (st[hi] - st[hi - 1]), 0, 1), A = G.ring[hi - 1], B = G.ring[hi]; return Array.from({ length: G.P }, (_, a) => [lerp(A[2 * a], B[2 * a], u), lerp(A[2 * a + 1], B[2 * a + 1], u)]); }
  function constellation(c, { step, upTo = 113, n = 0, x = W - 190, y = 165, U = 62, alpha = 1 }) {
    const P = GROK.P, N = nailsAt(step).map(q => [x + q[0] * U, y + q[1] * U]); c.save(); c.lineCap = 'round';
    c.beginPath(); for (let a = 0; a < Math.min(P, Math.ceil(upTo)); a++) { const p = N[a], q0 = N[(a + 1) % P], f = clamp(upTo - a, 0, 1); c.moveTo(p[0], p[1]); c.lineTo(lerp(p[0], q0[0], f), lerp(p[1], q0[1], f)); }
    c.strokeStyle = mixc('#a63d2f', '#ffd9a0', n); c.globalAlpha = alpha * lerp(.62, .5, n); c.lineWidth = 1.15; c.stroke();
    c.fillStyle = mixc(INK, '#fff6df', n); c.globalAlpha = alpha * .9; for (let a = 0; a < P; a++) { c.beginPath(); c.arc(N[a][0], N[a][1], lerp(1.7, 1.5, n), 0, TAU); c.fill(); }
    c.restore(); if (n > .2) glow(c, x, y, U * 2.6, .07 * n * alpha, '255,230,180');
  }
  return { PC, RX, RY, NH, HOME, MOLE, H, spot, depth, dims, house, ground, pond, night, snow, leaves, streaks, glow, constellation, nailsAt, sstep };
})();
