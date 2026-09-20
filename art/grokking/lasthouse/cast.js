'use strict';
// Cast: the postbird and the mole. Each exposed drawing is a whole pose built once from a pose
// description and held with its marks. Local units: origin on the ground under the body, facing right.
const Cast = (() => {
  const INK = '#3d362f', PAPER = '#f5f0e5', BREAST = '#e0a184', BACK = '#cdbfae', CAP = '#7f93ad', BAG = '#c9a66b', BEAK = '#d9a441', MOLE = '#b9ada3', SHAWL = '#a9b49a', LAMP = '#f2c55c';
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]], mul = (a, k) => [a[0] * k, a[1] * k], mixv = (a, b, u) => [lerp(a[0], b[0], u), lerp(a[1], b[1], u)];
  const rot = (p, a) => { const c = Math.cos(a), s = Math.sin(a); return [p[0] * c - p[1] * s, p[0] * s + p[1] * c]; };
  const soft = [[0, .12], [.14, .85], [.45, 1], [.8, .7], [1, .08]], even = [[0, .7], [.5, 1], [1, .7]];

  const BIRD = { tilt: 0, squash: 1, lift: 0, look: [0, 0], eyes: 'open', brow: 0, beak: 0, wing: .15, wingLen: 1, legs: 1, bag: 0, bagFull: .4, cap: 0, tail: 0, tear: 0, letter: 0 };
  const bird = (o = {}) => ({ ...BIRD, ...o });
  function between(a, b, u, faceAt = .5) { const out = {}; for (const k of Object.keys(BIRD)) out[k] = typeof BIRD[k] === 'string' ? (u < faceAt ? a[k] : b[k]) : Array.isArray(BIRD[k]) ? a[k].map((v, i) => lerp(v, b[k][i], u)) : lerp(a[k], b[k], u); return out; }

  function buildBird(p) {
    const groups = [], G = (name, fill, fillColor = PAPER, alpha = 1) => { const g = { name, fill, fillColor, alpha, strokes: [] }; groups.push(g); return g; };
    const S = (g, id, points, width = 1.8, opacity = .95, extra = {}) => g.strokes.push({ id, points, width, opacity, pressure: soft, ...extra });
    const legH = 20 * p.legs, C = [0, -legH - 50 * p.squash - p.lift], B = q => add(C, rot([q[0] * (2 - p.squash) ** .5, q[1] * p.squash], p.tilt));   // body-local -> figure
    // legs first (behind the body)
    if (p.legs > .05) for (const [k, x] of [['A', -9], ['B', 11]]) { const hip = B([x, 44]), foot = [hip[0] + 2, -p.lift * 0], g = G('leg' + k, null);
      S(g, 'shin', [hip, mixv(hip, foot, .55), foot], 1.6, .9, { pressure: even }); S(g, 'toes', [[foot[0] - 7, foot[1]], [foot[0] + 1, foot[1] - 1.5], [foot[0] + 11, foot[1]]], 1.6, .9, { pressure: even }); S(g, 'toe2', [[foot[0] + 1, foot[1] - 1], [foot[0] + 7, foot[1] + 3]], 1.2, .8, { pressure: even }); }
    // tail
    const t0 = B([-36, 22]), ta = -.35 + p.tail, tip = add(t0, rot([-30, 0], -ta + p.tilt)), tg = G('tail', [B([-30, 8]), add(tip, rot([0, -5], p.tilt)), add(tip, rot([-2, 5], p.tilt)), B([-26, 34])], BACK);
    S(tg, 'top', [B([-30, 8]), add(tip, rot([0, -5], p.tilt))], 1.8); S(tg, 'end', [add(tip, rot([0, -5], p.tilt)), add(tip, rot([-2, 5], p.tilt))], 1.6, .9); S(tg, 'bot', [add(tip, rot([-2, 5], p.tilt)), B([-26, 34])], 1.7);
    S(tg, 'mid', [B([-32, 20]), add(tip, rot([2, 0], p.tilt))], .9, .5);
    // body: one egg
    const egg = []; for (let i = 0; i < 26; i++) { const a = i / 26 * TAU, r = 1 + .09 * Math.cos(a - 1.2); egg.push(B([Math.cos(a) * 45 * r, Math.sin(a) * 50 * r])); }
    const body = G('body', egg, BACK); S(body, 'egg', egg, 2.2, 1, { close: true });
    const br = []; for (let i = 0; i <= 14; i++) { const a = -.5 + i / 14 * 2.5; br.push(B([Math.cos(a) * 43, Math.sin(a) * 48 + 1])); } br.push(B([4, 30]), B([14, -2]), B([30, -22]));
    const breast = G('breast', br, BREAST, .9);
    for (let i = 0; i < 11; i++) { const q = B([10 + hash(i, 3) * 26, -4 + i * 4.2]); S(breast, 'fluff/' + i, [q, add(q, [3, 5]), add(q, [4, 9])], .8, .35); }
    for (let i = 0; i < 7; i++) { const q = B([-30 + i * 3, -22 + i * 9]); S(body, 'shade/' + i, [q, add(q, [7, 6]), add(q, [10, 13])], .8, .3); }
    // satchel: strap across the body, bag on the front hip
    const s0 = B([-26, -34]), s1 = B([30, 22]); const sat = G('strap', null); S(sat, 'strap', [s0, B([2, -2]), s1], 2.6, .85, { color: '#8a6f45', pressure: even });
    const bc = add(s1, rot([2, 12], p.bag)), bw = 15 + 5 * p.bagFull, bagPts = [[-bw, -10], [bw, -10], [bw + 2, 12], [-bw - 2, 12]].map(q => add(bc, rot(q, p.bag + p.tilt * .4)));
    if (p.letter) { const L = [[-9, -22], [11, -24], [12, -8], [-8, -7]].map(q => add(bc, rot(q, p.bag + .15))), lg = G('letter', L, '#fbf7ee'); S(lg, 'env', L, 1.3, .9, { close: true, corner: .9 }); }
    const bag = G('bag', bagPts, BAG); S(bag, 'bag', bagPts, 1.8, .95, { close: true, corner: .9 }); S(bag, 'flap', [bagPts[0], add(bc, rot([0, 2], p.bag)), bagPts[1]], 1.4, .8);
    if (p.tear) S(bag, 'tear', [add(bc, rot([-6, 4], p.bag)), add(bc, rot([-1, 8], p.bag)), add(bc, rot([-5, 12], p.bag))], 1.3, .9, { corner: .8 });
    // wing: a teardrop hinged at the shoulder; wing = 0 folded .. 1 raised high, <0 drooped
    const sh = B([-10, -14]), wa = 2.2 - p.wing * 2.9 + p.tilt, wl = 46 * p.wingLen, wd = [Math.cos(wa), Math.sin(wa)], wn = [-wd[1], wd[0]];
    const wpts = [sh, add(add(sh, mul(wd, wl * .35)), mul(wn, -13)), add(add(sh, mul(wd, wl * .8)), mul(wn, -8)), add(sh, mul(wd, wl)), add(add(sh, mul(wd, wl * .7)), mul(wn, 9)), add(add(sh, mul(wd, wl * .3)), mul(wn, 12))];
    const wg = G('wing', wpts, BACK); S(wg, 'wing', wpts, 1.9, 1, { close: true });
    for (let i = 0; i < 3; i++) S(wg, 'pin/' + i, [add(add(sh, mul(wd, wl * (.45 + i * .14))), mul(wn, 7 - i)), add(add(sh, mul(wd, wl * (.72 + i * .1))), mul(wn, 3 - i * 2))], 1, .55);
    // face
    const [lx, ly] = p.look, e = B([24 + lx * 5, -30 + ly * 4]), face = G('face', null);
    if (p.eyes === 'closed') S(face, 'eye', [add(e, [-4.5, -1]), add(e, [0, 2]), add(e, [4.5, -1])], 1.8, 1, { pressure: even });
    else if (p.eyes === 'sad') { const r = []; for (let i = 0; i <= 10; i++) r.push(add(e, [Math.cos(i / 10 * TAU) * 5, Math.sin(i / 10 * TAU) * 5.4 + 1])); const eg = G('eyeball', r, '#2f2a26'); S(eg, 'rim', r, 1.2, 1, { pressure: even });
      const lid = [add(e, [-7, -6.5]), add(e, [7, -1]), add(e, [7, -8]), add(e, [-7, -9])]; G('lid', lid, BACK); S(face, 'lid', [add(e, [-7.5, -6.5]), add(e, [0, -3]), add(e, [7, -.5])], 1.7, 1, { pressure: even });
      const hl = []; for (let i = 0; i <= 6; i++) hl.push(add(e, [1.4 + Math.cos(i / 6 * TAU) * 1.4, 2.6 + Math.sin(i / 6 * TAU) * 1.4])); G('glint', hl, '#ffffff'); }
    else if (p.eyes === 'wide') { const r = []; for (let i = 0; i <= 10; i++) r.push(add(e, [Math.cos(i / 10 * TAU) * 6.6, Math.sin(i / 10 * TAU) * 7.4])); const wg = G('eyewhite', r, '#ffffff'); S(wg, 'rim', r, 1.4, 1, { pressure: even });
      const pu = []; for (let i = 0; i <= 8; i++) pu.push(add(e, [lx * 2.2 + Math.cos(i / 8 * TAU) * 3, ly * 2.4 + Math.sin(i / 8 * TAU) * 3.4])); G('pupil', pu, '#2f2a26'); }
    else { const r = []; for (let i = 0; i <= 10; i++) r.push(add(e, [Math.cos(i / 10 * TAU) * 5.2, Math.sin(i / 10 * TAU) * 6])); const eg = G('eyeball', r, '#2f2a26'); S(eg, 'rim', r, 1.2, 1, { pressure: even });
      const hl = []; for (let i = 0; i <= 6; i++) hl.push(add(e, [1.6 + Math.cos(i / 6 * TAU) * 1.7, -2.2 + Math.sin(i / 6 * TAU) * 1.7])); G('glint', hl, '#ffffff'); }
    if (p.brow) S(face, 'brow', [add(e, [-6, -9 - p.brow * 2.5]), add(e, [5, -9 + p.brow * 2.5])], 1.4, .85, { pressure: even });
    for (let i = 0; i < 3; i++) S(face, 'cheek/' + i, [add(e, [-12 + i * 3.5, 9]), add(e, [-14 + i * 3.5, 14])], .8, .35, { color: '#a63d2f', pressure: even });
    const bk = B([42, -22 + ly * 2]), up = [bk, add(bk, rot([15, 1 - p.beak * 5], p.tilt)), add(bk, rot([1, 6], p.tilt))], bg = G('beak', up, BEAK); S(bg, 'upper', up, 1.7, 1, { close: true, corner: .8 });
    if (p.beak > .1) S(bg, 'lower', [add(bk, rot([1, 6], p.tilt)), add(bk, rot([11, 6 + p.beak * 6], p.tilt)), add(bk, rot([0, 9], p.tilt))], 1.4, .9, { corner: .8 });
    // cap
    const cb = B([0, -45]), ca = p.tilt - .18 + p.cap, capPts = [[-17, 0], [-15, -12], [13, -13], [17, 0]].map(q => add(cb, rot(q, ca))), cg = G('cap', capPts, CAP);
    S(cg, 'cap', capPts, 1.7, 1, { close: true, corner: 1 }); S(cg, 'peak', [add(cb, rot([15, -1], ca)), add(cb, rot([29, 2], ca)), add(cb, rot([16, 3], ca))], 1.6, .95, { corner: .9 }); S(cg, 'band', [add(cb, rot([-16, -4], ca)), add(cb, rot([16, -4], ca))], 1, .6);
    if (p.tear2) S(face, 'teardrop', [add(e, [3, 7]), add(e, [1.5, 12]), add(e, [4.5, 12.5]), add(e, [3, 7])], 1.1, .8, { color: '#6f8fb5' });
    return { groups, wingTip: add(sh, mul(wd, wl)), bagAt: bc, head: B([10, -30]) };
  }

  const MOLEP = { nod: 0, lamp: .5, eyes: 'open', look: 0, wave: 0 };
  const mole = (o = {}) => ({ ...MOLEP, ...o });
  function buildMole(p) {
    const groups = [], G = (name, fill, fillColor = PAPER, alpha = 1) => { const g = { name, fill, fillColor, alpha, strokes: [] }; groups.push(g); return g; };
    const S = (g, id, points, width = 1.8, opacity = .95, extra = {}) => g.strokes.push({ id, points, width, opacity, pressure: soft, ...extra });
    // facing left, toward the pond
    const body = [[-30, 0], [-36, -40], [-26, -78], [-4, -92], [18, -80], [30, -40], [28, 0]], bg = G('body', body, MOLE); S(bg, 'body', body, 2.1); S(bg, 'base', [[-30, 0], [0, 2], [28, 0]], 1.6, .8);
    const sh = [[-30, -52], [-20, -74], [6, -80], [26, -66], [32, -34], [8, -42], [-12, -40]], sg = G('shawl', sh, SHAWL); S(sg, 'shawl', sh, 1.7, .95, { close: true }); for (let i = 0; i < 5; i++) S(sg, 'knit/' + i, [[-18 + i * 9, -68 + Math.abs(i - 2) * 4], [-20 + i * 9, -48 + Math.abs(i - 2) * 2]], .8, .4);
    const hc = [-6, -98 + p.nod * 6], hd = []; for (let i = 0; i < 20; i++) { const a = i / 20 * TAU, d = Math.atan2(Math.sin(a - Math.PI - .15), Math.cos(a - Math.PI - .15)); hd.push(add(hc, rot([Math.cos(a) * (24 + 20 * Math.exp(-d * d / .1)), Math.sin(a) * (22 + 3 * Math.exp(-d * d / .1))], p.nod * .25))); }
    const hg = G('head', hd, MOLE); S(hg, 'head', hd, 2, 1, { close: true });
    const nose = add(hc, rot([-43, 3], p.nod * .25)); S(hg, 'nose', [add(nose, [0, -3]), add(nose, [-3, 0]), add(nose, [0, 3]), add(nose, [2.5, 0]), add(nose, [0, -3])], 2.2, 1, { pressure: even, color: '#8a5a54' });
    for (let i = 0; i < 3; i++) S(hg, 'whisk/' + i, [add(nose, [8, 2 + i * 2]), add(nose, [-2, 8 + i * 5])], .7, .45, { pressure: even });
    const ey = add(hc, [-12 + p.look * 3, -5]); const ring = []; for (let i = 0; i <= 10; i++) ring.push(add(ey, [Math.cos(i / 10 * TAU) * 7, Math.sin(i / 10 * TAU) * 7])); S(hg, 'glasses', ring, 1.2, .85, { pressure: even }); S(hg, 'arm', [add(ey, [7, -1]), add(ey, [24, -4])], 1, .7, { pressure: even });
    if (p.eyes === 'closed') S(hg, 'eye', [add(ey, [-3, 0]), add(ey, [0, 1.6]), add(ey, [3, 0])], 1.5, 1, { pressure: even }); else S(hg, 'eye', [add(ey, [p.look, -1.6]), add(ey, [p.look, 1.6])], 2.2, 1, { pressure: even });
    // lamp arm toward the pond; lamp = 0 lowered .. 1 held high
    const s0 = [-22, -62], hand = [-52, -52 - p.lamp * 44], el = mixv(s0, hand, .5); el[1] += 12; const ag = G('arm', null); S(ag, 'arm', [s0, el, hand], 5.5, .9, { color: '#9c8f85', pressure: even });
    const lt = add(hand, [0, 6]), lampPts = [[-9, 0], [9, 0], [11, 22], [-11, 22]].map(q => add(lt, q)), lg = G('lamp', lampPts, p.lit === 0 ? '#ddd6c8' : LAMP); S(lg, 'lamp', lampPts, 1.6, 1, { close: true, corner: .9 }); S(lg, 'hook', [hand, add(lt, [0, -2])], 1.4, .9); S(lg, 'cap', [add(lt, [-6, 0]), add(lt, [0, -6]), add(lt, [6, 0])], 1.4, .9); S(lg, 'bar', [add(lt, [0, 2]), add(lt, [0, 21])], .9, .5);
    return { groups, lampAt: add(lt, [0, 11]) };
  }

  function compile(id, built) { built.id = id; built.cels = built.groups.map(g => compileCel({ strokes: g.strokes }, { id: id + '/' + g.name })); return built; }
  function draw(c, d, { material = 'pencil' } = {}) {
    d.groups.forEach((g, k) => { if (g.fill) { c.save(); c.globalAlpha *= g.alpha; c.fillStyle = g.fillColor; c.fill(curvePath(g.fill, true)); c.restore(); } if (g.strokes.length) drawCel(c, d.cels[k], { material, color: INK }); });
  }
  return { bird, mole, between, buildBird, buildMole, compile, draw, INK, PAPER, LAMP };
})();
