'use strict';
// The winder: a small figure in a bell coat and a red scarf, facing the board (screen right).
// Every exposed drawing is a whole pose built once from a pose description: contours, overlaps,
// face and scarf belong to that drawing, and a held drawing keeps its marks. Limb solving only
// locates elbows and knees; the visible outline is drawn around them.
// Load after core.js, studio.js, cels.js.
const Winder = (() => {
  const RED = '#a63d2f', RED_FILL = '#e3b5a6', GRAPHITE = '#3d362f', PAPER = '#f5f0e5';
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]], sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const mul = (a, k) => [a[0] * k, a[1] * k], mixv = (a, b, u) => [lerp(a[0], b[0], u), lerp(a[1], b[1], u)];
  const rot = (p, a) => { const c = Math.cos(a), s = Math.sin(a); return [p[0] * c - p[1] * s, p[0] * s + p[1] * c]; };
  const soft = [[0, .12], [.14, .85], [.45, 1], [.8, .7], [1, .08]], even = [[0, .7], [.5, 1], [1, .7]];

  // Local units: origin on the ground between the feet, y up is negative, about 250 tall.
  const REST = {
    pelvis: [0, -84], lean: 0, tilt: 0, look: [0, 0], eyes: 'open', brow: 0, mouth: 'none',
    footA: [-17, 0], footB: [15, 0], pitchA: 0, pitchB: 0,
    handA: [-16, -78], handB: [26, -82], bendA: 1, bendB: 1,
    scarf: { angle: 1.85, wave: .22, phase: 0, len: 78 }, hair: 0, hem: 0, flare: 0,
  };
  const pose = (o = {}) => ({ ...REST, ...o, scarf: { ...REST.scarf, ...(o.scarf || {}) } });

  // Numeric fields interpolate; discrete face fields switch at the authored fraction.
  function between(a, b, u, faceAt = .5) {
    const n = (x, y) => Array.isArray(x) ? x.map((v, i) => lerp(v, y[i], u)) : lerp(x, y, u);
    const out = {};
    for (const k of Object.keys(REST)) {
      if (k === 'scarf') out.scarf = Object.fromEntries(Object.keys(REST.scarf).map(f => [f, lerp(a.scarf[f], b.scarf[f], u)]));
      else if (typeof REST[k] === 'string' || k === 'bendA' || k === 'bendB') out[k] = u < faceAt ? a[k] : b[k];
      else out[k] = n(a[k], b[k]);
    }
    return out;
  }

  // A tube drawn as two edges around a centreline; returns edges and a fill polygon.
  function tube(centre, w0, w1, n = 7) {
    const path = motionPath(centre, { smooth: true }), top = [], bot = [];
    for (let i = 0; i <= n; i++) {
      const u = i / n, q = path.at(u), w = lerp(w0, w1, u), nx = -q.tangent[1], ny = q.tangent[0];
      top.push([q.p[0] + nx * w, q.p[1] + ny * w]); bot.push([q.p[0] - nx * w, q.p[1] - ny * w]);
    }
    return { top, bot, fill: [...top, ...bot.slice().reverse()], end: path.at(1), path };
  }

  function build(p) {
    const groups = [], G = (name, fill, fillColor = PAPER) => { const g = { name, fill, fillColor, strokes: [] }; groups.push(g); return g; };
    const S = (g, id, points, width = 1.8, opacity = .95, extra = {}) => g.strokes.push({ id, points, width, opacity, pressure: soft, ...extra });
    const P = p.pelvis, up = [Math.sin(p.lean), -Math.cos(p.lean)], fr = [Math.cos(p.lean), Math.sin(p.lean)];
    const N = add(P, mul(up, 80)), spine = s => mixv(N, P, s);
    const headA = p.lean * .45 + p.tilt, C = add(N, rot([6, -43], headA));
    const hl = q => add(C, rot(q, headA));                       // head-local to figure-local

    // limbs: far side first
    const arm = (key, sh, hand, bend) => {
      const L = solveLimb(sh, hand, 44, 42, bend), t = tube([L.root, mixv(L.root, L.joint, .6), L.joint, mixv(L.joint, L.end, .55), L.end], 10, 7.2);
      const g = G('arm' + key, t.fill);
      S(g, 'sleeve/top', t.top, 1.9); S(g, 'sleeve/bot', t.bot, 1.7, .9);
      const d = t.end.tangent, c0 = L.end, nrm = [-d[1], d[0]];
      S(g, 'cuff', [add(c0, mul(nrm, 8.2)), add(c0, mul(d, 1.5)), add(c0, mul(nrm, -8.2))], 1.5, .85);
      const hc = add(c0, mul(d, 8)), mit = [];
      for (let i = 0; i <= 10; i++) { const a = i / 10 * TAU; mit.push(add(hc, rot([Math.cos(a) * 8.4, Math.sin(a) * 7], Math.atan2(d[1], d[0])))); }
      const gh = G('hand' + key, mit); S(gh, 'mitt', mit, 1.7, .95, { close: true });
      S(gh, 'thumb', [add(hc, rot([-2, -6.5], Math.atan2(d[1], d[0]))), add(hc, rot([4, -10], Math.atan2(d[1], d[0]))), add(hc, rot([7, -5.5], Math.atan2(d[1], d[0])))], 1.3, .8);
      return { grip: add(c0, mul(d, 11)), limb: L };
    };
    const leg = (key, hip, foot, pitch) => {
      const ankle = add(foot, rot([0, -13], pitch)), L = solveLimb(hip, ankle, 31, 30, 1);
      const t = tube([L.root, L.joint, L.end], 7.2, 5.6, 5), g = G('leg' + key, t.fill);
      S(g, 'front', t.top, 1.6, .9); S(g, 'back', t.bot, 1.6, .9);
      const boot = [[-10, -16], [-11.5, -3], [-9, 0], [17, 0], [21, -4.5], [15, -10], [6.5, -12.5], [5.5, -17]].map(q => add(foot, rot(q, pitch)));
      const b = G('boot' + key, boot); S(b, 'boot', boot, 2, 1, { close: true, corner: 1.1 });
      S(b, 'sole', [add(foot, rot([-9, -3.2], pitch)), add(foot, rot([4, -2.6], pitch)), add(foot, rot([19, -3.4], pitch))], 1.1, .6);
      return L;
    };
    const shA = add(add(N, mul(up, -13)), mul(fr, 5)), shB = add(add(N, mul(up, -14)), mul(fr, -3));
    const farArm = arm('A', shA, p.handA, p.bendA);
    const legs = { A: leg('A', add(P, [-12 + p.hem * .2, 16]), p.footA, p.pitchA), B: leg('B', add(P, [9 + p.hem * .2, 16]), p.footB, p.pitchB) };

    // coat: one bell contour; the hem lags the body (hem) and opens in wind (flare)
    const hang = Math.sin(p.lean) * 36;                             // cloth hangs from the shoulders, so a lean carries the hem with it
    const hemB = add(P, [-47 + hang + p.hem - p.flare, 31]), hemF = add(P, [41 + hang * .8 + p.hem + p.flare * .4, 31]);
    const back = [add(N, mul(fr, -14)), add(spine(.33), mul(fr, -25)), add(spine(.68), mul(fr, -36)), hemB];
    const front = [add(N, mul(fr, 12)), add(spine(.33), mul(fr, 23)), add(spine(.68), mul(fr, 32)), hemF];
    const hem = [hemB, add(mixv(hemB, hemF, .32), [0, 8]), add(mixv(hemB, hemF, .7), [0, 9]), hemF];
    const coat = G('coat', [...back, ...hem.slice(1, 3), ...front.slice().reverse()]);
    S(coat, 'back', back, 2.2); S(coat, 'front', front, 2.1); S(coat, 'hem', hem, 1.9, .9);
    S(coat, 'placket', [add(N, mul(fr, 5)), add(spine(.4), mul(fr, 11)), add(spine(.75), mul(fr, 14)), add(mixv(hemB, hemF, .66), [0, 8])], 1.1, .55);
    for (let i = 0; i < 2; i++) { const c = add(spine(.34 + i * .26), mul(fr, 17 + i * 2)); S(coat, 'button/' + i, [add(c, [-2, -2]), add(c, [2, 0]), add(c, [-1, 2.5]), add(c, [-2, -2])], 1.2, .75, { pressure: even }); }
    for (let i = 0; i < 9; i++) {                                   // shadow hatching follows the back of the bell
      const s = .2 + i * .085, a = add(spine(s), mul(fr, -lerp(18, 32, s))), len = 13 + hash(i, 5) * 9;
      S(coat, 'shade/' + i, [a, add(a, [len * .55, len * .5]), add(a, [len * .8, len])], .8, .34 + hash(i, 6) * .2);
    }
    S(coat, 'back/search', back.slice(1).map(([x, y]) => [x - 2.5, y + 2]), .8, .25);

    // scarf tail streams from the back of the neck; drawn before the head
    const sc = p.scarf, t0 = add(N, add(mul(fr, -11), mul(up, 2))), cl = [t0];
    for (let k = 1; k <= 6; k++) { const a = sc.angle + sc.wave * Math.sin(sc.phase + k * .95) * (.35 + k / 6); cl.push(add(cl[k - 1], mul([Math.cos(a), Math.sin(a)], sc.len / 6))); }
    const tt = tube(cl, 7.5, 6, 8), tail = G('scarfTail', tt.fill, RED_FILL);
    S(tail, 'top', tt.top, 1.7, .95, { color: RED }); S(tail, 'bot', tt.bot, 1.7, .95, { color: RED });
    S(tail, 'end', [tt.top[8], add(mixv(tt.top[8], tt.bot[8], .5), mul(tt.end.tangent, 2)), tt.bot[8]], 1.5, .9, { color: RED });
    for (let i = 0; i < 3; i++) { const b = mixv(tt.top[8], tt.bot[8], .2 + i * .3); S(tail, 'fringe/' + i, [b, add(b, mul(tt.end.tangent, 7 + (i % 2) * 2))], 1, .8, { color: RED, pressure: even }); }
    for (let i = 0; i < 5; i++) { const u = .15 + i * .17, q = tt.path.at(u), n = [-q.tangent[1], q.tangent[0]]; S(tail, 'rib/' + i, [add(q.p, mul(n, 4.5)), add(q.p, mul(n, -4.5))], .9, .45, { color: RED, pressure: even }); }

    // head: one closed contour with the nose on the facing side
    const head = [];
    for (let i = 0; i < 22; i++) { const a = i / 22 * TAU, d = Math.atan2(Math.sin(a - .16), Math.cos(a - .16)); head.push(hl([Math.cos(a) * (40 + 8 * Math.exp(-d * d / .05)), Math.sin(a) * (38 + 5 * Math.exp(-d * d / .05))])); }
    const hd = G('head', head); S(hd, 'skull', head, 2.1, 1, { close: true });
    const [lx, ly] = p.look, eye = (id, x, y) => {
      const e = [x + lx * 6, y + ly * 5];
      if (p.eyes === 'closed') S(hd, id, [hl([e[0] - 4.5, e[1] - 1]), hl([e[0], e[1] + 2.2]), hl([e[0] + 4.5, e[1] - 1])], 1.7, 1, { pressure: even });
      else if (p.eyes === 'wide') { const r = []; for (let i = 0; i <= 8; i++) r.push(hl([e[0] + Math.cos(i / 8 * TAU) * 3.4, e[1] + Math.sin(i / 8 * TAU) * 4.4])); S(hd, id, r, 1.6, 1, { pressure: even }); S(hd, id + '/pupil', [hl([e[0] + lx, e[1] - 1.5 + ly]), hl([e[0] + lx, e[1] + 1.5 + ly])], 2.4, 1, { pressure: even }); }
      else S(hd, id, [hl([e[0], e[1] - 4.2]), hl([e[0] + .4, e[1]]), hl([e[0], e[1] + 4.2])], 2.6, 1, { pressure: even });
    };
    eye('eye/near', 13, -5); eye('eye/far', 29, -6);
    if (p.brow) for (const [id, x] of [['brow/near', 13], ['brow/far', 29]]) S(hd, id, [hl([x - 5 + lx * 5, -15 + ly * 4 - p.brow * (id === 'brow/near' ? -2.5 : 2.5) - Math.abs(p.brow) * 2]), hl([x + 5 + lx * 5, -15 + ly * 4 + p.brow * (id === 'brow/near' ? -2.5 : 2.5) - Math.abs(p.brow) * 2])], 1.4, .85, { pressure: even });
    if (p.mouth === 'o') { const m = []; for (let i = 0; i <= 8; i++) m.push(hl([24 + lx * 3 + Math.cos(i / 8 * TAU) * 3, 17 + Math.sin(i / 8 * TAU) * 4])); S(hd, 'mouth', m, 1.4, .9, { pressure: even }); }
    if (p.mouth === 'flat') S(hd, 'mouth', [hl([19, 17]), hl([24, 18]), hl([29, 16.5])], 1.4, .85, { pressure: even });
    if (p.mouth === 'smile') S(hd, 'mouth', [hl([18, 15]), hl([24, 19]), hl([30, 14.5])], 1.4, .9, { pressure: even });
    for (let i = 0; i < 4; i++) S(hd, 'cheek/' + i, [hl([2 + i * 4, 9]), hl([-1 + i * 4, 15])], .8, .32, { color: RED, pressure: even });
    for (let i = 0; i < 3; i++) { const b = [-8 + i * 7, -37.5 - (i === 1 ? 1.5 : 0)], tip = [b[0] - 9 + p.hair * 24 + i * 5, b[1] - 17 + Math.abs(p.hair) * 9 - (i === 1 ? 5 : 0)]; S(hd, 'hair/' + i, [hl(b), hl([lerp(b[0], tip[0], .4) + 3, lerp(b[1], tip[1], .65)]), hl(tip)], 1.6, .9); }

    // scarf wrap covers the neck joint
    const wa = add(N, add(mul(fr, -17), mul(up, 5))), wb = add(N, add(mul(fr, 16), mul(up, 1))), wrap = tube([wa, add(mixv(wa, wb, .5), mul(up, -5)), wb], 8, 7.5, 5);
    const w = G('scarfWrap', wrap.fill, RED_FILL); S(w, 'top', wrap.top, 1.7, .95, { color: RED }); S(w, 'bot', wrap.bot, 1.8, .95, { color: RED });
    S(w, 'a', [wrap.top[0], wrap.bot[0]], 1.4, .9, { color: RED, pressure: even }); S(w, 'b', [wrap.top[5], wrap.bot[5]], 1.4, .9, { color: RED, pressure: even });
    for (let i = 0; i < 4; i++) { const q = wrap.path.at(.2 + i * .2); S(w, 'rib/' + i, [add(q.p, [1.5, -5]), add(q.p, [-1.5, 5])], .9, .45, { color: RED, pressure: even }); }

    const nearArm = arm('B', shB, p.handB, p.bendB);
    const errs = [farArm.limb, nearArm.limb, legs.A, legs.B].map(l => l.error || 0);
    return { groups, gripA: farArm.grip, gripB: nearArm.grip, headCentre: C, reachError: Math.max(...errs) };
  }

  // Compile once per drawing id; the id seeds every mark, so a held drawing holds its marks.
  function compile(id, p) {
    const d = build(p);
    d.id = id; d.cels = d.groups.map(g => compileCel({ strokes: g.strokes }, { id: id + '/' + g.name }));
    return d;
  }
  function draw(c, d, { material = 'pencil' } = {}) {
    d.groups.forEach((g, k) => {
      if (g.fill) { c.save(); c.fillStyle = g.fillColor; c.fill(curvePath(g.fill, true)); c.restore(); }
      drawCel(c, d.cels[k], { material, color: GRAPHITE });
    });
  }
  return { pose, between, build, compile, draw, RED, RED_FILL, GRAPHITE, PAPER };
})();
