// Algorithm explorer: risk vs number of in-context examples n, for d = 10 isotropic regression.
// Curves are live closed forms (explorer_math.js); dots are trained models from train_lsa_deep.py (data_explorer.js).
(function () {
  const root = document.getElementById("explorer");
  if (!root || !window.ICL) return;
  const I = window.ICL, D = 10;
  const DATA = (window.EXPLORER_DATA || { points: [] }).points;
  const INK = "#1d1d1f", MUTED = "#6b6b70", RULE = "#e6e2da", GD = "#8f887c", GD1 = "#b9b2a6", LSA = "#2a78d6";
  const NS = Array.from({ length: 48 }, (_, i) => Math.round(11 * Math.pow(160 / 11, i / 47)));   // n in (d, 160]
  const NS_UNDER = [2, 3, 4, 5, 6, 7, 8, 9, 10];

  root.innerHTML = `
    <div class="ex-controls">
      <label>GD steps / attention layers <b>k</b> <input type="range" min="1" max="8" step="1" value="2" data-k><output></output></label>
      <label>label noise <b>σ</b> <input type="range" min="0" max="1" step="0.05" value="0" data-s><output></output></label>
    </div>
    <svg class="ex-svg" viewBox="0 0 760 420" role="img" aria-label="risk versus number of in-context examples"></svg>
    <div class="ex-readout"></div>`;
  const svg = root.querySelector("svg");
  const kIn = root.querySelector("[data-k]"), sIn = root.querySelector("[data-s]");
  const W = 760, H = 420, L = 62, R = 190, T = 18, B = 46, pw = W - L - R, ph = H - T - B;
  const xlo = Math.log(2), xhi = Math.log(160), ylo = Math.log10(1e-3), yhi = Math.log10(1.2);
  const X = n => L + (Math.log(n) - xlo) / (xhi - xlo) * pw;
  const Y = r => T + ph * (1 - (Math.log10(Math.max(1e-3, Math.min(1.2, r))) - ylo) / (yhi - ylo));
  const el = (tag, attrs, parent) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
  const cache = new Map();

  function gdk(n, s, k) {
    const key = `${n}|${s}|${k}`;
    if (!cache.has(key)) cache.set(key, n <= D ? null : I.gdkOptimal(D, n, s, k, 250).risk);
    return cache.get(key);
  }

  function draw() {
    const k = +kIn.value, s = +sIn.value;
    kIn.nextElementSibling.textContent = k; sIn.nextElementSibling.textContent = s.toFixed(2);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    // axes
    for (const r of [1e-3, 1e-2, 1e-1, 1]) {
      el("line", { x1: L, x2: L + pw, y1: Y(r), y2: Y(r), stroke: RULE });
      el("text", { x: L - 8, y: Y(r) + 4, "text-anchor": "end", fill: MUTED, "font-size": 11 }).textContent = r >= 0.1 ? r : r.toExponential(0);
    }
    for (const n of [2, 5, 10, 20, 40, 80, 160]) {
      el("line", { x1: X(n), x2: X(n), y1: T, y2: T + ph, stroke: RULE });
      el("text", { x: X(n), y: T + ph + 16, "text-anchor": "middle", fill: MUTED, "font-size": 11 }).textContent = n;
    }
    el("text", { x: L + pw / 2, y: H - 8, "text-anchor": "middle", fill: MUTED, "font-size": 12 }).textContent = "number of in-context examples n   (dimension d = 10)";
    const yl = el("text", { x: 16, y: T + ph / 2, "text-anchor": "middle", fill: MUTED, "font-size": 12, transform: `rotate(-90 16 ${T + ph / 2})` });
    yl.textContent = "excess risk / d  (0 = perfect, 1 = predict 0)";
    el("line", { x1: X(D), x2: X(D), y1: T, y2: T + ph, stroke: "#c9c4ba", "stroke-width": 1 });
    el("text", { x: X(D) + 4, y: T + 12, fill: MUTED, "font-size": 11 }).textContent = "n = d";

    const path = (pts, attrs) => {
      const dstr = pts.filter(p => p[1] !== null && isFinite(p[1])).map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      return el("path", Object.assign({ d: dstr, fill: "none", "stroke-linejoin": "round", "stroke-linecap": "round" }, attrs));
    };
    const all = NS_UNDER.concat(NS);
    const ridge = all.map(n => [n, I.ridgeRisk(D, n, s)]);
    const gd1 = all.map(n => [n, I.gd1Exact(D, n, s).risk]);
    const gdK = NS.map(n => [n, gdk(n, s, k)]);
    path(gd1, { stroke: GD1, "stroke-width": 2 });
    path(gdK, { stroke: GD, "stroke-width": 2.5 });
    path(ridge.filter(p => p[1] > 1.05e-3), { stroke: INK, "stroke-width": 2 });

    // trained models (only drawn where they exist for this sigma)
    const pts = DATA.filter(p => Math.abs(p.sigma - s) < 1e-6);
    for (const p of pts) {
      if (p.kind === "dense" && p.L === k) {
        el("circle", { cx: X(p.n), cy: Y(p.risk / D), r: 7, fill: LSA, stroke: "#fcfbf8", "stroke-width": 2 }).appendChild(
          Object.assign(document.createElementNS("http://www.w3.org/2000/svg", "title"), { textContent: `trained ${k}-layer linear attention, n=${p.n}: ${(p.risk / D).toFixed(4)}` }));
      }
      if (p.kind === "gd" && p.L === k) {
        el("circle", { cx: X(p.n), cy: Y(p.risk / D), r: 4.5, fill: "#fcfbf8", stroke: GD, "stroke-width": 2 }).appendChild(
          Object.assign(document.createElementNS("http://www.w3.org/2000/svg", "title"), { textContent: `trained step sizes, ${k}-step GD, n=${p.n}: ${(p.risk / D).toFixed(4)}` }));
      }
    }
    // direct labels at the right edge
    const lab = [[`ridge (Bayes optimal)`, ridge[ridge.length - 1][1], INK], [`GD, 1 step (= 1 LSA layer)`, gd1[gd1.length - 1][1], GD1], [`GD, ${k} tuned steps`, gdK[gdK.length - 1][1], GD]];
    const ys = lab.map(l => Y(Math.max(l[1], 1.1e-3)));
    for (let i = 1; i < ys.length; i++) for (let j = 0; j < i; j++) if (Math.abs(ys[i] - ys[j]) < 15) ys[i] = ys[j] + (ys[i] >= ys[j] ? 15 : -15);
    lab.forEach((l, i) => { el("text", { x: L + pw + 8, y: ys[i] + 4, fill: l[2] === GD1 ? "#8f887c" : l[2], "font-size": 12, "font-weight": 600 }).textContent = l[0]; });
    // legend for markers
    const ly = T + ph - 40;
    el("circle", { cx: L + pw + 14, cy: ly, r: 6, fill: LSA, stroke: "#fcfbf8", "stroke-width": 2 });
    el("text", { x: L + pw + 26, y: ly + 4, fill: INK, "font-size": 12 }).textContent = `trained ${k}-layer lin. attn`;
    el("circle", { cx: L + pw + 14, cy: ly + 20, r: 4.5, fill: "#fcfbf8", stroke: GD, "stroke-width": 2 });
    el("text", { x: L + pw + 26, y: ly + 24, fill: INK, "font-size": 12 }).textContent = `trained ${k}-step GD`;

    const have = pts.filter(p => p.L === k).length;
    const n20 = pts.find(p => p.kind === "dense" && p.L === k && p.n === 20), g20 = pts.find(p => p.kind === "gd" && p.L === k && p.n === 20);
    root.querySelector(".ex-readout").innerHTML = have
      ? `At n = 20: tuned ${k}-step GD (closed form) ${gdk(20, s, k).toFixed(3)}` + (g20 ? `, trained step sizes ${(g20.risk / D).toFixed(3)}` : "") + (n20 ? `, <b style="color:${LSA}">trained linear attention ${(n20.risk / D).toFixed(3)}</b>` : "") + "."
      : `No trained models for k = ${k}, σ = ${s.toFixed(2)} — curves only (trained dots exist for k ≤ 4 and σ ∈ {0, 0.5}).`;
  }
  kIn.addEventListener("input", draw); sIn.addEventListener("input", draw);
  draw();

  if (location.hash === "#selftest") {
    const ok = [];
    const g = I.gdkOptimal(D, 20, 0, 3, 3000).risk, f = 0.5 ** 3 * 0.5 / (1 - 0.5 ** 4);
    ok.push(Math.abs(g - f) < 1e-4);
    for (const kk of [1, 4, 8]) for (const ss of [0, 0.5, 1]) { kIn.value = kk; sIn.value = ss; draw(); }
    ok.push(svg.querySelectorAll("path").length === 3);
    console.log(`SELFTEST explorer gd3(gamma=.5)=${g.toFixed(5)} formula=${f.toFixed(5)} points=${DATA.length} ${ok.every(Boolean) ? "PASS" : "FAIL"}`);
    kIn.value = 2; sIn.value = 0; draw();
  }
})();
