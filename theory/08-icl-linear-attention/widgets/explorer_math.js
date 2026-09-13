// Closed forms for in-context regression with isotropic Gaussian inputs, used live by the explorer widget.
// Model: x ~ N(0, I_d), w ~ N(0, I_d), y = w.x + sigma*eps, n context examples. Risk = E (y_hat - w.x_q)^2 / d.
// GD-k and ridge use the Marchenko-Pastur (proportional) limit for the spectrum of S = X^T X / n; GD-1 is exact.
(function (root) {
  function mpNodes(gamma, m) {
    const a = (1 - Math.sqrt(gamma)) ** 2, b = (1 + Math.sqrt(gamma)) ** 2;
    const s = new Float64Array(m), wt = new Float64Array(m);
    let tot = 0;
    for (let i = 0; i < m; i++) {
      const th = Math.PI * (i + 0.5) / m;
      s[i] = a + (b - a) * (1 - Math.cos(th)) / 2;
      const ds = (b - a) * Math.sin(th) / 2 * (Math.PI / m);
      wt[i] = Math.sqrt(Math.max(0, (b - s[i]) * (s[i] - a))) / (2 * Math.PI * gamma * s[i]) * ds;
      tot += wt[i];
    }
    const atom = Math.max(0, 1 - 1 / gamma);
    for (let i = 0; i < m; i++) wt[i] *= (1 - atom) / tot;   // renormalise quadrature
    return { s, wt, atom, a, b };
  }

  function ridgeRisk(d, n, sigma) {
    const q = mpNodes(d / n, 400);
    let r = q.atom;
    if (sigma > 0) for (let i = 0; i < q.s.length; i++) r += q.wt[i] * sigma * sigma / (sigma * sigma + n * q.s[i]);
    return r;
  }

  function gd1Exact(d, n, sigma) {       // optimal single step (= trained 1-layer LSA optimum, ZFB/Ahn)
    const eta = 1 / (1 + (d + 1) / n + sigma * sigma / n);
    return { risk: 1 - eta, eta };
  }

  function gdkRiskGrad(eta, q, n, sigma) {
    const k = eta.length, ns2 = sigma * sigma / n;
    let r = q.atom;
    const g = new Float64Array(k);
    for (let i = 0; i < q.s.length; i++) {
      const s = q.s[i];
      const f = eta.map(e => 1 - e * s);
      let p = 1; for (const v of f) p *= v;
      const c = q.wt[i];
      r += c * (p * p + (1 - p) * (1 - p) * ns2 / s);
      const dr_dp = c * (2 * p - 2 * (1 - p) * ns2 / s);
      for (let l = 0; l < k; l++) {
        let pl = 1; for (let m = 0; m < k; m++) if (m !== l) pl *= f[m];
        g[l] += dr_dp * (-s * pl);
      }
    }
    return { r, g };
  }

  // tuned step sizes for k steps of GD (oracle-tuned per (n, sigma)), Chebyshev-root init + Adam
  function gdkOptimal(d, n, sigma, k, iters) {
    const q = mpNodes(d / n, 240);
    const lo = Math.max(q.a, 0.02 * q.b), hi = q.b;
    let eta = Array.from({ length: k }, (_, l) => {
      const root = (hi + lo) / 2 + (hi - lo) / 2 * Math.cos(Math.PI * (2 * l + 1) / (2 * k));
      return 1 / root;
    });
    let best = gdkRiskGrad(eta, q, n, sigma).r, bestEta = eta.slice();
    const m = new Float64Array(k), v = new Float64Array(k);
    const lr = 0.02;
    for (let it = 1; it <= (iters || 400); it++) {
      const { r, g } = gdkRiskGrad(eta, q, n, sigma);
      if (r < best) { best = r; bestEta = eta.slice(); }
      for (let l = 0; l < k; l++) {
        m[l] = 0.9 * m[l] + 0.1 * g[l]; v[l] = 0.999 * v[l] + 0.001 * g[l] * g[l];
        const mh = m[l] / (1 - 0.9 ** it), vh = v[l] / (1 - 0.999 ** it);
        eta[l] -= lr * mh / (Math.sqrt(vh) + 1e-12) * Math.max(0.05, Math.abs(eta[l]));
      }
    }
    return { risk: best, eta: bestEta };
  }

  root.ICL = { mpNodes, ridgeRisk, gd1Exact, gdkOptimal, gdkRiskGrad };
  if (typeof module !== "undefined") module.exports = root.ICL;
})(typeof window !== "undefined" ? window : globalThis);
