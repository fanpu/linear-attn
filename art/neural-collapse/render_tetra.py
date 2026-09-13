"""Terminal Phase I-V: the four centred class means of the C=4 run (exactly 3-D, so this is
faithful up to rotation and one global scale) at five checkpoints, plus STL meshes.

  python render_tetra.py c4 --epochs 0 1 8 83 250 --style brass|plotter|cyanotype|spectral
  python render_tetra.py c4 --epochs ... --stl
"""
import argparse, os, struct, sys
import numpy as np
import nclib as N
import artlib as A
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
IDEAL_DEG = np.degrees(np.arccos(-1 / 3))


def rot(az, el):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def tetra_state(tag, epoch, ref=None):
    meta, mets, ck = N.load_run(tag)
    e, path = N.pick_ckpts(ck, [epoch])[0]
    z = np.load(path)
    ytr, yte = N.labels(meta)
    al = N.Aligner(z["mu"], z["muG"], ref=ref)
    V = al.coords_means
    Mc = z["mu"] - z["muG"]
    cos = (Mc @ Mc.T) / np.outer(np.linalg.norm(Mc, axis=1), np.linalg.norm(Mc, axis=1))
    ang = {ed: np.degrees(np.arccos(np.clip(cos[ed], -1, 1))) for ed in EDGES}
    Wp = z["W"] @ al.proj
    Wp *= np.linalg.norm(al.F) / np.linalg.norm(Wp)
    return dict(epoch=e, al=al, V=V, W=Wp, ang=ang, Xtr=al(z["h_train"]), Xte=al(z["h_test"]), ytr=ytr, yte=yte,
                norms=np.linalg.norm(Mc, axis=1))


def project3(X, R):
    Y = X @ R.T
    return Y[:, :2], Y[:, 2]  # screen xy, depth (larger = toward viewer)


def draw_tetra(ax, V, R, style, lw, ghost=None, edge_colors=None, W=None):
    xy, dep = project3(V, R)
    dn = (dep - dep.min()) / (np.ptp(dep) + 1e-9)
    order = sorted(EDGES, key=lambda ed: dep[list(ed)].mean())
    if ghost is not None:
        gxy, _ = project3(ghost, R)
        for i, j in EDGES:
            ax.plot(*gxy[[i, j]].T, **style["ghost"], zorder=2)
    for ed in order:
        i, j = ed
        t = dep[list(ed)].mean()
        t = (t - dep.min()) / (np.ptp(dep) + 1e-9)
        w = lw * (0.65 + 0.35 * t)
        base = edge_colors[ed] if edge_colors is not None else style["rod"]
        if style.get("tube"):
            ax.plot(*xy[[i, j]].T, color=style["outline"], lw=w * 1.5, solid_capstyle="round", zorder=5)
            ax.plot(*xy[[i, j]].T, color=base, lw=w, solid_capstyle="round", zorder=6)
            ax.plot(*xy[[i, j]].T, color=style["hilite"], lw=w * 0.22, solid_capstyle="round", zorder=7, alpha=0.8)
        else:
            dashed = style.get("hidden_dashed") and t < 0.35
            ax.plot(*xy[[i, j]].T, color=base, lw=w, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="round", zorder=6)
    for v in np.argsort(dep):
        ax.scatter(*xy[v], s=(lw * style.get("node", 2.2)) ** 2, color=style["node_col"], zorder=8, linewidths=0)
    if W is not None:
        wxy, _ = project3(W, R)
        for v in range(4):
            ax.plot([0, wxy[v, 0]], [0, wxy[v, 1]], **style["wline"], zorder=4)


STYLES = {
    "brass": dict(bg="#0b0a09", rod="#c9a25a", outline="#2a1d0c", hilite="#fff1c9", node_col="#f3dca0", tube=True,
                  ghost=dict(color=(1, 1, 1, 0.10), lw=0.6, ls=(0, (1, 2))), text="#d9c7a0", dust=True),
    "plotter": dict(bg=A.PAPER, rod=A.INK, node_col=A.INK, hidden_dashed=True, node=1.8,
                    ghost=dict(color=(0.1, 0.1, 0.12, 0.35), lw=0.35, ls=(0, (1, 1.6))), text=A.INK, dust=False),
    "cyanotype": dict(bg="#153f6e", rod="#eef3f6", node_col="#eef3f6", hidden_dashed=True, node=1.8,
                      ghost=dict(color=(0.93, 0.95, 0.96, 0.35), lw=0.4, ls=(0, (4, 3))), text="#dce9f2", dust=False),
    "spectral": dict(bg="#101014", rod="#ffffff", outline="#000000", hilite="#ffffff", node_col="#f2efe6", tube=True,
                     ghost=dict(color=(1, 1, 1, 0.13), lw=0.5, ls=(0, (1, 2))), text="#e8e4d8", dust=False),
}


def render_series(states, style_name, R, panel_px=1400, captions=True, dust_colors=None, fname=None):
    st = STYLES[style_name]
    n = len(states)
    H = int(panel_px * (1.28 if captions else 1.0))
    fig = A.canvas(panel_px * n, H, st["bg"])
    ext = (-1.05, 1.05, -1.05, 1.05)
    lw = 7.0 * panel_px / 1400 if st.get("tube") else 1.6 * panel_px / 1400
    edge_colors = None
    if style_name == "spectral":
        dev = np.array([[s["ang"][ed] - IDEAL_DEG for ed in EDGES] for s in states])
        v = P.signed_rank_normalize(dev.ravel(), near_boundary="small").reshape(dev.shape)
        cmap = P.split_cmap(P.SIDES["spectral_purple"], P.SIDES["spectral_red"])
        edge_colors = [{ed: cmap((v[i, q] + 1) / 2) for q, ed in enumerate(EDGES)} for i in range(n)]
    for i, s in enumerate(states):
        top = (H - panel_px) / H
        ax = A.panel(fig, [i / n, top if captions else 0, 1 / n, panel_px / H], ext)
        if st["dust"]:
            cols = dust_colors
            img = np.zeros((panel_px, panel_px, 3))
            for X, y in [(s["Xte"], s["yte"]), (s["Xtr"], s["ytr"])]:
                xy, _ = project3(X, R)
                for j in range(4):
                    img += A.glow(A.splat(xy[y == j], ext, panel_px), (0.8, 3, 12), (1, .3, .1))[..., None] * cols[j]
            img = A.tonemap(img / np.percentile(img.max(-1), 99.7), 2.0)  # auto exposure (declared)
            ax.imshow(img, extent=ext, zorder=1, interpolation="lanczos")
        draw_tetra(ax, s["V"], R, st, lw, ghost=s["al"].F, edge_colors=None if edge_colors is None else edge_colors[i])
        if captions:
            cx = fig.add_axes([i / n, 0, 1 / n, top]); cx.set_axis_off(); cx.set_xlim(0, 1); cx.set_ylim(0, 1)
            angs = np.array([s["ang"][ed] for ed in EDGES])
            fs = 15 * panel_px / 1400
            cx.text(0.5, 0.78, ROMAN[i], ha="center", va="center", color=st["text"], fontsize=fs * 2.1, family="serif")
            ep = s["epoch"]
            eps = f"epoch {ep:.0f}" if ep >= 1 else (f"iteration {round(ep * 390.6)}" if ep > 0 else "initialisation")
            cx.text(0.5, 0.50, eps, ha="center", va="center", color=st["text"], fontsize=fs, family="serif", style="italic")
            cx.text(0.5, 0.30, "angles " + "  ".join(f"{a:.1f}" for a in angs) + "°", ha="center", va="center",
                    color=st["text"], fontsize=fs * 0.72, family="monospace")
            cx.text(0.5, 0.15, f"max |angle − 109.47°| = {np.abs(angs - IDEAL_DEG).max():.2f}°", ha="center",
                    va="center", color=st["text"], fontsize=fs * 0.72, family="monospace")
    return A.save(fig, fname)


# ------------------------------------------------------------------ STL
def _tri_bin(tris):
    out = bytearray(b"neural-collapse measured class-mean frame".ljust(80, b" "))
    out += struct.pack("<I", len(tris))
    for t in tris:
        nrm = np.cross(t[1] - t[0], t[2] - t[0]); nn = np.linalg.norm(nrm)
        nrm = nrm / nn if nn > 0 else nrm
        out += struct.pack("<12fH", *nrm, *t[0], *t[1], *t[2], 0)
    return bytes(out)


def cylinder(p, q, r, n=24):
    ax = q - p; L = np.linalg.norm(ax); ax = ax / L
    u = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
    A0, A1 = p + r * ring, q + r * ring
    tris = []
    for i in range(n):
        k = (i + 1) % n
        tris += [(A0[i], A1[i], A1[k]), (A0[i], A1[k], A0[k]), (p, A0[k], A0[i]), (q, A1[i], A1[k])]
    return tris


def sphere(c, r, n=16):
    tris = []
    th = np.linspace(0, np.pi, n + 1); ph = np.linspace(0, 2 * np.pi, 2 * n + 1)
    Pt = lambda a, b: c + r * np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
    for i in range(n):
        for j in range(2 * n):
            p00, p01, p10, p11 = Pt(th[i], ph[j]), Pt(th[i], ph[j + 1]), Pt(th[i + 1], ph[j]), Pt(th[i + 1], ph[j + 1])
            if i > 0: tris.append((p00, p10, p01))
            if i < n - 1: tris.append((p01, p10, p11))
    return tris


def export_stl(V, path, circum_mm=50.0, rod_mm=1.6, node_mm=3.2, hub=True):
    """Rods on the 6 edges + spheres at the 4 vertices (+ optional 4 spokes from the global mean).
    Scale: the IDEAL tetrahedron would have circumradius circum_mm (global scale of the aligned frame)."""
    V = np.asarray(V, float) * circum_mm / np.sqrt(3 / 4)
    tris = []
    for i, j in EDGES:
        tris += cylinder(V[i], V[j], rod_mm)
    for v in V:
        tris += sphere(v, node_mm)
    if hub:
        for v in V:
            tris += cylinder(np.zeros(3), v, rod_mm * 0.45, n=12)
        tris += sphere(np.zeros(3), node_mm * 0.6)
    tris = [tuple(np.asarray(x, np.float32) for x in t) for t in tris]
    open(path, "wb").write(_tri_bin(tris))
    return len(tris)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--epochs", type=float, nargs="+", required=True)
    ap.add_argument("--style", default="brass"); ap.add_argument("--az", type=float, default=20)
    ap.add_argument("--el", type=float, default=-20); ap.add_argument("--panel", type=int, default=1400)
    ap.add_argument("--stl", action="store_true")
    a = ap.parse_args()
    states = [tetra_state(a.tag, e) for e in a.epochs]
    for i, s in enumerate(states):
        angs = np.array(list(s["ang"].values()))
        print(ROMAN[i], f"ep {s['epoch']:.3f} resid {s['al'].residual:.4f} max|dAngle| {np.abs(angs-IDEAL_DEG).max():.3f} "
              f"norm cv {s['norms'].std()/s['norms'].mean():.4f}")
    if a.stl:
        os.makedirs(os.path.join(A.GAL, "stl"), exist_ok=True)
        for i, s in enumerate(states):
            p = os.path.join(A.GAL, "stl", f"terminal_phase_{ROMAN[i]}_ep{s['epoch']:.0f}.stl")
            print(p, export_stl(s["V"], p), "triangles")
    else:
        R = rot(a.az, a.el)
        dust = np.array([A.rgb(c) for c in ["#e8b04a", "#d9694a", "#6fb3c9", "#a7c96a"]])
        print(render_series(states, a.style, R, a.panel, dust_colors=dust, fname=f"tetra_series_{a.tag}_{a.style}.png"))
