"""Meshes for printing. Marching cubes smooths below the grid scale: printed fractal surfaces are declared as such (spec §0.5)."""
import numpy as np


def marching_cubes(field, level, lo, hi, closed=True):
    from skimage.measure import marching_cubes as _mc
    F = field.detach().cpu().numpy() if hasattr(field, "detach") else np.asarray(field)
    F = F.astype(np.float64)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    h = (hi - lo) / (np.array(F.shape[::-1]) - 1)                  # (hx, hy, hz)
    origin = lo.copy()
    if closed:
        # Mirror boundary values around `level` in the new shell so any solid touching
        # the box crosses exactly halfway into the pad layer (half a voxel outside);
        # non-solid boundaries just replicate outward and form no spurious cap. A flat
        # `level - 1` constant only lands on that exact half-voxel offset when the
        # boundary value happens to equal `level + 1`, which is not true in general
        # (e.g. a uniform field of 1.0 at level=0.5 lands a third of a voxel out).
        F = np.pad(F, 1, mode="edge")
        ring = np.ones_like(F, dtype=bool)
        ring[1:-1, 1:-1, 1:-1] = False
        b = F[ring]
        F[ring] = np.minimum(2 * level - b, b)
        origin = lo - h
    v, f, _, _ = _mc(F, level, spacing=(h[2], h[1], h[0]), gradient_direction="descent")
    verts = v[:, ::-1] + origin                                     # (z,y,x) -> (x,y,z): an odd permutation,
    faces = f[:, ::-1].astype(np.int64)                             # so the winding is reversed to stay outward
    if mesh_volume(verts, faces) < 0:
        faces = faces[:, ::-1]
    return verts, np.ascontiguousarray(faces)


def mesh_volume(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def is_watertight(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fwd = {tuple(x) for x in e.tolist()}
    if len(fwd) != len(e):
        return False
    return all((b, a) in fwd for a, b in fwd)


def write_stl(path, verts, faces, header="r3d"):
    T = verts[faces].astype(np.float32)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, T
    with open(path, "wb") as fh:
        fh.write(header.encode()[:80].ljust(80, b" "))
        fh.write(np.uint32(len(T)).tobytes())
        fh.write(rec.tobytes())


def read_stl(path):
    with open(path, "rb") as fh:
        fh.read(80)
        n = int(np.frombuffer(fh.read(4), "<u4")[0])
        rec = np.frombuffer(fh.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].copy()


def tube_mesh(P, radius, n_sides=16, cap=True):
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    N = [np.cross(T[0], a) / np.linalg.norm(np.cross(T[0], a))]
    for i in range(1, len(P)):                                      # parallel transport
        v = N[-1] - np.dot(N[-1], T[i]) * T[i]
        N.append(v / np.linalg.norm(v))
    N = np.array(N)
    B = np.cross(T, N)
    th = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring = np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]
    verts = (P[:, None, :] + radius * ring).reshape(-1, 3)
    faces = []
    m = n_sides
    for i in range(len(P) - 1):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            b0, b1 = a0 + m, a1 + m
            faces += [(a0, a1, b1), (a0, b1, b0)]
    if cap:
        c0, c1 = len(verts), len(verts) + 1
        verts = np.vstack([verts, P[0], P[-1]])
        last = (len(P) - 1) * m
        for j in range(m):
            faces.append((c0, (j + 1) % m, j))
            faces.append((c1, last + j, last + (j + 1) % m))
    faces = np.array(faces, dtype=np.int64)
    if mesh_volume(verts, faces) < 0:
        faces = faces[:, ::-1]
    return verts, np.ascontiguousarray(faces)


def scale_to_mm(verts, size_mm):
    v = verts - verts.min(0)
    return v * (size_mm / (v.max(0)).max())
