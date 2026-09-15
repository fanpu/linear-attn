# r3d Renderer + Serial GPU Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small, tested, device-agnostic torch 3D renderer at `art/_shared/r3d/` (volumes, isosurfaces, crisp voxels, tubes, meshes) plus a serial GPU queue, shared by the eight pieces in `art/ml-art-3d.md`.

**Architecture:** Pure functions over torch tensors. Marchers take flat rays `(N,3)` so they can be reused for shadows and ambient occlusion; thin camera wrappers reshape to images. Buffers store view-axis depth so volumes, surfaces and tubes composite together. Tests run on CPU at small resolution. The GPU is shared with `autonomous/`, which is at 95% right now.

**Tech Stack:** Python 3.12, torch 2.14 (`F.grid_sample` 5-D), numpy, matplotlib colormaps, PIL, scikit-image (marching cubes), pytest, ffmpeg.

**Spec:** `art/ml-art-3d.md` (§0 rules, §9 renderer, §11 verification, §12 scheduling)

## Global Constraints

- Python: `/home/fzeng/ml/research/art/.venv/bin/python`. Never use the repo-root `.venv`, never edit `pyproject.toml`.
- Install packages only as `flock /home/fzeng/ml/research/art/_shared/.pip.lock uv pip install --python /home/fzeng/ml/research/art/.venv/bin/python <pkg>`.
- Grid convention everywhere: `data[k, j, i]` is the value at world `(lo[0]+i*hx, lo[1]+j*hy, lo[2]+k*hz)`; axis 0 = z, 1 = y, 2 = x; `lo`/`hi` are the centres of the corner voxels; `h = (hi-lo)/(n-1)`.
- World is right-handed, z up. Images are `(H, W, ...)` with row 0 at the top. Depth buffers hold view-axis depth `z = (p - eye)·forward`, `inf` = empty.
- Colours in `[0,1]` float; volume and additive outputs are premultiplied.
- Tests: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests -q`, CPU only, each test < 20 s.
- Never run a GPU job except through `art/_shared/gpu1.sh`; never two GPU jobs at once.
- Commit with `art/_shared/commit.sh _shared "<msg>"` (commits only `art/_shared/`).

## File map

| File | Responsibility |
|---|---|
| `art/_shared/gpu1.sh` | serial GPU queue that also serialises with `autonomous/tools/gpu_run.sh` |
| `art/_shared/r3d/__init__.py` | re-exports the public API |
| `art/_shared/r3d/camera.py` | `Camera`, `orbit`, `turntable`, `stereo_pair`, `ray_box` |
| `art/_shared/r3d/grid.py` | `sample_grid`, `clip_keep` |
| `art/_shared/r3d/volume.py` | `TransferFunction`, `lut_tf`, `march_volume`, `render_volume`, `over` |
| `art/_shared/r3d/iso.py` | `march_iso`, `iso_normals`, `render_iso`, `iso_occluder` |
| `art/_shared/r3d/voxels.py` | `march_voxels`, `render_voxels`, `voxel_occluder` |
| `art/_shared/r3d/shade.py` | `lambert`, `hard_shadow`, `ambient_occlusion` |
| `art/_shared/r3d/tubes.py` | `sample_polyline`, `splat_spheres`, `splat_additive`, `visible_runs`, `write_svg` |
| `art/_shared/r3d/mesh.py` | `marching_cubes`, `mesh_volume`, `is_watertight`, `write_stl`, `read_stl`, `tube_mesh`, `scale_to_mm` |
| `art/_shared/r3d/io.py` | `save_png`, `glow_tonemap`, `write_film` |
| `art/_shared/r3d/tests/test_*.py` | one test file per module |
| `art/_shared/r3d/examples/nulls.py` | analytic null renders in four styles (§11.5 of spec) |
| `art/_shared/r3d/README.md` | API, conventions, honesty rules |

## Tasks

1. Environment, `.gitignore`, serial GPU queue
2. Camera and rays
3. Grid sampling and the volume renderer
4. Isosurface marcher
5. Crisp voxel marcher
6. Shading: Lambert, shadows, ambient occlusion
7. Tubes, glow and hidden-line SVG
8. Meshes, STL and films
9. Analytic null gallery, README, piece brief

---
### Task 1: Environment, `.gitignore`, serial GPU queue

**Files:**
- Modify: `art/.gitignore` (append `_shared/.gpu1.lock`)
- Create: `art/_shared/gpu1.sh`
- Create: `art/_shared/r3d/__init__.py` (empty for now), `art/_shared/r3d/tests/__init__.py` (empty)
- Test: `art/_shared/r3d/tests/test_gpu1.py`

**Interfaces:**
- Produces: `art/_shared/gpu1.sh <cmd...>`: exit status of `<cmd>`. Env `GPU1_IGNORE_OTHERS=1` skips the nvidia-smi wait; `GPU1_AUTONOMOUS_LOCK=<path>` overrides `autonomous/tools/.gpu.lock`.

- [ ] **Step 1: Install test and mesh dependencies**

Run:
```bash
flock /home/fzeng/ml/research/art/_shared/.pip.lock uv pip install --python /home/fzeng/ml/research/art/.venv/bin/python pytest scikit-image
/home/fzeng/ml/research/art/.venv/bin/python -c "import pytest, skimage; print(pytest.__version__, skimage.__version__)"
```
Expected: two version strings. If scikit-image has no aarch64 wheel, stop and report (Task 8 then implements marching cubes itself).

- [ ] **Step 2: Ignore the queue lock**

```bash
echo '_shared/.gpu1.lock' >> /home/fzeng/ml/research/art/.gitignore
```

- [ ] **Step 3: Write the failing test**

`art/_shared/r3d/tests/test_gpu1.py`:
```python
import os, subprocess, time, pathlib, stat

SCRIPT = "/home/fzeng/ml/research/art/_shared/gpu1.sh"


def _env(tmp, others_flag=None, ignore=False):
    fake = tmp / "bin"; fake.mkdir(exist_ok=True)
    smi = fake / "nvidia-smi"
    flag = others_flag or (tmp / "no_such_flag")
    smi.write_text(f"#!/usr/bin/env bash\n[ -e {flag} ] && echo '123, python'\nexit 0\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}",
               GPU1_AUTONOMOUS_LOCK=str(tmp / "autonomous" / "tools" / ".gpu.lock"),
               GPU1_POLL_S="0.2")
    (tmp / "autonomous" / "tools").mkdir(parents=True, exist_ok=True)
    if ignore:
        env["GPU1_IGNORE_OTHERS"] = "1"
    return env


def test_exit_status_passthrough(tmp_path):
    env = _env(tmp_path, ignore=True)
    r = subprocess.run([SCRIPT, "bash", "-c", "exit 7"], env=env, capture_output=True, text=True)
    assert r.returncode == 7


def test_two_jobs_are_serialised(tmp_path):
    env = _env(tmp_path, ignore=True)
    log = tmp_path / "log"
    cmd = f"echo start $(date +%s.%N) >> {log}; sleep 1; echo end $(date +%s.%N) >> {log}"
    a = subprocess.Popen([SCRIPT, "bash", "-c", cmd], env=env)
    time.sleep(0.2)
    b = subprocess.Popen([SCRIPT, "bash", "-c", cmd], env=env)
    assert a.wait(20) == 0 and b.wait(20) == 0
    ev = [l.split() for l in log.read_text().splitlines()]
    assert [e[0] for e in ev] == ["start", "end", "start", "end"]


def test_waits_for_other_gpu_process(tmp_path):
    flag = tmp_path / "busy"; flag.write_text("1")
    env = _env(tmp_path, others_flag=flag)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    time.sleep(1.0)
    assert not out.exists()
    flag.unlink()
    assert p.wait(20) == 0 and out.exists()


def test_waits_for_autonomous_lock_but_not_when_paused(tmp_path):
    env = _env(tmp_path, ignore=True)
    lock = tmp_path / "autonomous" / "tools" / ".gpu.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "3"])
    time.sleep(0.3)
    out = tmp_path / "ran"
    p = subprocess.Popen([SCRIPT, "bash", "-c", f"touch {out}"], env=env)
    time.sleep(1.0)
    assert not out.exists()            # blocked behind the autonomous job
    (tmp_path / "autonomous" / "GPU_PAUSE").write_text("")
    assert p.wait(20) == 0 and out.exists()   # pause means their jobs cannot start: stop waiting
    holder.wait()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_gpu1.py -q`
Expected: FAIL (`FileNotFoundError` / permission error: gpu1.sh does not exist).

- [ ] **Step 5: Write the implementation**

`art/_shared/gpu1.sh` (then `chmod +x`):
```bash
#!/usr/bin/env bash
# Serial GPU queue for the 3D art campaign (art/ml-art-3d.md §12).
#
#   setsid nohup art/_shared/gpu1.sh <cmd...> > <log> 2>&1 < /dev/null &
#
# 1. Takes art/_shared/.gpu1.lock: one art GPU job at a time, the rest queue.
# 2. Takes autonomous/tools/.gpu.lock (append mode, never truncated), so an autonomous/ job and an art job
#    never overlap. If autonomous/GPU_PAUSE exists its jobs cannot start, so we stop waiting for that lock.
# 3. Waits until nvidia-smi lists no compute process (two jobs at once -> driver OOM on the GB10).
# 4. Runs the command and exits with its status. Jobs must checkpoint so a rerun resumes.
#
# Env: GPU1_IGNORE_OTHERS=1 skips step 3 (tests). GPU1_AUTONOMOUS_LOCK overrides the step-2 lock path.
#      GPU1_POLL_S sets the poll interval (default 15 s).
set -u
ROOT=/home/fzeng/ml/research
POLL=${GPU1_POLL_S:-15}
ts() { date '+%F %T'; }

exec {artfd}>>"$ROOT/art/_shared/.gpu1.lock"
if ! flock -n "$artfd"; then
  echo "[gpu1 $(ts)] queued behind another art GPU job" >&2
  flock "$artfd"
fi

ALOCK=${GPU1_AUTONOMOUS_LOCK:-$ROOT/autonomous/tools/.gpu.lock}
APAUSE="$(dirname "$(dirname "$ALOCK")")/GPU_PAUSE"
if [ -d "$(dirname "$ALOCK")" ]; then
  exec {autfd}>>"$ALOCK"
  n=0
  until flock -n "$autfd"; do
    if [ -e "$APAUSE" ]; then
      echo "[gpu1 $(ts)] autonomous/ is paused; not waiting for its lock" >&2
      break
    fi
    (( n % 20 == 0 )) && echo "[gpu1 $(ts)] waiting for an autonomous/ GPU job to finish" >&2
    sleep "$POLL"; n=$((n + 1))
  done
fi

other_gpu_users() {
  [ "${GPU1_IGNORE_OTHERS:-0}" = 1 ] && return 1
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^\s*$')
  [ -n "$apps" ] && { echo "$apps"; return 0; }
  return 1
}
n=0
while other_gpu_users > "/tmp/gpu1_others.$$"; do
  (( n % 20 == 0 )) && echo "[gpu1 $(ts)] waiting for other GPU processes: $(tr '\n' ';' < "/tmp/gpu1_others.$$")" >&2
  sleep "$POLL"; n=$((n + 1))
done
rm -f "/tmp/gpu1_others.$$"

echo "[gpu1 $(ts)] start: $*" >&2
"$@"
rc=$?
echo "[gpu1 $(ts)] exit $rc: $*" >&2
exit "$rc"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/fzeng/ml/research/art/_shared && chmod +x gpu1.sh && ../.venv/bin/python -m pytest r3d/tests/test_gpu1.py -q`
Expected: `4 passed`.

- [ ] **Step 7: Commit**

```bash
cd /home/fzeng/ml/research && git add art/.gitignore && git commit -q -m "art: ignore gpu1 lock" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -- art/.gitignore
art/_shared/commit.sh _shared "art/_shared: gpu1.sh serial GPU queue (serialises with autonomous/), r3d test scaffold"
```

---
### Task 2: Camera and rays

**Files:**
- Create: `art/_shared/r3d/camera.py`
- Test: `art/_shared/r3d/tests/test_camera.py`

**Interfaces:**
- Produces:
  - `Camera(eye, target, up=(0,0,1), width=512, height=512, fov_deg=None, ortho_height=2.0)`; `fov_deg=None` means orthographic.
  - `Camera.basis() -> (forward, right, up)` numpy unit vectors.
  - `Camera.rays(device="cpu", dtype=torch.float32) -> (o (H,W,3), d (H,W,3) unit)`.
  - `Camera.project(pts (N,3)) -> (pix (N,2) as (col,row) with pixel centres at +0.5, depth (N,))` in float64.
  - `Camera.t_to_depth(t, d)`, `Camera.depth_to_t(depth, d)`: convert ray parameter ↔ view depth.
  - `Camera.pixel_scale(depth) -> float` world units per pixel.
  - `orbit(target, radius, az_deg, el_deg, **cam_kw) -> Camera`, `turntable(n_frames, target, radius, el_deg, az0_deg=0.0, **cam_kw) -> list[Camera]`, `stereo_pair(cam, separation) -> (left, right)`.
  - `ray_box(o (N,3), d (N,3), lo, hi) -> (tnear (N,), tfar (N,))`; miss where `tfar <= tnear`; `tnear >= 0`.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_camera.py`:
```python
import math
import numpy as np
import torch
from r3d.camera import Camera, orbit, turntable, stereo_pair, ray_box


def test_ortho_centre_ray_and_projection_roundtrip():
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=48, ortho_height=4.0)
    o, d = cam.rays()
    assert o.shape == (48, 64, 3)
    assert torch.allclose(d[0, 0], torch.tensor([-1.0, 0, 0]))
    pts = o.reshape(-1, 3).double() + 2.0 * d.reshape(-1, 3).double()
    pix, depth = cam.project(pts)
    rows, cols = torch.meshgrid(torch.arange(48) + 0.5, torch.arange(64) + 0.5, indexing="ij")
    assert torch.allclose(pix[:, 0], cols.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(pix[:, 1], rows.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(depth, torch.full_like(depth, 2.0), atol=1e-6)


def test_image_orientation_up_is_row_zero_right_is_last_col():
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=32, height=32, ortho_height=2.0)
    pix, _ = cam.project(torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.9, 0.0]]))
    # looking along -x with z up: +z is the top of the image, right = forward x up = +y
    assert pix[0, 1] < 4 and abs(pix[0, 0] - 16) < 1e-6
    assert pix[1, 0] > 28


def test_perspective_roundtrip_and_depth_conversion():
    cam = Camera(eye=(0, -6, 2), target=(0, 0, 0), width=40, height=30, fov_deg=35.0)
    o, d = cam.rays(dtype=torch.float64)
    t = torch.rand(30, 40, dtype=torch.float64) * 5 + 1
    pts = (o + t[..., None] * d).reshape(-1, 3)
    pix, depth = cam.project(pts)
    rows, cols = torch.meshgrid(torch.arange(30) + 0.5, torch.arange(40) + 0.5, indexing="ij")
    assert torch.allclose(pix[:, 0], cols.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(pix[:, 1], rows.reshape(-1).double(), atol=1e-6)
    assert torch.allclose(cam.t_to_depth(t, d).reshape(-1), depth, atol=1e-9)
    assert torch.allclose(cam.depth_to_t(depth.reshape(30, 40), d), t, atol=1e-9)


def test_orbit_turntable_stereo():
    cam = orbit((1, 2, 3), 10.0, az_deg=90, el_deg=0, width=8, height=8)
    assert np.allclose(cam.eye, (1, 12, 3))
    frames = turntable(4, (0, 0, 0), 5.0, el_deg=30, width=8, height=8)
    assert len(frames) == 4 and np.allclose(frames[2].eye[:2], -np.asarray(frames[0].eye[:2]))
    left, right = stereo_pair(frames[0], 0.5)
    assert math.isclose(np.linalg.norm(np.subtract(right.eye, left.eye)), 0.5)


def test_ray_box_hits_and_misses():
    o = torch.tensor([[-5.0, 0, 0], [-5.0, 3, 0], [0.0, 0, 0]])
    d = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 1]])
    tn, tf = ray_box(o, d, (-1, -1, -1), (1, 1, 1))
    assert torch.allclose(tn[0], torch.tensor(4.0)) and torch.allclose(tf[0], torch.tensor(6.0))
    assert tf[1] <= tn[1]
    assert tn[2] == 0 and torch.allclose(tf[2], torch.tensor(1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_camera.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.camera'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/camera.py`:
```python
"""Cameras and rays. World is right-handed with z up; images are (H, W, ...) with row 0 at the top.
Depth buffers everywhere in r3d hold view-axis depth z = (p - eye) . forward."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass
class Camera:
    eye: tuple
    target: tuple
    up: tuple = (0.0, 0.0, 1.0)
    width: int = 512
    height: int = 512
    fov_deg: float | None = None      # vertical field of view; None = orthographic
    ortho_height: float = 2.0         # world height of the view (orthographic only)

    def basis(self):
        f = np.asarray(self.target, float) - np.asarray(self.eye, float)
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(self.up, float))
        if np.linalg.norm(r) < 1e-9:
            raise ValueError("camera up is parallel to the view direction")
        r /= np.linalg.norm(r)
        return f, r, np.cross(r, f)

    def _screen(self):
        W, H = self.width, self.height
        xs = ((torch.arange(W, dtype=torch.float64) + 0.5) / W * 2 - 1) * (W / H)
        ys = 1 - (torch.arange(H, dtype=torch.float64) + 0.5) / H * 2
        return torch.meshgrid(ys, xs, indexing="ij")          # Y (H,W), X (H,W)

    def rays(self, device="cpu", dtype=torch.float32):
        f, r, u = (torch.tensor(v, dtype=torch.float64) for v in self.basis())
        e = torch.tensor(self.eye, dtype=torch.float64)
        Y, X = self._screen()
        H, W = Y.shape
        if self.fov_deg is None:
            s = self.ortho_height / 2
            o = e + X[..., None] * s * r + Y[..., None] * s * u
            d = f.expand(H, W, 3).clone()
        else:
            s = math.tan(math.radians(self.fov_deg) / 2)
            d = f + X[..., None] * s * r + Y[..., None] * s * u
            d = d / d.norm(dim=-1, keepdim=True)
            o = e.expand(H, W, 3).clone()
        return o.to(device, dtype), d.to(device, dtype)

    def project(self, pts):
        pts = torch.as_tensor(pts, dtype=torch.float64)
        f, r, u = (torch.tensor(v, dtype=torch.float64) for v in self.basis())
        rel = pts - torch.tensor(self.eye, dtype=torch.float64)
        z, x, y = rel @ f, rel @ r, rel @ u
        if self.fov_deg is None:
            s = self.ortho_height / 2
            X, Y = x / s, y / s
        else:
            s = math.tan(math.radians(self.fov_deg) / 2)
            X, Y = x / (z * s), y / (z * s)
        col = (X / (self.width / self.height) + 1) / 2 * self.width
        row = (1 - Y) / 2 * self.height
        return torch.stack([col, row], -1), z

    def _fdot(self, d):
        f = torch.tensor(self.basis()[0], dtype=d.dtype, device=d.device)
        return (d * f).sum(-1)

    def t_to_depth(self, t, d):
        return t * self._fdot(d)

    def depth_to_t(self, depth, d):
        return depth / self._fdot(d)

    def pixel_scale(self, depth=None):
        if self.fov_deg is None:
            return self.ortho_height / self.height
        return 2 * depth * math.tan(math.radians(self.fov_deg) / 2) / self.height


def orbit(target, radius, az_deg, el_deg, **cam_kw):
    a, e = math.radians(az_deg), math.radians(el_deg)
    t = np.asarray(target, float)
    eye = t + radius * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return Camera(eye=tuple(eye), target=tuple(t), **cam_kw)


def turntable(n_frames, target, radius, el_deg, az0_deg=0.0, **cam_kw):
    return [orbit(target, radius, az0_deg + 360.0 * i / n_frames, el_deg, **cam_kw) for i in range(n_frames)]


def stereo_pair(cam, separation):
    """Parallel-axis stereo: both eye and target shift sideways by +-separation/2."""
    _, r, _ = cam.basis()
    h = 0.5 * separation * r
    shift = lambda s: replace(cam, eye=tuple(np.asarray(cam.eye) + s * h), target=tuple(np.asarray(cam.target) + s * h))
    return shift(-1.0), shift(1.0)


def ray_box(o, d, lo, hi):
    lo = torch.as_tensor(lo, dtype=o.dtype, device=o.device)
    hi = torch.as_tensor(hi, dtype=o.dtype, device=o.device)
    safe = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t0, t1 = (lo - o) / safe, (hi - o) / safe
    tnear = torch.minimum(t0, t1).amax(-1).clamp_min(0)
    tfar = torch.maximum(t0, t1).amin(-1)
    return tnear, tfar
```

`art/_shared/r3d/tests/conftest.py` (so `import r3d` works from any cwd):
```python
import sys
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_camera.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: camera, rays, orbit/turntable/stereo, ray_box"
```

---
### Task 3: Grid sampling and the volume renderer

**Files:**
- Create: `art/_shared/r3d/grid.py`, `art/_shared/r3d/volume.py`
- Test: `art/_shared/r3d/tests/test_volume.py`

**Interfaces:**
- Consumes: `ray_box`, `Camera.rays`, `Camera.depth_to_t` (Task 2).
- Produces:
  - `sample_grid(data, lo, hi, pts, mode="linear"|"nearest", padding="zeros"|"border") -> (...)` for `data (D,H,W)`, `(...,C)` for `data (C,D,H,W)`; `pts (...,3)` world xyz; `data` must share `pts`' device; every grid axis needs ≥ 2 samples.
  - `clip_keep(pts, clip) -> bool (...)`: `clip` is a sequence of `(point, normal)`; keeps `(p - point)·normal <= 0` (the normal points into the removed half).
  - `TransferFunction(cmap, vmin, vmax, opacity, density=1.0, n=1024)`; `tf(v) -> (rgb (...,3), sigma (...))`; `opacity` is a callable on normalised `x ∈ [0,1]`.
  - `lut_tf(rgb (K,3), sigma (K,)) -> callable` for integer labels (use with `mode="nearest"`).
  - `march_volume(data, lo, hi, o (N,3), d (N,3), tf, step, *, tmax=None, clip=(), mode="linear", chunk_samples=2**22) -> (rgb (N,3) premultiplied, alpha (N,))`.
  - `render_volume(data, lo, hi, cam, tf, step, *, depth=None, clip=(), mode="linear", device="cpu") -> (rgb (H,W,3), alpha (H,W))`; `depth` is an opaque view-depth buffer that stops the rays.
  - `over(front_rgb, front_alpha, back_rgb) -> rgb`: premultiplied front over an opaque back.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_volume.py`:
```python
import math
import torch
from r3d.camera import Camera
from r3d.grid import sample_grid, clip_keep
from r3d.volume import TransferFunction, lut_tf, march_volume, render_volume, over

LO, HI = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)


def _linear_grid(D=7, H=6, W=5):
    z = torch.linspace(-1, 1, D); y = torch.linspace(-1, 1, H); x = torch.linspace(-1, 1, W)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
    return (2 * X + 3 * Y - Z).double()


def test_sample_grid_is_exact_for_linear_fields():
    g = _linear_grid()
    p = torch.rand(100, 3, dtype=torch.float64) * 2 - 1
    v = sample_grid(g, LO, HI, p)
    assert torch.allclose(v, 2 * p[:, 0] + 3 * p[:, 1] - p[:, 2], atol=1e-10)
    vc = sample_grid(torch.stack([g, -g]), LO, HI, p)
    assert vc.shape == (100, 2) and torch.allclose(vc[:, 1], -v)


def test_sample_grid_nearest_returns_voxel_values():
    g = _linear_grid()
    p = torch.tensor([[-1.0, -1.0, -1.0], [0.49, 0.39, 0.34]], dtype=torch.float64)  # x step .5, y .4, z 1/3
    v = sample_grid(g, LO, HI, p, mode="nearest")
    assert torch.allclose(v, torch.tensor([2 * -1 + 3 * -1 + 1, 2 * 0.5 + 3 * 0.2 - 1 / 3], dtype=torch.float64), atol=1e-9)


def test_clip_keep():
    p = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]])
    assert clip_keep(p, [((0, 0, 0), (1, 0, 0))]).tolist() == [False, True]


def _const_tf(s):
    return lambda v: (torch.tensor([1.0, 0.5, 0.25], dtype=v.dtype).expand(*v.shape, 3), torch.full_like(v, s))


def test_constant_cube_matches_beer_lambert():
    data = torch.ones(8, 8, 8, dtype=torch.float64)
    o = torch.tensor([[5.0, 0.1, -0.2]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    rgb, a = march_volume(data, LO, HI, o, d, _const_tf(0.7), step=0.013)
    assert math.isclose(a.item(), 1 - math.exp(-1.4), rel_tol=1e-9)
    assert torch.allclose(rgb[0], torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64) * a[0])


def test_clip_plane_and_depth_limit_shorten_the_path():
    data = torch.ones(8, 8, 8, dtype=torch.float64)
    o = torch.tensor([[5.0, 0, 0]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    _, a = march_volume(data, LO, HI, o, d, _const_tf(1.0), step=0.01, clip=[((0, 0, 0), (1, 0, 0))])
    assert math.isclose(a.item(), 1 - math.exp(-1.0), rel_tol=1e-3)
    _, a = march_volume(data, LO, HI, o, d, _const_tf(1.0), step=0.01, tmax=torch.tensor([4.5], dtype=torch.float64))
    assert math.isclose(a.item(), 1 - math.exp(-0.5), rel_tol=1e-3)


def test_lut_tf_labels_and_transfer_function():
    tf = lut_tf(torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]), torch.tensor([0.0, 5.0, 9.0]))
    rgb, s = tf(torch.tensor([2.0, 1.0, 0.0]))
    assert s.tolist() == [9.0, 5.0, 0.0] and rgb[0].tolist() == [0.0, 1.0, 0.0]
    t2 = TransferFunction("magma", 0.0, 2.0, opacity=lambda x: x, density=3.0)
    rgb, s = t2(torch.tensor([-1.0, 1.0, 5.0]))
    assert torch.allclose(s, torch.tensor([0.0, 1.5, 3.0])) and rgb.shape == (3, 3)


def test_render_volume_image_and_over():
    data = torch.ones(8, 8, 8)
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=32, height=32, ortho_height=4.0)
    rgb, a = render_volume(data, LO, HI, cam, _const_tf(50.0), step=0.02)
    assert rgb.shape == (32, 32, 3) and a.shape == (32, 32)
    assert a[16, 16] > 0.999 and a[0, 0] == 0          # centre hits the cube, corner misses it
    img = over(rgb, a, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(img[0, 0], torch.tensor([0.0, 0.0, 1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_volume.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.grid'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/grid.py`:
```python
"""Sampling a regular grid. data[k, j, i] sits at world (lo + (i, j, k) * h); lo/hi are corner-voxel centres."""
import torch
import torch.nn.functional as F


def sample_grid(data, lo, hi, pts, mode="linear", padding="zeros"):
    squeeze = data.dim() == 3
    x = data[None] if squeeze else data
    if x.dtype != pts.dtype:
        x = x.to(pts.dtype)
    lo_ = torch.as_tensor(lo, dtype=pts.dtype, device=pts.device)
    hi_ = torch.as_tensor(hi, dtype=pts.dtype, device=pts.device)
    g = (pts - lo_) / (hi_ - lo_) * 2 - 1
    shp = pts.shape[:-1]
    out = F.grid_sample(x[None], g.reshape(1, -1, 1, 1, 3), mode="bilinear" if mode == "linear" else "nearest",
                        padding_mode=padding, align_corners=True)
    out = out.reshape(x.shape[0], -1).T.reshape(*shp, x.shape[0])
    return out[..., 0] if squeeze else out


def clip_keep(pts, clip):
    keep = torch.ones(pts.shape[:-1], dtype=torch.bool, device=pts.device)
    for point, normal in clip:
        p = torch.as_tensor(point, dtype=pts.dtype, device=pts.device)
        n = torch.as_tensor(normal, dtype=pts.dtype, device=pts.device)
        keep &= ((pts - p) * n).sum(-1) <= 0
    return keep
```

`art/_shared/r3d/volume.py`:
```python
"""Emission-absorption volume rendering. The transfer function is a declared aesthetic choice (spec §0.6)."""
import math

import numpy as np
import torch

from .camera import ray_box
from .grid import clip_keep, sample_grid


class TransferFunction:
    def __init__(self, cmap, vmin, vmax, opacity, density=1.0, n=1024):
        import matplotlib
        cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.lut = torch.tensor(cm(np.linspace(0, 1, n))[:, :3], dtype=torch.float32)
        self.vmin, self.vmax, self.opacity, self.density, self.n = vmin, vmax, opacity, density, n

    def __call__(self, v):
        x = ((v - self.vmin) / (self.vmax - self.vmin)).clamp(0, 1)
        f = x * (self.n - 1)
        i0 = f.floor().long().clamp(max=self.n - 2)
        w = (f - i0)[..., None]
        lut = self.lut.to(v.device, v.dtype)
        return lut[i0] * (1 - w) + lut[i0 + 1] * w, self.density * self.opacity(x)


def lut_tf(rgb, sigma):
    def tf(v):
        i = v.round().long().clamp(0, rgb.shape[0] - 1)
        return rgb.to(v.device, v.dtype)[i], sigma.to(v.device, v.dtype)[i]
    return tf


def march_volume(data, lo, hi, o, d, tf, step, *, tmax=None, clip=(), mode="linear", chunk_samples=2 ** 22):
    N = o.shape[0]
    rgb = torch.zeros(N, 3, dtype=o.dtype, device=o.device)
    alpha = torch.zeros(N, dtype=o.dtype, device=o.device)
    tn, tfar = ray_box(o, d, lo, hi)
    if tmax is not None:
        tfar = torch.minimum(tfar, tmax)
    idx = (tfar > tn).nonzero().squeeze(1)
    if idx.numel() == 0:
        return rgb, alpha
    nmax = max(1, int(math.ceil(((tfar - tn)[idx].max().item()) / step)))
    per = max(1, chunk_samples // nmax)
    k = torch.arange(nmax, dtype=o.dtype, device=o.device)
    for s in range(0, idx.numel(), per):
        ii = idx[s:s + per]
        t0, t1 = tn[ii, None], tfar[ii, None]
        seg0 = t0 + k * step                                   # segment starts (R, S)
        ds = (t1 - seg0).clamp(0, step)                        # exact segment lengths; 0 past the exit
        ts = torch.minimum(seg0 + 0.5 * ds, t1)
        pts = o[ii, None, :] + ts[..., None] * d[ii, None, :]
        c, sig = tf(sample_grid(data, lo, hi, pts, mode))
        if clip:
            sig = sig * clip_keep(pts, clip)
        a = 1 - torch.exp(-sig * ds)
        T = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
        rgb[ii] = ((T * a)[..., None] * c).sum(1)
        alpha[ii] = 1 - T[:, -1] * (1 - a[:, -1])
    return rgb, alpha


def render_volume(data, lo, hi, cam, tf, step, *, depth=None, clip=(), mode="linear", device="cpu"):
    o, d = cam.rays(device=device, dtype=data.dtype if data.is_floating_point() else torch.float32)
    H, W = o.shape[:2]
    tmax = None if depth is None else cam.depth_to_t(depth.to(d), d).reshape(-1)
    rgb, a = march_volume(data, lo, hi, o.reshape(-1, 3), d.reshape(-1, 3), tf, step, tmax=tmax, clip=clip, mode=mode)
    return rgb.reshape(H, W, 3), a.reshape(H, W)


def over(front_rgb, front_alpha, back_rgb):
    back = torch.as_tensor(back_rgb, dtype=front_rgb.dtype, device=front_rgb.device)
    return front_rgb + (1 - front_alpha)[..., None] * back
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_volume.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: grid sampling, transfer functions, emission-absorption volume renderer"
```

---
### Task 4: Isosurface marcher

**Files:**
- Create: `art/_shared/r3d/iso.py`
- Test: `art/_shared/r3d/tests/test_iso.py`

**Interfaces:**
- Consumes: `ray_box` (Task 2), `sample_grid`, `clip_keep` (Task 3).
- Produces:
  - Solid convention: a point is solid where `field >= level` and every clip plane keeps it. Negate the field for "solid below".
  - `march_iso(field, lo, hi, o, d, level, step, *, refine=12, clip=(), tmax=None, chunk_samples=2**22) -> t (N,)`; `inf` = miss. A ray that enters the box already inside the solid hits at `tnear` (a cut face).
  - `iso_normals(field, lo, hi, pts, level, *, clip=(), tol=None) -> (N,3)` unit outward normals: `-∇field` normally; the clip-plane normal on clip faces; the box-face normal on box faces.
  - `render_iso(field, lo, hi, cam, level, step=None, *, clip=(), device="cpu") -> dict(depth (H,W), pos (H,W,3), normal (H,W,3), mask (H,W))`; `step` defaults to half the smallest voxel size.
  - `iso_occluder(field, lo, hi, level, step, clip=()) -> occluded(o, d, tmax) -> bool (N,)`, for `shade.py`.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_iso.py`:
```python
import math
import torch
from r3d.camera import Camera
from r3d.iso import march_iso, iso_normals, render_iso, iso_occluder

LO, HI = (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)


def _sphere(n=65):
    a = torch.linspace(-1.5, 1.5, n, dtype=torch.float64)
    Z, Y, X = torch.meshgrid(a, a, a, indexing="ij")
    return 1 - torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)            # solid (>= 0) is the unit ball


def test_centre_ray_hits_unit_sphere():
    f = _sphere()
    o = torch.tensor([[5.0, 0, 0], [5.0, 0.6, 0.0], [5.0, 1.2, 0]], dtype=torch.float64)
    d = torch.tensor([[-1.0, 0, 0]] * 3, dtype=torch.float64)
    t = march_iso(f, LO, HI, o, d, 0.0, step=0.02)
    assert abs(t[0].item() - 4.0) < 5e-3
    assert abs(t[1].item() - (5 - math.sqrt(1 - 0.36))) < 5e-3
    assert math.isinf(t[2].item())


def test_normals_point_outward():
    f = _sphere()
    p = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0.6, 0.8, 0]], dtype=torch.float64)
    n = iso_normals(f, LO, HI, p, 0.0)
    assert torch.allclose(n, p, atol=0.03)


def test_clip_plane_makes_a_flat_cap_with_plane_normal():
    f = _sphere()
    clip = [((0.5, 0, 0), (1, 0, 0))]                          # remove x > 0.5
    o = torch.tensor([[5.0, 0, 0]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    t = march_iso(f, LO, HI, o, d, 0.0, step=0.02, clip=clip)
    assert abs(t.item() - 4.5) < 1e-3
    n = iso_normals(f, LO, HI, o + t[:, None] * d, 0.0, clip=clip)
    assert torch.allclose(n, torch.tensor([[1.0, 0, 0]], dtype=torch.float64), atol=1e-6)


def test_solid_touching_the_box_gets_box_face_normal():
    f = torch.ones(9, 9, 9, dtype=torch.float64)             # everything solid
    o = torch.tensor([[0.0, 0, 5]], dtype=torch.float64); d = torch.tensor([[0.0, 0, -1.0]], dtype=torch.float64)
    t = march_iso(f, LO, HI, o, d, 0.5, step=0.05)
    assert abs(t.item() - 3.5) < 1e-9
    n = iso_normals(f, LO, HI, o + t[:, None] * d, 0.5)
    assert torch.allclose(n, torch.tensor([[0.0, 0, 1.0]], dtype=torch.float64))


def test_render_iso_silhouette_area_and_depth():
    f = _sphere().float()
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=96, height=96, ortho_height=3.0)
    hit = render_iso(f, LO, HI, cam, 0.0)
    px_area = (3.0 / 96) ** 2
    assert abs(hit["mask"].sum().item() * px_area - math.pi) / math.pi < 0.03
    assert abs(hit["depth"][48, 48].item() - 4.0) < 0.02
    assert torch.isinf(hit["depth"][0, 0])


def test_occluder():
    occ = iso_occluder(_sphere(), LO, HI, 0.0, 0.02)
    o = torch.tensor([[3.0, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
    d = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    assert occ(o, d, torch.tensor([10.0, 10.0], dtype=torch.float64)).tolist() == [True, False]
    assert occ(o, d, torch.tensor([1.0, 10.0], dtype=torch.float64)).tolist() == [False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_iso.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.iso'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/iso.py`:
```python
"""First crossing of field == level along rays, with bisection refinement on the trilinear field.
Features thinner than `step` can be missed; plates of fractal data should use voxels.py instead (spec §0.5)."""
import math

import torch

from .camera import ray_box
from .grid import clip_keep, sample_grid


def _solid(field, lo, hi, pts, level, clip):
    s = sample_grid(field, lo, hi, pts, padding="border") >= level
    return s & clip_keep(pts, clip) if clip else s


def march_iso(field, lo, hi, o, d, level, step, *, refine=12, clip=(), tmax=None, chunk_samples=2 ** 22):
    N = o.shape[0]
    out = torch.full((N,), math.inf, dtype=o.dtype, device=o.device)
    tn, tf = ray_box(o, d, lo, hi)
    if tmax is not None:
        tf = torch.minimum(tf, tmax)
    idx = (tf > tn).nonzero().squeeze(1)
    if idx.numel() == 0:
        return out
    nmax = int(math.ceil((tf - tn)[idx].max().item() / step)) + 1
    per = max(1, chunk_samples // nmax)
    k = torch.arange(nmax, dtype=o.dtype, device=o.device)
    for s in range(0, idx.numel(), per):
        ii = idx[s:s + per]
        ts = torch.minimum(tn[ii, None] + k * step, tf[ii, None])
        pts = o[ii, None, :] + ts[..., None] * d[ii, None, :]
        solid = _solid(field, lo, hi, pts, level, clip)
        has = solid.any(1)
        first = solid.to(torch.int8).argmax(1)
        r = torch.arange(ii.numel(), device=o.device)
        t_hi = ts[r, first]
        t_lo = torch.where(first > 0, ts[r, (first - 1).clamp_min(0)], t_hi)
        for _ in range(refine):
            mid = 0.5 * (t_lo + t_hi)
            sm = _solid(field, lo, hi, o[ii] + mid[:, None] * d[ii], level, clip)
            t_hi = torch.where(sm, mid, t_hi)
            t_lo = torch.where(sm, t_lo, mid)
        out[ii] = torch.where(has, t_hi, torch.full_like(t_hi, math.inf))
    return out


def iso_normals(field, lo, hi, pts, level, *, clip=(), tol=None):
    lo_t = torch.as_tensor(lo, dtype=pts.dtype, device=pts.device)
    hi_t = torch.as_tensor(hi, dtype=pts.dtype, device=pts.device)
    shape = torch.tensor(field.shape[::-1], dtype=pts.dtype, device=pts.device)   # (W, H, D) = x, y, z counts
    h = (hi_t - lo_t) / (shape - 1)
    tol = float(h.min()) * 0.25 if tol is None else tol
    g = torch.zeros_like(pts)
    for ax in range(3):
        e = torch.zeros(3, dtype=pts.dtype, device=pts.device); e[ax] = h[ax]
        g[:, ax] = (sample_grid(field, lo, hi, pts + e, padding="border") -
                    sample_grid(field, lo, hi, pts - e, padding="border")) / (2 * h[ax])
    n = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for ax in range(3):                                          # box faces
        for side, bound in ((-1.0, lo_t[ax]), (1.0, hi_t[ax])):
            on = (pts[:, ax] - bound).abs() < tol
            face = torch.zeros(3, dtype=pts.dtype, device=pts.device); face[ax] = side
            n = torch.where(on[:, None], face, n)
    for point, normal in clip:                                   # clip faces win over box faces
        p = torch.as_tensor(point, dtype=pts.dtype, device=pts.device)
        c = torch.as_tensor(normal, dtype=pts.dtype, device=pts.device)
        c = c / c.norm()
        on = ((pts - p) * c).sum(-1).abs() < tol
        n = torch.where(on[:, None], c, n)
    return n


def render_iso(field, lo, hi, cam, level, step=None, *, clip=(), device="cpu"):
    if step is None:
        h = [(b - a) / (n - 1) for a, b, n in zip(lo, hi, field.shape[::-1])]
        step = 0.5 * min(h)
    o, d = cam.rays(device=device, dtype=field.dtype)
    H, W = o.shape[:2]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    t = march_iso(field, lo, hi, o, d, level, step, clip=clip)
    mask = torch.isfinite(t)
    pos = o + torch.where(mask, t, torch.zeros_like(t))[:, None] * d
    normal = torch.zeros_like(pos)
    if mask.any():
        normal[mask] = iso_normals(field, lo, hi, pos[mask], level, clip=clip)
    depth = torch.where(mask, cam.t_to_depth(t, d), torch.full_like(t, math.inf))
    return dict(depth=depth.reshape(H, W), pos=pos.reshape(H, W, 3), normal=normal.reshape(H, W, 3), mask=mask.reshape(H, W))


def iso_occluder(field, lo, hi, level, step, clip=()):
    def occluded(o, d, tmax):
        return torch.isfinite(march_iso(field, lo, hi, o, d, level, step, clip=clip, tmax=tmax))
    return occluded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_iso.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: isosurface marcher with bisection, clip caps, outward normals, occluder"
```

---
### Task 5: Crisp voxel marcher

**Files:**
- Create: `art/_shared/r3d/voxels.py`
- Test: `art/_shared/r3d/tests/test_voxels.py`

**Interfaces:**
- Consumes: `ray_box` (Task 2), `clip_keep`, `sample_grid` (Task 3; tests only for sampling).
- Produces:
  - Cells are centred on grid points with size `h`, so a grid fills `[lo - h/2, hi + h/2]`. Clip planes keep or drop whole cells by their centre (a crisp staircase cut, declared).
  - `march_voxels(solid (D,H,W) bool, lo, hi, o, d, *, tmax=None, clip=(), chunk=2**20) -> (t (N,), normal (N,3), cell (N,3) long as (i,j,k))`; misses: `t = inf`, `cell = -1`.
  - `render_voxels(labels (D,H,W) int, lo, hi, cam, *, solid=None, clip=(), device="cpu") -> dict(depth, pos, normal, cell (H,W,3), label (H,W) with -1 = miss, mask)`; `solid` defaults to `labels > 0`.
  - `voxel_occluder(solid, lo, hi, clip=()) -> occluded(o, d, tmax) -> bool (N,)`.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_voxels.py`:
```python
import math
import torch
from r3d.camera import Camera, ray_box
from r3d.grid import sample_grid
from r3d.voxels import march_voxels, render_voxels, voxel_occluder


def test_single_voxel_exact_footprint_depth_and_normal():
    solid = torch.zeros(3, 3, 3, dtype=torch.bool); solid[1, 1, 1] = True
    cam = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=64, ortho_height=4.0)
    hit = render_voxels(solid.long(), (-1, -1, -1), (1, 1, 1), cam)
    assert hit["mask"].sum().item() == 256
    assert torch.allclose(hit["depth"][hit["mask"]].float(), torch.tensor(4.5))
    assert torch.allclose(hit["normal"][hit["mask"]].float(), torch.tensor([1.0, 0, 0]))
    assert (hit["label"][hit["mask"]] == 1).all() and (hit["label"][~hit["mask"]] == -1).all()


def test_matches_brute_force_on_random_rays():
    g = torch.Generator().manual_seed(0)
    solid = torch.rand(16, 16, 16, generator=g) < 0.03
    lo, hi = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    h = 2 / 15
    o = torch.randn(200, 3, generator=g, dtype=torch.float64) * 0.3 + torch.tensor([4.0, 3.0, 2.5], dtype=torch.float64)
    tgt = (torch.rand(200, 3, generator=g, dtype=torch.float64) * 2 - 1)
    d = tgt - o; d = d / d.norm(dim=-1, keepdim=True)
    t, n, cell = march_voxels(solid, lo, hi, o, d)
    blo, bhi = [-1 - h / 2] * 3, [1 + h / 2] * 3
    tn, tf = ray_box(o, d, blo, bhi)
    step = 2e-3
    for r in range(200):
        if tf[r] <= tn[r]:
            assert math.isinf(t[r]); continue
        ts = torch.arange(tn[r].item(), tf[r].item(), step, dtype=torch.float64)
        v = sample_grid(solid.double(), lo, hi, o[r] + ts[:, None] * d[r], mode="nearest") > 0.5
        if not v.any():
            assert math.isinf(t[r]); continue
        tb = ts[v.nonzero()[0, 0]].item()
        assert t[r].item() <= tb + 1e-9 and tb - t[r].item() < step + 1e-9
        p = o[r] + t[r] * d[r]
        ijk = torch.floor((p + 1e-7 * d[r] - torch.tensor(blo, dtype=torch.float64)) / h).long()
        assert cell[r].tolist() == ijk.tolist()
        assert abs(n[r].norm().item() - 1) < 1e-12 and (n[r] * d[r]).sum() < 0


def test_clip_drops_whole_cells_by_centre():
    solid = torch.ones(3, 3, 3, dtype=torch.bool)
    o = torch.tensor([[5.0, 0, 0]], dtype=torch.float64); d = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    t, n, _ = march_voxels(solid, (-1, -1, -1), (1, 1, 1), o, d, clip=[((0.5, 0, 0), (1, 0, 0))])
    assert abs(t.item() - 4.5) < 1e-9 and n[0].tolist() == [1.0, 0.0, 0.0]
    t, _, _ = march_voxels(solid, (-1, -1, -1), (1, 1, 1), o, d, clip=[((-0.1, 0, 0), (1, 0, 0))])
    assert abs(t.item() - 5.5) < 1e-9


def test_voxel_occluder():
    solid = torch.zeros(3, 3, 3, dtype=torch.bool); solid[1, 1, 1] = True
    occ = voxel_occluder(solid, (-1, -1, -1), (1, 1, 1))
    o = torch.tensor([[3.0, 0, 0]] * 2, dtype=torch.float64)
    d = torch.tensor([[-1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64)
    assert occ(o, d, torch.tensor([10.0, 10.0], dtype=torch.float64)).tolist() == [True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_voxels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.voxels'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/voxels.py`:
```python
"""Exact voxel traversal (Amanatides & Woo 1987), vectorised over rays. No interpolation: every face is a real cell face."""
import math

import torch

from .camera import ray_box
from .grid import clip_keep


def _geometry(solid, lo, hi, dtype, device):
    lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
    hi_t = torch.as_tensor(hi, dtype=dtype, device=device)
    n = torch.tensor(solid.shape[::-1], device=device)                 # (W, H, D) = counts along x, y, z
    h = (hi_t - lo_t) / (n - 1).to(dtype)
    return lo_t, hi_t, n, h


def _apply_clip(solid, lo_t, h, clip):
    if not clip:
        return solid
    D, H, W = solid.shape
    z, y, x = (torch.arange(m, dtype=h.dtype, device=h.device) for m in (D, H, W))
    Z, Y, X = torch.meshgrid(lo_t[2] + z * h[2], lo_t[1] + y * h[1], lo_t[0] + x * h[0], indexing="ij")
    return solid & clip_keep(torch.stack([X, Y, Z], -1), clip)


def march_voxels(solid, lo, hi, o, d, *, tmax=None, clip=(), chunk=2 ** 20):
    dt, dev = o.dtype, o.device
    solid = solid.to(dev)
    lo_t, hi_t, n, h = _geometry(solid, lo, hi, dt, dev)
    solid = _apply_clip(solid, lo_t, h, clip)
    blo, bhi = lo_t - h / 2, hi_t + h / 2
    N = o.shape[0]
    t_out = torch.full((N,), math.inf, dtype=dt, device=dev)
    n_out = torch.zeros(N, 3, dtype=dt, device=dev)
    c_out = torch.full((N, 3), -1, dtype=torch.long, device=dev)
    tn, tf = ray_box(o, d, blo, bhi)
    if tmax is not None:
        tf = torch.minimum(tf, tmax)
    idx = (tf > tn).nonzero().squeeze(1)
    for s in range(0, idx.numel(), chunk):
        ii = idx[s:s + chunk]
        oo, dd, tfar = o[ii], d[ii], tf[ii]
        t = tn[ii].clone()
        R = ii.numel()
        r_all = torch.arange(R, device=dev)
        safe = torch.where(dd == 0, torch.ones_like(dd), dd)
        ta, tb = (blo - oo) / safe, (bhi - oo) / safe
        entry = torch.where(dd == 0, torch.full_like(dd, -math.inf), torch.minimum(ta, tb))
        ax = entry.argmax(1)                                           # axis of the entry face
        sgn = -torch.sign(dd[r_all, ax])
        p = oo + t[:, None] * dd
        cell = torch.floor((p - blo) / h).long()
        cell = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), (n - 1).expand_as(cell))
        stp = torch.sign(dd).long()
        tdelta = torch.where(dd == 0, torch.full_like(dd, math.inf), h / dd.abs())
        nb = blo + (cell + (stp > 0).long()).to(dt) * h
        tmx = torch.where(dd == 0, torch.full_like(dd, math.inf), (nb - oo) / safe)
        active = r_all
        for _ in range(int(n.sum().item()) + 3):
            if active.numel() == 0:
                break
            c = cell[active]
            hit = solid[c[:, 2], c[:, 1], c[:, 0]]
            if hit.any():
                a = active[hit]
                t_out[ii[a]] = t[a]
                nv = torch.zeros(a.numel(), 3, dtype=dt, device=dev)
                nv[torch.arange(a.numel(), device=dev), ax[a]] = sgn[a]
                n_out[ii[a]] = nv
                c_out[ii[a]] = cell[a]
            rem = active[~hit]
            if rem.numel() == 0:
                break
            rr = torch.arange(rem.numel(), device=dev)
            axr = tmx[rem].argmin(1)
            t[rem] = tmx[rem, axr]
            cell[rem, axr] = cell[rem, axr] + stp[rem, axr]
            tmx[rem, axr] = tmx[rem, axr] + tdelta[rem, axr]
            ax[rem] = axr
            sgn[rem] = -stp[rem, axr].to(dt)
            cr = cell[rem]
            ok = (cr >= 0).all(1) & (cr < n).all(1) & (t[rem] <= tfar[rem])
            active = rem[ok]
    return t_out, n_out, c_out


def render_voxels(labels, lo, hi, cam, *, solid=None, clip=(), device="cpu"):
    labels = labels.to(device)
    solid = (labels > 0) if solid is None else solid.to(device)
    o, d = cam.rays(device=device, dtype=torch.float64)
    H, W = o.shape[:2]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    t, nrm, cell = march_voxels(solid, lo, hi, o, d, clip=clip)
    mask = torch.isfinite(t)
    pos = o + torch.where(mask, t, torch.zeros_like(t))[:, None] * d
    label = torch.full((o.shape[0],), -1, dtype=labels.dtype, device=device)
    if mask.any():
        c = cell[mask]
        label[mask] = labels[c[:, 2], c[:, 1], c[:, 0]]
    depth = torch.where(mask, cam.t_to_depth(t, d), torch.full_like(t, math.inf))
    return dict(depth=depth.reshape(H, W), pos=pos.reshape(H, W, 3), normal=nrm.reshape(H, W, 3),
                cell=cell.reshape(H, W, 3), label=label.reshape(H, W), mask=mask.reshape(H, W))


def voxel_occluder(solid, lo, hi, clip=()):
    def occluded(o, d, tmax):
        return torch.isfinite(march_voxels(solid, lo, hi, o, d, tmax=tmax, clip=clip)[0])
    return occluded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_voxels.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: exact vectorised voxel traversal for categorical and fractal volumes"
```

---
### Task 6: Shading — Lambert, shadows, ambient occlusion

**Files:**
- Create: `art/_shared/r3d/shade.py`
- Test: `art/_shared/r3d/tests/test_shade.py`

**Interfaces:**
- Consumes: any `occluded(o (M,3), d (M,3), tmax (M,)) -> bool (M,)`, e.g. `iso_occluder` (Task 4) or `voxel_occluder` (Task 5).
- Produces:
  - `lambert(normal (...,3), light_dir, ambient=0.2) -> (...)`; `light_dir` points toward the light.
  - `hemisphere_dirs(n_rays, seed=0, dtype=torch.float64, device="cpu") -> (n_rays,3)`: cosine-weighted, Fibonacci-stratified, around +z.
  - `ambient_occlusion(pos (N,3), normal (N,3), occluded, n_rays=16, radius=1.0, seed=0, bias=1e-4) -> (N,)`: 1 = fully open.
  - `hard_shadow(pos, normal, light_dir, occluded, bias=1e-4, tmax=1e9) -> (N,)`: 1 = lit, 0 = shadowed or facing away.
  - Lighting is form only, never data (spec §0.3).

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_shade.py`:
```python
import torch
from r3d.shade import lambert, hemisphere_dirs, ambient_occlusion, hard_shadow

UP = torch.tensor([[0.0, 0, 1]], dtype=torch.float64)


def never(o, d, tmax):
    return torch.zeros(o.shape[0], dtype=torch.bool)


def wall_at_x0(o, d, tmax):                     # solid half-space x < 0
    t = -o[:, 0] / torch.where(d[:, 0] == 0, torch.full_like(d[:, 0], 1e-30), d[:, 0])
    return (d[:, 0] < 0) & (t >= 0) & (t <= tmax)


def test_lambert():
    n = torch.tensor([[0.0, 0, 1], [1.0, 0, 0], [0, 0, -1.0]], dtype=torch.float64)
    v = lambert(n, (0, 0, 2), ambient=0.25)
    assert torch.allclose(v, torch.tensor([1.0, 0.25, 0.25], dtype=torch.float64))


def test_hemisphere_dirs_are_cosine_weighted():
    d = hemisphere_dirs(4096, seed=3)
    assert torch.allclose(d.norm(dim=1), torch.ones(4096, dtype=torch.float64))
    assert (d[:, 2] > 0).all()
    assert abs(d[:, 2].mean().item() - 2 / 3) < 0.01
    assert abs(d[:, 0].mean().item()) < 0.01


def test_ao_open_corner_and_radius():
    p = torch.tensor([[0.0, 0, 0]], dtype=torch.float64)
    assert ambient_occlusion(p, UP, never, n_rays=64).item() == 1.0
    ao = ambient_occlusion(p, UP, wall_at_x0, n_rays=256, radius=10.0)
    assert abs(ao.item() - 0.5) < 0.05
    far = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)
    assert ambient_occlusion(far, UP, wall_at_x0, n_rays=256, radius=0.5).item() == 1.0


def test_ao_handles_arbitrary_normals():
    p = torch.tensor([[0.5, 0, 0]], dtype=torch.float64)
    n = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)       # facing away from the wall
    assert ambient_occlusion(p, n, wall_at_x0, n_rays=64, radius=10.0).item() == 1.0


def test_hard_shadow():
    p = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    n = torch.tensor([[0.0, 0, 1], [0, 0, -1.0]], dtype=torch.float64)
    assert hard_shadow(p, n, (-1, 0, 1), wall_at_x0).tolist() == [0.0, 0.0]   # blocked, and facing away
    assert hard_shadow(p, n, (1, 0, 1), wall_at_x0).tolist() == [1.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_shade.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.shade'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/shade.py`:
```python
"""Form lighting. Brightness from these functions shows shape only; data lives in colour and position (spec §0.3)."""
import math

import torch


def _unit(v, like):
    v = torch.as_tensor(v, dtype=like.dtype, device=like.device)
    return v / v.norm()


def lambert(normal, light_dir, ambient=0.2):
    L = _unit(light_dir, normal)
    return ambient + (1 - ambient) * (normal * L).sum(-1).clamp_min(0)


def hemisphere_dirs(n_rays, seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    u0 = torch.rand(1, generator=g, dtype=torch.float64).item()
    i = torch.arange(n_rays, dtype=torch.float64) + 0.5
    r = torch.sqrt(i / n_rays)
    phi = 2 * math.pi * (i * (math.sqrt(5) - 1) / 2 + u0)
    d = torch.stack([r * torch.cos(phi), r * torch.sin(phi), torch.sqrt((1 - r * r).clamp_min(0))], 1)
    return d.to(device, dtype)


def _frame(n):
    ex = torch.tensor([1.0, 0, 0], dtype=n.dtype, device=n.device)
    ey = torch.tensor([0, 1.0, 0], dtype=n.dtype, device=n.device)
    a = torch.where((n[:, :1].abs() < 0.9), ex, ey)
    t = torch.linalg.cross(n, a)
    t = t / t.norm(dim=-1, keepdim=True)
    return t, torch.linalg.cross(n, t)


def ambient_occlusion(pos, normal, occluded, n_rays=16, radius=1.0, seed=0, bias=1e-4):
    N = pos.shape[0]
    loc = hemisphere_dirs(n_rays, seed, pos.dtype, pos.device)
    t, b = _frame(normal)
    w = loc[None, :, 0:1] * t[:, None] + loc[None, :, 1:2] * b[:, None] + loc[None, :, 2:3] * normal[:, None]
    o = (pos + bias * normal)[:, None, :].expand(N, n_rays, 3).reshape(-1, 3)
    tmax = torch.full((N * n_rays,), float(radius), dtype=pos.dtype, device=pos.device)
    hit = occluded(o, w.reshape(-1, 3), tmax).reshape(N, n_rays)
    return 1 - hit.to(pos.dtype).mean(1)


def hard_shadow(pos, normal, light_dir, occluded, bias=1e-4, tmax=1e9):
    L = _unit(light_dir, pos)
    facing = (normal * L).sum(-1) > 0
    o = pos + bias * normal
    hit = occluded(o, L.expand_as(pos).contiguous(), torch.full((pos.shape[0],), float(tmax), dtype=pos.dtype, device=pos.device))
    return (facing & ~hit).to(pos.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_shade.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: Lambert, hard shadows, cosine-weighted ambient occlusion"
```

---
### Task 7: Tubes, glow and hidden-line SVG

**Files:**
- Create: `art/_shared/r3d/tubes.py`
- Test: `art/_shared/r3d/tests/test_tubes.py`

**Interfaces:**
- Consumes: `Camera.project`, `Camera.basis`, `Camera.pixel_scale` (Task 2).
- Produces:
  - `sample_polyline(P (N,3), spacing) -> (pts (M,3), s (M,))`; consecutive samples are ≤ `spacing` apart; both endpoints are included; `s` is segment index + fraction.
  - `splat_spheres(centers (M,3), radius float|(M,), cam, *, attrs (M,C)=None, chunk=2**22) -> dict(depth (H,W), normal (H,W,3), attr (H,W,C) or None, mask (H,W))`. A tube is spheres sampled at spacing ≤ 0.2·radius (ripple ≈ 0.005·radius). Exact union-of-spheres depth under orthographic cameras; under perspective the disc radius uses the centre's depth (declared approximation). Needs radius ≥ ~0.75 px; use `splat_additive` for hairlines.
  - `splat_additive(points (M,3), cam, *, weight=None, color (M,3)=None, sigma_px=0.7, depth=None, eps=0.0, chunk=2**22) -> (H,W) or (H,W,3)`: Gaussian splats that each deposit ≈ `weight` in total; points behind `depth + eps` are dropped.
  - `visible_runs(P (N,3), cam, depth (H,W), eps) -> list[np.ndarray (M,2)]`: visible stretches of a dense polyline in pixel coordinates. Pass `eps ≥` the tube radius when testing a tube's own centreline.
  - `write_svg(path, polylines, width, height, stroke="#000000", stroke_width=1.0, background=None)`.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_tubes.py`:
```python
import math
import xml.etree.ElementTree as ET
import torch
from r3d.camera import Camera
from r3d.tubes import sample_polyline, splat_spheres, splat_additive, visible_runs, write_svg

CAM = Camera(eye=(5, 0, 0), target=(0, 0, 0), width=64, height=64, ortho_height=4.0)   # 16 px per unit


def test_sample_polyline_spacing_and_endpoints():
    P = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 2, 0]], dtype=torch.float64)
    pts, s = sample_polyline(P, 0.3)
    assert torch.allclose(pts[0], P[0]) and torch.allclose(pts[-1], P[-1])
    assert (pts[1:] - pts[:-1]).norm(dim=1).max() <= 0.3 + 1e-12
    assert (s[1:] >= s[:-1]).all() and s[-1].item() == 2.0


def test_single_sphere_area_depth_normal():
    hit = splat_spheres(torch.zeros(1, 3, dtype=torch.float64), 1.0, CAM)
    assert abs(hit["mask"].sum().item() - math.pi * 16 ** 2) / (math.pi * 16 ** 2) < 0.03
    assert abs(hit["depth"][32, 32].item() - 4.0) < 0.01
    assert torch.allclose(hit["normal"][32, 32], torch.tensor([1.0, 0, 0], dtype=torch.float64), atol=0.05)
    top = hit["normal"][17, 32]                                  # near the top edge the normal tilts to +z
    assert top[2] > 0.7


def test_nearer_sphere_wins():
    c = torch.tensor([[0.0, 0, 0], [1.0, 0.2, 0]], dtype=torch.float64)   # second is nearer the camera at +x
    hit = splat_spheres(c, 0.8, CAM, attrs=torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    assert hit["attr"][32, 32, 0].item() == 2.0
    assert hit["attr"][hit["mask"]].unique().tolist() == [1.0, 2.0]


def test_additive_conserves_mass_and_respects_depth():
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(100, 3, generator=g, dtype=torch.float64) - 0.5)
    acc = splat_additive(pts, CAM, sigma_px=1.0)
    assert abs(acc.sum().item() - 100) < 2
    wall = torch.full((64, 64), 4.0, dtype=torch.float64)       # opaque wall at x = 1, in front of every point
    assert splat_additive(pts, CAM, sigma_px=1.0, depth=wall).sum().item() == 0.0
    col = splat_additive(pts, CAM, color=torch.tensor([[1.0, 0, 0]] * 100, dtype=torch.float64))
    assert col.shape == (64, 64, 3) and col[..., 1].sum() == 0


def test_visible_runs_and_svg(tmp_path):
    hit = splat_spheres(torch.zeros(1, 3, dtype=torch.float64), 0.5, CAM)
    line, _ = sample_polyline(torch.tensor([[-1.0, -1.5, 0], [-1.0, 1.5, 0]], dtype=torch.float64), 0.01)
    runs = visible_runs(line, CAM, hit["depth"], eps=1e-6)       # line at x = -1 passes behind the sphere
    assert len(runs) == 2
    front, _ = sample_polyline(torch.tensor([[1.0, -1.5, 0], [1.0, 1.5, 0]], dtype=torch.float64), 0.01)
    assert len(visible_runs(front, CAM, hit["depth"], eps=1e-6)) == 1
    p = tmp_path / "x.svg"
    write_svg(p, runs, 64, 64, background="#ffffff")
    root = ET.parse(p).getroot()
    assert len([e for e in root.iter() if e.tag.endswith("polyline")]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_tubes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.tubes'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/tubes.py`:
```python
"""Curves in space: tubes as z-buffered sphere impostors, hairlines as additive Gaussian splats, and hidden-line SVG."""
import math

import numpy as np
import torch


def sample_polyline(P, spacing):
    seg = P[1:] - P[:-1]
    L = seg.norm(dim=1)
    k = torch.ceil(L / spacing).clamp_min(1).long()
    sid = torch.repeat_interleave(torch.arange(len(seg), device=P.device), k)
    start = torch.cumsum(k, 0) - k
    frac = (torch.arange(int(k.sum()), device=P.device) - start[sid]).to(P.dtype) / k[sid].to(P.dtype)
    pts = torch.cat([P[sid] + frac[:, None] * seg[sid], P[-1:]], 0)
    s = torch.cat([sid.to(P.dtype) + frac, torch.tensor([float(len(seg))], dtype=P.dtype, device=P.device)])
    return pts, s


def _offsets(K, device):
    o = torch.arange(-K, K + 1, device=device)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    return ox.reshape(-1), oy.reshape(-1)


def splat_spheres(centers, radius, cam, *, attrs=None, chunk=2 ** 22):
    dt, dev = centers.dtype, centers.device
    H, W = cam.height, cam.width
    M = centers.shape[0]
    R = torch.as_tensor(radius, dtype=dt, device=dev).expand(M)
    pix, z = cam.project(centers)
    pix, z = pix.to(dev, dt), z.to(dev, dt)
    ps = torch.full_like(z, cam.pixel_scale()) if cam.fov_deg is None else cam.pixel_scale(z)
    Rpx = R / ps
    f, r, u = (torch.tensor(v, dtype=dt, device=dev) for v in cam.basis())
    K = int(math.ceil(Rpx.max().item()))
    ox, oy = _offsets(K, dev)
    per = max(1, chunk // ox.numel())
    zbuf = torch.full((H * W,), math.inf, dtype=dt, device=dev)
    nbuf = torch.zeros(H * W, 3, dtype=dt, device=dev)
    abuf = None if attrs is None else torch.zeros(H * W, attrs.shape[1], dtype=dt, device=dev)
    for s in range(0, M, per):
        sl = slice(s, s + per)
        col, row = pix[sl, 0:1], pix[sl, 1:2]
        pc = torch.floor(col) + ox + 0.5
        pr = torch.floor(row) + oy + 0.5
        rp = Rpx[sl, None]
        dx, dy = (pc - col) / rp, (pr - row) / rp
        rho2 = dx * dx + dy * dy
        inside = (rho2 <= 1) & (pc >= 0) & (pc < W) & (pr >= 0) & (pr < H) & (z[sl, None] > 0)
        if not inside.any():
            continue
        nz = torch.sqrt((1 - rho2).clamp_min(0))
        depth = (z[sl, None] - R[sl, None] * nz)[inside]
        flat = (pr.long() * W + pc.long())[inside]
        nrm = (dx[..., None] * r - dy[..., None] * u - nz[..., None] * f)[inside]
        cmin = torch.full((H * W,), math.inf, dtype=dt, device=dev).scatter_reduce(0, flat, depth, "amin")
        win = (depth == cmin[flat]) & (depth < zbuf[flat])
        fw = flat[win]
        zbuf[fw] = depth[win]
        nbuf[fw] = nrm[win]
        if abuf is not None:
            sid = torch.arange(s, min(s + per, M), device=dev)[:, None].expand_as(inside)[inside]
            abuf[fw] = attrs[sid[win]].to(dt)
    mask = torch.isfinite(zbuf)
    return dict(depth=zbuf.reshape(H, W), normal=nbuf.reshape(H, W, 3),
                attr=None if abuf is None else abuf.reshape(H, W, -1), mask=mask.reshape(H, W))


def splat_additive(points, cam, *, weight=None, color=None, sigma_px=0.7, depth=None, eps=0.0, chunk=2 ** 22):
    dt, dev = points.dtype, points.device
    H, W = cam.height, cam.width
    M = points.shape[0]
    pix, z = cam.project(points)
    pix, z = pix.to(dev, dt), z.to(dev, dt)
    w = torch.ones(M, dtype=dt, device=dev) if weight is None else torch.as_tensor(weight, dtype=dt, device=dev).expand(M)
    C = 1 if color is None else 3
    acc = torch.zeros(H * W, C, dtype=dt, device=dev)
    K = int(math.ceil(3 * sigma_px))
    ox, oy = _offsets(K, dev)
    per = max(1, chunk // ox.numel())
    norm = 1.0 / (2 * math.pi * sigma_px ** 2)
    for s in range(0, M, per):
        sl = slice(s, s + per)
        col, row = pix[sl, 0:1], pix[sl, 1:2]
        pc = torch.floor(col) + ox + 0.5
        pr = torch.floor(row) + oy + 0.5
        g = norm * torch.exp(-((pc - col) ** 2 + (pr - row) ** 2) / (2 * sigma_px ** 2)) * w[sl, None]
        ok = (pc >= 0) & (pc < W) & (pr >= 0) & (pr < H) & (z[sl, None] > 0)
        flat = (pr.long().clamp(0, H - 1) * W + pc.long().clamp(0, W - 1))
        if depth is not None:
            ok &= z[sl, None] <= depth.reshape(-1).to(dt)[flat] + eps
        val = g[ok][:, None]
        if color is not None:
            val = val * color[sl].to(dt)[:, None, :].expand(-1, ox.numel(), 3)[ok]
        acc.index_add_(0, flat[ok], val)
    out = acc.reshape(H, W, C)
    return out[..., 0] if color is None else out


def visible_runs(P, cam, depth, eps):
    pix, z = cam.project(P)
    col, row = pix[:, 0], pix[:, 1]
    H, W = depth.shape
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H) & (z > 0)
    ci, ri = col.long().clamp(0, W - 1), row.long().clamp(0, H - 1)
    vis = inb & (z <= depth.double().cpu()[ri, ci] + eps)
    runs, cur = [], []
    xy = pix.numpy()
    for i, v in enumerate(vis.tolist()):
        if v:
            cur.append(xy[i])
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    return [r for r in runs if len(r) >= 2]


def write_svg(path, polylines, width, height, stroke="#000000", stroke_width=1.0, background=None):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    if background:
        out.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')
    for pl in polylines:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in pl)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_tubes.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: sphere-impostor tubes, additive glow splats, hidden-line SVG"
```

---
### Task 8: Meshes, STL and films

**Files:**
- Create: `art/_shared/r3d/mesh.py`, `art/_shared/r3d/io.py`
- Test: `art/_shared/r3d/tests/test_mesh_io.py`

**Interfaces:**
- Produces:
  - `marching_cubes(field (D,H,W) np|torch, level, lo, hi, closed=True) -> (verts (V,3) float64 world xyz, faces (F,3) int64)`, outward-oriented for solid = `field >= level`. `closed=True` pads with `level - 1` so solids touching the box are capped.
  - `mesh_volume(verts, faces) -> float` (signed; positive when outward).
  - `is_watertight(faces) -> bool`: every directed edge appears once and its reverse once.
  - `write_stl(path, verts, faces, header="r3d")`, `read_stl(path) -> (F,3,3) float32`.
  - `tube_mesh(P (N,3), radius, n_sides=16, cap=True) -> (verts, faces)`, with parallel-transport frames, closed.
  - `scale_to_mm(verts, size_mm) -> verts`: largest extent becomes `size_mm`, min corner at 0.
  - `save_png(path, img)`: float `[0,1]` `(H,W)`, `(H,W,3)` or `(H,W,4)`, tensor or array; clipped; 8-bit.
  - `glow_tonemap(x, exposure) -> 1 - exp(-exposure * x)`.
  - `write_film(pattern, mp4, fps=30, gif=None, gif_width=540)`: pattern like `dir/f_%05d.png`.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_mesh_io.py`:
```python
import math, subprocess
import numpy as np
import torch
from PIL import Image
from r3d.mesh import marching_cubes, mesh_volume, is_watertight, write_stl, read_stl, tube_mesh, scale_to_mm
from r3d.io import save_png, glow_tonemap, write_film


def _ball(n=64):
    a = np.linspace(-1.5, 1.5, n)
    Z, Y, X = np.meshgrid(a, a, a, indexing="ij")
    return 1 - np.sqrt(X ** 2 + Y ** 2 + Z ** 2)


def test_marching_cubes_ball_volume_orientation_watertight():
    v, f = marching_cubes(_ball(), 0.0, (-1.5,) * 3, (1.5,) * 3)
    assert is_watertight(f)
    assert abs(mesh_volume(v, f) - 4 / 3 * math.pi) / (4 / 3 * math.pi) < 0.02


def test_marching_cubes_axis_order_and_box_capping():
    a = np.linspace(0, 1, 11)
    Z, Y, X = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 1, 11), np.linspace(0, 4, 41), indexing="ij")
    field = np.ones_like(X)                                         # solid fills a 4 x 1 x 2 box
    v, f = marching_cubes(field, 0.5, (0, 0, 0), (4, 1, 2))
    assert is_watertight(f)
    ext = v.max(0) - v.min(0)
    assert np.allclose(ext, [4 + 0.1, 1 + 0.1, 2 + 0.1], atol=1e-6)  # capped half a voxel outside
    assert mesh_volume(v, f) > 0


def test_stl_roundtrip_and_scale(tmp_path):
    v, f = marching_cubes(_ball(24), 0.0, (-1.5,) * 3, (1.5,) * 3)
    v = scale_to_mm(v, 50.0)
    assert np.isclose((v.max(0) - v.min(0)).max(), 50.0) and np.allclose(v.min(0), 0)
    p = tmp_path / "b.stl"
    write_stl(p, v, f)
    tri = read_stl(p)
    assert tri.shape == (len(f), 3, 3)
    assert np.allclose(tri, v[f].astype(np.float32))


def test_tube_mesh_volume():
    P = np.array([[0.0, 0, 0], [0, 0, 1], [0, 0, 2]])
    v, f = tube_mesh(P, 0.1, n_sides=64)
    assert is_watertight(f)
    poly = 0.5 * 64 * math.sin(2 * math.pi / 64) * 0.01
    assert abs(mesh_volume(v, f) - poly * 2) / (poly * 2) < 1e-6
    bent = np.array([[math.cos(t), math.sin(t), 0.3 * t] for t in np.linspace(0, 6, 200)])
    vb, fb = tube_mesh(bent, 0.05, n_sides=12)
    assert is_watertight(fb) and mesh_volume(vb, fb) > 0


def test_png_tonemap_and_film(tmp_path):
    img = torch.zeros(10, 12, 3); img[..., 0] = 2.0
    save_png(tmp_path / "a.png", img)
    arr = np.asarray(Image.open(tmp_path / "a.png"))
    assert arr.shape == (10, 12, 3) and arr[0, 0, 0] == 255 and arr[0, 0, 1] == 0
    assert np.isclose(float(glow_tonemap(torch.tensor(1.0), 2.0)), 1 - math.exp(-2))
    for i in range(3):
        save_png(tmp_path / f"f_{i:05d}.png", np.full((32, 32), i / 2))
    mp4, gif = tmp_path / "x.mp4", tmp_path / "x.gif"
    write_film(str(tmp_path / "f_%05d.png"), mp4, fps=10, gif=gif, gif_width=32)
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.strip()
    assert n == "3" and gif.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_mesh_io.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'r3d.mesh'`.

- [ ] **Step 3: Write the implementation**

`art/_shared/r3d/mesh.py`:
```python
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
        F = np.pad(F, 1, constant_values=level - 1.0)
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
```

`art/_shared/r3d/io.py`:
```python
"""Image and film output. PNG only for data images (never JPEG)."""
import subprocess

import numpy as np
from PIL import Image


def save_png(path, img):
    a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def glow_tonemap(x, exposure):
    import torch
    x = torch.as_tensor(x)
    return 1 - torch.exp(-exposure * x)


def write_film(pattern, mp4, fps=30, gif=None, gif_width=540):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(pattern),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(mp4)], check=True)
    if gif is not None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(pattern), "-vf",
                        f"fps={min(fps, 15)},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_mesh_io.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: marching cubes, watertight STL, tube meshes, PNG and film output"
```

---
### Task 9: Analytic null gallery, package exports, README

**Files:**
- Modify: `art/_shared/r3d/__init__.py`
- Create: `art/_shared/r3d/examples/nulls.py`, `art/_shared/r3d/README.md`
- Output: `art/_shared/r3d/examples/gallery/*.png`, `*.svg`
- Test: `art/_shared/r3d/tests/test_package.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces: `import r3d` exposing the whole public API. Four reference images, used by every piece as the "smooth null through the same pipeline" check (spec §11.5):
  - `null_plaster_torus.png`: iso + AO + shadow;
  - `null_voxels_checker.png`: crisp voxels with a clip cut;
  - `null_volume_split.png`: a smooth two-sided field in Spectral-split fog;
  - `null_glow_knot.png` + `null_plotter_knot.svg`: a torus knot as tube, glow and hidden line.

- [ ] **Step 1: Write the failing test**

`art/_shared/r3d/tests/test_package.py`:
```python
import r3d

API = ["Camera", "orbit", "turntable", "stereo_pair", "ray_box", "sample_grid", "clip_keep", "TransferFunction",
       "lut_tf", "march_volume", "render_volume", "over", "march_iso", "iso_normals", "render_iso", "iso_occluder",
       "march_voxels", "render_voxels", "voxel_occluder", "lambert", "hemisphere_dirs", "ambient_occlusion",
       "hard_shadow", "sample_polyline", "splat_spheres", "splat_additive", "visible_runs", "write_svg",
       "marching_cubes", "mesh_volume", "is_watertight", "write_stl", "read_stl", "tube_mesh", "scale_to_mm",
       "save_png", "glow_tonemap", "write_film"]


def test_public_api():
    missing = [n for n in API if not hasattr(r3d, n)]
    assert missing == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests/test_package.py -q`
Expected: FAIL with a non-empty `missing` list.

- [ ] **Step 3: Write `__init__.py`**

```python
"""r3d: a small torch 3D renderer for the gallery. See README.md for conventions and the honesty rules."""
from .camera import Camera, orbit, turntable, stereo_pair, ray_box
from .grid import sample_grid, clip_keep
from .volume import TransferFunction, lut_tf, march_volume, render_volume, over
from .iso import march_iso, iso_normals, render_iso, iso_occluder
from .voxels import march_voxels, render_voxels, voxel_occluder
from .shade import lambert, hemisphere_dirs, ambient_occlusion, hard_shadow
from .tubes import sample_polyline, splat_spheres, splat_additive, visible_runs, write_svg
from .mesh import marching_cubes, mesh_volume, is_watertight, write_stl, read_stl, tube_mesh, scale_to_mm
from .io import save_png, glow_tonemap, write_film
```

Run: `cd /home/fzeng/ml/research/art/_shared && ../.venv/bin/python -m pytest r3d/tests -q`
Expected: all tests pass (`42 passed`).

- [ ] **Step 4: Write the null gallery script**

`art/_shared/r3d/examples/nulls.py`:
```python
"""Analytic nulls through the r3d pipeline (spec art/ml-art-3d.md §11.5). CPU, ~1-3 min.

  cd art/_shared && ../.venv/bin/python r3d/examples/nulls.py [--size 480]
"""
import argparse, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import r3d  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "gallery")
PLASTER, INK, NIGHT = torch.tensor([0.93, 0.91, 0.87]), torch.tensor([0.13, 0.12, 0.11]), torch.tensor([0.035, 0.035, 0.05])


def grid(n, a=1.5, dtype=torch.float32):
    t = torch.linspace(-a, a, n, dtype=dtype)
    Z, Y, X = torch.meshgrid(t, t, t, indexing="ij")
    return X, Y, Z


def plaster_torus(S):
    X, Y, Z = grid(129)
    f = 0.35 - torch.sqrt((torch.sqrt(X ** 2 + Y ** 2) - 0.9) ** 2 + Z ** 2)       # solid torus, R 0.9, r 0.35
    lo, hi = (-1.5,) * 3, (1.5,) * 3
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=35, el_deg=35, width=S, height=S, ortho_height=3.2)
    hit = r3d.render_iso(f, lo, hi, cam, 0.0)
    m = hit["mask"]
    occ = r3d.iso_occluder(f, lo, hi, 0.0, 0.02)
    light = (0.4, -0.6, 1.0)
    pos, nrm = hit["pos"][m], hit["normal"][m]
    ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=24, radius=0.6)
    sh = r3d.hard_shadow(pos, nrm, light, occ)
    lum = (0.25 + 0.75 * r3d.lambert(nrm, light, ambient=0.0) * (0.35 + 0.65 * sh)) * (0.4 + 0.6 * ao)
    img = PLASTER.expand(S, S, 3).clone() * 0.97
    img[m] = PLASTER * lum[:, None]
    r3d.save_png(os.path.join(OUT, "null_plaster_torus.png"), img)


def voxel_checker(S):
    n = 16
    k = torch.arange(n)
    Z, Y, X = torch.meshgrid(k, k, k, indexing="ij")
    labels = ((X // 4 + Y // 4 + Z // 4) % 2 + 1).long()
    lo, hi = (-1.0,) * 3, (1.0,) * 3
    clip = [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=40, el_deg=30, width=S, height=S, ortho_height=3.6)
    hit = r3d.render_voxels(labels, lo, hi, cam, clip=clip)
    m = hit["mask"]
    pal = torch.tensor([[0, 0, 0], [0.85, 0.45, 0.25], [0.25, 0.45, 0.75]], dtype=torch.float64)
    shade = r3d.lambert(hit["normal"][m], (0.3, -0.5, 1.0), ambient=0.35)
    img = NIGHT.double().expand(S, S, 3).clone()
    img[m] = pal[hit["label"][m]] * shade[:, None]
    r3d.save_png(os.path.join(OUT, "null_voxels_checker.png"), img)


def volume_split(S):
    import matplotlib
    X, Y, Z = grid(96)
    v = X ** 2 / 0.8 + Y ** 2 / 0.5 + Z ** 2 / 0.3 - 1.0                  # smooth two-sided field, seam at 0
    spec = matplotlib.colormaps["Spectral"]
    neg = r3d.TransferFunction(matplotlib.colors.ListedColormap(spec(torch.linspace(0.5, 1.0, 256).numpy())), -1.0, 0.0,
                               opacity=lambda x: 0.15 + 0.85 * (x < 0.97).float() * x, density=2.5)
    rgb_lut = torch.tensor(spec(torch.linspace(0.0, 1.0, 1024).numpy())[:, :3], dtype=torch.float32)

    def tf(val):
        inside = val < 0
        c_in, s_in = neg(val)
        x = (val / 3.0).clamp(0, 1)
        c_out = rgb_lut[(0.5 * (1 - x) * 1023).long()]                     # pale near the seam side, red far out
        s_out = 0.08 * (1 - x)
        return torch.where(inside[..., None], c_in, c_out), torch.where(inside, s_in, s_out)

    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=30, el_deg=25, width=S, height=S, ortho_height=3.2)
    rgb, a = r3d.render_volume(v, (-1.5,) * 3, (1.5,) * 3, cam, tf, step=0.01, clip=[((0, 0, 0), (0, -1, 1))])
    r3d.save_png(os.path.join(OUT, "null_volume_split.png"), r3d.over(rgb, a, NIGHT))


def knot(S):
    t = torch.linspace(0, 2 * math.pi, 4000, dtype=torch.float64)
    p, q = 2, 3
    r = 0.9 + 0.35 * torch.cos(q * t)
    P = torch.stack([r * torch.cos(p * t), r * torch.sin(p * t), 0.35 * torch.sin(q * t)], 1)
    cam = r3d.orbit((0, 0, 0), 6.0, az_deg=20, el_deg=55, width=S, height=S, ortho_height=3.0)
    pts, s = r3d.sample_polyline(P, 0.004)
    hit = r3d.splat_spheres(pts, 0.035, cam, attrs=s[:, None])
    glow = r3d.splat_additive(pts, cam, sigma_px=1.2, weight=0.6)
    m = hit["mask"]
    lum = r3d.lambert(hit["normal"], (0.2, -0.4, 1.0), ambient=0.25)
    img = NIGHT.double().expand(S, S, 3).clone() + r3d.glow_tonemap(glow, 0.8)[..., None] * torch.tensor([0.9, 0.55, 0.25], dtype=torch.float64)
    img[m] = (torch.tensor([0.95, 0.75, 0.45], dtype=torch.float64) * lum[m][:, None])
    r3d.save_png(os.path.join(OUT, "null_glow_knot.png"), img)
    runs = r3d.visible_runs(P, cam, hit["depth"], eps=0.04)
    r3d.write_svg(os.path.join(OUT, "null_plotter_knot.svg"), runs, S, S, stroke="#1f1d1b", stroke_width=1.2, background="#f3efe6")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=480)
    S = ap.parse_args().size
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    for fn in (plaster_torus, voxel_checker, volume_split, knot):
        fn(S); print("done", fn.__name__)
```

- [ ] **Step 5: Run it and inspect the images**

Run: `cd /home/fzeng/ml/research/art/_shared && OMP_NUM_THREADS=4 ../.venv/bin/python r3d/examples/nulls.py --size 480`
Expected: four `done` lines and five files in `r3d/examples/gallery/`. Open each PNG and check:
- the torus is smooth, lit from the upper left, with contact darkening in the hole;
- the checker has crisp cubic faces and a stepped diagonal cut;
- the ellipsoid fog has a smooth seam with no banding or ring artefacts;
- the knot's crossings occlude correctly, and the SVG shows gaps where a strand passes behind.

A renderer bug shows up here as structure in a smooth object. Fix it before continuing (spec §11.5).

- [ ] **Step 6: Write the README**

`art/_shared/r3d/README.md` must contain these sections, with the content below:
1. **What it is:** the one-paragraph summary from the plan Goal; import with `sys.path.insert(0, "/home/fzeng/ml/research/art/_shared"); import r3d`.
2. **Conventions:** the grid, world, image and depth-buffer bullets copied from this plan's Global Constraints; the solid convention `field >= level`; voxel cells centred on grid points; clip normals point into the removed half.
3. **Module table:** the File map rows for the r3d modules.
4. **Honesty rules:** spec `art/ml-art-3d.md` §0 items 2–6 in one line each; "use `render_voxels` for categorical and fractal data"; "every hero ships with a cutaway or slice plate"; "render the null through the same function and parameters".
5. **Performance notes:** everything is chunked; pass `device="cuda"` and tensors on the GPU for big renders, but only inside a job launched with `setsid nohup art/_shared/gpu1.sh ...`; CPU is fine for ≤ 512² stills.
6. **Nulls:** the four images from Step 5, embedded with `<img width="45%">`.

- [ ] **Step 7: Commit**

```bash
/home/fzeng/ml/research/art/_shared/commit.sh _shared "r3d: package API, analytic null gallery, README"
```

---

## After this plan

Each piece in `art/ml-art-3d.md` then gets its own directory, brief and plan (`docs/superpowers/plans/2026-09-15-3d-<piece>.md`), starting with the free-data trio: §1 space-time solid, §4 Stage A, §5 tori.
