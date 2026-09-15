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
