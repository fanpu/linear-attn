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
