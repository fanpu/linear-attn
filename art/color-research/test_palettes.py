"""Checks: Lab round trip, CIEDE2000 against Sharma et al. (2005) test pairs, and that
render_split(M, 'sd_spectral', near_boundary='large') reproduces Sohl-Dickstein's cdf_img + Spectral."""
import numpy as np, matplotlib as mpl
import palettes as P

rgb = np.random.default_rng(0).random((1000, 3))
assert np.abs(P.lab_to_rgb(P.rgb_to_lab(rgb)) - rgb).max() < 1e-6
# Sharma, Wu & Dalal (2005) pairs 1, 7, 17, 25
pairs = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
         ((50, 0, 0), (50, -1, 2), 2.3669),
         ((50, 2.5, 0), (73, 25, -18), 27.1492),
         ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]
for a, b, e in pairs:
    assert abs(P.deltaE2000(np.array(a), np.array(b)) - e) < 1e-3, (a, b, P.deltaE2000(np.array(a), np.array(b)), e)
print("Lab + CIEDE2000 ok")


def cdf_img(x, buffer=0.25):  # verbatim logic of his colab
    u = np.sort(x.ravel())
    nn = np.sum(u < 0); npos = u.shape[0] - nn
    v = np.concatenate((np.linspace(-1, -buffer, nn), np.linspace(buffer, 1, npos)))
    return -np.interp(x, u, v)


M = np.load('/home/fzeng/ml/research/art/trainability-fractal/cache/zoom_zoomA/kf_000.npz')['measure']
ref = mpl.colormaps['Spectral']((cdf_img(M) + 1) / 2)[..., :3]
mine = P.render_split(M, 'sd_spectral', near_boundary='large')
err = np.abs(ref - mine).max() * 255
print("sd_spectral reproduction max abs error (0-255):", round(err, 2))
assert err < 6  # residual = LUT quantisation + average-rank tie handling
x = np.linspace(-1, 1, 11)
v = P.signed_rank_normalize(x, near_boundary='small')
assert np.all(v[x < 0] <= 0) and np.all(v[x > 0] >= 0) and v[5] == 0 and v[0] == -0.75 and v[-1] == 0.75
print("all ok")
