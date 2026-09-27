"""v2 deliverables (CPU): export the dense-solid stills, composite the split boil film, encode the
films with ffmpeg (H.264, 1920x1080, yuv420p), and build contact_sheet_dense.jpg.

    python gallery2.py [--boil_frames 96]
"""
import argparse, os, subprocess
from PIL import Image, ImageDraw, ImageFont
from dl_common import CACHE, GALLERY

p = argparse.ArgumentParser()
p.add_argument("--boil_frames", type=int, default=96)
p.add_argument("--crf", type=int, default=24)
p.add_argument("--maxrate", default="4M", help="capped-CRF: keeps each film ~6 MB (gallery < 60 MB)")
p.add_argument("--skip_films", action="store_true")
a = p.parse_args()
R2 = os.path.join(CACHE, "renders2"); FD = os.path.join(CACHE, "film")
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except OSError:
    font = small = ImageFont.load_default()

STILLS = {"G1_fur256_acc": 5120, "G2_fur256_low": 3840, "G3_fur512_acc": 3840, "G4_fur256_fid": 3840}
ims = {}
for n, w in STILLS.items():
    f = os.path.join(R2, n + ".png")
    if not os.path.exists(f):
        continue
    im = Image.open(f).convert("RGB"); ims[n] = im
    out = im if im.width == w else im.resize((w, im.height * w // im.width), Image.LANCZOS)
    out.save(os.path.join(GALLERY, n + ".jpg"), quality=88, optimize=True)

# density study: the v1 hairline render and the v2 solid render, same fraction of the frame
v1 = os.path.join(CACHE, "renders", "F4_noise_acc_top.png")
if os.path.exists(v1) and "G1_fur256_acc" in ims:
    A = Image.open(v1).convert("RGB"); B = ims["G1_fur256_acc"]
    def mid(im, fw=0.30):
        w, h = im.size; cw, ch = int(w * fw), int(w * fw * 0.625)
        return im.crop((w // 2 - cw // 2, h // 2 - ch // 2, w // 2 + cw // 2, h // 2 + ch // 2)).resize((1400, 875), Image.LANCZOS)
    D = Image.new("RGB", (2820, 875), (0, 0, 0)); D.paste(mid(A), (0, 0)); D.paste(mid(B), (1420, 0))
    d = ImageDraw.Draw(D)
    d.text((24, 20), "v1: every 1024² point, bilinear, one light (needles 1 cell ≈ 5 px wide)", fill=(235, 235, 235), font=font)
    d.text((1444, 20), "v2: every 4th point (256²), bicubic, shadows + AO + fill", fill=(235, 235, 235), font=font)
    D.save(os.path.join(GALLERY, "study_density_v1_v2.jpg"), quality=88, optimize=True)


def encode(pattern, fps_in, out, extra_vf=""):
    vf = "scale=1920:1080:flags=lanczos" + extra_vf
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps_in), "-i", pattern, "-vf", vf, "-r", "24",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
           "-maxrate", a.maxrate, "-bufsize", "8M", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print(out, round(os.path.getsize(out) / 1e6, 2), "MB")


if not a.skip_films:
    # boil: each measurement held for 3 video frames (8 measurements per second)
    bd = os.path.join(FD, "boil")
    if os.path.exists(os.path.join(bd, f"f{a.boil_frames - 1:04d}.png")):
        tmp = os.path.join(FD, "boil_seq"); os.makedirs(tmp, exist_ok=True)
        for k in range(a.boil_frames):
            dst = os.path.join(tmp, f"f{k:04d}.png")
            if not os.path.exists(dst):
                os.symlink(os.path.join(bd, f"f{k:04d}.png"), dst)
        encode(os.path.join(tmp, "f%04d.png"), 8, os.path.join(GALLERY, "film_boil.mp4"))
        # split: left half = one fixed minibatch (C, static), right half = the boiling fresh measurement
        sd = os.path.join(FD, "boil_split"); os.makedirs(sd, exist_ok=True)
        C = Image.open(os.path.join(bd, "C.png")).convert("RGB")
        W, H = C.size
        for k in range(a.boil_frames):
            dst = os.path.join(sd, f"f{k:04d}.png")
            if os.path.exists(dst):
                continue
            Bk = Image.open(os.path.join(bd, f"f{k:04d}.png")).convert("RGB")
            S = Bk.copy(); S.paste(C.crop((0, 0, W // 2, H)), (0, 0))
            d = ImageDraw.Draw(S)
            d.rectangle((W // 2 - 1, 0, W // 2 + 1, H), fill=(0, 0, 0))
            d.text((28, H - 48), "one fixed minibatch of 16 (the same draw every frame)", fill=(235, 235, 235), font=small)
            d.text((W // 2 + 28, H - 48), "a fresh minibatch of 16 at every point, redrawn every frame", fill=(235, 235, 235), font=small)
            S.save(dst, compress_level=1)
        encode(os.path.join(sd, "f%04d.png"), 8, os.path.join(GALLERY, "film_boil_split.mp4"))
    # settle: video frame f shows the mean of the first m(f) measurements, m geometric in f, then held
    st = os.path.join(FD, "settle")
    if os.path.exists(os.path.join(st, f"f{a.boil_frames - 1:04d}.png")):
        seq = os.path.join(FD, "settle_seq"); os.makedirs(seq, exist_ok=True)
        n_rise, n_hold = 192, 48
        for f in range(n_rise + n_hold):
            m = min(a.boil_frames, max(1, round(a.boil_frames ** (min(f, n_rise - 1) / (n_rise - 1)))))
            dst = os.path.join(seq, f"f{f:04d}.png")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.join(st, f"f{m - 1:04d}.png"), dst)
        encode(os.path.join(seq, "f%04d.png"), 24, os.path.join(GALLERY, "film_settle.mp4"))
    for name in ("sweep", "flyover"):
        d = os.path.join(FD, name)
        if os.path.exists(os.path.join(d, "f0000.png")):
            encode(os.path.join(d, "f%04d.png"), 24, os.path.join(GALLERY, f"film_{name}.mp4"))

# averaging study: 1 fresh measurement, mean of 16, mean of 96 (same camera, light, colour)
av = [os.path.join(R2, f"avg_{m}.png") for m in (1, 16, 96)]
if all(os.path.exists(f) for f in av):
    tw = 1280; th = 720
    D = Image.new("RGB", (3 * tw + 32, th), (0, 0, 0))
    for i, (f, t) in enumerate(zip(av, ("one fresh measurement (bs 16 per point)", "mean of 16 measurements", "mean of 96 measurements"))):
        D.paste(Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS), (i * (tw + 16), 0))
        ImageDraw.Draw(D).text((i * (tw + 16) + 24, 20), t, fill=(235, 235, 235), font=font)
    D.save(os.path.join(GALLERY, "study_averaging.jpg"), quality=88, optimize=True)

# contact sheet of the dense-solid work
order = [("G1_fur256_acc", "fresh bs16 per point, 256² subgrid, colour = accuracy"),
         ("G2_fur256_low", "same, low close camera"),
         ("G3_fur512_acc", "512² subgrid, K = 0.25"),
         ("G4_fur256_fid", "256², colour = function identity")]
order = [(n, t) for n, t in order if n in ims]
if order:
    tw, th = 1000, 563; cols = 2; rows = (len(order) + 1) // 2
    S = Image.new("RGB", (cols * tw + (cols + 1) * 16, rows * (th + 40) + 16), (8, 8, 9))
    for i, (n, t) in enumerate(order):
        x = 16 + (i % cols) * (tw + 16); y = 16 + (i // cols) * (th + 40)
        S.paste(ims[n].resize((tw, th), Image.LANCZOS), (x, y))
        ImageDraw.Draw(S).text((x, y + th + 8), f"{n}: {t}", fill=(200, 200, 200), font=small)
    S.save(os.path.join(GALLERY, "contact_sheet_dense.jpg"), quality=90, optimize=True)
tot = sum(os.path.getsize(os.path.join(GALLERY, f)) for f in os.listdir(GALLERY))
print("gallery MB", round(tot / 1e6, 1))
