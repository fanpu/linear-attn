"""Continuous bit depth: the same 30 s log chirp on a uniform grid whose step shrinks smoothly, peak/Delta from 1 to 512
("effective bits" log2(2 A/Delta + 1) from 1.6 to 10), undithered. 150 frames. STFT 2048 Blackman-Harris, hop 1024.
Saves cache/bitsweep.npz (frames x bins x times, float16) and SINAD per frame."""
import numpy as np
import dsp

x, _ = dsp.log_chirp(30.0, 20.0, 24000.0)
F = 150
ratio = 2.0 ** np.linspace(0, 9, F)          # peak amplitude in LSB
frames, sinad = [], []
for r in ratio:
    y = np.round(x * r) / r
    S, _, _ = dsp.stft_db(y, nfft=2048, hop=1024)
    frames.append(S.T[::-1].astype(np.float16))
    sinad.append(10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2)))
np.savez("cache/bitsweep.npz", frames=np.array(frames), ratio=ratio, sinad=np.array(sinad))
print("done", np.array(frames).shape)
