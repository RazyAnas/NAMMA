# Make the "degraded" copy of the ICL-NUIM frames: motion blur + lagging auto-exposure / white balance + 10 Hz flicker + noise.
# Blur averages consecutive 30 fps frames in linear light (continuous blur, no ghost copies at this frame spacing).
# Usage: python degrade.py <color_raw dir> <out dir> <frame> [<frame> ...]
import sys, os, numpy as np, cv2
src, out, frames = sys.argv[1], sys.argv[2], [int(f) for f in sys.argv[3:]]
os.makedirs(out, exist_ok=True)
lin = lambda x: (x / 255.) ** 2.2
srgb = lambda x: np.clip(x, 0, 1) ** (1 / 2.2) * 255.
n_all = len(os.listdir(src))
for f in frames:
    rng = np.random.default_rng(f)
    t = f / 30.
    k = int(round(1.5 + 1.0 * np.sin(2 * np.pi * t / 6.1)))        # blur half-width 0..2 frames (33-133 ms of camera motion)
    stack = [lin(cv2.imread(f'{src}/{g:05d}.jpg').astype(np.float32)) for g in range(max(0, f - k), min(n_all, f + k + 1))]
    im = np.mean(stack, 0)
    gain = np.exp(0.35 * np.sin(2 * np.pi * t / 5.3 + 0.7))         # auto-exposure drifting about +-0.5 stop
    wb = np.array([1 + 0.08 * np.sin(2 * np.pi * t / 7.1 + 1.3), 1.0, 1 + 0.08 * np.sin(2 * np.pi * t / 7.1)])  # BGR
    flick = 1 + 0.06 * np.sin(2 * np.pi * 10 * t + 0.4)             # mains-light flicker
    im = im * gain * flick * wb[None, None]
    im = srgb(im) + rng.normal(0, 2.5, im.shape)
    cv2.imwrite(f'{out}/{f:05d}.jpg', np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
