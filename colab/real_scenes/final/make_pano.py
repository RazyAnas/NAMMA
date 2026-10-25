"""Render 360 equirectangular panoramas from the REAL reconstruction (cloud.npz + walk.npz), holes left as holes. CPU, numpy + pillow.
Usage: python make_pano.py <model_dir> [--n 3] [--at X Z] [--w 4096]
Writes pano_<k>_raw.jpg (holes black), pano_<k>_mask.png (white = NOT filmed), pano_<k>_report.txt. Convention: x = u (floor frame),
y up, z = v; image column 0.5 looks along -z, rows: top = up. Checks coverage = share of the sphere that is real."""
import argparse, json, pathlib, sys, numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "experiments" / "nav_from_mesh"))
import nav_from_mesh as N

def load(model):
    c = np.load(model / "cloud.npz"); w = np.load(model / "walk.npz"); up = w["up"]; t, b = N.floor_frame(up)
    P = c["pts"]; xyz = np.stack([P @ t, P @ up - float(w["floor"]), P @ b], 1).astype(np.float32)
    return xyz, c["col"], w["walk"], w["grid"]

def render(xyz, col, eye, W=4096, near=0.15):
    H = W // 2; d = xyz - eye; r = np.linalg.norm(d, axis=1); ok = r > near; d, r, col = d[ok], r[ok], col[ok]
    lon = np.arctan2(d[:, 0], -d[:, 2]); lat = np.arcsin(np.clip(d[:, 1] / r, -1, 1))
    x = ((lon / (2 * np.pi) + 0.5) * W).astype(int) % W; y = np.clip(((0.5 - lat / np.pi) * H).astype(int), 0, H - 1)
    flat = y * W + x; order = np.argsort(-r); flat, r, col = flat[order], r[order], col[order]      # far first, near overwrites
    img = np.zeros((H * W, 3), np.uint8); dep = np.full(H * W, np.inf, np.float32); img[flat] = col; dep[flat] = r
    img, dep = img.reshape(H, W, 3), dep.reshape(H, W); hit = np.isfinite(dep)
    for _ in range(2):                                             # close pin-holes: fill an empty pixel from its nearest-depth hit neighbour
        pad = np.pad(dep, 1, constant_values=np.inf); padi = np.pad(img, ((1, 1), (1, 1), (0, 0))); best = dep.copy(); bi = img.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nd = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]; m = (nd < best) & ~hit
                best = np.where(m, nd, best); bi = np.where(m[..., None], padi[1 + dy:1 + dy + H, 1 + dx:1 + dx + W], bi)
        newly = np.isfinite(best) & ~hit; img = bi; dep = np.where(newly, best, dep); hit = np.isfinite(dep)
    return img, hit

def coverage_quick(xyz, eye):
    img, hit = render(xyz[::6], None if False else np.zeros((len(xyz[::6]), 3), np.uint8), eye, W=256); return hit.mean()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--n", type=int, default=3); ap.add_argument("--at", type=float, nargs=2)
    ap.add_argument("--w", type=int, default=4096); a = ap.parse_args(); model = pathlib.Path(a.model)
    xyz, col, walk, (u0, v0, cell) = load(model); idx = np.argwhere(walk); rng = np.random.default_rng(0)
    if a.at: eyes = [np.array([a.at[0], 1.5, a.at[1]], np.float32)]
    else:
        cand = idx[rng.choice(len(idx), min(60, len(idx)), replace=False)]
        pos = [np.array([u0 + (i + .5) * cell, 1.5, v0 + (j + .5) * cell], np.float32) for i, j in cand]
        cov = np.array([coverage_quick(xyz, e) for e in pos]); picked = []
        for k in np.argsort(-cov):                                  # best coverage first, but spread out (>= 1.5 m apart)
            if all(np.linalg.norm(pos[k] - p) > 1.5 for p in picked): picked.append(pos[k])
            if len(picked) == a.n: break
        eyes = picked
    for k, e in enumerate(eyes):
        img, hit = render(xyz, col, e, a.w); out = model / f"pano_{k}"
        Image.fromarray(img).save(f"{out}_raw.jpg", quality=95); Image.fromarray((~hit * 255).astype(np.uint8)).save(f"{out}_mask.png")
        txt = f"pano {k}: eye (x,y,z)={np.round(e, 2).tolist()} | REAL share of sphere {hit.mean()*100:.1f}% | not filmed {100-hit.mean()*100:.1f}%"
        open(f"{out}_report.txt", "w").write(txt + "\n"); print(txt)
