"""Dense depth for EVERY verified frame (namma-prep ran Depth Anything 3 on every 3rd frame only), put on the SfM cameras.
Scale per new frame: warp the already-aligned depth of the nearest frames that have it into the new frame through the SfM cameras,
and take the median ratio against the new DA3 depth (the same idea as sfm_align, with aligned depth instead of SfM points).
Writes <out>/da3_sfm.npz (same layout as the cache, kept frames only) and links the frames. GPU (DA3), ~1 s per frame on a T4.
Usage: python densify.py <cache> <keep.json> <out>"""
import gc, glob, json, os, subprocess, sys, numpy as np

def warp_depth(depth_r, K, c2w_r, w2c_t):
    """Depth of reference frame r as seen from frame t (z-buffered, nearest pixel); 0 = not seen."""
    H, W = depth_r.shape; v, u = np.mgrid[0:H, 0:W]; z = depth_r.ravel(); ok = z > 0
    P = np.stack([(u.ravel() - K[0, 2]) / K[0, 0] * z, (v.ravel() - K[1, 2]) / K[1, 1] * z, z], 1)[ok]
    Q = (P @ c2w_r[:3, :3].T + c2w_r[:3, 3]) @ w2c_t[:3, :3].T + w2c_t[:3, 3]; Q = Q[Q[:, 2] > 0.05]
    ut = np.round(Q[:, 0] / Q[:, 2] * K[0, 0] + K[0, 2]).astype(int); vt = np.round(Q[:, 1] / Q[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    m = (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H); out = np.full(H * W, np.inf, np.float32)
    np.minimum.at(out, vt[m] * W + ut[m], Q[m, 2].astype(np.float32)); out[np.isinf(out)] = 0
    return out.reshape(H, W)

def scale_for(newz, d, j, refs):
    """Median (aligned reference depth / new DA3 depth) over pixels both see; also returns how many pixels agreed."""
    w2c = np.linalg.inv(d['c2w'][j]); r = []
    for k in refs:
        wz = warp_depth(d['depth'][k], d['K'][k], d['c2w'][k], w2c); m = (wz > 0) & (newz > 0)
        r.append(wz[m] / newz[m])
    r = np.concatenate(r) if r else np.zeros(0)
    if len(r) < 500: return None, len(r)
    s = np.median(r); spread = np.median(np.abs(np.log(r / s)))
    return (s if spread < 0.1 else None), len(r)                                   # disagreement > ~10%: do not trust this frame

def da3():
    sh = lambda c: subprocess.call(c, shell=True)
    sh('pip -q install --no-deps --ignore-requires-python "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" 2>&1 | tail -3')
    sh('pip -q install einops omegaconf huggingface_hub safetensors pillow imageio plyfile trimesh scipy addict moviepy 2>&1 | tail -2')
    for _ in range(25):                                                              # same self-heal as run.py stage 2
        try: from depth_anything_3.api import DepthAnything3; break
        except ModuleNotFoundError as e:
            pkg = {'cv2': 'opencv-python-headless', 'PIL': 'pillow', 'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}.get(e.name.split('.')[0], e.name.split('.')[0])
            sh(f'pip -q install {pkg} 2>&1 | tail -1')
    return DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE').to('cuda').eval()

def run(cache, keep, out, chunk=8):
    import torch, cv2
    z = np.load(f'{cache}/da3_sfm.npz'); d = {k: z[k] for k in z.files}; files = sorted(glob.glob(f'{cache}/frames/*.jpg'))
    ok = np.isin(d['idx'], json.load(open(keep))['keep']); d = {k: v[ok] for k, v in d.items()}
    has = d['depth'].reshape(len(d['idx']), -1).max(1) > 0; todo = np.where(~has)[0]; Hd, Wd = d['depth'].shape[1:]
    print(f'{len(d["idx"])} verified frames, {has.sum()} with depth, {len(todo)} to add', flush=True)
    model = da3(); added = 0; skipped = []
    for a in range(0, len(todo), chunk):
        js = todo[a:a + chunk]; ims = [files[int(d['idx'][j])] for j in js]
        with torch.no_grad(): pred = model.inference(ims)
        for j, nz in zip(js, np.asarray(pred.depth)):
            nz = cv2.resize(nz.astype(np.float32), (Wd, Hd), interpolation=cv2.INTER_NEAREST) if nz.shape != (Hd, Wd) else nz
            near = [k for k in np.argsort(np.abs(d['idx'] - d['idx'][j])) if has[k]][:3]   # nearest frames (by walk order) that have aligned depth
            s, n = scale_for(nz, d, j, near)
            if s is None: skipped.append((int(d['idx'][j]), n)); continue
            d['depth'][j] = nz * s; added += 1
        del pred; gc.collect(); torch.cuda.empty_cache(); print(f'{min(a + chunk, len(todo))}/{len(todo)} done, {added} added', flush=True)
    os.makedirs(out, exist_ok=True); np.savez(f'{out}/da3_sfm.npz', **d)
    if not os.path.exists(f'{out}/frames'): os.symlink(f'{cache}/frames', f'{out}/frames')
    print(f'dense depth: {added} frames added, {len(skipped)} skipped (no agreement with neighbours): {skipped[:30]}', flush=True)

if __name__ == '__main__':
    run(*sys.argv[1:4])
