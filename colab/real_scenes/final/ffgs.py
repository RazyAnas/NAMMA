"""Feed-forward Gaussians (no per-scene training): Depth Anything 3's Gaussian head on the REAL frames, conditioned on our SfM
cameras so every chunk lands in one world frame. Merged, de-duplicated per voxel (keep the most opaque), written as one splat .ply.
This is the 'generator' route (like Marble / HY-World: a network outputs the splats) but driven by real views, so it stays true.
The DA3 Gaussian API is discovered at run time (signature + prediction fields are printed), because it was never run here before."""
import glob, inspect, os, time, numpy as np

def load_model(DepthAnything3):
    last = None
    for name in ('depth-anything/DA3NESTED-GIANT-LARGE', 'depth-anything/DA3-GIANT', 'depth-anything/DA3-LARGE'):
        try:
            m = DepthAnything3.from_pretrained(name).to('cuda').eval(); print('model', name, flush=True); yield name, m
        except Exception as e: print('model', name, 'not usable:', repr(e)[:200]); last = e

def read_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)['vertex'].data; return v

def run(cache, out_ply, every=6, chunk=8, voxel=0.015, cap=3_000_000):
    import torch
    from depth_anything_3.api import DepthAnything3
    d = np.load(f'{cache}/da3_sfm.npz'); files = sorted(glob.glob(f'{cache}/frames/*.jpg'))
    sel = np.arange(0, len(d['idx']), every); fid = d['idx'][sel]
    Hd, Wd = d['depth'].shape[1:]
    from PIL import Image
    w0, h0 = Image.open(files[fid[0]]).size
    K = d['K'][sel].copy(); K[:, 0] *= w0 / Wd; K[:, 1] *= h0 / Hd          # intrinsics back at frame resolution
    w2c = np.linalg.inv(d['c2w'][sel]).astype(np.float32)
    print(f'{len(sel)} views (every {every}th SfM frame), frame {w0}x{h0}', flush=True)
    for name, model in load_model(DepthAnything3):
        sig = inspect.signature(model.inference); print('inference', sig, flush=True)
        kw = {}
        if 'infer_gs' in sig.parameters: kw['infer_gs'] = True
        plys = []; t0 = time.time(); ok = True
        for a in range(0, len(sel), chunk):
            b = min(a + chunk, len(sel)); ims = [files[g] for g in fid[a:b]]; ex = f'/kaggle/temp/ffgs/{a:04d}'; os.makedirs(ex, exist_ok=True)
            tries = [dict(extrinsics=w2c[a:b], intrinsics=K[a:b].astype(np.float32)), dict(extrinsics=w2c[a:b, :3], intrinsics=K[a:b].astype(np.float32)), {}]
            pred = None
            for extra in tries:
                if extra and 'extrinsics' not in sig.parameters: continue
                for fmt in ('gs_ply', None):
                    try:
                        args = dict(kw, **extra)
                        if fmt and 'export_format' in sig.parameters: args.update(export_dir=ex, export_format=fmt)
                        with torch.no_grad(): pred = model.inference(ims, **args)
                        if a == 0: print('call ok with', sorted(k for k in args if k not in ('extrinsics', 'intrinsics')), '| posed' if extra else '| UNPOSED (chunks will not line up)', flush=True)
                        break
                    except Exception as e:
                        if a == 0: print('try', sorted(extra), fmt, '->', repr(e)[:300], flush=True)
                        torch.cuda.empty_cache()
                if pred is not None: break
            if pred is None: ok = False; break
            if a == 0:
                fields = {k: (tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__) for k, v in vars(pred).items()}
                print('prediction fields', fields, flush=True)
                g = getattr(pred, 'gaussians', None)
                if g is not None: print('gaussians', {k: (tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__) for k, v in (vars(g).items() if hasattr(g, '__dict__') else g.items())}, flush=True)
                print('files written', [os.path.relpath(p, ex) for p in glob.glob(f'{ex}/**/*', recursive=True)][:20], flush=True)
            plys += glob.glob(f'{ex}/**/*.ply', recursive=True)
            torch.cuda.empty_cache()
        print(f'{name}: {len(plys)} ply files in {time.time()-t0:.0f}s', flush=True)
        if ok and plys: return merge(plys, out_ply, voxel, cap, d['c2w'][sel][:, :3, 3])
        del model; torch.cuda.empty_cache()
    raise RuntimeError('no DA3 checkpoint produced Gaussians (see the tries above)')

def merge(plys, out_ply, voxel, cap, cams):
    from plyfile import PlyData, PlyElement
    parts = [read_ply(p) for p in plys]; names = parts[0].dtype.names; print('ply properties', names, flush=True)
    v = np.concatenate([p[list(names)] for p in parts if p.dtype.names == names])
    xyz = np.stack([v['x'], v['y'], v['z']], 1); n0 = len(v)
    op = v['opacity'] if 'opacity' in names else np.zeros(len(v))
    key = np.floor(xyz / voxel).astype(np.int64); key = key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791
    o = np.lexsort((-op, key)); keep = o[np.r_[True, key[o][1:] != key[o][:-1]]]      # most opaque splat per voxel
    if len(keep) > cap: keep = keep[np.argsort(-op[keep])[:cap]]
    v = v[keep]; xyz = xyz[keep]
    dist = np.median(np.linalg.norm(xyz[:, None, :] - cams[None, ::max(1, len(cams) // 20)], axis=2).min(1))
    print(f'merged {n0} -> {len(v)} splats (voxel {voxel*100:.1f} cm) | median distance splat->nearest camera {dist:.2f} m '
          f'(~1-3 m means the chunks share the camera frame; huge = unposed/unscaled)', flush=True)
    PlyData([PlyElement.describe(v, 'vertex')]).write(out_ply); return len(v)
