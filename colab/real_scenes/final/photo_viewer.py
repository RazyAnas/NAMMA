"""Matterport-style viewer from REAL photos: every verified frame with dense depth becomes a photo-textured depth mesh; the browser
draws the nearest real frame first and lets the next ones fill only what it did not see (stencil), so on the walked path you
see the photo itself and nothing is invented. Holes stay visible (H key: magenta = never filmed from here).
Usage: python photo_viewer.py <cache> <keep.json> <out_dir>. CPU only. Open with: cd out_dir && python3 -m http.server"""
import json, os, shutil, sys, urllib.request, numpy as np

def build(cache, keep, out, stride=2):
    z = np.load(f'{cache}/da3_sfm.npz'); d = {k: z[k] for k in z.files}
    files = sorted(os.listdir(f'{cache}/frames')); ok = set(json.load(open(keep))['keep']) if keep else None
    has = d['depth'].reshape(len(d['idx']), -1).max(1) > 0
    sel = [j for j in range(len(d['idx'])) if has[j] and (ok is None or int(d['idx'][j]) in ok)]
    os.makedirs(f'{out}/f', exist_ok=True); Hd, Wd = d['depth'].shape[1:]; cams = []
    for j in sel:
        g = int(d['idx'][j]); shutil.copy(f'{cache}/frames/{files[g]}', f'{out}/f/{g:04d}.jpg')
        (np.clip(d['depth'][j][::stride, ::stride] * 1000, 0, 65535).astype('<u2')).tofile(f'{out}/f/{g:04d}.bin')   # millimetres
        cams.append(dict(g=g, K=d['K'][j][:2].ravel().round(3).tolist(), c2w=d['c2w'][j][:3].ravel().round(5).tolist()))
    x = d['c2w'][sel][:, :3, 0]; up = np.linalg.eigh(x.T @ x)[1][:, 0]                 # same as fuse_nav.camera_up (that module needs open3d)
    if up @ (-d['c2w'][sel][:, :3, 1].mean(0)) < 0: up = -up
    json.dump(dict(Hd=int(Hd), Wd=int(Wd), stride=stride, up=up.round(5).tolist(), cams=cams), open(f'{out}/cams.json', 'w'))
    if not os.path.exists(f'{out}/three.min.js'):
        urllib.request.urlretrieve('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js', f'{out}/three.min.js')
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photo_viewer.html'), f'{out}/index.html')
    print(f'photo viewer: {len(cams)} real frames with depth ({len(sel)} of {len(d["idx"])} cameras) -> {out}', flush=True)

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2] if sys.argv[2] != '-' else None, sys.argv[3])
