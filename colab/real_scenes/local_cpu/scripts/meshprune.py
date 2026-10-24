# Mesh-guided pruning: drop Gaussians whose centre is more than tau from the fused mesh surface (fog / floaters in free
# space), score the held-out stretch (never trained on) and the on-path views, with the CPU renderer.
import sys, json, numpy as np, cv2, open3d as o3d
sys.path.insert(0, '.')
from plyio import read; from render import gaussians, render
W = sys.argv[1]; hdr, names, a = read(f'{W}/b3k/splat.ply'); ix = {n: i for i, n in enumerate(names)}
fin = np.isfinite(a).all(1); a = a[fin]
m = o3d.io.read_triangle_mesh(f'{W}/mesh_boot/mesh.ply'); sc = o3d.t.geometry.RaycastingScene()
sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
xyz = a[:, [ix['x'], ix['y'], ix['z']]].astype(np.float32)
dist = sc.compute_distance(o3d.core.Tensor(xyz)).numpy()
st = json.load(open(f'{W}/clean/stats.json')); P = json.load(open(f'{W}/clean/poses_w2c.json')); K, Wd, H = st['K'], st['width'], st['height']
test = st['test_frames']; onpath = st['train_frames'][::12]
gt = lambda f: cv2.cvtColor(cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
psnr = lambda x, y: 10 * np.log10(1 / ((x - y) ** 2).mean())
print('gaussians', len(a), 'dist pct 50/90/99', np.percentile(dist, [50, 90, 99]).round(3).tolist(), flush=True)
res = {}
for tau in [None, 0.30, 0.15, 0.08, 0.04]:
    keep = np.ones(len(a), bool) if tau is None else dist < tau; g = gaussians(names, a[keep])
    off = [psnr(render(g, P[str(f)], K, Wd, H), gt(f)) for f in test]
    on = [psnr(render(g, P[str(f)], K, Wd, H), gt(f)) for f in onpath]
    res[str(tau)] = dict(kept=int(keep.sum()), off=off, on=on)
    print(tau, int(keep.sum()), 'off', round(np.mean(off), 3), 'on(train views)', round(np.mean(on), 3), flush=True)
json.dump(res, open(f'{W}/meshprune.json', 'w'))
