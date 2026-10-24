# MODP-style Gaussian initialisation (Wu et al., SIGGRAPH 2025, Sec. 3.2) from our anchored monocular depth.
#  - candidates per frame: tracker-like feature pixels (corners + strong edges) for complex regions, a coarse grid for
#    textureless regions, and optional error-compensation pixels (low SSIM between a render and the frame)
#  - each Gaussian gets the pixel-footprint scale s = k * d / f (Eq. 1, with k the pixel spacing it stands for)
#  - a Multi-level Occupancy Hash Voxel (MOHV) rejects Gaussians whose voxel at their own level is already occupied;
#    accepting one marks its voxel at that level and every coarser level (Sec. 3.2.3)
# Writes a full 3DGS PLY that OpenSplat can --resume from, or appends to an existing trained PLY.
# Usage: python modp_init.py <work> <out.ply> [--ec <errmask dir>] [--base <trained.ply> --base-step N] [--stats out.json]
import sys, os, json, argparse, numpy as np, cv2
ap = argparse.ArgumentParser()
ap.add_argument('work'); ap.add_argument('out')
ap.add_argument('--ec', help='dir of <frame>.png masks (255 = high error) for error compensation')
ap.add_argument('--ec-only', action='store_true', help='only add error-compensation points (use with --base)')
ap.add_argument('--base', help='trained PLY to append to (its Gaussians pre-occupy the MOHV)')
ap.add_argument('--base-step', type=int, default=0)
ap.add_argument('--k-feat', type=float, default=2.0, help='pixel spacing represented by a feature-region Gaussian')
ap.add_argument('--k-grid', type=float, default=8.0, help='pixel spacing represented by a textureless-region Gaussian')
ap.add_argument('--k-ec', type=float, default=1.5)
ap.add_argument('--s-init', type=float, default=1.28, help='coarsest MOHV voxel (m)')
ap.add_argument('--levels', type=int, default=9)
ap.add_argument('--stats')
a = ap.parse_args()
W = a.work; st = json.load(open(f'{W}/clean/stats.json')); K = np.array(st['K']); f = K[0, 0]; H, Wd = st['height'], st['width']
P = json.load(open(f'{W}/clean/poses_w2c.json'))
train = st['train_frames']

occ = [set() for _ in range(a.levels)]
def keys(X, lvl):
    v = a.s_init / 2.0 ** lvl
    q = np.floor(X / v[:, None]).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

def level_of(spacing):  # finest level whose voxel is still >= the spacing this Gaussian stands for
    spacing = np.nan_to_num(spacing, nan=a.s_init, posinf=a.s_init)  # trained PLYs can hold a few non-finite scales
    return np.clip(np.floor(np.log2(a.s_init / np.maximum(spacing, 1e-6))), 0, a.levels - 1).astype(int)

def mohv_insert(X, spacing):
    """Return mask of accepted points (in order), updating the occupancy."""
    lvl = level_of(spacing); acc = np.zeros(len(X), bool)
    for l in np.unique(lvl):
        idx = np.where(lvl == l)[0]; k = keys(X[idx], np.full(len(idx), l, float))
        _, first = np.unique(k, return_index=True)             # one per voxel within this batch
        first = np.sort(first); kk = k[first]
        new = np.array([x not in occ[l] for x in kk.tolist()], bool)
        sel = idx[first[new]]; acc[sel] = True
        for m in range(l + 1):                                   # mark own level and every coarser level
            occ[m].update(keys(X[sel], np.full(len(sel), m, float)).tolist())
    return acc

def backproject(us, vs, d, w2c):
    Xc = np.stack([(us - K[0, 2]) / K[0, 0] * d, (vs - K[1, 2]) / K[1, 1] * d, d], 1)
    R, t = w2c[:3, :3], w2c[:3, 3]; return (Xc - t) @ R

def read_ply(path):
    with open(path, 'rb') as fh:
        hdr = []
        while True:
            l = fh.readline().decode().strip(); hdr.append(l)
            if l == 'end_header': break
        n = int([h for h in hdr if h.startswith('element vertex')][0].split()[-1])
        props = [h.split()[-1] for h in hdr if h.startswith('property')]
        a_ = np.frombuffer(fh.read(), np.float32).reshape(n, len(props))
    return props, a_.copy()

PROPS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(45)] + \
        ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
rows = []; counts = dict(feature=0, grid=0, ec=0, base=0); growth = []
if a.base:
    props, base = read_ply(a.base); assert props == PROPS, props
    sc = np.exp(base[:, PROPS.index('scale_0'):PROPS.index('scale_0') + 3]).max(1)
    mohv_insert(base[:, :3].astype(np.float64), 2 * sc); counts['base'] = len(base)

def add(X, rgb, spacing, kind):
    m = mohv_insert(X, spacing); n = int(m.sum()); counts[kind] += n
    if not n: return
    r = np.zeros((n, len(PROPS)), np.float32); r[:, :3] = X[m]
    r[:, 6:9] = (rgb[m] / 255. - 0.5) / 0.28209479177387814
    r[:, PROPS.index('opacity')] = np.log(0.1 / 0.9)
    r[:, PROPS.index('scale_0'):PROPS.index('scale_0') + 3] = np.log(np.maximum(0.5 * spacing[m], 2e-4))[:, None]
    r[:, PROPS.index('rot_0')] = 1.0; rows.append(r)

for i, fr in enumerate(train):
    D = cv2.imread(f'{W}/clean/boot_depth/{fr:05d}.png', -1).astype(np.float64) / 1000.
    img = cv2.imread(f'{W}/data/color_raw/{fr:05d}.jpg'); rgbim = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    w2c = np.array(P[str(fr)]); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def take(us, vs, k, kind):
        us, vs = np.asarray(us, int), np.asarray(vs, int); d = D[vs, us]; ok = (d > 0.1) & (d < 15)
        us, vs, d = us[ok], vs[ok], d[ok]
        add(backproject(us, vs, d, w2c), rgbim[vs, us], k * d / f, kind)
    if not a.ec_only:
        # complex regions: corners (what a feature tracker keeps) plus strong image edges
        c = cv2.goodFeaturesToTrack(g, 4000, 0.005, 2)
        if c is not None: take(c[:, 0, 0], c[:, 0, 1], a.k_feat, 'feature')
        mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        vs, us = np.nonzero(mag[::2, ::2] > np.percentile(mag, 92)); take(us * 2, vs * 2, a.k_feat, 'feature')
        # textureless regions: coarse grid, each Gaussian standing for a k_grid-pixel patch
        vv, uu = np.mgrid[4:H:8, 4:Wd:8]; take(uu.ravel(), vv.ravel(), a.k_grid, 'grid')
    if a.ec and os.path.exists(f'{a.ec}/{fr:05d}.png'):
        m = cv2.imread(f'{a.ec}/{fr:05d}.png', 0); vs, us = np.nonzero(m > 127)
        take(us, vs, a.k_ec, 'ec')
    growth.append([i + 1, int(sum(len(r) for r in rows))])

new = np.concatenate(rows) if rows else np.zeros((0, len(PROPS)), np.float32)
allg = np.concatenate([base, new]) if a.base else new
with open(a.out, 'wb') as fh:
    fh.write((f'ply\nformat binary_little_endian 1.0\ncomment Generated by opensplat at iteration {a.base_step}\n'
              f'element vertex {len(allg)}\n' + ''.join(f'property float {p}\n' for p in PROPS) + 'end_header\n').encode())
    fh.write(allg.astype(np.float32).tobytes())
res = dict(counts=counts, total=int(len(allg)), new=int(len(new)), growth=growth, args=vars(a))
print(json.dumps({k: v for k, v in res.items() if k != 'growth'}))
if a.stats: json.dump(res, open(a.stats, 'w'))
