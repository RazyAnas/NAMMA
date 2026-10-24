# Second opinion on our camera poses: VGGT-1B solves short windows of consecutive frames on its own, each window is
# aligned to our poses with a similarity transform (Umeyama on camera centres), and the leftover per-frame error says
# where our poses and VGGT disagree. A stretch where every window disagrees (e.g. the lobby's ~20 dB frames 879-5389)
# points at bad poses there; agreement points at blur / exposure / coverage instead.
# Usage: python vggt_pose_check.py <frames dir> <poses_w2c.json> <out json> [window=24] [step=16]
import sys, os, json, numpy as np, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
FR, POSES, OUT = sys.argv[1:4]; WIN = int(sys.argv[4]) if len(sys.argv) > 4 else 24; STEP = int(sys.argv[5]) if len(sys.argv) > 5 else 16
P = json.load(open(POSES)); frames = sorted(int(k) for k in P if os.path.exists(f'{FR}/{int(k):05d}.jpg'))
def c2w(m): m = np.asarray(m, dtype=np.float64); m = m if m.shape[0] == 4 else np.vstack([m, [0, 0, 0, 1.]]); return np.linalg.inv(m)
ours = {f: c2w(P[str(f)]) for f in frames}
dev = 'cuda'; dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = VGGT.from_pretrained('facebook/VGGT-1B').to(dev).eval()

def umeyama(src, dst):   # s, R, t with dst ~ s R src + t
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ D @ Vt; s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean(); return s, Rm, md - s * Rm @ ms

windows, per_frame = [], {}
for i in range(0, max(1, len(frames) - WIN + 1), STEP):
    fs = frames[i:i + WIN]
    imgs = load_and_preprocess_images([f'{FR}/{f:05d}.jpg' for f in fs]).to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype): pred = model(imgs[None] if imgs.dim() == 4 else imgs)
    ext, _ = pose_encoding_to_extri_intri(pred['pose_enc'], imgs.shape[-2:])
    theirs = [c2w(e) for e in ext[0].float().cpu().numpy()]
    src = np.stack([m[:3, 3] for m in theirs]); dst = np.stack([ours[f][:3, 3] for f in fs])
    s, Rm, t = umeyama(src, dst)
    pos = np.linalg.norm((s * (Rm @ src.T)).T + t - dst, axis=1)
    rot = [np.degrees(np.arccos(np.clip((np.trace((Rm @ th[:3, :3]).T @ ou[:3, :3]) - 1) / 2, -1, 1))) for th, ou in zip(theirs, (ours[f] for f in fs))]
    path = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())
    windows.append(dict(frames=[fs[0], fs[-1]], path_m=round(path, 3), pos_median_cm=round(float(np.median(pos)) * 100, 2),
                        pos_max_cm=round(float(pos.max()) * 100, 2), rot_median_deg=round(float(np.median(rot)), 2), rot_max_deg=round(float(np.max(rot)), 2)))
    for f, p, r in zip(fs, pos, rot): per_frame.setdefault(f, []).append((float(p), float(r)))
    print(json.dumps(windows[-1]), flush=True)
pf = {f: dict(pos_cm=round(100 * float(np.median([x[0] for x in v])), 2), rot_deg=round(float(np.median([x[1] for x in v])), 2)) for f, v in sorted(per_frame.items())}
json.dump(dict(window=WIN, step=STEP, windows=windows, per_frame=pf), open(OUT, 'w'), indent=1)
worst = sorted(windows, key=lambda w: -w['pos_median_cm'] / max(w['path_m'], 1e-3))[:5]
print('worst windows (error relative to path length):'); [print(' ', w) for w in worst]
