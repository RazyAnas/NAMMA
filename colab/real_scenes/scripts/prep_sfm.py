# Stage 1 for a real video (same as prep_whole.py's SfM step): COLMAP global SfM on every frame (train + held-out),
# fixed pinhole intrinsics, then Sim3-aligned onto reference poses (ARCore for the phone, the earlier metric-scaled
# COLMAP model for the lobby) so the scene is in metres, as the ICL run was aligned to its ground truth.
# Writes <work>/clean/{stats.json, poses_w2c.json, sfm_obs.npz} and <work>/data/color_raw/<frame>.jpg.
# Usage: python prep_sfm.py <work> <images dir> <cameras.json>
import sys, os, re, json, shutil, time, numpy as np, pycolmap
W, IMG, CJ = sys.argv[1:4]; t0 = time.time()
cam = json.load(open(CJ)); K = np.array(cam['K']); Wd, H = cam['width'], cam['height']
num = lambda s: int(re.sub(r'\D', '', s))
ref = {num(e['frame']): np.array(e['w2c']) for e in cam['train'] + cam['test']}
train = sorted(num(e['frame']) for e in cam['train']); test = sorted(num(e['frame']) for e in cam['test'])
src_name = {num(e['frame']): os.path.basename(e['image']) for e in cam['train'] + cam['test']}
cr = f'{W}/data/color_raw'; sd = f'{W}/clean/sfm_images'
for d in (cr, sd): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
for f in train + test:
    shutil.copy(f'{IMG}/{src_name[f]}', f'{cr}/{f:05d}.jpg'); os.symlink(f'{cr}/{f:05d}.jpg', f'{sd}/{f:05d}.jpg')
db = f'{W}/clean/db.db'
if os.path.exists(db): os.remove(db)
opts = pycolmap.FeatureExtractionOptions(); opts.sift.max_num_features = 4096
reader = pycolmap.ImageReaderOptions(); reader.camera_model = 'PINHOLE'
reader.camera_params = f'{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}'
pycolmap.extract_features(db, sd, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=opts)
print('features', time.time() - t0, flush=True)
pycolmap.match_exhaustive(db); print('matched', time.time() - t0, flush=True)
so = f'{W}/clean/sfm'; shutil.rmtree(so, ignore_errors=True); os.makedirs(so)
recs = pycolmap.global_mapping(db, sd, so)
rec = max(recs.values(), key=lambda r: r.num_reg_images())
reg = {int(im.name[:5]): im for im in rec.images.values()}
c2 = lambda w2c: -w2c[:3, :3].T @ w2c[:3, 3]
fr = sorted(reg); src = np.array([reg[f].projection_center() for f in fr]); dst = np.array([c2(ref[f]) for f in fr])
mu_s, mu_d = src.mean(0), dst.mean(0); A = (dst - mu_d).T @ (src - mu_s) / len(src)
U, Sg, Vt = np.linalg.svd(A); Dg = np.eye(3); Dg[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = U @ Dg @ Vt; s = (Sg * np.diag(Dg)).sum() / ((src - mu_s) ** 2).sum(1).mean(); t = mu_d - s * R @ mu_s
err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
est = {}
for f, im in reg.items():
    w2c = im.cam_from_world().matrix(); Rc, tc = w2c[:, :3], w2c[:, 3]; Rn = Rc @ R.T; tn = s * tc - Rn @ t
    est[f] = np.vstack([np.hstack([Rn, tn[:, None]]), [0, 0, 0, 1]])
rot = [np.degrees(np.arccos(np.clip((np.trace(est[f][:3, :3] @ ref[f][:3, :3].T) - 1) / 2, -1, 1))) for f in fr]
pid2i = {pid: i for i, pid in enumerate(rec.points3D.keys())}
X3 = np.array([s * R @ p.xyz + t for p in rec.points3D.values()]); C3 = np.array([p.color for p in rec.points3D.values()])
obs = {}
for f, im in reg.items():
    uv, ii = [], []
    for p2 in im.points2D:
        if p2.has_point3D(): uv.append(p2.xy); ii.append(pid2i[p2.point3D_id])
    obs[f'uv_{f}'] = np.array(uv, np.float32).reshape(-1, 2); obs[f'id_{f}'] = np.array(ii, np.int32)
np.savez_compressed(f'{W}/clean/sfm_obs.npz', X3=X3.astype(np.float32), C3=C3.astype(np.uint8), **obs)
json.dump({str(f): est[f].tolist() for f in sorted(est)}, open(f'{W}/clean/poses_w2c.json', 'w'))
json.dump({str(f): ref[f].tolist() for f in sorted(ref)}, open(f'{W}/clean/poses_ref_w2c.json', 'w'))
stats = dict(K=K.tolist(), width=Wd, height=H, registered=len(reg), of=len(train + test),
             missing=[f for f in train + test if f not in reg], sfm_points=rec.num_points3D(), sim3_scale=float(s),
             vs_reference=dict(centre_rmse_cm=float(100 * np.sqrt((err ** 2).mean())), centre_median_cm=float(100 * np.median(err)),
                               rot_median_deg=float(np.median(rot)), rot_p90_deg=float(np.percentile(rot, 90))),
             train_frames=[f for f in train if f in reg], test_frames=[f for f in test if f in reg], seconds=time.time() - t0)
json.dump(stats, open(f'{W}/clean/stats.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in stats.items() if 'frames' not in k}))
