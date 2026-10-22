#!/usr/bin/env python3
"""Phone video -> mesh + surface-held Gaussian splat, CPU only. Every stage is resumable (skipped if its output exists).

Usage: python run.py <video> <work_dir> [options]   (see --help)

Stages (each writes <work>/<stage>/... and a done-marker):
  frames   sharpest frame per window (~5 fps), native resolution; fixed held-out test frames (off-path block + near-path every Nth)
  sfm      COLMAP (pycolmap) SIFT 8192, exhaustive matching, global mapper, one shared SIMPLE_RADIAL camera (focal solved)
  undist   undistort to PINHOLE with solved intrinsics, crop black borders; optional Sim3 to reference (AR) poses for metric scale
  depth    Depth Anything V2 (ONNX, aspect preserved) per frame, affine-in-disparity fit to SfM points, multi-view check
  mesh     TSDF fusion (open3d) of train depth -> mesh.ply (+ SDF grid for the splat trainer)
  splat    CPU 3DGS (OpenSplat CPU rasteriser via a torch extension) initialised from SfM + mesh samples, held to the mesh
           surface by a rendered-depth loss and periodic pruning of Gaussians far from the mesh; per-image exposure (affine colour)
  score    held-out views: mesh / splat / hybrid (mesh fills where splat alpha is low); PSNR/SSIM raw and colour-fitted on half
  strips   photo | splat | hybrid | mesh for 4 test views + same camera moved 0.3 m right, 0.3 m up, rotated 30 deg
"""
import argparse, json, os, sys, time, shutil, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def log(W, msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(f'{W}/log.txt', 'a') as f:
        f.write(line + '\n')


def done(W, stage):
    return os.path.exists(f'{W}/.done_{stage}')


def mark(W, stage, info):
    info = dict(info)
    json.dump(info, open(f'{W}/.done_{stage}', 'w'), indent=1)
    t = json.load(open(f'{W}/timings.json')) if os.path.exists(f'{W}/timings.json') else {}
    t[stage] = info.get('seconds')
    json.dump(t, open(f'{W}/timings.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------------------------------- frames
def stage_frames(a, W):
    import cv2
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    sharp = []
    while True:
        ok, im = cap.read()
        if not ok: break
        size = list(im.shape[1::-1]); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape; g = cv2.resize(g, (320, round(h * 320 / w)), interpolation=cv2.INTER_AREA)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var() / (g.var() + 1e-6)))   # exposure-invariant blur score
    n = len(sharp); win = max(1, round(fps / a.fps)) if a.window <= 0 else a.window
    picks = [max(range(s, min(n, s + win)), key=lambda i: sharp[i]) for s in range(0, n, win)]
    lo, hi = a.test_block if a.test_block else (n + 1, n + 1)
    test_off = list(range(lo, min(hi, n - 1) + 1, a.test_step)) if a.test_block else []
    ban = set(range(lo - a.gap, hi + a.gap + 1))
    rest = [p for p in picks if p not in ban]
    near = rest[a.near_every // 2::a.near_every] if a.near_every > 0 else []
    train = [p for p in rest if p not in set(near)]
    keep = sorted(set(train) | set(test_off) | set(near))
    os.makedirs(f'{W}/frames', exist_ok=True)
    cap = cv2.VideoCapture(a.video); i = 0; ks = set(keep)
    while True:
        ok, im = cap.read()
        if not ok: break
        if i in ks: cv2.imwrite(f'{W}/frames/{i:05d}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    sel_sharp = np.array([sharp[p] for p in picks]); stride_sharp = np.array([sharp[s] for s in range(0, n, win)])
    info = dict(fps=fps, n_video_frames=n, window=win, train=train, test_offpath=test_off, test_near=near,
                sharpness_selected_median=float(np.median(sel_sharp)), sharpness_fixed_stride_median=float(np.median(stride_sharp)),
                image_size=size)
    json.dump(info, open(f'{W}/split.json', 'w'), indent=1)
    return dict(train=len(train), test_offpath=len(test_off), test_near=len(near), window=win,
                sharp_sel=info['sharpness_selected_median'], sharp_stride=info['sharpness_fixed_stride_median'])


# -------------------------------------------------------------------------------------------------------------------- sfm
def stage_sfm(a, W):
    import pycolmap
    sp = json.load(open(f'{W}/split.json'))
    d = f'{W}/sfm'; os.makedirs(d, exist_ok=True); db = f'{d}/db.db'
    if os.path.exists(db): os.remove(db)
    eo = pycolmap.FeatureExtractionOptions(); eo.sift.max_num_features = a.sift_features; eo.max_image_size = 2000; eo.sift.peak_threshold = a.sift_peak
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = 'SIMPLE_RADIAL'
    if a.focal_prior:
        w, h = sp['image_size']; ro.camera_params = f'{a.focal_prior},{w / 2},{h / 2},0'
    t = time.time()
    pycolmap.extract_features(db, f'{W}/frames', camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
    log(W, f'sfm: features {time.time() - t:.0f}s')
    # No watermark test: on slow phone motion consecutive frames move almost by a pure 2D shift and COLMAP labels the pair
    # WATERMARK (dropped), which left the end of the room walk with ~0 tracks. Guided matching then marked such pairs DEGENERATE.
    mo = pycolmap.FeatureMatchingOptions(); mo.guided_matching = False
    vo = pycolmap.TwoViewGeometryOptions(); vo.detect_watermark = False
    pycolmap.match_exhaustive(db, matching_options=mo, verification_options=vo)
    log(W, f'sfm: exhaustive matching {time.time() - t:.0f}s')
    out = f'{d}/models'; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    gm = pycolmap.GlobalPipelineOptions() if hasattr(pycolmap, 'GlobalPipelineOptions') else None
    recs = pycolmap.global_mapping(db, f'{W}/frames', out) if gm is None else pycolmap.global_mapping(db, f'{W}/frames', out, options=gm)
    log(W, f'sfm: global mapping {time.time() - t:.0f}s, models: {[r.num_reg_images() for r in recs.values()]}')
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    # global mapper's own BA already refines the shared focal length and k1 (refine_focal_length / refine_extra_params)
    best = f'{d}/best'; os.makedirs(best, exist_ok=True); rec.write(best)
    n_all = len(os.listdir(f'{W}/frames'))
    cam = list(rec.cameras.values())[0]
    pts = {im.name: im.num_points3D for im in rec.images.values()}
    weak = sorted(k for k, v in pts.items() if v < 50)
    return dict(registered=rec.num_reg_images(), total=n_all, weak_images_lt50_points=len(weak),
                median_points_per_image=float(np.median(list(pts.values()))), models=len(recs), points=rec.num_points3D(),
                reproj_px=float(rec.compute_mean_reprojection_error()), camera=cam.model.name, params=[float(x) for x in cam.params])


# ----------------------------------------------------------------------------------------------------------------- undist
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); A = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, Sg, Vt = np.linalg.svd(A); Dg = np.eye(3); Dg[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Dg @ Vt; s = (Sg * np.diag(Dg)).sum() / ((src - mu_s) ** 2).sum(1).mean(); t = mu_d - s * R @ mu_s
    return s, R, t


def stage_undist(a, W):
    import pycolmap, cv2
    d = f'{W}/undist'; shutil.rmtree(d, ignore_errors=True)
    uo = pycolmap.UndistortCameraOptions(); uo.blank_pixels = 0.0     # crop: no black borders
    pycolmap.undistort_images(d, f'{W}/sfm/best', f'{W}/frames', undistort_options=uo)
    rec = pycolmap.Reconstruction(f'{d}/sparse')
    cam = list(rec.cameras.values())[0]; fx, fy, cx, cy = [float(x) for x in cam.params]
    Wd, H = cam.width, cam.height
    num = lambda s: int(os.path.splitext(s)[0])
    reg = {num(im.name): im for im in rec.images.values()}
    w2c = {f: np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]]) for f, im in reg.items()}
    X = np.array([p.xyz for p in rec.points3D.values()]); C = np.array([p.color for p in rec.points3D.values()])
    pid = {k: i for i, k in enumerate(rec.points3D.keys())}
    trk = np.array([p.track.length() for p in rec.points3D.values()]); err = np.array([p.error for p in rec.points3D.values()])
    # Sim3 to reference poses (metric, gravity up) if given, else scale so the median camera->point depth is a.depth_guess m
    S = dict(kind='none')
    cc = lambda M: -M[:3, :3].T @ M[:3, 3]
    if a.ref_poses:
        import re
        dig = lambda x: int(re.sub(r'\D', '', str(x)))       # 'frame_000008' -> 8
        rj = json.load(open(a.ref_poses))
        ref = {dig(e['frame']): np.array(e['w2c']) for e in rj.get('train', []) + rj.get('test', [])} if 'train' in rj else \
              {dig(k): np.array(v) for k, v in rj.items()}
        common = sorted(set(ref) & set(w2c))
        src = np.array([cc(w2c[f]) for f in common]); dst = np.array([cc(ref[f]) for f in common])
        s, R, t = umeyama(src, dst); res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        rot = []
        for f in common:
            Rn = w2c[f][:3, :3] @ R.T
            rot.append(np.degrees(np.arccos(np.clip((np.trace(Rn @ ref[f][:3, :3].T) - 1) / 2, -1, 1))))
        S = dict(kind='sim3_to_reference', n=len(common), scale=float(s), centre_median_cm=float(100 * np.median(res)),
                 centre_rmse_cm=float(100 * np.sqrt((res ** 2).mean())), rot_median_deg=float(np.median(rot)))
    else:
        dep = []
        for f, im in reg.items():
            Mx = w2c[f]; ids = [pid[p.point3D_id] for p in im.points2D if p.has_point3D()]
            if ids: dep.append(np.median((X[ids] @ Mx[:3, :3].T + Mx[:3, 3])[:, 2]))
        s = a.depth_guess / float(np.median(dep)); R = np.eye(3); t = np.zeros(3)
        # gravity guess: mean camera "down" (+y in OpenCV camera) -> world -y so that +y is up
        downs = np.array([w2c[f][:3, :3].T @ np.array([0, 1., 0]) for f in w2c]); g = downs.mean(0); g /= np.linalg.norm(g)
        up = -g; ax = np.cross(up, [0, 1., 0]); sn = np.linalg.norm(ax); cs = up @ np.array([0, 1., 0])
        if sn > 1e-8:
            ax /= sn; Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); R = np.eye(3) + sn * Kx + (1 - cs) * Kx @ Kx
        S = dict(kind='median_depth_guess', median_depth_m=a.depth_guess, scale=float(s))
    Rt = R
    for f in w2c:
        M = w2c[f]; Rn = M[:3, :3] @ Rt.T; tn = s * M[:3, 3] - Rn @ t
        w2c[f] = np.vstack([np.hstack([Rn, tn[:, None]]), [0, 0, 0, 1]])
    X = (s * (Rt @ X.T)).T + t
    obs = {}
    for f, im in reg.items():
        uv, ii = [], []
        for p2 in im.points2D:
            if p2.has_point3D(): uv.append(p2.xy); ii.append(pid[p2.point3D_id])
        obs[f'uv_{f}'] = np.array(uv, np.float32).reshape(-1, 2); obs[f'id_{f}'] = np.array(ii, np.int32)
    np.savez_compressed(f'{W}/scene_points.npz', X=X.astype(np.float32), C=C.astype(np.uint8), track=trk.astype(np.int32),
                        err=err.astype(np.float32), **obs)
    sp = json.load(open(f'{W}/split.json'))
    # working resolution images (train width a.width)
    sc = a.width / Wd; w2, h2 = a.width, int(round(H * sc / 2) * 2)
    os.makedirs(f'{W}/img', exist_ok=True)
    for f in reg:
        im = cv2.imread(f'{d}/images/{f:05d}.jpg')
        cv2.imwrite(f'{W}/img/{f:05d}.png', cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA))
    sx, sy = w2 / Wd, h2 / H
    scene = dict(K_full=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]], size_full=[Wd, H],
                 K=[[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], width=w2, height=h2,
                 w2c={str(f): w2c[f].tolist() for f in sorted(w2c)}, scale=S,
                 train=[f for f in sp['train'] if f in w2c], test_offpath=[f for f in sp['test_offpath'] if f in w2c],
                 test_near=[f for f in sp['test_near'] if f in w2c],
                 missing=dict(train=[f for f in sp['train'] if f not in w2c], test_offpath=[f for f in sp['test_offpath'] if f not in w2c],
                              test_near=[f for f in sp['test_near'] if f not in w2c]))
    json.dump(scene, open(f'{W}/scene.json', 'w'), indent=1)
    return dict(size_full=[Wd, H], work_size=[w2, h2], solved_K=[fx, fy, cx, cy], scale=S,
                missing={k: len(v) for k, v in scene['missing'].items()})


STAGES = ['frames', 'sfm', 'undist', 'depth', 'mesh', 'splat', 'score', 'strips']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video'); p.add_argument('work')
    p.add_argument('--fps', type=float, default=5.0, help='target frames per second (one sharpest frame per window)')
    p.add_argument('--window', type=int, default=0, help='explicit window in frames (overrides --fps)')
    p.add_argument('--test-block', type=int, nargs=2, default=None, help='off-path held-out video frame range [lo, hi]')
    p.add_argument('--test-step', type=int, default=2)
    p.add_argument('--gap', type=int, default=8, help='frames around the block also kept out of training')
    p.add_argument('--near-every', type=int, default=8, help='hold out every Nth remaining pick as near-path test')
    p.add_argument('--sift-features', type=int, default=8192)
    p.add_argument('--sift-peak', type=float, default=0.0067, help='SIFT peak threshold; lower (0.001-0.002) for low-texture, low-bitrate video')
    p.add_argument('--focal-prior', type=float, default=0, help='initial focal in px at native resolution (refined by SfM)')
    p.add_argument('--ref-poses', default='', help='JSON with metric reference w2c (AR), for Sim3 scale/gravity only')
    p.add_argument('--depth-guess', type=float, default=2.5, help='median scene depth in m when no reference poses')
    p.add_argument('--width', type=int, default=432, help='working width for depth/mesh/splat/scoring')
    p.add_argument('--voxel', type=float, default=0.01)
    p.add_argument('--trunc', type=float, default=4.0, help='TSDF truncation in voxels')
    p.add_argument('--depth-max', type=float, default=4.0)
    p.add_argument('--mesh-depth', default='depth_cons', help='depth_cons (multi-view checked) or depth (all pixels)')
    p.add_argument('--weight-thresh', type=float, default=2.0, help='TSDF min weight for surface extraction')
    p.add_argument('--steps', type=int, default=1000, help='CPU budget; on the room 1000 beat 2000 and 3000 on held-out views (overfit with ~70 views)')
    p.add_argument('--train-scale', type=float, default=1.0, help='splat training resolution relative to --width')
    p.add_argument('--depth-w', type=float, default=0.2, help='weight of the mesh-depth loss in splat training')
    p.add_argument('--prune-dist', type=float, default=0.05, help='prune Gaussians farther than this (m) from the mesh during training')
    p.add_argument('--max-gauss', type=int, default=300000)
    p.add_argument('--grad-thresh', type=float, default=0.0008, help='absgrad densification threshold (gsplat absgrad default)')
    p.add_argument('--prune-every', type=int, default=500)
    p.add_argument('--opacity-reset', type=int, default=0)
    p.add_argument('--expo-lr', type=float, default=1e-2, help='Adam lr of the per-image colour matrices (each image is seen only steps/n_images times)')
    p.add_argument('--sh-degree', type=int, default=1, help='max SH degree (3 overfits view-dependent colour with ~70 views)')
    p.add_argument('--init-mesh-pts', type=int, default=150000)
    p.add_argument('--depth-model', default=os.path.expanduser('~/models/depth_anything_v2_vits.onnx'))
    p.add_argument('--only', default='', help='comma list of stages to run (default all)')
    p.add_argument('--redo', default='', help='comma list of stages to force')
    a = p.parse_args()
    W = a.work; os.makedirs(W, exist_ok=True)
    json.dump(vars(a), open(f'{W}/args.json', 'w'), indent=1)
    only = a.only.split(',') if a.only else STAGES; redo = set(a.redo.split(',')) if a.redo else set()
    from stages import stage_depth, stage_mesh, stage_splat, stage_score, stage_strips
    fns = dict(frames=stage_frames, sfm=stage_sfm, undist=stage_undist, depth=stage_depth, mesh=stage_mesh,
               splat=stage_splat, score=stage_score, strips=stage_strips)
    for s in STAGES:
        if s not in only: continue
        if done(W, s) and s not in redo:
            log(W, f'{s}: done already, skipping'); continue
        t = time.time(); log(W, f'{s}: start')
        info = fns[s](a, W) or {}
        info['seconds'] = round(time.time() - t, 1)
        mark(W, s, info); log(W, f'{s}: {json.dumps(info)}')


if __name__ == '__main__':
    main()
