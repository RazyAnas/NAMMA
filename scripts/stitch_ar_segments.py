"""Stitch an AR recording whose ARCore origin reset several times, using image-based SfM pieces as the glue.

ARCore pieces break where tracking resets (each piece is metric and gravity-aligned, but shifted); SfM pieces from the
video break where the walls have no texture. They break in different places, so they can be chained:
  - AR piece 0 (the one with most frames) defines the world.
  - An SfM model that shares frames with already-placed AR pieces gets a similarity transform (7 DOF, Umeyama on
    camera centres).
  - An AR piece that shares frames with an already-placed SfM model gets a yaw + translation (4 DOF; ARCore already
    has metric scale and gravity), yaw from the camera orientations, translation from the centres.
Repeat until nothing new is placed, then rewrite every tracked pose of every placed piece into one frame.

Writes <out>/frames_stitched.csv (the recorder's frames.csv with m0..m15 replaced, plus 'piece') and stitch_report.json.
Usage: python stitch_ar_segments.py <frames.csv> <video.csv> <sfm models dir (inc/0, inc/1 ...)> <frame stride> <out dir>
"""
import csv
import json
import os
import sys

import numpy as np
import pycolmap

FRAMES, VIDEO, MODELS, STRIDE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
MAX_SPEED = 3.0      # m/s; faster pose changes are origin resets (same rule as quick-model.js arSegments)
MIN_SHARED = 4       # frames two pieces must share to be linked
MAX_FIT_M = 0.10     # an SfM model is placed only if its fit to placed AR centres has a median error below this
MAX_PIECE_M = 0.15   # an AR piece is placed only if its centres land within this (median) of the placed SfM centres
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(FRAMES)))
video_ts = [int(r["ts_us"]) for r in csv.DictReader(open(VIDEO))]
row_by_ts = {int(float(r["video_ts_us"])): r for r in rows if r["video_ts_us"]}
GL_TO_CV = np.diag([1.0, -1.0, -1.0])   # WebXR camera (y up, -z forward) <-> COLMAP camera (y down, +z forward)


def ar_c2w(r):
    return np.array([float(r[f"m{i}"]) for i in range(16)]).reshape(4, 4).T   # column-major in the CSV


# AR pieces
piece_of, piece, last = {}, 0, None
for r in rows:
    if r["tracking"] != "0" or not r["m12"]:
        continue
    p, t = ar_c2w(r)[:3, 3], float(r["t_ms"]) / 1000
    if last is not None and np.linalg.norm(p - last[1]) / max(1e-3, t - last[0]) > MAX_SPEED:
        piece += 1
    piece_of[r["frame"]] = piece
    last = (t, p)
sizes = {}
for p in piece_of.values():
    sizes[p] = sizes.get(p, 0) + 1
root = max(sizes, key=sizes.get)

# SfM observations: (model, AR piece, AR camera-to-world, SfM camera-to-world)
obs = []
models = sorted(d for d in os.listdir(MODELS) if d.isdigit())
for m in models:
    rec = pycolmap.Reconstruction(os.path.join(MODELS, m))
    for img in rec.images.values():
        k = int(os.path.splitext(img.name)[0]) * STRIDE
        r = row_by_ts.get(video_ts[k]) if k < len(video_ts) else None
        if r is None or r["frame"] not in piece_of:
            continue
        w2c = img.cam_from_world().matrix() if callable(img.cam_from_world) else img.cam_from_world.matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = w2c[:3, :3].T
        c2w[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
        obs.append((m, piece_of[r["frame"]], ar_c2w(r), c2w))


def umeyama(x, y):
    mx, my = x.mean(0), y.mean(0)
    a, b = x - mx, y - my
    U, D, Vt = np.linalg.svd(b.T @ a / len(x))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = (D * np.diag(E)).sum() / (a ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


def yaw_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


ar_T = {root: np.eye(4)}      # AR piece -> world (yaw + translation)
rejected_models, rejected_pieces = {}, {}
sfm_S = {}                    # SfM model -> world (scale, R, t)
changed = True
while changed:
    changed = False
    for m in models:
        if m in sfm_S:
            continue
        pairs = [(o[3][:3, 3], (ar_T[o[1]] @ o[2])[:3, 3]) for o in obs if o[0] == m and o[1] in ar_T]
        if len(pairs) >= MIN_SHARED:
            x, y = map(np.array, zip(*pairs))
            s_, R_, t_ = umeyama(x, y)
            fit = np.median(np.linalg.norm((s_ * (R_ @ x.T)).T + t_ - y, axis=1))
            if fit <= MAX_FIT_M:
                sfm_S[m] = (s_, R_, t_)
                changed = True
            else:
                rejected_models[m] = round(float(fit) * 100, 1)
    for p in sizes:
        if p in ar_T:
            continue
        links = [o for o in obs if o[1] == p and o[0] in sfm_S]
        if len(links) < MIN_SHARED:
            continue
        yaws, centres_ar, centres_w = [], [], []
        for m, _, a, c in links:
            s, R, t = sfm_S[m]
            Rw = R @ c[:3, :3] @ GL_TO_CV          # world-from-camera in WebXR camera axes, via SfM
            D = Rw @ a[:3, :3].T                   # maps AR-piece orientation to world orientation
            yaws.append(np.arctan2(D[0, 2] - D[2, 0], D[0, 0] + D[2, 2]))
            centres_ar.append(a[:3, 3])
            centres_w.append(s * R @ c[:3, 3] + t)
        theta = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
        Ry = yaw_rot(theta)
        tr = np.mean(np.array(centres_w) - (Ry @ np.array(centres_ar).T).T, axis=0)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Ry, tr
        err = np.median(np.linalg.norm(np.array(centres_w) - ((Ry @ np.array(centres_ar).T).T + tr), axis=1))
        if err <= MAX_PIECE_M:
            ar_T[p] = T
            changed = True
        else:
            rejected_pieces[p] = round(float(err) * 100, 1)

# Residuals: placed AR camera centres against the placed SfM centres for the same frames
res = {}
for m, p, a, c in obs:
    if p in ar_T and m in sfm_S:
        s, R, t = sfm_S[m]
        res.setdefault(p, []).append(np.linalg.norm((ar_T[p] @ a)[:3, 3] - (s * R @ c[:3, 3] + t)))

with open(os.path.join(OUT, "frames_stitched.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(list(rows[0].keys()) + ["piece"])
    placed = dropped = 0
    for r in rows:
        p = piece_of.get(r["frame"])
        out = dict(r)
        if p is not None and p in ar_T:
            M = ar_T[p] @ ar_c2w(r)
            for i, v in enumerate(M.T.reshape(-1)):
                out[f"m{i}"] = f"{v:.6f}"
            placed += 1
        elif p is not None:
            out["tracking"] = "1"     # unplaced piece: mark untracked so nothing uses its pose
            dropped += 1
        w.writerow(list(out.values()) + ["" if p is None else p])

report = dict(
    pieces=len(sizes), root=root, placed_pieces=sorted(ar_T), unplaced_pieces=sorted(set(sizes) - set(ar_T)),
    sfm_models=len(models), placed_models=sorted(sfm_S), tracked_frames_placed=placed, tracked_frames_dropped=dropped,
    residual_cm_per_piece={str(p): round(float(np.median(v)) * 100, 1) for p, v in sorted(res.items())},
    rejected_models_fit_cm=rejected_models, rejected_pieces_fit_cm={str(k): v for k, v in rejected_pieces.items()},
)
json.dump(report, open(os.path.join(OUT, "stitch_report.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
