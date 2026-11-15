"""Frames for v3: from a video (sharpest frame per time window) or from a folder of already extracted frames (DL3DV, photos).
Every output frame is resized to `long_side` px on its long side and named 0000.jpg, 0001.jpg, ... in walk order.
Usage: python frames.py <video file | image folder> <out dir> [n=300] [long_side=1200]"""
import glob, os, sys
import cv2

def from_video(video, out, n=300, long_side=1200):
    """Sharpest (variance of Laplacian) frame in each of n equal time windows: motion blur is the main enemy of phone video."""
    os.makedirs(out, exist_ok=True); cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = max(1, total // n); best, bs, k, i = None, -1, 0, 0
    while k < n:
        if not cap.grab(): break
        ok, f = cap.retrieve()
        if ok:
            g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)), cv2.COLOR_BGR2GRAY); sc = cv2.Laplacian(g, cv2.CV_64F).var()
            if sc > bs: best, bs = f, sc
        if (i + 1) % win == 0 and best is not None:
            save(best, f'{out}/{k:04d}.jpg', long_side); k += 1; best, bs = None, -1
        i += 1
    return k

def from_video_fast(video, out, n=300, long_side=1200, total=0):
    """Same choice as from_video (sharpest frame per window), but ffmpeg decodes on all CPU cores and scales to long_side
    itself (30 Oct: OpenCV decoded a 17-min 4K 60 fps video at ~5 frames/s = 3.6 h). Returns the number of frames saved."""
    import json, subprocess, tempfile, numpy as np
    vf = f"scale='if(gt(iw,ih),min(iw,{long_side}),-2)':'if(gt(iw,ih),-2,min(ih,{long_side}))':flags=area"
    if not total:
        pr = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=nb_read_packets',
                             '-of', 'json', video], capture_output=True, text=True)
        total = int(json.loads(pr.stdout)['streams'][0]['nb_read_packets'])
    with tempfile.TemporaryDirectory() as td:                                        # output size after rotation + scaling
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vf', vf, '-frames:v', '1', f'{td}/f.png'], check=True)
        h, w = cv2.imread(f'{td}/f.png').shape[:2]
    os.makedirs(out, exist_ok=True); win = max(1, total // n); size = w * h * 3
    hw = ['-hwaccel', 'cuda'] if subprocess.run(['ffmpeg', '-v', 'error', '-hwaccel', 'cuda', '-i', video, '-t', '1', '-f', 'null', '-'],
                                                 capture_output=True).returncode == 0 else []   # the T4's video decoder, if ffmpeg has it
    print('frames: ffmpeg decode on', 'GPU (NVDEC)' if hw else 'CPU (all cores)', flush=True)
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '0', *hw, '-i', video, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                         stdout=subprocess.PIPE, bufsize=size * 4)
    best, bs, k, i = None, -1, 0, 0
    while k < n:
        buf = p.stdout.read(size)
        if len(buf) < size: break
        f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        g = cv2.cvtColor(cv2.resize(f, (w // 4, h // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); sc = cv2.Laplacian(g, cv2.CV_64F).var()
        if sc > bs: best, bs = f.copy(), sc
        if (i + 1) % win == 0 and best is not None:
            save(best, f'{out}/{k:04d}.jpg', long_side); k += 1; best, bs = None, -1
        i += 1
    p.stdout.close(); p.kill(); p.wait()
    return k

def pick_by_motion(sharp, motion, n, mix=0.5, drop=0.25):
    """Which of m candidate frames to keep (at most n), given each candidate's sharpness and how much the picture moved during
    its stretch of video. Equal time windows (from_video) give a fast turn as few frames as a pause; here half of the frames
    follow the motion and half the time (`mix`), so a turn gets up to ~2.5x the frames and a pause still gets some. Inside each
    group the sharpest candidate wins. A kept frame far less sharp than its neighbours (< drop x their median) is left out
    (blur smears the model): never the first or last, never two in a row, at most a tenth of the frames.
    Returns (sorted candidate indices, info). Pure numpy: tests/v3/test_budget.py."""
    import numpy as np
    S, M = np.asarray(sharp, float), np.asarray(motion, float); m = len(S)
    if m == 0: return [], dict(kept=0, dropped_blurred=0, active=0.0, fast=[], blurry=[])
    n = max(1, min(int(n), m)); pos = M[M > 0]; med = float(np.median(pos)) if len(pos) else 0.0
    mo = np.minimum(M, 4 * med) if med > 0 else np.ones(m)                           # one cut or glitch must not take the frames
    w = mix * mo / mo.sum() + (1 - mix) / m; c = np.cumsum(w); g = np.minimum(n - 1, ((c - w / 2) * n).astype(int))
    keep = [int(i[np.argmax(S[i])]) for i in (np.flatnonzero(g == k) for k in range(n)) if len(i)]
    ks = S[keep]; loc = np.array([np.median(ks[max(0, i - 5):i + 6]) for i in range(len(keep))]); out = []
    for i in range(1, len(keep) - 1):
        if ks[i] < drop * loc[i] and len(out) < len(keep) // 10 and (not out or out[-1] != i - 1): out.append(i)
    kept = [k for i, k in enumerate(keep) if i not in set(out)]
    runs = lambda mask: [[int(a[0]), int(a[-1])] for a in np.split(np.flatnonzero(mask), np.flatnonzero(np.diff(np.flatnonzero(mask)) > 1) + 1) if len(a)]
    return kept, dict(kept=len(kept), dropped_blurred=len(out), active=float((M >= 0.15 * med).mean()) if med > 0 else 1.0,
                      fast=runs(M >= 4 * med) if med > 0 else [], blurry=runs(S < 0.4 * np.median(S)))


def from_video_smart(video, out, cap=1000, long_side=1200, total=0, fps=30.0, rate=411 / 123.3, over=4, cand=None):
    """Frames by motion, not by the clock (16 Nov). One decode pass like from_video_fast, but it keeps the sharpest frame of
    `over` x as many short windows as frames are wanted (the candidates, in `cand`), with the picture motion of each window
    (mean difference of consecutive frames at 64 px, brightness removed). Then: seconds the camera actually moved x `rate`
    frames (run 4's density), at most `cap`, chosen by pick_by_motion. Frames are named by their window number (walk order),
    so candidates added later (run.py: a camera solve that broke apart) sort into place; unused candidates stay in `cand`.
    Returns (frames saved, report); the report is also written to <cand>/pick.json."""
    import json, shutil, subprocess, tempfile, numpy as np
    vf = f"scale='if(gt(iw,ih),min(iw,{long_side}),-2)':'if(gt(iw,ih),-2,min(ih,{long_side}))':flags=area"
    if not total:
        pr = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=nb_read_packets',
                             '-of', 'json', video], capture_output=True, text=True)
        total = int(json.loads(pr.stdout)['streams'][0]['nb_read_packets'])
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vf', vf, '-frames:v', '1', f'{td}/f.png'], check=True)
        h, w = cv2.imread(f'{td}/f.png').shape[:2]
    cand = cand or out.rstrip('/') + '_cand'; shutil.rmtree(cand, ignore_errors=True); os.makedirs(cand); os.makedirs(out, exist_ok=True)
    dur = total / (fps or 30.0); n0 = int(max(50, min(cap, round(dur * rate)))); win = max(1, total // (n0 * over)); size = w * h * 3
    hw = ['-hwaccel', 'cuda'] if subprocess.run(['ffmpeg', '-v', 'error', '-hwaccel', 'cuda', '-i', video, '-t', '1', '-f', 'null', '-'],
                                                 capture_output=True).returncode == 0 else []
    print(f'frames: motion-aware, {total} video frames, windows of {win}, ffmpeg decode on', 'GPU (NVDEC)' if hw else 'CPU (all cores)', flush=True)
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '0', *hw, '-i', video, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                         stdout=subprocess.PIPE, bufsize=size * 4)
    S, M, best, bs, mo, prev, i = [], [], None, -1.0, 0.0, None, 0
    th = max(8, round(64 * h / w))
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size: break
        f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        g = cv2.cvtColor(cv2.resize(f, (w // 4, h // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); sc = cv2.Laplacian(g, cv2.CV_64F).var()
        t = cv2.resize(g, (64, th), interpolation=cv2.INTER_AREA).astype(np.float32); t -= t.mean()      # auto-exposure steps are not motion
        if prev is not None: mo += float(np.abs(t - prev).mean())
        prev = t
        if sc > bs: best, bs = f.copy(), sc
        if (i + 1) % win == 0:
            save(best, f'{cand}/{len(S):05d}.jpg', long_side); S.append(float(bs)); M.append(mo); best, bs, mo = None, -1.0, 0.0
        i += 1
    p.stdout.close(); p.kill(); p.wait()
    if not S: return 0, {}
    _, info = pick_by_motion(S, M, n0); n = int(max(50, min(cap, round(dur * info['active'] * rate))))   # paused seconds are not walking
    kept, info = pick_by_motion(S, M, n)
    for j in kept: shutil.move(f'{cand}/{j:05d}.jpg', f'{out}/{j:05d}.jpg')
    sec = lambda r: [[round(a * win / fps, 1), round((b + 1) * win / fps, 1)] for a, b in r]
    rep = dict(mode='smart', video_frames=total, fps=round(fps, 2), seconds=round(dur, 1), active_seconds=round(dur * info['active'], 1),
               window=win, candidates=len(S), frames=len(kept), dropped_blurred=info['dropped_blurred'],
               too_fast_s=sec(info['fast'])[:40], blurry_s=sec(info['blurry'])[:40], kept=kept)
    json.dump(dict(rep, sharp=[round(x, 1) for x in S], motion=[round(x, 3) for x in M]), open(f'{cand}/pick.json', 'w'))
    return len(kept), rep


def add_candidates(cand, out, names, around=2, most=0):
    """More frames where the camera solve lost the video: for each frame in `names` (not registered), the unused candidates of
    the `around` windows before and after it move into `out`. At most `most` frames (0 = no limit). Returns the names added."""
    import shutil
    have = {int(os.path.splitext(f)[0]) for f in os.listdir(cand) if f.endswith('.jpg') and f[:5].isdigit()}; add = []
    for nm in sorted(names):
        j = int(os.path.splitext(nm)[0])
        for k in range(j - around, j + around + 1):
            if k in have and (not most or len(add) < most): have.discard(k); add.append(f'{k:05d}.jpg')
    for a in add: shutil.move(f'{cand}/{a}', f'{out}/{a}')
    return sorted(add)


def thin_dir(folder, keep):
    """Keep `keep` evenly spaced frames of a folder (walk order), delete the rest: a camera solve that ran out of time."""
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.jpg')); keep = max(2, min(keep, len(files)))
    pick = {files[round(i * (len(files) - 1) / (keep - 1))] for i in range(keep)}
    for f in files:
        if f not in pick: os.remove(f'{folder}/{f}')
    return len(pick)


def resize_dir(folder, long_side):
    """Every frame of a folder down to long_side px (in place). gsplat's loader scales the camera to the image size it
    finds (datasets/colmap.py: actual size / COLMAP size), so the camera model stays as solved."""
    n = 0
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        img = cv2.imread(f'{folder}/{f}')
        if img is not None and max(img.shape[:2]) > long_side: save(img, f'{folder}/{f}', long_side); n += 1
    return n


def from_images(folder, out, n=300, long_side=1200):
    """Evenly spaced subset (keeps walk order) of an image folder, e.g. frames a dataset already extracted from a video."""
    os.makedirs(out, exist_ok=True)
    files = sorted(f for e in ('jpg', 'jpeg', 'png', 'JPG', 'PNG') for f in glob.glob(f'{folder}/*.{e}'))
    step = max(1, len(files) / n); pick = [files[int(i * step)] for i in range(min(n, len(files)))]
    for k, f in enumerate(pick): save(cv2.imread(f), f'{out}/{k:04d}.jpg', long_side)
    return len(pick)

def save(img, path, long_side):
    h, w = img.shape[:2]; s = long_side / max(h, w)
    if s < 1: img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300; ls = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    k = from_images(src, out, n, ls) if os.path.isdir(src) else from_video(src, out, n, ls)
    print(f'{k} frames -> {out}', flush=True)
