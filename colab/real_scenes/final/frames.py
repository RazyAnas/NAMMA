"""Frames for v3: from a video (sharpest frame per time window) or from a folder of already extracted frames (DL3DV, photos).
Every output frame is resized to `long_side` px on its long side and named 0000.jpg, 0001.jpg, ... in walk order.
Usage: python frames.py <video file | image folder> <out dir> [n=300] [long_side=1200]"""
import glob, os, sys
import cv2

# ---- stretched edges (16 Nov): a video whose picture is squeezed, with its edge rows smeared over the gap
BANDS = None                                    # set by find_bands(video): every reader below then repairs each frame of that video
SIDES = ('top', 'bottom', 'left', 'right')


def _band(g, max_frac):
    """Rows from the top edge of g (grey, float32) that only repeat one row: from row to row almost nothing changes (a) while
    along the row the picture's detail is still there (b). Measured 16 Nov: a / b is under 0.25 in a smear (HEVC, 20 Mbit/s)
    and 0.4 or more in real picture rows, where the sensor's noise alone changes as much down as across. The band ends where
    3 of 4 rows in a row are picture. None: this frame cannot tell (an edge without detail, or no end within max_frac)."""
    import numpy as np
    lim = int(max_frac * g.shape[0]) + 8
    vd = np.abs(np.diff(g[:lim + 1], axis=0)).mean(1); hd = np.abs(np.diff(g[:lim], axis=1)).mean(1); pic = vd > 0.3 * hd + 0.05
    end = next((r for r in range(lim - 4) if pic[r:r + 4].sum() >= 3), None)
    if end is None or (end >= 4 and float(np.median(hd[:end])) < 0.3): return None
    return end


def edge_bands(imgs, min_frac=0.03, max_frac=0.3):
    """Smeared bands along the edges of a video, from a few of its frames (BGR, all one size): {side: px}, 0 = none. A band is
    one row (or column) of the picture repeated to the edge: across it nothing changes, along it the picture's detail remains.
    Real content can look like that in one frame (curtains, a door frame), so a side counts only when most frames that can tell
    agree on the same width, and that width is at least min_frac of the picture."""
    import numpy as np
    H, W = imgs[0].shape[:2]; out = {}
    for side in SIDES:
        v = []
        for im in imgs:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            n = _band(np.ascontiguousarray({'top': g, 'bottom': g[::-1], 'left': g.T, 'right': g.T[::-1]}[side]), max_frac)
            if n is not None: v.append(n)
        L = H if side in ('top', 'bottom') else W; out[side] = 0
        if len(v) >= max(3, 0.4 * len(imgs)):
            med = float(np.median(v))
            if med >= min_frac * L and np.mean(np.abs(np.array(v) - med) <= max(2.0, 0.01 * L)) >= 0.7: out[side] = int(round(med))
    return out


def band_fix(b, w, h, margin=3):
    """The repair for bands b on a w x h picture: cut them off (+ margin px: the rows where smear and picture blend), and if
    two opposite sides are smeared alike the picture was squeezed between them, so the other axis is shrunk by the same factor
    and shapes are right again (shrinking invents no pixels). {size, crop [x, y, w, h], out [w, h], squeeze}, None if no bands."""
    if not any(b[s] for s in SIDES): return None
    m = {s: b[s] + margin if b[s] else 0 for s in SIDES}
    cw, ch = w - m['left'] - m['right'], h - m['top'] - m['bottom']; ow, oh = float(cw), float(ch); sq = [1.0, 1.0]
    pair = lambda p, q: p > 0 and q > 0 and abs(p - q) <= 0.25 * max(p, q)
    # a band is found a row or two short (the codec blends its last rows into the picture), so a factor that close to a
    # simple one is that one, e.g. 3/4: a 4:3 image drawn on 16:9
    snap = lambda f, L: next((r for r in (0.75, 0.5625, 0.8, 2 / 3) if abs(f - r) < max(0.006, 6 / L)), f)
    if pair(b['top'], b['bottom']): sq[1] = snap((h - b['top'] - b['bottom']) / h, h); ow = cw * sq[1]
    if pair(b['left'], b['right']): sq[0] = snap((w - b['left'] - b['right']) / w, w); oh = ch * sq[0]
    even = lambda x: max(2, int(round(x / 2)) * 2)
    return dict(size=[w, h], bands=dict(b), crop=[m['left'], m['top'], cw, ch], out=[even(ow), even(oh)], squeeze=[round(sq[0], 4), round(sq[1], 4)])


def sample_frames(video, k=16):
    """k frames spread over the video, as ffmpeg shows them (rotation applied); OpenCV if ffmpeg cannot."""
    import json, subprocess, numpy as np
    out = []
    try:
        pr = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', video], capture_output=True, text=True)
        dur = float(json.loads(pr.stdout)['format']['duration'])
        for i in range(k):
            r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{dur * (i + 0.5) / k:.2f}', '-i', video, '-frames:v', '1', '-f', 'image2pipe',
                                '-vcodec', 'png', '-'], capture_output=True)
            im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None
            if im is not None: out.append(im)
    except Exception: out = []
    if len(out) < 4:
        cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
        for i in range(k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / k)); ok, f = cap.read()
            if ok: out.append(f)
    return [f for f in out if f.shape == out[0].shape]


def find_bands(video, k=16):
    """Look for smeared edges in this video and remember the repair (BANDS) for the readers below; returns it, None if the
    video is clean. First seen 16 Nov: the app's live-coverage recorder drew ARCore's 4:3 camera image onto the 16:9 video,
    so the picture was squeezed to 3/4 of its height and the top and bottom eighth were smears (app fixed: TexQuad)."""
    global BANDS; BANDS = None
    imgs = sample_frames(video, k)
    if len(imgs) >= 4: BANDS = band_fix(edge_bands(imgs), imgs[0].shape[1], imgs[0].shape[0])
    return BANDS


def unband(img):
    """One decoded frame repaired (only a frame of the size the bands were found on)."""
    B = BANDS
    if not B or [img.shape[1], img.shape[0]] != B['size']: return img
    x, y, cw, ch = B['crop']; c = img[y:y + ch, x:x + cw]
    return c if [cw, ch] == B['out'] else cv2.resize(c, tuple(B['out']), interpolation=cv2.INTER_AREA)


def _vf(long_side):
    """ffmpeg's filter for every frame: the band repair (if any), then the long side down to long_side."""
    pre = 'crop={2}:{3}:{0}:{1},scale={4}:{5}:flags=area,'.format(*BANDS['crop'], *BANDS['out']) if BANDS else ''
    return pre + f"scale='if(gt(iw,ih),min(iw,{long_side}),-2)':'if(gt(iw,ih),-2,min(ih,{long_side}))':flags=area"


def from_video(video, out, n=300, long_side=1200):
    """Sharpest (variance of Laplacian) frame in each of n equal time windows: motion blur is the main enemy of phone video."""
    os.makedirs(out, exist_ok=True); cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = max(1, total // n); best, bs, k, i = None, -1, 0, 0
    while k < n:
        if not cap.grab(): break
        ok, f = cap.retrieve()
        if ok:
            f = unband(f)
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
    vf = _vf(long_side)
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
    vf = _vf(long_side)
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
