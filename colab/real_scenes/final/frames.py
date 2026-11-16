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
