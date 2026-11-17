"""Smeared edges (frames.find_bands): the app's live-coverage recorder drew a 4:3 camera image on a 16:9 video (16 Nov), so the
picture was squeezed to 3/4 of its height and the top and bottom eighth were the edge rows repeated. Found, cut, un-squeezed."""
import os, shutil, subprocess, sys
import cv2, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'colab', 'real_scenes', 'final'))
import frames


def scene(k, w=960, h=540):
    """A textured picture that moves a little from frame to frame (fine noise like a sensor's + shapes)."""
    r = np.random.default_rng(7); base = cv2.GaussianBlur(r.integers(0, 255, (h + 80, w + 200, 3), dtype=np.uint8), (0, 0), 3)
    img = base[10:10 + h, 4 * k:4 * k + w].copy()
    for i in range(12): cv2.circle(img, (80 * i + 40 - 3 * k, 60 + 37 * (i % 5) * 2), 28, (40 * (i % 6), 255 - 30 * (i % 7), 90), -1)
    return np.clip(img.astype(np.int16) + np.random.default_rng(k).integers(-4, 5, img.shape), 0, 255).astype(np.uint8)


def squeezed(img, frac=0.75):
    h, w = img.shape[:2]; ch = int(round(h * frac)); pad = (h - ch) // 2
    return cv2.copyMakeBorder(cv2.resize(img, (w, ch), interpolation=cv2.INTER_AREA), pad, h - ch - pad, 0, 0, cv2.BORDER_REPLICATE)


def test_clean_frames_have_no_bands():
    imgs = [scene(k) for k in range(12)]
    assert frames.edge_bands(imgs) == dict(top=0, bottom=0, left=0, right=0) and frames.band_fix(frames.edge_bands(imgs), 960, 540) is None
    flat = [np.full((540, 960, 3), 200, np.uint8) for _ in range(12)]                         # a blank picture cannot tell: no bands
    assert not any(frames.edge_bands(flat).values())


def test_one_frame_of_vertical_stripes_is_not_a_band():
    imgs = [scene(k) for k in range(12)]; imgs[3][:90] = imgs[3][90]                          # curtains in one frame only
    assert not any(frames.edge_bands(imgs).values())


def test_squeezed_frames_are_found_and_restored():
    imgs = [squeezed(scene(k)) for k in range(12)]; b = frames.edge_bands(imgs)
    assert abs(b['top'] - 67) <= 2 and abs(b['bottom'] - 67) <= 2 and b['left'] == b['right'] == 0     # 540 / 8 = 67.5
    fix = frames.band_fix(b, 960, 540); assert fix['squeeze'] == [1.0, 0.75] and fix['out'][0] == 720 and abs(fix['out'][1] - 400) <= 6
    frames.BANDS = fix
    try:
        out = frames.unband(imgs[5]); want = cv2.resize(scene(5), (720, 405), interpolation=cv2.INTER_AREA)
        y = (405 - out.shape[0]) // 2; assert out.shape[1] == 720 and cv2.PSNR(out, want[y:y + out.shape[0]]) > 30   # the same picture, right shape
        other = np.zeros((300, 400, 3), np.uint8); assert frames.unband(other) is other                  # another size: untouched
    finally: frames.BANDS = None


def test_sideways_bands_squeeze_the_other_axis():
    imgs = [np.ascontiguousarray(squeezed(scene(k)).transpose(1, 0, 2)) for k in range(12)]; b = frames.edge_bands(imgs)
    assert b['top'] == b['bottom'] == 0 and abs(b['left'] - 67) <= 2 and abs(b['right'] - 67) <= 2
    assert frames.band_fix(b, 540, 960)['squeeze'] == [0.75, 1.0]


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg')
def test_a_squeezed_video_through_both_readers(tmp_path):
    for k in range(40): cv2.imwrite(str(tmp_path / f'{k:03d}.png'), squeezed(scene(k)))
    v = str(tmp_path / 'v.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '30', '-i', str(tmp_path / '%03d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', v], check=True)
    try:
        fix = frames.find_bands(v, k=10); assert fix and fix['squeeze'] == [1.0, 0.75] and fix['size'] == [960, 540]
        assert frames.from_video_fast(v, str(tmp_path / 'fast'), 4, 1440) == 4 and frames.from_video(v, str(tmp_path / 'cv'), 4, 1440) == 4
        for d in ('fast', 'cv'):
            o = cv2.imread(str(tmp_path / d / '0001.jpg')); assert o.shape[1] == 720 and abs(o.shape[0] - 400) <= 6
            g = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32)                         # no smear left at the top or bottom
            for rows in (g[:6], g[-6:]): assert np.abs(np.diff(rows, axis=0)).mean() > 0.3 * np.abs(np.diff(rows, axis=1)).mean()
        clean = str(tmp_path / 'clean.mp4')
        for k in range(40): cv2.imwrite(str(tmp_path / f'{k:03d}.png'), scene(k))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '30', '-i', str(tmp_path / '%03d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', clean], check=True)
        assert frames.find_bands(clean, k=10) is None and frames.BANDS is None
    finally: frames.BANDS = None
