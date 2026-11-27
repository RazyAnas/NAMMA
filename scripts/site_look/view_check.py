"""Checks of a spot's view that the live check lacked (27 Nov): the old one compared only the part of the scene that view and
photo share, so a view zoomed in 1.6x (vertical field of view applied across the window) passed for four weeks, and it never
looked at where the camera points. Pure numpy / PIL: no browser, so it is unit-tested (tests/v3/test_view_check.py)."""
import numpy as np

SCALES = (0.55, 0.62, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8)


def _grey(im, box):
    from PIL import ImageFilter
    return np.asarray(im.crop(box).convert('L').resize((110, 70)).filter(ImageFilter.GaussianBlur(1)), np.float32)


def match(view, thumb, fv, ff):
    """Correlation of view and photo over the part of the scene both cover; fv / ff = their horizontal fields of view, degrees."""
    W, H = view.size; tw, th = thumb.size
    txv = np.tan(np.radians(fv) / 2); tyv = txv * H / W; txf = np.tan(np.radians(ff) / 2); tyf = txf * th / tw
    ox, oy = min(txv, txf), min(tyv, tyf)
    box = lambda im, fx, fy: (int(im.size[0] / 2 * (1 - fx)), int(im.size[1] / 2 * (1 - fy)), int(im.size[0] / 2 * (1 + fx)), int(im.size[1] / 2 * (1 + fy)))
    a, b = _grey(view, box(view, ox / txv, oy / tyv)), _grey(thumb, box(thumb, ox / txf, oy / tyf)); a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def zoom(view, thumb, fv, ff):
    """How much the view is zoomed against what its settings say: the factor z for which 'the view really spans fv / z' matches
    the photo best (1.0 = right; 1.6 = the 27 Nov bug). Returns (z, match at z, match at 1.0)."""
    t = np.tan(np.radians(fv) / 2); m = {z: match(view, thumb, float(np.degrees(2 * np.arctan(t / z))), ff) for z in SCALES}
    z = max(m, key=m.get); return z, round(m[z], 3), round(m[1.0], 3)


def zoom_wrong(view, thumb, fv, ff, lo=0.85, hi=1.2, clear=0.08):
    """A message when the view is clearly zoomed (best factor outside lo..hi AND matching clearly better than 1.0), else None."""
    z, mz, m1 = zoom(view, thumb, fv, ff)
    return f'shown {z:.2f}x zoomed (match {mz:.2f} against {m1:.2f} at 1.0x): wrong field of view' if (z < lo or z > hi) and mz - m1 > clear else None


def off_direction(position, target, forward):
    """Degrees between where the spot should look (position -> target) and where the camera looks (its forward vector)."""
    a = np.asarray(target, float) - np.asarray(position, float); b = np.asarray(forward, float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)); return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
