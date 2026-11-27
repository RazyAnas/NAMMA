"""The live check's new view tests (scripts/site_look/view_check.py): a zoomed spot view and a wrong direction are caught."""
import os, sys
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'site_look')))
import view_check                                                                         # noqa: E402

F = 600.0                                                                                 # the scene is a flat picture seen with this focal length


def scene():
    rng = np.random.default_rng(3); a = rng.random((90, 160)) * 255
    return Image.fromarray(a.astype(np.uint8)).resize((3200, 1800), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6)).convert('RGB')


def look(sc, fov, w, h):
    """What a camera with this horizontal fov and window shape sees of the scene (centre crop, resized)."""
    hw = F * np.tan(np.radians(fov) / 2); hh = hw * h / w; cx, cy = sc.size[0] / 2, sc.size[1] / 2
    return sc.crop((int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))).resize((w, h))


def test_a_right_view_is_not_called_zoomed():
    sc = scene(); photo = look(sc, 71.4, 240, 135)
    for w, h in ((1280, 720), (1024, 768)):
        z, mz, m1 = view_check.zoom(look(sc, 71.4, w, h), photo, 71.4, 71.4)
        assert z == 1.0 and m1 > 0.9 and view_check.zoom_wrong(look(sc, 71.4, w, h), photo, 71.4, 71.4) is None


def test_the_bug_of_27_nov_is_caught():
    # settings said 43.6 deg "horizontal" and the photo's field was taken as the same number: the old check compared the same crop
    # of both and passed. The frame is really 71.4 deg wide, the view really showed 43.6 deg: zoomed 1.6x
    sc = scene(); photo = look(sc, 71.4, 240, 135); view = look(sc, 43.6, 1280, 720)
    z, mz, m1 = view_check.zoom(view, photo, 71.4, 71.4)
    assert 1.4 <= z <= 1.8 and mz > m1 + 0.2
    assert 'zoomed' in view_check.zoom_wrong(view, photo, 71.4, 71.4)


def test_direction():
    assert view_check.off_direction([0, 1.35, 0], [1, 1.35, 0], [1, 0, 0]) < 1e-3
    assert abs(view_check.off_direction([0, 1.35, 0], [1, 1.35, 0], [0, 0, 1]) - 90) < 1e-3
    assert abs(view_check.off_direction([0, 0, 0], [1, 0, 0], [np.cos(np.radians(4)), 0, np.sin(np.radians(4))]) - 4) < 1e-3
