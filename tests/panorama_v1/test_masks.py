import numpy as np
from PIL import Image

import masks
import scene_io


def test_static_vehicle_mask_finds_fixed_bottom_blob(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(8):
        img = rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)
        img[200:, 150:350] = 90  # the capture car: identical in every panorama
        Image.fromarray(img).save(tmp_path / f"{i:04d}.jpg", quality=98)
    keep = masks.static_vehicle_mask(tmp_path, width=512, std_thresh=8.0)
    assert not keep[230, 250]          # car masked
    assert keep[230, 50] and keep[60, 250]  # moving road and sky-side kept


def test_erp_mask_to_view_front_and_back():
    erp = np.ones((100, 200), bool)
    erp[:, 90:110] = False  # a strip straight ahead (longitude 0)
    K = np.array([[20.0, 0, 20], [0, 20.0, 20], [0, 0, 1]])
    front = masks.erp_mask_to_view(erp, scene_io.legacy_crop_rotation(0), K, 40, 40)
    back = masks.erp_mask_to_view(erp, scene_io.legacy_crop_rotation(180), K, 40, 40)
    assert not front[20, 20] and back[20, 20]


def test_detected_vehicle_mask_projects_downward_view_back_to_panorama(tmp_path, monkeypatch):
    # fake detector: masks a disc in the centre of the downward view (= straight below the camera)
    class Fake:
        def __init__(self, *a, **k):
            pass

        def __call__(self, img, prompt=None, **kw):
            h, w = img.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            return ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) > (0.2 * w) ** 2

    monkeypatch.setattr(masks, "DynamicMasker", Fake)
    for i in range(4):
        Image.fromarray(np.full((256, 512, 3), 128, np.uint8)).save(tmp_path / f"{i}.jpg")
    keep = masks.detected_vehicle_mask(tmp_path, device="cpu", samples=4, width=512)
    assert keep.shape == (256, 512)
    assert not keep[-1, 100] and not keep[-3, 400]   # nadir rows masked at every azimuth
    assert keep[0, 100] and keep[128, 256]           # top and horizon kept
