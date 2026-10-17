import cv2
import numpy as np

from app.services.frame_quality import measure_frame, select_sharpest, summarize


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = (rng.random((240, 320, 3)) * 200 + 20).astype(np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_sharpness_ignores_exposure_but_detects_blur():
    scene = _scene()
    darker = (scene.astype(np.float32) * 0.5).astype(np.uint8)
    blurred = cv2.blur(scene, (9, 1))
    sharp_score = measure_frame(scene).sharpness
    assert abs(measure_frame(darker).sharpness - sharp_score) < 0.05 * sharp_score
    assert measure_frame(blurred).sharpness < 0.75 * sharp_score


def test_select_sharpest_keeps_one_frame_per_window():
    scene = _scene()
    frames = [cv2.blur(scene, (k, 1)) for k in (7, 1, 9, 5, 11, 3, 1)]
    scores = [measure_frame(image, f"{i}.jpg") for i, image in enumerate(frames)]
    selected = select_sharpest(scores, 3)
    assert [item.name for item in selected] == ["1.jpg", "5.jpg", "6.jpg"]


def test_summary_flags_lighting_drift():
    scene = _scene()
    frames = [scene, (scene * 0.25).astype(np.uint8)]
    scores = [measure_frame(image, f"{i}.jpg") for i, image in enumerate(frames)]
    report = summarize(scores, scores)
    assert any("Brightness changed" in warning for warning in report["warnings"])
