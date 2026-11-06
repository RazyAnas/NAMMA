"""Masks for moving objects and the capture car (commercial-safe models).

Dynamic objects: Grounding DINO tiny (IDEA-Research, Apache-2.0) finds boxes
for "car. person. bus. truck. bicycle. motorcycle." and SAM 2.1 (Meta,
Apache-2.0) turns each box into a pixel mask. Both load through
transformers, so nothing is compiled and nothing is gated.
Capture car: views pitched down have the car body / tripod at the bottom of
the panorama; pixels whose ray points more than nadir_deg below the horizon
are masked.

Output per view: PNG, 255 = use pixel, 0 = ignore (written to cam["mask"]).
T4 estimate: ~0.4 s/view (DINO-tiny + SAM2-large fp16) -> 480 views ~3-4 min.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PROMPT = "car. person. bus. truck. bicycle. motorcycle. van."


def static_vehicle_mask(pano_dir, max_panos: int = 60, width: int = 1024,
                        std_thresh: float = 6.0, min_elev_deg: float = -5.0) -> np.ndarray:
    """ERP mask (True = keep) removing pixels that never change along the route.

    The capture vehicle (bonnet, roof rack, blur patch, black no-data band) is
    fixed in the panorama frame while the world moves past it, so its pixels
    have almost no variation across the sequence. Only pixels below the
    horizon are considered, so a uniform sky is never masked.
    """
    import cv2
    from pathlib import Path

    files = sorted(p for p in Path(pano_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) < 5:
        return np.ones((width // 2, width), bool)
    step = max(1, len(files) // max_panos)
    stack = []
    for f in files[::step]:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        stack.append(cv2.resize(im, (width, width // 2), interpolation=cv2.INTER_AREA).astype(np.float32))
    stack = np.stack(stack)
    std = stack.std(0)
    h = width // 2
    elev = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    below = (elev < min_elev_deg)[:, None].repeat(width, 1)
    static = (std < std_thresh) & below
    static = cv2.morphologyEx(static.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # keep only big connected blobs touching the bottom rows (the vehicle), not random flat road
    n, lab, stats, _ = cv2.connectedComponentsWithStats(static, 8)
    keep = np.ones((h, width), bool)
    for k in range(1, n):
        x, y, w_, h_, area = stats[k]
        if y + h_ >= h - 2 and area > 0.002 * h * width:
            keep &= lab != k
    keep = cv2.erode(keep.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return keep


def erp_mask_to_view(erp_keep: np.ndarray, cam_from_pano: np.ndarray, K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Sample an ERP mask into a perspective view (pycolmap panorama conventions)."""
    H, W = erp_keep.shape
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    d = d @ cam_from_pano  # R^T d
    yaw = np.arctan2(d[..., 0], d[..., 2])
    pitch = -np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    u = ((1 + yaw / np.pi) / 2 * W).astype(int) % W
    v = np.clip(((1 - pitch * 2 / np.pi) / 2 * H).astype(int), 0, H - 1)
    return erp_keep[v, u]


def nadir_mask(K: np.ndarray, w2c: np.ndarray, up_world: np.ndarray, w: int, h: int,
               nadir_deg: float = 55.0) -> np.ndarray:
    """True = keep. Masks rays pointing more than nadir_deg below the horizon."""
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    d = d @ w2c[:3, :3]  # camera -> world directions
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    elev = np.degrees(np.arcsin(np.clip(d @ up_world, -1, 1)))
    return elev > -nadir_deg


class DynamicMasker:
    def __init__(self, device: str = "cuda", box_threshold: float = 0.3):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)

        self.device = device
        self.box_threshold = box_threshold
        self.det_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny").to(device).eval()
        self.sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large",
                                             torch_dtype=torch.float32).to(device).eval()

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """True = keep pixel (no moving object)."""
        import torch
        from PIL import Image

        im = Image.fromarray(image_rgb)
        with torch.no_grad():
            inputs = self.det_proc(images=im, text=PROMPT, return_tensors="pt").to(self.device)
            out = self.det(**inputs)
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.box_threshold, text_threshold=0.25,
                target_sizes=[im.size[::-1]])[0]
            boxes = res["boxes"].cpu().numpy().tolist()
            keep = np.ones(image_rgb.shape[:2], dtype=bool)
            if not boxes:
                return keep
            sin = self.sam_proc(images=im, input_boxes=[boxes], return_tensors="pt").to(self.device)
            so = self.sam(**sin, multimask_output=False)
            masks = self.sam_proc.post_process_masks(so.pred_masks.cpu(), sin["original_sizes"])[0]
            m = masks.numpy().reshape(len(boxes), *image_rgb.shape[:2]).any(0)
        # grow a little: motion blur and shadows sit just outside the outline
        import cv2

        m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return ~m


def build_masks(cameras: list[dict], out_dir: Path, up_world: np.ndarray, long_side: int = 960,
                device: str = "cuda", dynamic: bool = True, nadir_deg: float = 55.0, log=print,
                vehicle_erp: np.ndarray | None = None, rotations=None) -> dict:
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masker, masker_error = None, None
    if dynamic:
        try:
            masker = DynamicMasker(device)
        except Exception as exc:  # noqa: BLE001 - keep vehicle/nadir masks even if the detector fails
            masker_error = f"{type(exc).__name__}: {exc}"[:400]
            log(f"[masks] dynamic masker unavailable, vehicle/nadir masks only: {masker_error}")
    masked_frac = []
    for i, cam in enumerate(cameras):
        im = Image.open(cam["path"]).convert("RGB")
        s = min(1.0, long_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        rgb = np.asarray(im)
        K = np.array(cam["K"], dtype=np.float64).copy()
        K[:2] *= s
        keep = nadir_mask(K, np.asarray(cam["w2c"]), up_world, rgb.shape[1], rgb.shape[0], nadir_deg)
        if vehicle_erp is not None and rotations is not None and cam["name"].startswith("pano_camera"):
            k = int(cam["name"].split("/")[0].replace("pano_camera", ""))
            keep &= erp_mask_to_view(vehicle_erp, np.asarray(rotations[k]), K, rgb.shape[1], rgb.shape[0])
        if masker is not None:
            keep &= masker(rgb)
        stem = cam["name"].replace("/", "__").rsplit(".", 1)[0]
        path = out_dir / f"{stem}.png"
        Image.fromarray((keep * 255).astype(np.uint8)).save(path)
        cam["mask"] = str(path)
        masked_frac.append(1 - keep.mean())
        if i % 50 == 0:
            log(f"[masks] {i}/{len(cameras)} masked {masked_frac[-1]:.3f}")
    return {"views": len(cameras), "masked_fraction_mean": float(np.mean(masked_frac)),
            "dynamic_masker": None if masker is None else "grounding-dino-tiny + sam2.1-hiera-large",
            "dynamic_masker_error": masker_error}
