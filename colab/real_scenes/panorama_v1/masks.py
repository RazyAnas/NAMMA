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
                device: str = "cuda", dynamic: bool = True, nadir_deg: float = 55.0, log=print) -> dict:
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masker = DynamicMasker(device) if dynamic else None
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
        if masker is not None:
            keep &= masker(rgb)
        stem = cam["name"].replace("/", "__").rsplit(".", 1)[0]
        path = out_dir / f"{stem}.png"
        Image.fromarray((keep * 255).astype(np.uint8)).save(path)
        cam["mask"] = str(path)
        masked_frac.append(1 - keep.mean())
        if i % 50 == 0:
            log(f"[masks] {i}/{len(cameras)} masked {masked_frac[-1]:.3f}")
    return {"views": len(cameras), "masked_fraction_mean": float(np.mean(masked_frac))}
