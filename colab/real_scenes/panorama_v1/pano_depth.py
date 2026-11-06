"""Multi-view-consistent panorama depth from PanoVGGT (CVPR 2026, MIT; trained on many panorama worlds).

MoGe-2 sees one perspective view at a time, so its scale wanders between neighbouring views
(log-std ~0.25 on the Paris street) and the splats average the disagreement into blur.
PanoVGGT reads several equirectangular panoramas jointly and predicts depth that is consistent
across them. We run it on overlapping windows of neighbouring panoramas, keep for each panorama
the window where it sits most centrally, and resample its radial depth into every rig view
(z-depth = radial depth x cos(angle to the view axis)). Scale is then fitted to SfM tracks exactly
as for MoGe, so both depth sources are interchangeable in depth_memory.build_memory.

Depth is indexed by panorama pixel, so no axis convention of PanoVGGT itself is involved: the
same pycolmap spherical mapping that rendered the rig views maps view pixels to panorama pixels.
T4: ~1B-parameter aggregator, fp16 autocast, 8 panoramas per window at 518x1036.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = "https://github.com/YijingGuo-June/PanoVGGT"
WEIGHTS = "https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt"


def setup(work: Path, log=print) -> Path:
    work = Path(work)
    code = work / "PanoVGGT"
    if not code.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(code)], check=True)
    ck = work / "panovggt_model.pt"
    if not ck.exists():
        from huggingface_hub import hf_hub_download

        p = hf_hub_download("YijingGuo/PanoVGGT", "model.pt", local_dir=str(work / "hf"))
        Path(p).rename(ck)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "omegaconf", "einops", "kornia", "hydra-core"],
                   check=False)
    log(f"[panovggt] code {code}, weights {ck.stat().st_size / 1e9:.2f} GB")
    return code


def load(work: Path, device: str = "cuda"):
    import torch
    from omegaconf import OmegaConf

    code = setup(work)
    sys.path.insert(0, str(code))
    from panovggt.models.panovggt_model import PanoVGGTModel

    cfg = OmegaConf.load(code / "training" / "config" / "default.yaml")
    OmegaConf.resolve(cfg)
    mc = cfg.model
    model = PanoVGGTModel(img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
                          enable_camera=mc.enable_camera, enable_depth=mc.enable_depth, enable_point=mc.enable_point,
                          aggregator=OmegaConf.to_container(mc.aggregator, resolve=True))
    ck = torch.load(Path(work) / "panovggt_model.pt", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ck:
            ck = ck[key]
            break
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck.items()}, strict=False)
    return model.to(device).eval()


def radial_depths(model, pano_paths: list[Path], device: str = "cuda") -> np.ndarray:
    """[N, 518, 1036] radial depth (model units) for panoramas processed jointly."""
    import torch
    from PIL import Image

    imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((1036, 518), Image.BICUBIC)) for p in pano_paths])
    x = torch.tensor(imgs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)[None] / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        preds = model(x)
    if preds.get("local_points") is not None:
        r = torch.linalg.norm(preds["local_points"].float(), dim=-1)
    elif preds.get("depth") is not None:
        r = preds["depth"].float()
        if r.ndim == 5:
            r = r[..., 0]
    else:
        cp = preds["camera_poses"][..., :3, 3][:, :, None, None, :]
        r = torch.linalg.norm(preds["world_points"].float() - cp, dim=-1)
    return r[0].cpu().numpy()


def windows(n: int, size: int = 8, overlap: int = 3) -> list[list[int]]:
    if n <= size:
        return [list(range(n))]
    step = max(1, size - overlap)
    starts = list(range(0, n - size + 1, step))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [list(range(s, s + size)) for s in starts]


def pano_radial_depths(model, pano_paths: list[Path], device: str = "cuda", size: int = 8,
                       overlap: int = 3, log=print) -> dict[str, np.ndarray]:
    """Per panorama stem: radial depth from the window where that panorama is most central."""
    best: dict[int, tuple[float, np.ndarray]] = {}
    for w in windows(len(pano_paths), size, overlap):
        r = radial_depths(model, [pano_paths[i] for i in w], device)
        mid = (len(w) - 1) / 2
        for j, i in enumerate(w):
            centrality = -abs(j - mid)
            if i not in best or centrality > best[i][0]:
                best[i] = (centrality, r[j])
        log(f"[panovggt] window {w[0]}..{w[-1]} done")
    return {Path(pano_paths[i]).stem: best[i][1] for i in sorted(best)}


def view_zdepth(radial_erp: np.ndarray, cam_from_pano: np.ndarray, K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resample a panorama's radial depth into a perspective rig view as z-depth (pycolmap axes)."""
    import cv2

    H, W = radial_erp.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ray = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    d = ray @ cam_from_pano  # camera -> panorama direction
    yaw = np.arctan2(d[..., 0], d[..., 2])
    pitch = -np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    u = ((1 + yaw / math.pi) / 2 * W - 0.5).astype(np.float32)
    v = ((1 - pitch * 2 / math.pi) / 2 * H - 0.5).astype(np.float32)
    r = cv2.remap(radial_erp.astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return (r * ray[..., 2]).astype(np.float32)
