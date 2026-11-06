"""Dense metric depth memory for panorama scenes (H3), plus free-space carving.

Steps (all on the perspective views the splats are trained on):
  1. MoGe-2 (MIT, Ruicheng/moge-2-vitl-normal) metric z-depth per view with
     the known 90 deg field of view. Its validity mask marks sky / infinitely
     far pixels; those become the sky mask.
  2. Per-view scale alignment to sparse reference points that are known to be
     visible in that panorama (SfM tracks or openMVG observations): robust
     median of z_ref / z_moge over inliers.
  3. Fusion: back-project every aligned view (strided) into one coloured world
     point cloud = the scene's geometric memory and the splat initialisation.
  4. Carving: rays from real views mark voxels in front of the measured surface
     as empty; a voxel is empty when >= 2 views see through it and no view
     sees a surface in it. Splats whose centre falls in empty space are
     floaters by construction and are deleted (or killed during training).

T4 cost (estimate): MoGe-2 ViT-L at 512 px ~0.15 s/view fp16 -> 480 views
~1.5 min plus model load; carving 480 views at stride 8 ~1 min on GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ MoGe-2
def load_moge(device: str = "cuda"):
    import torch
    from moge.model.v2 import MoGeModel

    # fp32 weights; MoGe runs its own fp16 autocast internally where safe (as in World v1)
    return MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()


def moge_view_depth(model, image_rgb: np.ndarray, hfov_deg: float, device: str = "cuda",
                    level: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Metric z-depth [H, W] and validity mask (False = sky / unknown)."""
    import torch

    x = torch.tensor(image_rgb, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = model.infer(x, fov_x=float(hfov_deg), resolution_level=level)
    depth = out["depth"].float().cpu().numpy()
    mask = out["mask"].cpu().numpy().astype(bool)
    depth[~mask] = 0.0
    return depth.astype(np.float32), mask


# --------------------------------------------------------------- alignment
def project(points_xyz: np.ndarray, K: np.ndarray, w2c: np.ndarray, w: int, h: int):
    X = points_xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    front = z > 1e-6
    uv = (X[front] @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    idx = np.nonzero(front)[0][ok]
    return idx, uv[ok], z[front][ok]


def robust_scale(z_ref: np.ndarray, z_est: np.ndarray, iters: int = 3, tol: float = 0.15):
    """Scale s with z_ref ~= s * z_est; returns (s, inlier_ratio)."""
    good = (z_est > 0) & (z_ref > 0)
    if good.sum() < 8:
        return None, 0.0
    r = z_ref[good] / z_est[good]
    s = float(np.median(r))
    inl = np.ones_like(r, dtype=bool)
    for _ in range(iters):
        inl = np.abs(np.log(r / s)) < tol
        if inl.sum() < 8:
            break
        s = float(np.median(r[inl]))
    return s, float(inl.mean())


def align_view(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray, ref_xyz: np.ndarray):
    h, w = depth.shape
    idx, uv, z = project(ref_xyz, K, w2c, w, h)
    if len(idx) == 0:
        return None, 0.0, 0
    zd = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    s, inl = robust_scale(z, zd)
    return s, inl, int(len(idx))


# ------------------------------------------------------------------ fusion
def backproject(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                stride: int = 4, keep: np.ndarray | None = None) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    d = depth[ys, xs]
    ok = d > 0
    if keep is not None:
        ok &= keep[ys, xs]
    xs, ys, d = xs[ok], ys[ok], d[ok]
    cam = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * d, (ys + 0.5 - K[1, 2]) / K[1, 1] * d, d], 1)
    R, t = w2c[:3, :3], w2c[:3, 3]
    world = (cam - t) @ R
    col = rgb[ys, xs].astype(np.float32) / 255.0
    return np.concatenate([world, col], 1).astype(np.float32)


def depth_edge_mask(depth: np.ndarray, rtol: float = 0.04) -> np.ndarray:
    """True where depth is smooth (drop flying pixels at depth jumps)."""
    d = depth.copy()
    d[d <= 0] = np.nan
    dx = np.abs(np.diff(d, axis=1, append=d[:, -1:]))
    dy = np.abs(np.diff(d, axis=0, append=d[-1:, :]))
    rel = np.nanmax(np.stack([dx, dy]), axis=0) / np.where(np.isnan(d), 1, d)
    return np.nan_to_num(rel, nan=1.0) < rtol


# ----------------------------------------------------------------- carving
class CarveGrid:
    """Voxel free-space grid in a given (normalised) frame."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, max_voxels: int = 48_000_000, device: str = "cuda",
                 dims=None):
        import torch

        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)
        ext = np.maximum(self.hi - self.lo, 1e-6)
        if dims is None:
            vox = float((np.prod(ext) / max_voxels) ** (1 / 3))
            dims = np.maximum(1, np.ceil(ext / vox)).astype(int)
        self.dims = np.asarray(dims, dtype=int)
        self.size = (ext / self.dims).astype(np.float32)
        self.res = int(self.dims.max())
        self.device = device
        shape = tuple(int(d) for d in self.dims)
        self.empty = torch.zeros(shape, dtype=torch.int16, device=device)
        self.surface = torch.zeros(shape, dtype=torch.int16, device=device)

    def _index(self, pts):
        import torch

        lo = torch.tensor(self.lo, device=pts.device)
        size = torch.tensor(self.size, device=pts.device)
        ijk = torch.floor((pts - lo) / size).long()
        dims = torch.tensor(self.dims, device=pts.device)
        ok = ((ijk >= 0) & (ijk < dims)).all(-1)
        return ijk, ok

    def add_view(self, depth, K, w2c, stride: int = 8, margin: float = 0.92, max_samples: int = 256):
        """Mark voxels along rays up to margin*depth as seen-empty, voxel at depth as surface."""
        import torch

        dev = self.device
        h, w = depth.shape
        ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        d = depth[ys, xs]
        ok = d > 0
        if not ok.any():
            return
        xs, ys, d = xs[ok], ys[ok], d[ok]
        R = torch.tensor(w2c[:3, :3], dtype=torch.float32, device=dev)
        t = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=dev)
        dirs_cam = torch.tensor(np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1],
                                          np.ones_like(d)], 1), dtype=torch.float32, device=dev)
        origin = -(R.T @ t)
        dirs = dirs_cam @ R  # world direction for unit z
        dz = torch.tensor(d, dtype=torch.float32, device=dev)
        step = float(np.min(self.size))
        # surface voxels
        surf = origin + dirs * dz[:, None]
        ijk, inside = self._index(surf)
        ijk = ijk[inside]
        self.surface.index_put_((ijk[:, 0], ijk[:, 1], ijk[:, 2]),
                                torch.ones(len(ijk), dtype=torch.int16, device=dev), accumulate=True)
        # empty samples (chunked)
        ray_len = dz * torch.linalg.norm(dirs, dim=-1) * margin
        n = torch.clamp((ray_len / step).long(), 1, max_samples)
        mx = int(n.max())
        ts = torch.arange(mx, device=dev, dtype=torch.float32)
        for c0 in range(0, len(dz), 4096):
            sl = slice(c0, c0 + 4096)
            frac = (ts[None, :] + 0.5) / n[sl, None].float()
            valid = ts[None, :] < n[sl, None]
            pts = origin + dirs[sl, None, :] * (dz[sl, None, None] * margin * frac[..., None])
            pts = pts[valid]
            ijk, inside = self._index(pts)
            ijk = ijk[inside]
            # count each voxel once per view
            d1, d2 = int(self.dims[1]), int(self.dims[2])
            flat = torch.unique(ijk[:, 0] * d1 * d2 + ijk[:, 1] * d2 + ijk[:, 2])
            i0 = flat // (d1 * d2)
            i1 = (flat // d2) % d1
            i2 = flat % d2
            self.empty.index_put_((i0, i1, i2), torch.ones(len(flat), dtype=torch.int16, device=dev),
                                  accumulate=True)

    def is_empty(self, pts, min_views: int = 2):
        """Bool per point: inside carved free space (>= min_views see through, no surface)."""
        import torch

        ijk, inside = self._index(pts)
        out = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
        ii = ijk[inside]
        e = self.empty[ii[:, 0], ii[:, 1], ii[:, 2]] >= min_views
        s = self.surface[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
        out[inside] = e & ~s
        return out

    def save(self, path: Path):
        import torch

        torch.save({"lo": self.lo, "hi": self.hi, "dims": self.dims,
                    "empty": self.empty.cpu(), "surface": self.surface.cpu()}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cuda"):
        import torch

        d = torch.load(path, map_location=device, weights_only=False)
        g = cls(d["lo"], d["hi"], device=device, dims=d["dims"])
        g.empty, g.surface = d["empty"].to(device), d["surface"].to(device)
        return g


def scene_io_centre(cam) -> np.ndarray:
    w2c = np.asarray(cam["w2c"], dtype=np.float64)
    return -w2c[:3, :3].T @ w2c[:3, 3]


# ----------------------------------------------------------- orchestration
def build_memory(cameras: list[dict], ref_points_by_pano: dict[str, np.ndarray], out_dir: Path,
                 long_side: int = 960, device: str = "cuda", fuse_stride: int = 4,
                 max_points: int = 1_500_000, log=print) -> dict:
    """Run MoGe-2 on every view, align, write per-view depth/sky files, fuse points.

    Adds "depth" (npy, metric world units, 0 = unknown) and "sky_mask" (png,
    255 = not sky) to each camera dict in place. Returns a report.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "sky").mkdir(parents=True, exist_ok=True)
    model = load_moge(device)
    scales, inliers, fused = [], [], []
    failed = 0
    for i, cam in enumerate(cameras):
        im = Image.open(cam["path"]).convert("RGB")
        s = min(1.0, long_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        rgb = np.asarray(im)
        K = np.array(cam["K"], dtype=np.float64).copy()
        K[:2] *= s
        hfov = float(np.degrees(2 * np.arctan(rgb.shape[1] / (2 * K[0, 0]))))
        depth, valid = moge_view_depth(model, rgb, hfov, device)
        ref = ref_points_by_pano.get(cam["pano"])
        scale, inl, n = (None, 0.0, 0) if ref is None else align_view(depth, K, cam["w2c"], ref)
        if scale is None or inl < 0.3:
            failed += 1
            depth = np.zeros_like(depth)
        else:
            depth = depth * scale
            scales.append(scale)
            inliers.append(inl)
        stem = cam["name"].replace("/", "__").rsplit(".", 1)[0]
        np.save(out_dir / "depth" / f"{stem}.npy", depth.astype(np.float32))
        Image.fromarray((valid * 255).astype(np.uint8)).save(out_dir / "sky" / f"{stem}.png")
        cam["depth"] = str(out_dir / "depth" / f"{stem}.npy")
        cam["sky_mask"] = str(out_dir / "sky" / f"{stem}.png")
        if cam["split"] == "train" and depth.any():
            fused.append(backproject(depth, rgb, K, cam["w2c"], fuse_stride, depth_edge_mask(depth)))
        if i % 50 == 0:
            log(f"[depth] {i}/{len(cameras)} views, last scale {scale} inliers {inl:.2f} refs {n}")
    pts = np.concatenate(fused) if fused else np.zeros((0, 6), np.float32)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    sc = np.asarray(scales)
    report = {"views": len(cameras), "aligned": len(scales), "failed": failed,
              "scale_median": float(np.median(sc)) if len(sc) else None,
              "scale_spread_log_std": float(np.std(np.log(sc))) if len(sc) > 1 else None,
              "inlier_median": float(np.median(inliers)) if inliers else None,
              "fused_points": int(len(pts))}
    np.save(out_dir / "fused_points.npy", pts)
    (out_dir / "depth_report.json").write_text(json.dumps(report, indent=2))
    del model
    return report


def build_carve_grid(cameras: list[dict], centre, scale, out_path: Path, points: np.ndarray | None = None,
                     device: str = "cuda") -> CarveGrid:
    """Grid in the trainer's normalised frame, bounding the scene's points (p1-p99, padded)."""
    C = np.stack([(scene_io_centre(c) - centre) * scale for c in cameras if c["split"] == "train"])
    if points is not None and len(points):
        P = (points[:, :3] - centre) * scale
        lo = np.minimum(np.percentile(P, 1, 0), C.min(0))
        hi = np.maximum(np.percentile(P, 99, 0), C.max(0))
    else:
        lo, hi = C.min(0) - 1.0, C.max(0) + 1.0
    pad = 0.05 * (hi - lo) + 1e-3
    grid = CarveGrid(lo - pad, hi + pad, device=device)
    for cam in cameras:
        if cam["split"] != "train" or not cam.get("depth"):
            continue
        depth = np.load(cam["depth"])
        if not depth.any():
            continue
        h, w = depth.shape
        K = np.array(cam["K"], dtype=np.float64).copy()
        s = w / cam["width"]
        K[:2] *= s
        w2c = np.asarray(cam["w2c"], dtype=np.float64)
        Vn = np.eye(4)
        Vn[:3, :3] = w2c[:3, :3]
        Vn[:3, 3] = (w2c[:3, :3] @ centre + w2c[:3, 3]) * scale
        grid.add_view(depth * scale, K, Vn)
    grid.save(out_path)
    return grid
