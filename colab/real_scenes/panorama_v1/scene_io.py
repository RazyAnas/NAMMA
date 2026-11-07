"""One scene format for every panorama experiment and the final pipeline.

A scene directory holds:
  scene.json   cameras (one entry per perspective view) and metadata
  points.npy   float32 [N, 6] initial points: xyz, rgb in 0..1
  images/...   the perspective views referenced by scene.json

Each camera entry: {"name", "path", "pano", "width", "height",
"K": 3x3, "w2c": 4x4 (OpenCV, world->camera), "split": "train" | "val"}.

Converters exist for a pycolmap Reconstruction (rig-aware panorama SfM or the
legacy per-crop SfM) and for the openMVG ground truth shipped with GGPS.
Holdout is by whole panorama, so no validation view shares a camera centre
with a training view.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

SCENE_VERSION = 1


# ----------------------------------------------------------------- holdout
def holdout_panos(pano_ids: Sequence[str], every: int = 8, offset: int = 4) -> set[str]:
    """Hold out whole panoramas: the ones at route index offset, offset+every, ...

    The first and last panoramas are never held out, so evaluation views
    always lie between trained positions (interpolation, not extrapolation).
    """
    ordered = sorted(set(pano_ids))
    if len(ordered) < 3:
        return set()
    picked = set()
    for i, pano in enumerate(ordered):
        if 0 < i < len(ordered) - 1 and i % every == offset % every:
            picked.add(pano)
    return picked


def assign_splits(cameras: list[dict], every: int = 8, offset: int = 4) -> list[dict]:
    val = holdout_panos([c["pano"] for c in cameras], every, offset)
    for cam in cameras:
        cam["split"] = "val" if cam["pano"] in val else "train"
    return cameras


# ------------------------------------------------------------ geometry utils
def camera_centre(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:3, :3], w2c[:3, 3]
    return -R.T @ t


def world_up_from_cameras(cameras: Iterable[dict]) -> np.ndarray:
    """Average camera 'up' (-y in OpenCV camera axes) mapped to world.

    Panorama rigs contain views pitched up and down symmetrically, so the
    average is the panorama's up axis even though single views are tilted.
    """
    ups = []
    for cam in cameras:
        R = np.asarray(cam["w2c"], dtype=np.float64)[:3, :3]
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))
    up = np.mean(ups, axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        raise ValueError("Cannot infer a world up direction from the cameras.")
    return up / norm


def pano_centres(cameras: Iterable[dict]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for cam in cameras:
        grouped.setdefault(cam["pano"], []).append(camera_centre(np.asarray(cam["w2c"])))
    return {k: np.mean(v, axis=0) for k, v in grouped.items()}


def within_pano_spread(cameras: Iterable[dict]) -> float:
    """Largest distance between a view's centre and its panorama's mean centre."""
    grouped: dict[str, list[np.ndarray]] = {}
    for cam in cameras:
        grouped.setdefault(cam["pano"], []).append(camera_centre(np.asarray(cam["w2c"])))
    worst = 0.0
    for centres in grouped.values():
        c = np.stack(centres)
        worst = max(worst, float(np.max(np.linalg.norm(c - c.mean(0), axis=1))))
    return worst


# ----------------------------------------------------------------- writing
def write_scene(scene_dir: Path, cameras: list[dict], points: np.ndarray,
                meta: dict[str, Any] | None = None) -> Path:
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 6:
        raise ValueError("points must be [N, 6] (xyz + rgb 0..1).")
    if len(pts) < 100:
        raise ValueError(f"Only {len(pts)} initial points; refusing to train.")
    for cam in cameras:
        for key in ("name", "path", "pano", "width", "height", "K", "w2c", "split"):
            if key not in cam:
                raise ValueError(f"Camera entry missing {key}: {cam.get('name')}")
    body = {
        "version": SCENE_VERSION,
        "meta": meta or {},
        "within_pano_centre_spread": within_pano_spread(cameras),
        "cameras": [
            {**cam, "K": np.asarray(cam["K"]).tolist(), "w2c": np.asarray(cam["w2c"]).tolist()}
            for cam in cameras
        ],
    }
    (scene_dir / "scene.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    np.save(scene_dir / "points.npy", pts)
    return scene_dir / "scene.json"


def read_scene(scene_dir: Path) -> tuple[list[dict], np.ndarray, dict]:
    body = json.loads((Path(scene_dir) / "scene.json").read_text(encoding="utf-8"))
    if body.get("version") != SCENE_VERSION:
        raise ValueError("Unsupported scene.json version.")
    cams = body["cameras"]
    for cam in cams:
        cam["K"] = np.asarray(cam["K"], dtype=np.float64)
        cam["w2c"] = np.asarray(cam["w2c"], dtype=np.float64)
    return cams, np.load(Path(scene_dir) / "points.npy"), body.get("meta", {})


# ------------------------------------------------------- pycolmap converter
def _intrinsics(camera) -> np.ndarray:
    model = str(camera.model_name)
    p = np.asarray(camera.params, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f"Unsupported camera model for training: {model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _pose_matrix(image) -> np.ndarray:
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    w2c = np.eye(4)
    w2c[:3, :] = np.asarray(pose.matrix())
    return w2c


def from_pycolmap(rec, image_root: Path, pano_of: Callable[[str], str],
                  max_points: int = 2_000_000) -> tuple[list[dict], np.ndarray]:
    """Convert a registered pycolmap Reconstruction into scene cameras + points."""
    cameras = []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        cam = rec.cameras[image.camera_id]
        cameras.append({
            "name": image.name,
            "path": str(Path(image_root) / image.name),
            "pano": pano_of(image.name),
            "width": int(cam.width),
            "height": int(cam.height),
            "K": _intrinsics(cam),
            "w2c": _pose_matrix(image),
        })
    cameras.sort(key=lambda c: c["name"])
    pts = [np.r_[p.xyz, np.asarray(p.color, dtype=np.float64) / 255.0]
           for p in rec.points3D.values()]
    points = np.asarray(pts, dtype=np.float32)
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points = points[keep]
    return assign_splits(cameras), points


_LEGACY = re.compile(r"^(p\d+)_v\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_RIG = re.compile(r"^pano_camera\d+/(.+)$")


def pano_of_name(name: str) -> str:
    m = _RIG.match(name)
    if m:
        return Path(m.group(1)).stem
    m = _LEGACY.match(Path(name).name)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot tell which panorama view {name!r} belongs to.")


# ---------------------------------------------------------- openMVG (GGPS)
def _openmvg_items(node) -> list:
    return node if isinstance(node, list) else []


def read_openmvg_poses(sfm_data: dict) -> dict[str, np.ndarray]:
    """filename -> pano_from_world 4x4 from an openMVG sfm_data.json.

    openMVG Pose3 stores rotation R and centre C with X_cam = R (X - C).
    Spherical intrinsics use x right, y down, z forward (longitude 0 at the
    image centre), the same axes as pycolmap's panorama code.
    """
    poses = {}
    for item in _openmvg_items(sfm_data.get("extrinsics")):
        value = item["value"]
        R = np.asarray(value["rotation"], dtype=np.float64)
        C = np.asarray(value["center"], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ C
        poses[int(item["key"])] = T
    out = {}
    for item in _openmvg_items(sfm_data.get("views")):
        data = item["value"]["ptr_wrapper"]["data"]
        pose_id = data.get("id_pose")
        if pose_id in poses:
            out[Path(data["filename"]).name] = poses[pose_id]
    if not out:
        raise ValueError("No posed views found in sfm_data.json.")
    return out


def rig_views_from_poses(pano_poses: dict[str, np.ndarray], view_dir: Path,
                         rotations: Sequence[np.ndarray], camera_K: np.ndarray,
                         width: int, height: int) -> list[dict]:
    """Cameras for perspective views rendered from posed panoramas.

    rotations[k] is cam_from_pano for virtual camera k (pycolmap convention),
    and the view file is view_dir/pano_camera{k}/<pano filename>.
    """
    cameras = []
    for pano_name, pano_from_world in sorted(pano_poses.items()):
        for k, R in enumerate(rotations):
            cam_from_pano = np.eye(4)
            cam_from_pano[:3, :3] = np.asarray(R)
            name = f"pano_camera{k}/{pano_name}"
            cameras.append({
                "name": name,
                "path": str(Path(view_dir) / name),
                "pano": Path(pano_name).stem,
                "width": int(width),
                "height": int(height),
                "K": np.asarray(camera_K, dtype=np.float64),
                "w2c": cam_from_pano @ pano_from_world,
            })
    return assign_splits(cameras)


def legacy_crop_rotation(yaw_deg: float) -> np.ndarray:
    """cam_from_pano rotation of a prepare_panorama.py crop (pitch 0).

    prepare_panorama maps camera ray (x, -y, 1) to pano direction
    d = (c x + s z, y, -s x + c z) with y up. In pycolmap pano axes (y down)
    this is a yaw by the same angle about the vertical axis.
    """
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    pano_from_cam = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return pano_from_cam.T


def read_ply_points(path: Path, max_points: int = 2_000_000) -> np.ndarray:
    """Read xyz + rgb from an ASCII or binary PLY (openMVG colorized.ply)."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    names = v.dtype.names
    if {"red", "green", "blue"} <= set(names):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    pts = np.concatenate([xyz, rgb], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return pts


def colour_check(cameras: list[dict], points: np.ndarray, samples: int = 3000,
                 views: int = 24, seed: int = 0) -> dict:
    """Do posed views see the right colours at projected 3D points?

    Projects random coloured points into a few views and compares the point
    colour with the image colour there. Wrong axis conventions give a median
    difference near that of random pixels (~0.25); correct poses give a much
    lower value. Returns both so the gap is visible in logs.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    pts = points[rng.choice(len(points), min(samples, len(points)), replace=False)]
    chosen = [cameras[i] for i in rng.choice(len(cameras), min(views, len(cameras)), replace=False)]
    diffs, rand = [], []
    for cam in chosen:
        img = np.asarray(Image.open(cam["path"]).convert("RGB"), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        sx, sy = w / cam["width"], h / cam["height"]
        X = pts[:, :3] @ cam["w2c"][:3, :3].T + cam["w2c"][:3, 3]
        front = X[:, 2] > 1e-6
        uv = X[front] @ np.asarray(cam["K"]).T
        uv = uv[:, :2] / uv[:, 2:3]
        u, v = uv[:, 0] * sx, uv[:, 1] * sy
        ok = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if not ok.any():
            continue
        col = img[v[ok].astype(int), u[ok].astype(int)]
        diffs.append(np.abs(col - pts[front][ok, 3:6]).mean(1))
        rr = img[rng.integers(0, h, ok.sum()), rng.integers(0, w, ok.sum())]
        rand.append(np.abs(rr - pts[front][ok, 3:6]).mean(1))
    if not diffs:
        return {"projected": 0}
    d, r = np.concatenate(diffs), np.concatenate(rand)
    return {"projected": int(len(d)), "median_diff": float(np.median(d)),
            "median_diff_random_pixels": float(np.median(r))}


def openmvg_points_by_view(sfm_data: dict, max_per_view: int = 20000) -> dict[str, np.ndarray]:
    """filename -> xyz of structure points observed in that view (openMVG)."""
    names = {}
    for item in _openmvg_items(sfm_data.get("views")):
        data = item["value"]["ptr_wrapper"]["data"]
        names[int(item["key"])] = Path(data["filename"]).name
    by_view: dict[str, list] = {}
    for item in _openmvg_items(sfm_data.get("structure")):
        X = item["value"]["X"]
        for obs in item["value"].get("observations", []):
            n = names.get(int(obs["key"]))
            if n is not None:
                by_view.setdefault(n, []).append(X)
    rng = np.random.default_rng(0)
    out = {}
    for n, pts in by_view.items():
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) > max_per_view:
            arr = arr[rng.choice(len(arr), max_per_view, replace=False)]
        out[n] = arr
    return out


def pycolmap_points_by_pano(rec, pano_of: Callable[[str], str], max_per_pano: int = 20000) -> dict[str, np.ndarray]:
    """pano id -> xyz of 3D points with a track element in any view of that panorama."""
    names = {img.image_id: pano_of(img.name) for img in rec.images.values()}
    by: dict[str, list] = {}
    for p in rec.points3D.values():
        seen = {names[e.image_id] for e in p.track.elements if e.image_id in names}
        for pano in seen:
            by.setdefault(pano, []).append(p.xyz)
    rng = np.random.default_rng(0)
    out = {}
    for k, v in by.items():
        arr = np.asarray(v, dtype=np.float64)
        if len(arr) > max_per_pano:
            arr = arr[rng.choice(len(arr), max_per_pano, replace=False)]
        out[k] = arr
    return out
