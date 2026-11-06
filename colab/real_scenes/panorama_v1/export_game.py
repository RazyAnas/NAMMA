"""Game-ready export of a trained panorama world (metric, y-up, browser assets).

Inputs: a scene dir (scene.json with aligned depth/mask paths per view), the
trained model dir (splats.pt [+ carve.pt]), the route manifest (optional GPS).
Outputs in out_dir:
  scene.ply        splats in metres, y up (standard 3DGS PLY; SH degree 1 rotated)
  collider.glb     TSDF surface from the aligned real depth (walls, kerbs, ground)
  ground.json      heightfield of the drivable/walkable ground (metres)
  ground.glb       the same heightfield as a mesh (debug / visual)
  sky.jpg          equirectangular sky in the game frame (row 0 = straight up)
  ibl/pNNNN.jpg    each panorama resampled into the game frame (avatar/car lighting)
  game.json        units, spawn, route, IBL probe positions, scale provenance
Frame: x = east (when GPS exists), y = up, z = south (three.js convention).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import scene_io

SH_C1 = 0.4886025119029199


# ---------------------------------------------------------------- frames
def enu(lat0, lon0, lat, lon):
    east = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    north = (lat - lat0) * 110_540.0
    return east, north


def metric_frame(cameras, up_world, meta, manifest):
    """Return (scale m/unit, R world->game rows, origin world, provenance)."""
    pc = scene_io.pano_centres([c for c in cameras if c.get("kind") != "generated"])
    names = sorted(pc)
    C = np.stack([pc[n] for n in names])
    up = up_world / np.linalg.norm(up_world)
    fwd = C[-1] - C[0]
    fwd = fwd - up * fwd.dot(up)
    if np.linalg.norm(fwd) < 1e-9:
        fwd = np.cross(up, [1.0, 0, 0])
    x_axis = fwd / np.linalg.norm(fwd)
    z_axis = np.cross(x_axis, up)
    R = np.stack([x_axis, up, z_axis])  # game = R @ world (before scale)
    prov = {}
    scale = None
    # metric scale from MoGe-2 metric depth alignment (z_sfm = s * z_moge)
    dm = (meta or {}).get("depth_memory") or {}
    if dm.get("scale_median"):
        prov["scale_from_moge_m_per_unit"] = 1.0 / dm["scale_median"]
        scale = prov["scale_from_moge_m_per_unit"]
    # GPS: horizontal distances and heading
    gps = []
    if manifest:
        by_index = {f"{i:04d}": p for i, p in enumerate(manifest.get("panoramas", []))}
        for n in names:
            key = n.split("_")[0]
            p = by_index.get(key)
            if p and p.get("lat") is not None:
                gps.append((n, p["lat"], p["lon"]))
    if len(gps) >= 8:
        lat0, lon0 = gps[0][1], gps[0][2]
        E = np.array([enu(lat0, lon0, la, lo) for _, la, lo in gps])          # east, north
        G = np.stack([(R @ (pc[n] - C[0]))[[0, 2]] for n, _, _ in gps])        # game x, z (unscaled)
        d_gps = np.linalg.norm(np.diff(E, axis=0), axis=1)
        d_sfm = np.linalg.norm(np.diff(G, axis=0), axis=1)
        ok = (d_gps > 2) & (d_sfm > 1e-6)
        if ok.sum() >= 5 and float(np.sum(d_gps)) > 30:
            s_gps = float(np.median(d_gps[ok] / d_sfm[ok]))
            prov["scale_from_gps_m_per_unit"] = s_gps
            scale = s_gps
            # rotate about up so game x = east, game -z = north
            target = np.stack([E[:, 0], -E[:, 1]], 1)
            src = G * s_gps
            H = (src - src.mean(0)).T @ (target - target.mean(0))
            U, _, Vt = np.linalg.svd(H)
            Rot = (U @ Vt).T
            if np.linalg.det(Rot) < 0:
                Vt[-1] *= -1
                Rot = (U @ Vt).T
            a = math.atan2(Rot[1, 0], Rot[0, 0])
            ca, sa = math.cos(a), math.sin(a)
            Ryaw = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
            R = Ryaw @ R
            res = (Rot @ src.T).T + target.mean(0) - (Rot @ src.mean(0))
            prov["gps_fit_rms_m"] = float(np.sqrt(np.mean(np.sum((res - target) ** 2, 1))))
    if scale is None:
        scale = 1.0
        prov["scale"] = "unknown (no GPS, no metric depth): 1 unit = 1 m assumed"
    return scale, R, C[0], prov


def quat_mul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def rot_to_quat(R):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def rotate_sh1(shN: np.ndarray, R: np.ndarray) -> np.ndarray:
    """shN [N, 3 coeffs, 3 colours] (degree 1). Colour term = SH_C1 * (-y c0 + z c1 - x c2)."""
    a = np.stack([-shN[:, 2], -shN[:, 0], shN[:, 1]], 1)  # [N, 3(xyz), 3 colours]
    a2 = np.einsum("ij,njc->nic", R, a)
    return np.stack([-a2[:, 1], a2[:, 2], -a2[:, 0]], 1)


# ---------------------------------------------------------------- splats
def export_splats(ck, R, origin, scale_m, out_ply: Path, drop_far_m: float = 2000.0,
                  min_opacity: float = 0.01, max_aniso: float = 6.0) -> dict:
    """Normalised trainer splats -> metres, y-up; prune faint, absurdly far, clamp needles."""
    from plyfile import PlyData, PlyElement

    p = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in ck["params"].items()}
    centre, s_norm = np.asarray(ck["centre"]), float(ck["scale"])
    world = p["means"] / s_norm + centre
    pos = scale_m * (world - origin) @ R.T
    log_s = p["scales"] - math.log(s_norm) + math.log(scale_m)
    sc = np.exp(log_s)
    srt = np.sort(sc, 1)
    cap = srt[:, 1:2] * max_aniso
    clamped = int((sc > cap).any(1).sum())
    sc = np.minimum(sc, cap)
    op = 1 / (1 + np.exp(-p["opacities"]))
    keep = (op >= min_opacity) & (np.linalg.norm(pos, axis=1) < drop_far_m) & np.isfinite(pos).all(1)
    q = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    qR = rot_to_quat(R)
    q = quat_mul(np.repeat(qR[None], len(q), 0), q)
    sh0 = p["sh0"].reshape(len(pos), 3)
    shN = p["shN"]
    if shN.shape[1] == 3:
        shN = rotate_sh1(shN, R)
    elif shN.shape[1] > 0:
        shN = np.zeros((len(pos), 3, 3))  # higher degrees: drop view dependence rather than rotate wrongly
    rest = shN.transpose(0, 2, 1).reshape(len(pos), -1)
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(rest.shape[1])]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([pos, sh0, rest, p["opacities"][:, None], np.log(sc), q], 1)[keep].astype(np.float32)
    arr = np.empty(len(data), dtype=[(n, "f4") for n in names])
    for j, n in enumerate(names):
        arr[n] = data[:, j]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(out_ply))
    return {"splats_in": int(len(pos)), "splats_out": int(keep.sum()), "needles_clamped": clamped}


# ------------------------------------------------------- collider + ground
def fuse_collider(cameras, R, origin, scale_m, out_dir: Path, corridor_m: float = 30.0,
                  vox: float = 0.2, max_voxels: int = 110_000_000, device: str = "cuda") -> dict:
    import torch
    import trimesh
    from PIL import Image

    import tsdf

    real = [c for c in cameras if c.get("kind") != "generated" and c["split"] == "train" and c.get("depth")]
    centres = np.stack([scale_m * (scene_io.camera_centre(np.asarray(c["w2c"])) - origin) @ R.T for c in real])
    lo = centres.min(0) - np.array([corridor_m, 6.0, corridor_m])
    hi = centres.max(0) + np.array([corridor_m, 25.0, corridor_m])
    while np.prod(np.ceil((hi - lo) / vox)) > max_voxels:
        vox *= 1.15
    frames = []
    for c in real:
        d = np.load(c["depth"]).astype(np.float32) * scale_m
        if c.get("mask") and Path(c["mask"]).is_file():
            m = np.asarray(Image.open(c["mask"]).convert("L").resize((d.shape[1], d.shape[0]))) > 127
            d = d * m
        K = np.array(c["K"], dtype=np.float64).copy()
        K[:2] *= d.shape[1] / c["width"]
        w2c = np.asarray(c["w2c"], dtype=np.float64)
        c2w_world = np.linalg.inv(w2c)
        c2w = np.eye(4)
        c2w[:3, :3] = R @ c2w_world[:3, :3]
        c2w[:3, 3] = scale_m * (R @ (c2w_world[:3, 3] - origin))
        rgb = np.asarray(Image.open(c["path"]).convert("RGB").resize((d.shape[1], d.shape[0])))
        frames.append((torch.tensor(d, device=device), K, c2w, torch.tensor(rgb, device=device)))
    t, w, col = tsdf.fuse(frames, lo, hi, np.eye(3), vox, trunc=3, device=device)
    mesh = tsdf.extract(t, w, col, lo, np.eye(3), vox, centres, min_faces=200)
    faces_in = len(mesh.faces)
    if faces_in > 400_000:
        try:
            import fast_simplification

            v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces),
                                                target_reduction=1 - 400_000 / faces_in)
            mesh = trimesh.Trimesh(v, f, process=False)
        except Exception as exc:  # noqa: BLE001 - keep the full mesh if simplification is unavailable
            print("[export] simplification skipped:", exc, flush=True)
    mesh.export(out_dir / "collider.glb")
    ground = heightfield(t, w, lo, vox, centres)
    (out_dir / "ground.json").write_text(json.dumps(ground))
    ground_mesh(ground).export(out_dir / "ground.glb")
    return {"voxel_m": vox, "collider_faces": int(len(mesh.faces)), "collider_faces_before": faces_in,
            "ground_cells": [ground["nx"], ground["nz"]], "ground_coverage": ground["coverage"]}


def heightfield(tsdf_vol, wt, lo, vox, centres, cell_vox: int = 2, smooth: float = 1.5) -> dict:
    """Top-down: highest free->solid crossing below the camera height in each column."""
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    nx, ny, nz = tsdf_vol.shape
    cam_y = float(np.median(centres[:, 1]))
    top = int(np.clip((cam_y + 0.5 - lo[1]) / vox, 1, ny - 1))
    sub = tsdf_vol[:, :top, :]
    obs = wt[:, :top, :] > 0
    solid = (sub < 0) & obs
    free_above = np.zeros_like(solid)
    free_above[:, :-1, :] = (sub[:, 1:, :] > 0) & obs[:, 1:, :]
    cross = solid & free_above
    idx = np.where(cross.any(1), top - 1 - np.argmax(cross[:, ::-1, :], axis=1), -1)
    h = np.where(idx >= 0, lo[1] + (idx + 1) * vox, np.nan)
    h = h[::cell_vox, ::cell_vox]
    known = np.isfinite(h)
    coverage = float(known.mean())
    if known.any():
        _, (ii, jj) = distance_transform_edt(~known, return_indices=True)
        h = h[ii, jj]
        h = gaussian_filter(h, smooth)
    else:
        h = np.full_like(h, cam_y - 2.0)
    return {"origin": [float(lo[0]), float(lo[2])], "cell": float(vox * cell_vox), "nx": int(h.shape[0]),
            "nz": int(h.shape[1]), "heights": np.round(h, 3).ravel().tolist(), "coverage": coverage}


def ground_mesh(g: dict):
    import trimesh

    nx, nz, cell = g["nx"], g["nz"], g["cell"]
    h = np.asarray(g["heights"]).reshape(nx, nz)
    xs = g["origin"][0] + np.arange(nx) * cell
    zs = g["origin"][1] + np.arange(nz) * cell
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    V = np.stack([X.ravel(), h.ravel(), Z.ravel()], 1)
    i = np.arange(nx * nz).reshape(nx, nz)
    a, b, c, d = i[:-1, :-1].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel(), i[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, d, b], 1), np.stack([b, d, c], 1)])
    return trimesh.Trimesh(V, F, process=False)


# --------------------------------------------------------------- sky + IBL
def game_dirs(w: int, h: int) -> np.ndarray:
    """Unit directions for an equirect in the game frame: u -> azimuth, row 0 = up."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    lon = (u - 0.5) * 2 * math.pi
    lat = (0.5 - v) * math.pi
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), -np.cos(LAT) * np.cos(LON)], -1)


def export_sky(ck, R, out_path: Path, w: int = 2048, h: int = 1024) -> bool:
    import torch
    from PIL import Image

    if ck.get("env") is None:
        return False
    env = ck["env"][0].detach().cpu().numpy().transpose(1, 2, 0)  # [He, We, 3]
    up = np.asarray(ck["up"], dtype=np.float64)
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    fwd = np.cross(up, helper)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    d_world = game_dirs(w, h) @ R  # game -> world (R rows are game axes in world)
    x, y, z = d_world @ right, d_world @ up, d_world @ fwd
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1, 1))
    He, We = env.shape[:2]
    cu = np.clip(((lon / math.pi + 1) / 2 * We).astype(int), 0, We - 1)
    cv = np.clip(((lat / (math.pi / 2) + 1) / 2 * He).astype(int), 0, He - 1)
    Image.fromarray((np.clip(env[cv, cu], 0, 1) * 255).astype(np.uint8)).save(out_path, quality=92)
    return True


def export_ibl(cameras, pano_dir: Path, R, origin, scale_m, out_dir: Path, w: int = 1024, h: int = 512) -> list:
    """Each panorama resampled into the game frame (pycolmap panorama axes: x right, y down, z forward)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    rig0 = {}
    for c in cameras:
        if c.get("kind") == "generated" or not c["name"].startswith("pano_camera0/"):
            continue
        rig0[c["pano"]] = c
    probes = []
    if not rig0:
        return probes
    import pycolmap.panorama as P

    R0 = np.asarray(P.get_virtual_rotations(4, (-35.0, 0.0, 35.0))[0])  # cam0_from_pano
    files = {p.stem: p for p in Path(pano_dir).glob("*.jpg")}
    for k, (pano, c) in enumerate(sorted(rig0.items())):
        if pano not in files:
            continue
        w2c = np.asarray(c["w2c"])
        pano_from_world = R0.T @ w2c[:3, :3]
        img = np.asarray(Image.open(files[pano]).convert("RGB").resize((2048, 1024)))
        d = game_dirs(w, h) @ R @ pano_from_world.T  # game -> world -> pano
        yaw = np.arctan2(d[..., 0], d[..., 2])
        pitch = -np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
        u = (((1 + yaw / math.pi) / 2) * img.shape[1]).astype(int) % img.shape[1]
        v = np.clip(((1 - pitch * 2 / math.pi) / 2 * img.shape[0]).astype(int), 0, img.shape[0] - 1)
        name = f"p{k:04d}.jpg"
        Image.fromarray(img[v, u]).save(out_dir / name, quality=88)
        pos = scale_m * (scene_io.camera_centre(w2c) - origin) @ R.T
        probes.append({"file": f"ibl/{name}", "pos": [round(float(x), 3) for x in pos], "pano": pano})
    return probes


# ------------------------------------------------------------------ main
def export(scene_dir: Path, model_dir: Path, out_dir: Path, pano_dir: Path | None = None,
           manifest: dict | None = None, device: str = "cuda", log=print) -> dict:
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras, _, meta = scene_io.read_scene(scene_dir)
    ck = torch.load(Path(model_dir) / "splats.pt", map_location="cpu", weights_only=False)
    scale_m, R, origin, prov = metric_frame(cameras, np.asarray(ck["up"]), meta, manifest)
    log("[export] frame", json.dumps(prov))
    rep = {"units": "m", "up": "y", "frame": "x=east (if GPS), y=up, z=south", "scale_m_per_unit": scale_m, **prov}
    rep["splats"] = export_splats(ck, R, origin, scale_m, out_dir / "scene.ply")
    log("[export] splats", json.dumps(rep["splats"]))
    rep["collider"] = fuse_collider(cameras, R, origin, scale_m, out_dir, device=device)
    log("[export] collider", json.dumps(rep["collider"]))
    rep["sky"] = export_sky(ck, R, out_dir / "sky.jpg")
    probes = export_ibl(cameras, pano_dir, R, origin, scale_m, out_dir / "ibl") if pano_dir else []
    g = json.loads((out_dir / "ground.json").read_text())
    route = [p["pos"] for p in probes]
    if route:
        x0, z0 = route[0][0], route[0][2]
        hh = np.asarray(g["heights"]).reshape(g["nx"], g["nz"])
        i = int(np.clip((x0 - g["origin"][0]) / g["cell"], 0, g["nx"] - 1))
        j = int(np.clip((z0 - g["origin"][1]) / g["cell"], 0, g["nz"] - 1))
        ground_y = float(hh[i, j])
        heading = route[min(1, len(route) - 1)]
        rep["camera_height_m"] = round(route[0][1] - ground_y, 2)
        rep["spawn"] = {"pos": [x0, ground_y + 1.7, z0],
                        "look_at": [heading[0], ground_y + 1.7, heading[2]]}
    rep["route"] = route
    rep["ibl"] = probes
    rep["files"] = {"splats": "scene.ply", "collider": "collider.glb", "ground": "ground.json",
                    "ground_mesh": "ground.glb", "sky": "sky.jpg" if rep["sky"] else None}
    (out_dir / "game.json").write_text(json.dumps(rep, indent=1))
    return rep
