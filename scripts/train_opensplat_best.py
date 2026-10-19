"""OpenSplat with the measured "best" recipe (phone studio Best quality, CPU or Mac).

Measured on the whole ICL-NUIM living room (309 train / 44 held-out views, 3,000 CPU steps): 31.33 -> 32.26 dB
held-out PSNR, 0.877 -> 0.900 SSIM, about +14% time (/mnt/project-files/modp-pipeline/README.md). Steps:

1. MODP pixel placement (Wu et al. 2025, Sec. 3.2): per training frame, corners + strong edges (every 2 px) and an
   8 px grid for textureless areas, back-projected with the anchored metric depth, deduplicated by a multi-level
   occupancy hash voxel (MOHV). They become the dataset's 3D points, so OpenSplat sets its own neighbour scales.
2. Train to step 1,000 with 3DGS-paper densification (OPENSPLAT_PAPER_ADC=1, refine every 100 from 500 to 5/6 of
   the run).
3. MODP error compensation at step 1,000: render every training view, mask pixels with SSIM < 0.5 (every 8th pixel),
   add Gaussians there from the metric depth through the MOHV, then resume to the end.

Needs an OpenSplat build with the NAMMA patches (--densify-from/--densify-until, --resume, --val-render,
OPENSPLAT_TEST_PREFIX and OPENSPLAT_PAPER_ADC; see docs/PHONE_STUDIO.md). With a stock build it falls back to the
plain recipe and says so in train_info.json.

    python scripts/train_opensplat_best.py --dataset D --depth DEPTH_DIR --output OUT [--iterations 3000]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import cv2
import numpy as np
import pycolmap

PAPER_ADC = {"OPENSPLAT_PAPER_ADC": "1"}


def opensplat_features(binary: str) -> set[str]:
    try:
        text = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {flag for flag in ("densify-from", "densify-until", "resume", "val-render") if f"--{flag}" in text}


# ---- MOHV (multi-level occupancy hash voxel) -----------------------------------------------------

class Mohv:
    def __init__(self, coarsest: float = 1.28, levels: int = 9):
        self.coarsest, self.levels = coarsest, levels
        self.occupied = [set() for _ in range(levels)]

    def _keys(self, xyz: np.ndarray, level: int) -> np.ndarray:
        q = np.floor(xyz / (self.coarsest / 2.0 ** level)).astype(np.int64) + (1 << 20)
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def insert(self, xyz: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        """Accept points whose voxel at their own level is free; mark that level and every coarser one."""
        spacing = np.nan_to_num(spacing, nan=self.coarsest, posinf=self.coarsest)
        level = np.clip(np.floor(np.log2(self.coarsest / np.maximum(spacing, 1e-6))), 0, self.levels - 1).astype(int)
        accepted = np.zeros(len(xyz), bool)
        for lvl in np.unique(level):
            idx = np.where(level == lvl)[0]
            keys = self._keys(xyz[idx], lvl)
            _, first = np.unique(keys, return_index=True)
            first = np.sort(first)
            fresh = np.array([k not in self.occupied[lvl] for k in keys[first].tolist()], bool)
            chosen = idx[first[fresh]]
            accepted[chosen] = True
            for coarser in range(lvl + 1):
                self.occupied[coarser].update(self._keys(xyz[chosen], coarser).tolist())
        return accepted


# ---- Dataset helpers -----------------------------------------------------------------------------

def frames(reconstruction: pycolmap.Reconstruction, dataset: Path, depth_dir: Path):
    for image in sorted(reconstruction.images.values(), key=lambda im: im.name):
        depth_path = depth_dir / f"{Path(image.name).stem}.npy"
        color = cv2.imread(str(dataset / "images" / image.name))
        if color is None or not depth_path.is_file():
            continue
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != color.shape[:2]:
            depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
        K = np.asarray(reconstruction.cameras[image.camera_id].calibration_matrix(), float)
        yield image, color, depth, K, image.cam_from_world().matrix()


def backproject(us, vs, d, K, cam_from_world):
    camera = np.stack([(us - K[0, 2]) / K[0, 0] * d, (vs - K[1, 2]) / K[1, 1] * d, d], 1)
    R, t = cam_from_world[:, :3], cam_from_world[:, 3]
    return (camera - t) @ R


def sample(depth, us, vs):
    us, vs = np.asarray(us).astype(int), np.asarray(vs).astype(int)
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > 0.05)
    return us[ok], vs[ok], d[ok]


def modp_points(reconstruction, dataset, depth_dir, mohv, k_feature=2.0, k_grid=8.0):
    xyz, rgb, counts = [], [], {"feature": 0, "grid": 0}
    for image, color, depth, K, pose in frames(reconstruction, dataset, depth_dir):
        grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape
        corners = cv2.goodFeaturesToTrack(grey, 4000, 0.005, 2)
        cu, cv_ = (corners[:, 0, 0], corners[:, 0, 1]) if corners is not None else (np.zeros(0), np.zeros(0))
        magnitude = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
        ev, eu = np.nonzero(magnitude[::2, ::2] > np.percentile(magnitude, 92))
        gv, gu = np.mgrid[4:h:8, 4:w:8]
        for kind, us, vs, k in (("feature", np.r_[cu, eu * 2], np.r_[cv_, ev * 2], k_feature),
                                ("grid", gu.ravel(), gv.ravel(), k_grid)):
            us, vs, d = sample(depth, us, vs)
            points = backproject(us, vs, d, K, pose)
            keep = mohv.insert(points, k * d / K[0, 0])
            xyz.append(points[keep])
            rgb.append(color[vs[keep], us[keep], ::-1])
            counts[kind] += int(keep.sum())
    return np.concatenate(xyz), np.concatenate(rgb), counts


def dataset_with_points(dataset: Path, target: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    if target.exists():
        shutil.rmtree(target)
    (target / "sparse" / "0").mkdir(parents=True)
    (target / "images").symlink_to((dataset / "images").resolve(), target_is_directory=True)
    reconstruction = pycolmap.Reconstruction(str(dataset / "sparse" / "0"))
    for point_id in list(reconstruction.points3D):
        reconstruction.delete_point3D(point_id)
    for point, colour in zip(xyz, rgb.astype(np.uint8)):
        reconstruction.add_point3D(point, pycolmap.Track(), colour)
    reconstruction.write(str(target / "sparse" / "0"))
    return target


def render_project(dataset: Path, target: Path, names: list[str], keep_train: list[str]) -> Path:
    """OpenSplat project whose cameras `names` are held out (test_ prefix) so --val-render renders them all."""
    if target.exists():
        shutil.rmtree(target)
    (target / "images").mkdir(parents=True)
    (target / "txt").mkdir()
    (target / "sparse" / "0").mkdir(parents=True)
    reconstruction = pycolmap.Reconstruction(str(dataset / "sparse" / "0"))
    by_name = {im.name: im for im in reconstruction.images.values()}
    with open(target / "txt" / "cameras.txt", "w") as cameras:
        for camera in reconstruction.cameras.values():
            params = " ".join(str(v) for v in camera.params)
            cameras.write(f"{camera.camera_id} {camera.model.name} {camera.width} {camera.height} {params}\n")
    with open(target / "txt" / "images.txt", "w") as images:
        items = [(n, "") for n in keep_train] + [(n, "test_") for n in names]
        for i, (name, prefix) in enumerate(items, 1):
            image = by_name[name]
            pose = image.cam_from_world()
            q = pose.rotation.quat  # x y z w
            t = pose.translation
            images.write(f"{i} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {image.camera_id} {prefix}{name}\n\n")
            os.symlink((dataset / "images" / name).resolve(), target / "images" / f"{prefix}{name}")
    rng = np.random.default_rng(0)
    (target / "txt" / "points3D.txt").write_text(
        "".join(f"{i} {x} {y} {z} 128 128 128 0\n" for i, (x, y, z) in enumerate(rng.uniform(-2, 2, (500, 3)), 1)))
    pycolmap.Reconstruction(str(target / "txt")).write_binary(str(target / "sparse" / "0"))
    return target


# ---- Error compensation --------------------------------------------------------------------------

def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def read_gaussians(path: Path) -> tuple[list[str], np.ndarray, str]:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    count = int(next(line for line in header.split("\n") if line.startswith("element vertex")).split()[-1])
    props = [line.split()[-1] for line in header.split("\n") if line.startswith("property")]
    return props, np.frombuffer(raw[end:end + count * 4 * len(props)], "<f4").reshape(count, len(props)).copy(), header


def write_gaussians(path: Path, props: list[str], table: np.ndarray, step: int) -> None:
    header = (f"ply\nformat binary_little_endian 1.0\ncomment Generated by opensplat at iteration {step}\n"
              f"element vertex {len(table)}\n" + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    path.write_bytes(header.encode() + table.astype("<f4").tobytes())


def compensate(base_ply: Path, out_ply: Path, renders: Path, reconstruction, dataset, depth_dir, step: int,
               threshold=0.5, stride=8, k_ec=1.5) -> dict:
    props, table, _ = read_gaussians(base_ply)
    col = {p: i for i, p in enumerate(props)}
    mohv = Mohv()
    scales = np.exp(table[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1)
    mohv.insert(table[:, :3].astype(np.float64), 2 * scales)
    rows, masked = [], []
    for image, color, depth, K, pose in frames(reconstruction, dataset, depth_dir):
        render = cv2.imread(str(renders / f"test_{image.name}"))
        if render is None:
            render = cv2.imread(str(renders / f"test_{Path(image.name).stem}.png"))
        if render is None:
            continue
        frame = cv2.resize(color, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA)
        low = ssim_map(cv2.cvtColor(render, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < threshold
        masked.append(float(low.mean()))
        grid = np.zeros_like(low)
        grid[::stride, ::stride] = True
        vs, us = np.nonzero(low & grid)
        sx, sy = color.shape[1] / render.shape[1], color.shape[0] / render.shape[0]
        us, vs, d = sample(depth, np.minimum(us * sx, color.shape[1] - 1), np.minimum(vs * sy, color.shape[0] - 1))
        points = backproject(us, vs, d, K, pose)
        spacing = k_ec * d / K[0, 0]
        keep = mohv.insert(points, spacing)
        if not keep.any():
            continue
        new = np.zeros((int(keep.sum()), len(props)), np.float32)
        new[:, :3] = points[keep]
        new[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]] = (color[vs[keep], us[keep], ::-1] / 255.0 - 0.5) / 0.28209479177387814
        new[:, col["opacity"]] = np.log(0.1 / 0.9)
        new[:, [col["scale_0"], col["scale_1"], col["scale_2"]]] = np.log(np.maximum(0.5 * spacing[keep], 2e-4))[:, None]
        new[:, col["rot_0"]] = 1.0
        rows.append(new)
    added = np.concatenate(rows) if rows else np.zeros((0, len(props)), np.float32)
    write_gaussians(out_ply, props, np.concatenate([table, added]), step)
    return {"base": len(table), "added": len(added), "views": len(masked),
            "mean_masked_fraction": round(float(np.mean(masked)), 4) if masked else None}


# ---- Training ------------------------------------------------------------------------------------

def run(command: list, env: dict, log, stop_when=None) -> None:
    print("Running:", subprocess.list2cmdline([str(c) for c in command]), flush=True)
    process = subprocess.Popen([str(c) for c in command], env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
    while process.poll() is None:
        if stop_when and stop_when():
            process.send_signal(signal.SIGTERM)
            process.wait()
            return
        time.sleep(2)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True, help="metric depth per frame (<image stem>.npy)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--ec-step", type=int, default=1000)
    parser.add_argument("--render-chunk", type=int, default=30, help="views per render call (300 at once runs out of memory)")
    parser.add_argument("--max-points", type=int, default=500_000, help="random cap on placed points (0 = no cap)")
    parser.add_argument("--opensplat", default=os.environ.get("OPENSPLAT_BIN", "opensplat"))
    parser.add_argument("--extra", default=os.environ.get("OPENSPLAT_EXTRA", ""), help="extra OpenSplat flags, e.g. --cpu")
    args = parser.parse_args()

    binary = shutil.which(args.opensplat) or args.opensplat
    out = args.output.resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    reconstruction = pycolmap.Reconstruction(str(dataset / "sparse" / "0"))
    features = opensplat_features(binary)
    patched = {"densify-from", "densify-until", "resume", "val-render"} <= features
    n = args.iterations
    common = [str(dataset), "-n", n, "-d", args.downscale, *args.extra.split()]
    info = {"trainer": "opensplat", "recipe": "best", "iterations": n, "patched_opensplat": patched, "measured": False}
    started = time.time()

    mohv = Mohv()
    xyz, rgb, counts = modp_points(reconstruction, dataset, args.depth, mohv)
    info["init"] = {"points": len(xyz), **counts}
    if args.max_points and len(xyz) > args.max_points:
        # Not part of the measured recipe: a phone capture can place ~1M points, too many for CPU training.
        keep = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
        info["init"]["capped_to"] = args.max_points
    print(f"MODP placement: {len(xyz)} points ({counts})", flush=True)
    placed = dataset_with_points(dataset, work / "dataset-modp", xyz, rgb)
    common[0] = str(placed)
    log = open(out / "opensplat.log", "a")

    if not patched:
        print("This OpenSplat has no NAMMA patches: training the placed points with default densification.", flush=True)
        run([binary, *common, "-o", out / "splat.ply"], {}, log)
    else:
        until = max(args.ec_step + 1, round(n * 2500 / 3000))
        adc = ["--refine-every", 100, "--densify-from", 500, "--densify-until", until]
        info["densification"] = {"paper_adc": True, "refine_every": 100, "from": 500, "until": until}
        if n > args.ec_step:
            stage1 = work / "splat.ply"
            checkpoint = work / f"splat_{args.ec_step}.ply"
            run([binary, *common, *adc, "--save-every", args.ec_step, "-o", stage1], PAPER_ADC, log,
                stop_when=lambda: checkpoint.is_file() and time.time() - checkpoint.stat().st_mtime > 5)
            renders = work / "renders"
            renders.mkdir(exist_ok=True)
            names = sorted(im.name for im in reconstruction.images.values())
            for start in range(0, len(names), args.render_chunk):
                chunk = names[start:start + args.render_chunk]
                project = render_project(dataset, work / "render-project", chunk, names[:4])
                run([binary, project, "-n", args.ec_step, "-d", args.downscale, *args.extra.split(), "--resume", checkpoint,
                     "--val-render", renders, "-o", work / "render.ply"], {"OPENSPLAT_TEST_PREFIX": "test_"}, log)
            compensated = work / f"splat_{args.ec_step}_ec.ply"
            info["error_compensation"] = compensate(checkpoint, compensated, renders, reconstruction, dataset, args.depth,
                                                    args.ec_step)
            print(f"Error compensation: {info['error_compensation']}", flush=True)
            run([binary, *common, *adc, "--resume", compensated, "-o", out / "splat.ply"], PAPER_ADC, log)
        else:
            run([binary, *common, *adc, "-o", out / "splat.ply"], PAPER_ADC, log)
    info["seconds"] = round(time.time() - started, 1)
    # Intermediate splats would otherwise be picked up as the result (the newest .ply under the output wins).
    shutil.rmtree(work, ignore_errors=True)
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"Exported {out / 'splat.ply'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
