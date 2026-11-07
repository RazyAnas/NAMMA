"""Lean Gaussian-splat trainer for panorama scenes (gsplat library API only).

Why not gsplat's examples/simple_trainer.py: the panorama branch spent most of
its commits patching that script's parser, imports and split at run time.
This trainer reads scene_io's scene.json directly, so:
  * the train/val split is by whole panorama (scene.json "split");
  * optional per-pixel masks (moving objects, capture car, sky) and per-pixel
    weights (generated fill pixels count less than real ones);
  * optional dense depth supervision (aligned metric depth maps);
  * optional sky dome: sky pixels are explained by a learned environment
    texture, never by splats, so sky floaters cannot form;
  * MCMC strategy with the optimizer stepped BEFORE relocation (video
    pipeline lesson, NOTES.md 2026-10-25), scale/opacity regularisers and an
    anisotropy penalty against needle splats.

Outputs (in --out): model.ply (world coordinates, standard 3DGS layout),
splats.pt (normalised params + transform), metrics.json, train.log lines on
stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import scene_io

C0 = 0.28209479177387814  # SH band-0 constant


# ------------------------------------------------------------------ data
def _load_rgb(path: str, long_side: int):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, long_side / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8), s


def _load_mask(path: str | None, size: tuple[int, int]):
    """Mask PNG: 255 = use pixel, 0 = ignore. Missing file = all valid."""
    if not path or not Path(path).is_file():
        return None
    from PIL import Image

    m = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def load_views(cameras, long_side: int):
    views = []
    for cam in cameras:
        rgb, s = _load_rgb(cam["path"], long_side)
        K = np.array(cam["K"], dtype=np.float64).copy()
        K[:2] *= s
        h, w = rgb.shape[:2]
        view = {"cam": cam, "rgb": rgb, "K": K, "w": w, "h": h}
        for key in ("mask", "sky_mask", "weight"):
            view[key] = _load_mask(cam.get(key), (w, h))
        if cam.get("depth") and Path(cam["depth"]).is_file():
            from PIL import Image

            d = np.load(cam["depth"]) if cam["depth"].endswith(".npy") else None
            if d is not None:
                d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((w, h), Image.NEAREST))
            view["depth"] = d
        views.append(view)
    return views


# --------------------------------------------------------- normalisation
def normalisation(cameras):
    """Similarity mapping world -> unit-ish frame centred on the route."""
    centres = np.stack([scene_io.camera_centre(np.asarray(c["w2c"])) for c in cameras])
    centre = centres.mean(0)
    radius = float(np.percentile(np.linalg.norm(centres - centre, axis=1), 95))
    radius = max(radius, 1e-3)
    return centre, 1.0 / radius


def pano_subset(cameras, points, spec: str, margin: float = 4.0):
    """Keep the views of panoramas whose route index (leading digits of the panorama name) lies in spec "a-b",
    and the initial points within margin x the median panorama spacing of a kept panorama centre."""
    a, b = (int(x) for x in spec.split("-"))

    def idx(c):
        head = str(c.get("pano", "")).split("_")[0]
        return int(head) if head.isdigit() else -1

    kept = [c for c in cameras if a <= idx(c) <= b]
    if not kept:
        raise ValueError(f"no panoramas in range {spec}")
    if points is None or not len(points):
        return kept, points
    pc = scene_io.pano_centres([c for c in kept if c.get("kind") != "generated"])
    C = np.stack(list(pc.values()))
    from scipy.spatial import cKDTree

    if len(C) > 1:
        dd, _ = cKDTree(C).query(C, k=2)
        spacing = float(np.median(dd[:, 1]))
    else:
        spacing = 1.0
    d, _ = cKDTree(C).query(points[:, :3], k=1)
    return kept, points[d < margin * spacing]


def max_splat_size(cameras, scale: float, frac: float = 0.3) -> float:
    """Largest allowed splat scale in normalised units: 0.3 x the median gap between neighbouring
    panoramas (~3 m for Street View's ~10 m spacing). The normalised unit itself is the route radius
    (~150 m on a 40-panorama street), so a fixed number in normalised units would be far too loose."""
    pc = scene_io.pano_centres([c for c in cameras if c.get("kind") != "generated"])
    C = np.stack(list(pc.values())) * scale
    if len(C) < 2:
        return 0.15
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(max(frac * np.median(d.min(1)), 1e-4))


# ----------------------------------------------------------------- losses
def ssim(a, b):
    """a, b: [1, 3, H, W] in 0..1. Gaussian-window SSIM (11, 1.5)."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    mu1 = F.conv2d(a, w, padding=5, groups=3)
    mu2 = F.conv2d(b, w, padding=5, groups=3)
    s1 = F.conv2d(a * a, w, padding=5, groups=3) - mu1 ** 2
    s2 = F.conv2d(b * b, w, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=5, groups=3) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)))


def sky_colour(env, dirs):
    """Equirectangular environment texture lookup. env: [1,3,He,We]; dirs: [H,W,3] world (unit)."""
    import torch
    import torch.nn.functional as F

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1, 1))
    grid = torch.stack([lon / math.pi, lat / (math.pi / 2)], -1)[None]
    return F.grid_sample(env, grid, align_corners=False, padding_mode="border")[0].permute(1, 2, 0)


# ----------------------------------------------------------------- train
def train(scene_dir: Path, out: Path, steps: int = 7000, cap: int = 1_000_000,
          long_side: int = 960, sh_degree: int = 1, depth_weight: float = 0.0,
          sky: bool = False, aniso_max: float = 10.0, aniso_weight: float = 0.0,
          scale_reg: float = 0.01, opacity_reg: float = 0.01, max_minutes: float = 600.0,
          seed: int = 0, device: str = "cuda", carve: bool = False, carve_every: int = 500,
          init_ckpt: Path | None = None, lr_scale: float = 1.0, crop_px: int = 1024,
          snapshot: Path | None = None, snapshot_minutes: float = 30.0, packed: bool = True,
          depth_loss: str = "rel", scale_cap_frac: float = 0.3, noise_lr: float = 5e5,
          init_opacity: float = 0.1, panos: str = "") -> dict:
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cameras, points, meta = scene_io.read_scene(scene_dir)
    if panos:
        cameras, points = pano_subset(cameras, points, panos)
        print(f"[train] panorama subset {panos}: {len(cameras)} views, {len(points)} points", flush=True)
    ck0 = None
    if init_ckpt:
        ck0 = torch.load(init_ckpt, map_location=device, weights_only=False)
        centre, scale, up_world = np.asarray(ck0["centre"]), float(ck0["scale"]), np.asarray(ck0["up"])
    else:
        centre, scale = normalisation(cameras)
        up_world = scene_io.world_up_from_cameras([c for c in cameras if c.get("kind") != "generated"])

    views = load_views(cameras, long_side)
    train_v = [v for v in views if v["cam"]["split"] == "train"]
    val_v = [v for v in views if v["cam"]["split"] == "val"]
    print(f"[train] {len(train_v)} train views, {len(val_v)} val views, {len(points)} init points", flush=True)

    def viewmat(cam):
        w2c = np.asarray(cam["w2c"], dtype=np.float64)
        R, t = w2c[:3, :3], w2c[:3, 3]
        # x_n = (x - centre) * scale  ->  x_cam = R x + t = R (x_n/scale + centre) + t
        Vn = np.eye(4)
        Vn[:3, :3] = R
        Vn[:3, 3] = (R @ centre + t) * scale
        return torch.tensor(Vn, dtype=torch.float32, device=device)

    for v in views:
        v["V"] = viewmat(v["cam"])
        v["Kt"] = torch.tensor(v["K"], dtype=torch.float32, device=device)

    # ---- initial splats
    xyz = (points[:, :3].astype(np.float64) - centre) * scale
    keep = np.linalg.norm(xyz, axis=1) < 50.0          # drop absurd SfM outliers before they seed floaters
    xyz, rgb = xyz[keep], points[keep, 3:6]
    from scipy.spatial import cKDTree

    d, _ = cKDTree(xyz).query(xyz, k=4)
    dist = np.sqrt(np.mean(d[:, 1:] ** 2, axis=1)).clip(1e-4, 0.05)
    n = len(xyz)
    K_sh = (sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(dist, dtype=torch.float32, device=device))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opacity, device=device))),
        "sh0": torch.nn.Parameter(((torch.tensor(rgb, dtype=torch.float32, device=device) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros((n, K_sh - 1, 3), device=device)),
    })
    if ck0 is not None:
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device).float())
                                         for k, v in ck0["params"].items()})
    scene_scale = 1.0
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    lrs = {k: v * lr_scale for k, v in lrs.items()}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}

    # per-view colour affine (exposure / white balance), identity at start
    n_tr = len(train_v)
    cc = torch.nn.Parameter(torch.zeros((n_tr, 3, 4), device=device))
    opt_cc = torch.optim.Adam([cc], lr=1e-3)

    env = None
    if sky:
        init_env = (torch.logit(ck0["env"].to(device).clamp(1e-3, 1 - 1e-3))
                    if ck0 is not None and ck0.get("env") is not None
                    else torch.full((1, 3, 256, 512), 0.4, device=device))
        env = torch.nn.Parameter(init_env)
        opt_env = torch.optim.Adam([env], lr=1e-2)
    up_n = torch.tensor(up_world, dtype=torch.float32, device=device)

    grid = None
    if carve:
        import depth_memory

        if init_ckpt and (Path(init_ckpt).parent / "carve.pt").is_file():
            grid = depth_memory.CarveGrid.load(Path(init_ckpt).parent / "carve.pt", device)
            grid.save(out / "carve.pt")
        else:
            grid = depth_memory.build_carve_grid(cameras, centre, scale, out / "carve.pt", points=points,
                                                 device=device)
        print(f"[train] carve grid: {int((grid.empty >= 2).sum())} seen-empty voxels", flush=True)

    def carve_now(final=False):
        with torch.no_grad():
            dead = grid.is_empty(params["means"])
            if final:
                return dead
            params["opacities"][dead] = torch.logit(torch.tensor(1e-3, device=device))
            return dead

    strat = MCMCStrategy(cap_max=cap, refine_stop_iter=int(steps * 0.8), noise_lr=noise_lr, verbose=False)
    strat.check_sanity(params, opts)
    state = strat.initialize_state()

    def to_tensor(arr):
        return torch.from_numpy(arr).to(device)

    def render(v, sh_deg):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        img, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, v["V"][None], v["Kt"][None],
            v["w"], v["h"], sh_degree=sh_deg, render_mode="RGB+ED" if depth_weight > 0 else "RGB",
            packed=packed, absgrad=False, tile_size=TILE[0],
        )
        return img[0], alpha[0], info

    def ray_dirs_world(v):
        h, w = v["h"], v["w"]
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        K = v["Kt"]
        d_cam = torch.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], torch.ones_like(xs, dtype=torch.float32)], -1)
        R = v["V"][:3, :3]
        d = d_cam @ R  # R^T d
        d = F.normalize(d, dim=-1)
        # express in a gravity-aligned frame: y = up
        upv = up_n
        helper = torch.tensor([1.0, 0, 0], device=device) if abs(float(upv[0])) < 0.9 else torch.tensor([0, 0, 1.0], device=device)
        fwd = F.normalize(torch.cross(upv, helper, dim=0), dim=0)
        right = torch.cross(fwd, upv, dim=0)
        return torch.stack([d @ right, d @ upv, d @ fwd], -1)

    def step_loss(v, c, i, sh_deg, step):
        # high-res views: train on a random window (principal point shifted) so the tile-intersection
        # buffer stays inside a 16 GB T4 while every pixel keeps its full resolution
        if CROP > 0 and v["w"] * v["h"] > CROP * CROP:
            cw, ch = min(CROP, v["w"]), min(CROP, v["h"])
            x0, y0 = np.random.randint(0, v["w"] - cw + 1), np.random.randint(0, v["h"] - ch + 1)
            v = crop_view(v, x0, y0, cw, ch)
            c = {k: (None if t is None else t[y0:y0 + ch, x0:x0 + cw]) for k, t in c.items()}
        out_img, alpha, info = render(v, sh_deg)
        rgb = out_img[..., :3]
        if env is not None:
            sky_rgb = sky_colour(torch.sigmoid(env), ray_dirs_world(v))
            rgb = rgb + (1 - alpha) * sky_rgb
        A = cc[i]
        rgb_c = rgb @ (torch.eye(3, device=device) + A[:, :3]).T + A[:, 3]
        gt = c["rgb"]
        w = torch.ones_like(gt[..., 0])
        if c["mask"] is not None:
            w = w * c["mask"]
        if c["weight"] is not None:
            w = w * (0.3 + 0.7 * c["weight"])
        l1 = ((rgb_c - gt).abs().mean(-1) * w).sum() / w.sum().clamp_min(1)
        s = ssim(rgb_c.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])[0].mean(0)
        l_ssim = ((1 - s) * w).sum() / w.sum().clamp_min(1)
        loss = 0.8 * l1 + 0.2 * l_ssim
        if c["sky"] is not None:
            # sky pixels (mask==0 in sky_mask) must stay transparent to splats
            loss = loss + 0.05 * (alpha[..., 0] * (1 - c["sky"])).mean()
        if depth_weight > 0 and c["depth"] is not None:
            ed = out_img[..., 3]
            gd = c["depth"] * scale
            valid = (gd > 0) & (alpha[..., 0] > 0.5)
            if c["mask"] is not None:
                valid = valid & (c["mask"] > 0)
            if valid.any():
                dw = depth_weight * (0.01 ** (step / steps)) ** 0.5
                if depth_loss == "inv":
                    loss = loss + dw * (1 / ed[valid].clamp_min(1e-3) - 1 / gd[valid].clamp_min(1e-3)).abs().mean()
                else:
                    loss = loss + dw * (gd[valid] / ed[valid].clamp_min(1e-4) - 1).abs().clamp(max=1.0).mean()
        loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).mean()
        loss = loss + scale_reg * torch.exp(params["scales"]).mean()
        if aniso_weight > 0:
            sc = torch.exp(params["scales"])
            ratio = sc.max(-1).values / sc.min(-1).values.clamp_min(1e-8)
            loss = loss + aniso_weight * torch.relu(ratio - aniso_max).mean()
        loss = loss + 1e-3 * (cc[i] ** 2).sum()
        return loss, info

    def stepped(step, info, loss):
        lr_m = lrs["means"] * 0.01 ** (step / steps)
        for g in opts["means"].param_groups:
            g["lr"] = lr_m
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opt_cc.step(); opt_cc.zero_grad(set_to_none=True)
        if env is not None:
            opt_env.step(); opt_env.zero_grad(set_to_none=True)
        strat.step_post_backward(params, opts, state, step, info, lr=lr_m)
        if grid is not None and step % carve_every == 0 and step > 0:
            carve_now()
        with torch.no_grad():
            if MAX_LOG_SCALE is not None:
                params["scales"].clamp_(max=MAX_LOG_SCALE)

    CROP = crop_px
    TILE = [32]  # 32 px tiles cut intersections ~4x for large splats; falls back to 16 if the kernel refuses
    # ~3 m at the default 0.3 x panorama spacing; stops screen-filling splats (0 = no cap)
    MAX_LOG_SCALE = math.log(max_splat_size(cameras, scale, scale_cap_frac)) if scale_cap_frac > 0 else None
    print(f"[train] splat size cap {'none' if MAX_LOG_SCALE is None else f'{math.exp(MAX_LOG_SCALE):.4f}'} "
          f"normalised units; packed {packed}, window {CROP}, depth loss {depth_loss}", flush=True)
    oom_skips = 0
    last_snap = time.time()
    log_every = max(1, steps // 20)
    for step in range(steps):
        if (time.time() - t0) / 60 > max_minutes:
            print(f"[train] time budget reached at step {step}", flush=True)
            break
        i = np.random.randint(n_tr)
        v = train_v[i]
        # views live on the CPU as uint8 (16 GB T4: 400 float views on the GPU ran out of memory)
        c = {
            "rgb": to_tensor(v["rgb"]).float() / 255.0,
            "mask": None if v["mask"] is None else to_tensor(v["mask"]).float(),
            "sky": None if v["sky_mask"] is None else to_tensor(v["sky_mask"]).float(),
            "weight": None if v["weight"] is None else to_tensor(v["weight"]).float(),
            "depth": None if v.get("depth") is None else to_tensor(v["depth"]).float(),
        }
        sh_deg = min(sh_degree, step // 1000)
        try:
            loss, info = step_loss(v, c, i, sh_deg, step)
            loss.backward()
        except torch.OutOfMemoryError:
            # a few huge on-screen splats can blow the tile-intersection buffer; skip the view, keep training
            for o in list(opts.values()) + [opt_cc] + ([opt_env] if env is not None else []):
                o.zero_grad(set_to_none=True)
            del c
            torch.cuda.empty_cache()
            oom_skips += 1
            if oom_skips % 10 == 0 and CROP > 512:
                # memory pressure grows as splats spread: train on smaller windows rather than stop
                CROP = max(512, int(CROP * 0.85))
                print(f"[train] out-of-memory x{oom_skips}: training window now {CROP} px", flush=True)
            if oom_skips > 300:
                print(f"[train] too many out-of-memory skips, stopping at step {step}", flush=True)
                break
            continue
        except RuntimeError as exc:
            if TILE[0] == 16:
                raise
            print(f"[train] tile size 32 failed ({str(exc)[:200]}), using 16", flush=True)
            TILE[0] = 16
            for o in list(opts.values()) + [opt_cc] + ([opt_env] if env is not None else []):
                o.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        stepped(step, info, loss)
        if snapshot and (time.time() - last_snap) / 60 > snapshot_minutes:
            # a run cut off by the Kaggle quota or time limit still leaves a usable model behind
            tmp = Path(str(snapshot) + ".tmp")
            torch.save({"params": {k: p.detach().cpu() for k, p in params.items()}, "centre": centre,
                        "scale": scale, "up": up_world, "step": step, "sh_degree": sh_degree,
                        "env": None if env is None else torch.sigmoid(env).detach().cpu()}, tmp)
            tmp.replace(snapshot)
            last_snap = time.time()
            print(f"[train] snapshot at step {step} -> {snapshot}", flush=True)
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                op = torch.sigmoid(params["opacities"])
                big = torch.exp(params["scales"]).max(-1).values
                at_cap = 0.0 if MAX_LOG_SCALE is None else float((big > 0.98 * math.exp(MAX_LOG_SCALE)).float().mean())
            print(f"[train] step {step}/{steps} loss {loss.item():.4f} splats {len(params['means'])} "
                  f"oom-skips {oom_skips} {(time.time() - t0) / 60:.1f} min | transparent {float((op < 0.05).float().mean()):.3f} "
                  f"size p50 {float(big.median()):.5f} p99 {float(big[::max(1, len(big) // 1_000_000)].quantile(0.99)):.4f} "
                  f"at-cap {at_cap:.3f}", flush=True)

    opts.clear(); state.clear()
    torch.cuda.empty_cache()
    carved = 0
    if grid is not None:
        dead = carve_now(final=True)
        carved = int(dead.sum())
        with torch.no_grad():
            keepm = ~dead
            for k in list(params.keys()):
                params[k] = torch.nn.Parameter(params[k][keepm].detach())
        print(f"[train] carved {carved} splats in proven-empty space", flush=True)

    # ---------------------------------------------------------------- eval
    metrics = evaluate(params, val_v, render, env, ray_dirs_world, device, tile=min(CROP, 768) if CROP > 0 else 768)
    metrics["carved_splats"] = carved
    metrics.update({"steps": steps, "splats": int(len(params["means"])),
                    "train_views": len(train_v), "val_views": len(val_v),
                    "minutes": round((time.time() - t0) / 60, 1),
                    "within_pano_centre_spread": scene_io.within_pano_spread(cameras)})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if env is not None:
        with torch.no_grad():
            He = env.shape[2]
            lat = (torch.arange(He, device=device) + 0.5) / He * 180 - 90   # row 0 = straight down
            for pole_rows, ring in ((lat > 78, int(((78 + 90) / 180) * He) - 1), (lat < -78, int(((-78 + 90) / 180) * He))):
                env[0, :, pole_rows, :] = env[0, :, ring:ring + 1, :].mean(-1, keepdim=True)
    torch.save({"params": {k: p.detach().cpu() for k, p in params.items()},
                "centre": centre, "scale": scale, "up": up_world,
                "env": None if env is None else torch.sigmoid(env).detach().cpu(),
                "sh_degree": sh_degree}, out / "splats.pt")
    write_ply(out / "model.ply", params, centre, scale)
    print("[train] metrics", json.dumps(metrics), flush=True)
    return metrics


def evaluate(params, val_v, render, env, ray_dirs_world, device, tile=768):
    import torch

    if not val_v:
        return {"psnr": None}
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except Exception as exc:  # pragma: no cover - optional metric
        print("[eval] LPIPS unavailable:", exc, flush=True)
        lpips = None
    psnr, psnr_cc, ssims, lp = [], [], [], []
    with torch.no_grad():
        for v in val_v:
            rgb = torch.zeros((v["h"], v["w"], 3), device=device)
            for y0 in range(0, v["h"], tile):
                for x0 in range(0, v["w"], tile):
                    t = crop_view(v, x0, y0, min(tile, v["w"] - x0), min(tile, v["h"] - y0))
                    out_img, alpha, _ = render(t, _sh_of(params))
                    part = out_img[..., :3]
                    if env is not None:
                        part = part + (1 - alpha) * sky_colour(torch.sigmoid(env), ray_dirs_world(t))
                    rgb[y0:y0 + t["h"], x0:x0 + t["w"]] = part
            rgb = rgb.clamp(0, 1)
            gt = torch.from_numpy(v["rgb"]).to(device).float() / 255.0
            mask = None if v["mask"] is None else torch.from_numpy(v["mask"]).to(device).bool()
            sel = (lambda t: t[mask]) if mask is not None else (lambda t: t.reshape(-1, 3))
            mse = ((sel(rgb) - sel(gt)) ** 2).mean()
            psnr.append(float(-10 * torch.log10(mse)))
            # exposure-fair: best affine colour map fitted on this image
            X = torch.cat([sel(rgb), torch.ones_like(sel(rgb)[:, :1])], 1)
            sol = torch.linalg.lstsq(X, sel(gt)).solution
            mse_cc = ((X @ sol - sel(gt)) ** 2).mean()
            psnr_cc.append(float(-10 * torch.log10(mse_cc)))
            ssims.append(float(ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean()))
            if lpips is not None:
                lp.append(float(lpips(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
    return {"psnr": float(np.mean(psnr)), "psnr_exposure_fair": float(np.mean(psnr_cc)),
            "ssim": float(np.mean(ssims)), "lpips": float(np.mean(lp)) if lp else None}


def crop_view(v: dict, x0: int, y0: int, cw: int, ch: int) -> dict:
    """The same camera restricted to a pixel window: shift the principal point, keep the focal length."""
    K = v["Kt"].clone()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return {**v, "Kt": K, "w": int(cw), "h": int(ch)}


def _sh_of(params) -> int:
    k = params["shN"].shape[1] + 1
    return int(round(math.sqrt(k))) - 1


def write_ply(path: Path, params, centre, scale):
    """Standard 3DGS PLY in WORLD coordinates (undo the normalisation)."""
    import torch
    from plyfile import PlyData, PlyElement

    with torch.no_grad():
        means = params["means"].cpu().numpy() / scale + centre
        scales = params["scales"].cpu().numpy() - math.log(scale)
        quats = torch.nn.functional.normalize(params["quats"], dim=-1).cpu().numpy()
        op = params["opacities"].cpu().numpy()
        sh0 = params["sh0"].cpu().numpy().reshape(len(means), 3)
        shN = params["shN"].cpu().numpy().transpose(0, 2, 1).reshape(len(means), -1)
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(shN.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, op[:, None], scales, quats], 1).astype(np.float32)
    arr = np.empty(len(data), dtype=[(n, "f4") for n in names])
    for j, n in enumerate(names):
        arr[n] = data[:, j]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--cap", type=int, default=1_000_000)
    p.add_argument("--long-side", type=int, default=960)
    p.add_argument("--sh-degree", type=int, default=1)
    p.add_argument("--depth-weight", type=float, default=0.0)
    p.add_argument("--sky", action="store_true")
    p.add_argument("--aniso-weight", type=float, default=0.0)
    p.add_argument("--max-minutes", type=float, default=600)
    p.add_argument("--carve", action="store_true")
    p.add_argument("--init-ckpt", type=Path)
    p.add_argument("--lr-scale", type=float, default=1.0)
    p.add_argument("--crop", type=int, default=1024)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--packed", type=int, default=1)
    p.add_argument("--depth-loss", choices=["rel", "inv"], default="rel")
    p.add_argument("--scale-cap-frac", type=float, default=0.3)
    p.add_argument("--opacity-reg", type=float, default=0.01)
    p.add_argument("--noise-lr", type=float, default=5e5)
    p.add_argument("--init-opacity", type=float, default=0.1)
    p.add_argument("--panos", default="", help="train on panoramas with route index in a range, e.g. 16-39")
    a = p.parse_args()
    train(a.scene, a.out, a.steps, a.cap, a.long_side, a.sh_degree, a.depth_weight,
          a.sky, aniso_weight=a.aniso_weight, max_minutes=a.max_minutes, carve=a.carve,
          init_ckpt=a.init_ckpt, lr_scale=a.lr_scale, crop_px=a.crop, snapshot=a.snapshot,
          packed=bool(a.packed), depth_loss=a.depth_loss, scale_cap_frac=a.scale_cap_frac,
          opacity_reg=a.opacity_reg, noise_lr=a.noise_lr, init_opacity=a.init_opacity, panos=a.panos)


if __name__ == "__main__":
    main()
