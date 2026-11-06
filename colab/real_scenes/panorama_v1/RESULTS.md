# Panorama → 3DGS: measured results (newest first)

All runs: Kaggle 2×T4, trainer `train_splats.py` (gsplat 1.5.3 library, MCMC, SH1, 960 px views),
held-out = whole panoramas (every 8th along the route, never the first or last), PSNR/SSIM/LPIPS on all
views of those panoramas. "Exposure-fair" fits one colour affine per held-out image first.
Contact sheets (held-out photo/render pairs + novel off-path cameras) are checked by eye for every run.

## Round 2 — priors (GGPS FTP, 40 panoramas, 7,000 steps, rig-aware SfM poses)

| Test | Priors | PSNR | exp.-fair | SSIM | LPIPS | needles | aniso p50 | Off-path by eye |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H2 | none | 24.09 | 26.31 | 0.824 | 0.253 | 45 % | 16.5 | coherent; streaky sky, edge smears |
| **H3 (no carving)** | **MoGe-2 dense init + depth loss, sky dome + sky mask, vehicle/people masks, needle penalty** | **25.31** | **28.49** | **0.879** | **0.157** | **16 %** | **4.5** | **near-photo; clean sky with clouds; no floaters seen** |

- MoGe-2 aligned on 476/480 views (median inlier ratio 0.82); per-view scale spread 0.55 (log std) —
  per-view alignment is required, one global scale would not do.
- H3-full (with carving) crashed: two experiments loaded CUDA models from threads of one process
  (lazy CUDA init is not thread-safe). Priors now run in their own subprocess; rerun queued.

## Browser game (synthetic level, published through the full chain)
- Chain: export_game -> Spark build-lod (.rad) -> Vite build of the MIT third-person splat template
  + NAMMA layer -> HF Space `world/<tag>/`.
- Chrome on Apple M2, 1470x780 at DPR 2: **59.9 FPS average (vsync cap), 45.9 FPS 1 %-low**, no errors.
- Built-in browser checks: walking 1.6 m/s at 1.62 m eye height; car 41 km/h straight, no roll in a hard
  turn, stops upright at a wall (fixed-timestep physics).

## Round 1 — GGPS FTP (Insta360, aerial 360 over a park), first 40 panoramas, 7,000 steps

| Test | Poses | Views | PSNR | exp.-fair | SSIM | LPIPS | SfM min | Off-path by eye |
|---|---|---|---:|---:|---:|---:|---:|---|
| H0 | legacy hloc + median rig closure (branch head) | 12×60° horizon band | 22.07 | 24.79 | 0.780 | 0.282 | 72.5 | black blobs top and bottom (sky and ground never trained) |
| H1b | openMVG ground truth | 12×60° horizon band | 21.22 | 23.48 | 0.715 | 0.372 | 0.7 | — |
| H1 | openMVG ground truth | rig 4 yaw × 3 pitch × 90° | 22.10 | 24.60 | 0.760 | 0.346 | 2.7 | — |
| **H2** | **pycolmap.panorama (rig-aware, incremental, sequential)** | **rig** | **24.09** | **26.31** | **0.824** | **0.253** | **26.8** | coherent; streaky sky, edge smears |

Findings
- H1 (as written, ≥3 dB from GT poses) **fails**: GGPS's openMVG poses are not better than our SfM.
  Rig-aware SfM on the exact training views is self-consistent (0.47 px) and gives 2.3× the
  initial points; it is the production pose solver.
- Rig views vs horizon band (same GT poses): +0.9 dB and no black holes off-path.
- Branch-head pipeline → H2: +2.0 dB PSNR, SSIM 0.78 → 0.82, LPIPS 0.28 → 0.25, SfM 2.7× faster.
- Remaining defects at H2: 41 % near-transparent splats, 45 % needles, sky made of splats.

## Diagnostics
- openMVG spherical convention == pycolmap panorama convention: openMVG's own observations
  reproject at 0.34 px median (alternative transpose: 184 px). GGPS FTP has 354 panoramas,
  3840×1920, 274,870 structure points; `colorized.ply` = structure + camera centres.
- Smoke (10 panoramas): pycolmap.panorama registered 120/120 views, 0.39 px, within-panorama
  centre spread 0 (the old per-crop SfM had up to 9.4 units).
