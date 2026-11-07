# Panorama → 3DGS: measured results (newest first)

All runs: Kaggle 2×T4, trainer `train_splats.py` (gsplat 1.5.3 library, MCMC, SH1, 960 px views),
held-out = whole panoramas (every 8th along the route, never the first or last), PSNR/SSIM/LPIPS on all
views of those panoramas. "Exposure-fair" fits one colour affine per held-out image first.
Contact sheets (held-out photo/render pairs + novel off-path cameras) are checked by eye for every run.

## Round 4 — H7c street run (GPS spatial matching, vehicle removal, 15k steps)

| Test | SfM | Views registered | PSNR | SSIM | LPIPS | Notes |
|---|---|---:|---:|---:|---:|---|
| H7-full | spatial + incremental | 288/480 (5 models) | 22.15 | 0.757 | 0.352 | base model; Fixer round: 22.47 / 0.757 / 0.397 |
| H7-global | spatial + **global** | **480/480 (1 model)** | — | — | — | crashed at step 9000 (OOM: views cached on GPU) |

- Metric scale cross-check: GPS 19.82 m/unit vs MoGe-2 19.84 m/unit (0.1 %), GPS fit RMS 0.21 m,
  camera height 2.07 m (roof-mounted car camera).
- NVIDIA Fixer changed a median 0.8 % of pixels and made LPIPS worse (0.352 -> 0.397): off by default
  (the video pipeline saw the same).
- Contact sheet bugs, all fixed + tested: ground/car-spot cameras pointed up (sign), unobserved sky
  zenith left grey, MoGe depth of the blanked vehicle fused into grey blobs, fill ray march OOM.
- Floaters: 0 % beyond 10x route radius, 0 % in proven-empty space.

## Round 3 — first real streets (Panoramax, Paris, car-mounted 360, 40 panoramas at 10 m)

| Test | Setup | PSNR | SSIM | LPIPS | Notes |
|---|---|---:|---:|---:|---|
| H7-plain | rig SfM (sequential), no priors | 18.93 | 0.650 | 0.627 | SfM split into 4 models; 24/40 panoramas used |
| H7-priors | + MoGe-2, sky, masks, carving | 22.33 | 0.690 | 0.455 | +3.4 dB; carving removed 11,064 splats, 0 % left in empty space; bonnet baked in |
| H7b | spatial matcher | — | — | — | failed: no GPS reached COLMAP (needs lat+lon+alt; OpenCV re-save dropped EXIF) |

Fixes verified before the next run:
- COLMAP 4.2.1 stores a GPS pose prior only with latitude, longitude and altitude (checked locally: 3/3).
- Vehicle mask = static pixels ∪ Grounding DINO + SAM 2 vote on a 140° downward view of 12 panoramas,
  instances >70 % of the view ignored. Overlay on a real panorama: bonnet, roof and no-data band
  masked (30 %), road, pavement and pedestrians kept (`vehicle_mask_overlay.jpg`).
- GGPS H3-full (carving on): 25.19 dB / 0.876 / 0.162 vs 25.31 without carving: carving costs nothing
  measurable and keeps proven-empty space at 0 % floaters.
- H4/H5 failed loading the LCM-LoRA (Kaggle torchao 0.10 vs peft); torchao removed, LCM falls back to
  normal sampling, and optional rounds no longer abort a run.

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

## H7d (fill vs no fill, 20k steps): both crashed
Both arms ran out of GPU memory at step 10k (no fill) and 15k (fill) inside gsplat `intersect_tile` (3.3 GB request):
splats grew large on screen late in training. Priors were fine (480 views masked, 386/480 depth-aligned,
scale log-std 0.19). Fix (trainer): packed rasterisation, max splat axis 10 m, skip a view on OOM, and a scale-free
depth loss (the old inverse-depth L1 spiked on close road views). H8 relaunched with the fix, plus H9 densify on the MoGe arm.

## H8 v2 (8K panoramas, 1600 px, 6M cap): stopped at step ~5.1k
Both arms: 200 out-of-memory skips by step 5.1k once splats reached 6M (2.7-3.1 GB intersection buffer), then
the final carve ran out of memory too. PanoVGGT did not run (omegaconf mismatch on Kaggle), so both arms used MoGe.
Fix: train on random 1024 px windows of each full-res view (principal point shifted), render eval/off-path frames
in 1024 px tiles, free optimiser state before the final carve, pin omegaconf 2.3.0 / hydra-core 1.3.2. Cap 5M.

## H8 v3 (1024 px windows, 5M cap): stopped at step ~9k
Got further (no out-of-memory until step ~6k), then skips climbed as splats spread and it stopped at 9.2k / 8.7k.
PanoVGGT still fell back to MoGe (pinning omegaconf did not help). Fixes: gsplat 32 px tiles (about 4x fewer
tile intersections for big splats, falls back to 16), max splat axis 3 m, the training window shrinks by 15% per
10 out-of-memory skips (floor 512 px) instead of stopping, 768 px eval tiles; PanoVGGT config read with plain YAML
(no OmegaConf), and checkpoint key mismatches are now reported instead of ignored.

## H8 v4 (30k steps, 5M cap, 1024 px windows): trained to the end, but the model collapsed
Both arms finished training (no crash; 10 out-of-memory skips), then Kaggle stopped the job at its 12 h limit during export.
Held-out PSNR **11.4 dB** (MoGe) / 11.3 dB (PanoVGGT label, still MoGe in practice), SSIM 0.63, LPIPS 0.70 - versus 22 dB for H7c.
99.9 % of splats have opacity < 0.05 and splats are near-spheres (anisotropy p50 1.03 vs 9 in H7c): every render is the sky
dome's flat colour. Densify could not help (real-pixel fraction 0). The size cap was also wrong: log(0.15) normalised units is
~22 m here, not 3 m (1 normalised unit = route radius, 147 m); the cap now comes from panorama spacing (0.3 x ~10 m).
Ablation `dbg` queued: H8 settings vs old/new cap, no depth loss, inverse depth loss, unpacked raster, no training window,
aniso 0.01, and an H7c-like reference, 4k steps each on the same scene.
