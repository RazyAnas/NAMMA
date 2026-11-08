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

## Ablation `dbg` (same scene, 4k steps each): the collapse comes from growing past ~1.6M splats
| Variant | PSNR | transparent (<0.05) |
|---|---:|---:|
| H8 settings, old ~22 m cap | 15.7 | 99 % |
| H8 settings, 3 m cap | 17.2 | 99 % |
| no depth loss | 12.1 | 99.7 % |
| inverse depth loss | 16.7 | 99 % |
| unpacked raster | 15.2 | 99 % |
| aniso 0.01 | 14.0 | 99 % |
| no training window | eval crashed (tile 0 bug, fixed) | 96 % at step 2.4k and rising |
| H7c-like (1.6M cap, 960 px, unpacked) | eval crashed (same bug) | **69 %, stable** |
Per-step logs: with a 5M cap the transparent share climbs in step with MCMC growth (66 % at 1.7M splats, 90 % at 2.1M,
100 % at 3.1M); with the 1.6M cap it settles at ~0.6-0.7. Window, packed raster and depth-loss form are not the cause.
The 3 m cap helps (+1.5 dB). PanoVGGT now loads (0 missing keys) and gives slightly steadier depth than MoGe-2
(per-view scale log-std 0.199 vs 0.213).
H10 queued: PanoVGGT, 1.6M cap, 3 m cap; arm A 8K panoramas / 1600 px windows + densify, arm B 960 px H7c-like.

## H10 (PanoVGGT, 1.6M cap, 3 m size cap, 30k steps): trained, still collapsed; export stalled
| Arm | PSNR | exp.-fair | SSIM | LPIPS | transparent (<0.05) | aniso p50 | minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| H10-safe (960 px, unpacked, inv depth) | 17.81 | 19.20 | 0.689 | 0.587 | 94 % | 5.8 | 92 |
| H10-hires (8K panos, 1024 px windows) | 16.56 | 17.75 | 0.678 | 0.668 | 99 % | 1.6 | 86 |
- The 1.6M cap did not stop the collapse: H10-safe went 65 % transparent at step 1.5k, 96 % at 4.5k, ~100 % from 6k to 22k,
  back to 94 % once MCMC refinement stopped; median splat grew from ~15 cm to ~2.4 m, single-view loss spiked to 0.7.
  Renders: wide streets readable but soft, near-camera blobs, narrow-street held-out views are grey fog.
- Not like-for-like with H7c: H7c's incremental SfM kept only 24 panoramas of one straight street (252 views). Since H7d the
  global mapper registers all 40, a route with two sharp turns (pano 6 and 16); held-out pano 16 sits on a corner and pano 8
  right after a turn. PanoVGGT depth aligned well (388/480 views, inlier median 0.87).
- Densify crashed on a missing package (ftfy, fixed). Export ran 5+ h in the collider/texture step (H8 stalled at the same point)
  until the job was stopped: UV unwrap and the export stage now have hard time limits.
- H11 queued: same scene, 8 variants stopped at ~8k steps of the 30k schedule: no carving, no sky dome, no opacity
  regulariser, 10x less MCMC noise, initial opacity 0.5, no size cap, and the straight street only (pano 16-39).

## H11 ablation (same scene as H10-safe, 30k schedule cut at 25 min ≈ 8–9k steps): the opacity regulariser causes the collapse
| Variant | PSNR | Exposure-fair | SSIM | LPIPS | Opacity < 0.05 | Aniso p50 | Off-path sharpness |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (H10-safe settings) | 16.28 | 17.58 | 0.668 | 0.648 | 95% | 3.8 | 7.2 |
| no carving | 16.27 | 17.79 | 0.670 | 0.602 | 99% | 4.6 | 19.8 |
| no sky dome | 16.10 | 17.70 | 0.651 | 0.642 | 93% | 3.1 | 1.7 |
| **opacity reg 0** | **19.56** | **20.42** | **0.709** | **0.549** | **23%** | **1.7** | **48.3** |
| straight street only (panos 16–39) | 20.02 | 20.86 | 0.702 | 0.550 | 79% | 5.6 | 67.1 |
| MCMC noise ÷10 | 15.70 | 17.33 | 0.653 | 0.598 | 100% | 4.8 | 33.4 |
| initial opacity 0.5 | 16.14 | 17.49 | 0.672 | 0.634 | 97% | 4.0 | 3.9 |
| no 3 m size cap | 12.91 | 15.58 | 0.636 | 0.676 | 98% | 4.4 | 0.4 |

- Only removing the opacity regulariser stops the collapse: transparent fraction holds at ~24% from step 4.5k to 9k
  (base: 96% at 4.5k) and median splat size stays ~15 cm (base grows to ~0.9 m). +3.3 dB on the same scene.
- Likely mechanism (not measured): with the dense 1.5M-point init the 0.01 × mean-opacity term outweighs the photometric gradient per splat, so MCMC
  relocates dead splats, they grow to cover the gap, and the reg pushes them transparent again.
- Contact sheets: opreg0 shows readable Paris facades, trees and street furniture from mid-path, side-step and
  raised views; base is grey fog. The road under the capture car (car-spot-down) is still a grey patch in both:
  that is the vehicle mask hole, for densify/fill to cover.
- The straight-street subset scores higher but still collapses (79%), so the turns are not the cause.
- Default `opacity_reg` is now 0. H12 queued: full route and straight street, opacity reg 0, full 30k steps,
  densify 1 round, export with time limits.

## H12 (opacity reg 0, full 30k steps): no collapse; densify then ran out the 12 h limit
| Arm | PSNR | Exposure-fair | SSIM | LPIPS | Opacity < 0.05 | Aniso p50 | Needles | Off-path sharpness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H12-full (40 panos, 420/60 views) | 18.90 | 19.97 | 0.694 | 0.444 | 11% | 5.3 | 17% | 243 |
| H12-straight (panos 16–39, 252/36 views) | 21.04 | 21.86 | 0.736 | 0.370 | 9% | 6.7 | 21% | 414 |

- Training is stable to the end: transparent fraction 24% → 11% (full) and 26% → 9% (straight), median splat
  size ~15 cm throughout, 79–82 min for 30k steps. LPIPS 0.444 vs 0.587 for H10-safe on the same scene.
- Contact sheets: mid-path, side-step and raised views show sharp, readable Paris facades, trees, shopfronts and
  road markings. Problems left: spiky needles on the near road, and the grey vehicle-mask hole under the car.
  The straight arm's held-out panos 8 and 16 sit outside its training range, so those tiles are not meaningful.
- Densify (Wan VACE) took ~21 min per clip; `max_minutes` only applied to the fine-tune, so 27–29 of 40 clips ran
  for ~10 h and Kaggle stopped the job before export. No model was kept (`keep_ply` false). The straight arm's
  densify also walked the whole route, so its clips outside panos 16–39 were 75–100% holes.
- Fixed: densify now has a time budget (`stage_minutes`, default 60) and a hard subprocess timeout.
- H13 queued: full route trained again with export and the model kept; arm B uses aniso weight 0.05 against the needles.

## H13 (H12-full settings, export on, model kept): first exported world since the collapse fix
| Arm | PSNR | Exposure-fair | SSIM | LPIPS | Opacity < 0.05 | Aniso p50 | Needles | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H13-full (aniso 0.01) | 19.07 | 20.18 | 0.701 | 0.434 | 10% | 5.2 | 17% | 178 min |
| H13-aniso05 (aniso 0.05) | 19.12 | 20.21 | 0.703 | 0.436 | 10% | 5.2 | 16% | 177 min |

- Aniso weight 0.05 changed nothing measurable or visible; the near-road needles are not fixed by it.
- Export finished in ~69 min. The new step logs show where H8/H10 stalled: TSDF fuse 57 s, **mesh extraction
  53 min** (792×106×959 grid at 0.30 m, 2.7M faces), simplify to 400k 7 s, then the xatlas UV unwrap hit its
  15 min limit, so there is no textured surface layer. Splats (1.53M out, 208k needles clamped), collider and
  ground heightfield were written. Ground heightfield covers only 15% of its grid.
- Next for export: extract the mesh on the GPU or at a coarser voxel, and unwrap the simplified 150k mesh only.
- Published H13-full as world `paris-h13`.

## paris-h13 published and tested in Chrome (2026-11-08)
World: https://anasrazy-namma-street.static.hf.space/world/paris-h13/index.html (new Space: `namma-tour` hit its 1 GB limit).
- Readable Paris street around the avatar: facades, windows, shopfronts, crosswalk, scaffolding. Raw PLY in a bare
  Spark page renders the same, so the exported splats are correct.
- Four publish bugs found and fixed on the way:
  1. The GitHub secret `KAGGLE_TOKEN` cannot read `anasrazy` kernels; the workflow now tries each stored token.
  2. Spark `build-lod` `.rad`: published, but the scene never showed. Now `"rad": false` ships `scene.ply`
     (140 MB, ~2 min to load) and Spark builds LoD in the browser.
  3. The player spawned in the collider hole left by the vehicle mask and fell forever (camera at y = −142 km), so
     only the sky dome was ever visible — this was the "blocky" look. Client now adds a 6 m road strip collider
     along the route and resets the player if they fall.
  4. Mesh cleanup `trimesh.split` took ~53 min in export; replaced with one sparse components pass (19 min → 0.2 s
     on a 1.9M-face test mesh, identical output).
- Measured: benchmark (B) 30.2 fps average, 1%-low 30.1, on this Mac in Chrome.
- Still wrong: the learned sky dome (`sky.jpg`, 2048×1024) has the buildings baked into it, so blocky buildings
  show wherever splats are thin (raised views, holes); smeared near-camera clutter (parked scooters); the grey
  vehicle-mask patch on the road. Not yet checked: whether `.rad` itself is broken or was only hidden by bug 3.

## Sky backdrop fixed (2026-11-08)
- Cause: the trainer's sky dome (256×512) learned from every see-through pixel, so facades were baked into
  `sky.jpg`; export also sampled it nearest-neighbour. In the game this showed as blocky buildings wherever
  splats were thin.
- Trainer: the dome now gets gradients only from MoGe sky pixels (it is still composited everywhere, frozen),
  and records which texels were seen as sky (`env_hits` in `splats.pt`). Export samples bilinearly and fills
  unseen texels from the sky above, fading to haze at the horizon (`sky_clean.py`). Not yet run on a GPU.
- Old exports: the publish workflow runs `sky_clean.py` (guesses sky texels: bright, smooth, connected to the
  zenith) when `game.json` has no `sky_clean` flag. paris-h13 republished this way; checked in Chrome: plain sky
  at street level and from the raised benchmark view, no buildings in it.
- Also fixed: the avatar still fell through the road strip after the splats loaded (it spawned exactly on the
  floor's top face). Spawn is now 0.3 m above a 2 m thick strip, the fall reset runs before the move, and fall
  speed is capped. Stable on the street after load; benchmark 30.1 fps average, 1%-low 30.

## H14: why roads and buildings look bad (2026-11-08)
Same Paris route as H13. New in the trainer: fit on 40 training views and a photo|render sheet (`fit_sheet.jpg`).

| arm | splats | held-out PSNR / SSIM / LPIPS | training-view PSNR / SSIM | train min |
|---|---|---|---|---|
| H14-base (H13 settings) | 1.6M | 19.42 / 0.713 / 0.422 | 26.07 / 0.844 | 78 |
| H14-big (long side 1024) | 4.0M | 19.36 / 0.704 / 0.419 | 26.55 / 0.861 | 118 |

- Poses are not the main problem: training views line up and facades are sharp (26 dB).
- 2.5× more splats changes nothing held-out, so capacity is not the problem either.
- Road: the capture car is cut out of every panorama below about −46° (`vehicle_mask_erp.png`, 30 % of the
  panorama). Nothing ever trains the road within ~2.5 m of each capture spot. It stays see-through, and the pale
  sky dome shows as a grey blob with shards (visible in every training view and every `car-spot-down` off-path
  view).
- Buildings and near objects: good where photographed, shards and smears between panoramas 10 m apart (bins,
  scooters, close facades). This is view sparsity, not fitting.
- Fix in code (not yet run): `ground_seed.py` puts an opaque flat road disk (radius 0.35 × pano spacing) at the
  road height measured from nearby cloud points under every panorama. A `--solid-weight` loss makes every pixel
  below the horizon or not sky opaque, masked or not. H15 (road fix; road fix + depth weight 0.5) is
  configured in `experiments/h15.json` but blocked: the Kaggle account's 30 h weekly GPU quota is used up.
