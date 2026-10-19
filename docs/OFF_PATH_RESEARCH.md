# Off-path quality: what we measured, what the field does, what to try next

Status 2026-10-19. "Off-path" means a view the phone never filmed from. The walking demo's bad screenshots (streaks,
smeared walls, floating blobs) are off-path views; on the camera path the same splats score 31-36 dB.

Labels: **[M]** measured in this project, **[Q]** quoted from a paper, not reproduced, **[P]** pending.

## 1. What we measured

| Test | Result | Label |
|---|---|---|
| Phone room 30k splat, 36 held-out views on the path (our CPU renderer, `scratchpad/clean/render.py`) | 33.8 dB mean (the training run reported 35.5 dB; per-view within ~0.1-4 dB, so the renderer is usable for A/B) | M |
| Lobby 30k splat, same renderer, frames 48 / 5549 | 31.2 / 31.9 dB (training run: 31.5 / 33.7) | M |
| Post-hoc cleanup, phone room: shrink needles (long axis > 6x middle) | -1.19 dB on the path | M |
| ... needles > 4x middle | -2.25 dB | M |
| ... drop Gaussians within 25 cm of a training camera | -4.21 dB (worst view 15.8 dB): in a small room those are real surfaces | M |
| ... drop large faint Gaussians (> 10 cm, opacity < 0.35) | -1.59 dB | M |
| Same cleanups, rendered 0.6 m back and 0.4 m sideways from four test cameras | No visible improvement: the streaks and smears are the same | M |
| Hold out a stretch of the walk (frames 448-516, 35 views, median 18.5 cm from the nearest training camera), OpenSplat 3k on the Mac CPU, default recipe | **15.40 dB** (20.14 after an affine colour fit), SSIM 0.752. Same scene on the path: ~34 dB | M |
| ... with the anisotropy cap (max/min 10, weight 0.1) | 15.52 dB (20.33): +0.11 dB, 95% CI +0.05 to +0.18, better on 20 of 35. Real but small | M |
| ... TSDF mesh from our anchored depth (other work stream; reproduced here) | 18.76 dB, SSIM 0.811: the plain mesh beats the splat off the path by ~3.4 dB, so geometry is the weak part | M |
| ... same mesh textured from the 4 nearest frames (photogrammetry / SkyeBrowse style) | 18.84 dB, SSIM 0.824 | M |
| ... splat initialised on that mesh, same 3k OpenSplat recipe | 14.59 dB: worse than default. The start alone does not hold without a geometry loss (depth / normal / surfel) during training | M |
| ... anisotropy cap 5, weight 1.0 | 15.48 dB: +0.08, 95% CI -0.08 to +0.23, not significant; training less stable | M |
| ... SH degree 1 | not run locally (the error-compensation step needs SH 3); in Kaggle round 3 (`SH_MAX=1`) | P |
| Same on Colab / Kaggle T4: gsplat default / MCMC / depth / DN-style / 2DGS, OpenSplat 30k with and without the cap | scripts ready (`colab_bakeoff.sh`, `kaggle_bakeoff.ipynb`), not run yet | P |
| Difix3D+ on the best round-1 splat, same held-out stretch | scripts ready (`colab_round2.sh`) | P |
| VGGT second opinion on the lobby poses (is the 20 dB middle a pose problem?) | scripts ready (`colab_round2.sh`) | P |
| Earlier: VGGT depth vs DA-V2 for the walkable floor (ICL) | 199 / 300 true floor cells vs 22; splat 0.22 dB worse, so floors only | M |

### GPU bake-off, same held-out stretch (Kaggle T4 x2, 2026-10-19) [M]

| Run | PSNR | colour-fit PSNR | SSIM |
|---|---|---|---|
| OpenSplat 3k (reproduces the Mac CPU run, 15.40) | 15.34 | 19.92 | 0.741 |
| OpenSplat 30k | 16.48 | 20.94 | 0.759 |
| OpenSplat 30k + anisotropy cap 10 | 16.13 | 20.61 | 0.745 |
| gsplat default, 7k | 16.29 | 21.06 | 0.758 |
| gsplat MCMC, 7k | 15.94 | 20.63 | 0.745 |
| gsplat MCMC + depth loss, 7k | **16.61** | 20.78 | **0.768** |
| gsplat DN-style (depth + normals), 7k | 15.64 | 20.19 | 0.754 |
| gsplat 2DGS | crashed (distortion loss on an RGB-only test render; fixed in `gsplat_train.py`) | | |
| TSDF mesh from our depth (CPU) | **18.76** | | **0.811** |

Every training recipe lands within 15.3-16.6 dB; the best (depth loss) gains ~1.3 dB over the 3k baseline, and
the plain mesh is still ~2 dB ahead. Longer training (3k to 30k) helps a little off the path (+1.1 dB), so it is not
the overfitting trap it could have been, but no recipe closes the gap. Next: round 3 (mesh-initialised + depth loss,
video depth, per-image colour) and round 2 (Difix3D+) are queued in the same session.

**Conclusion so far.** Off the path we lose ~19 dB within 18 cm. A finished splat cannot be edited into good off-path views: every cleanup costs on-path quality
and none fixes the off-path renders. The fix has to come from (a) more viewpoints at capture, (b) training that
constrains geometry, or (c) a generative prior that supplies what was never filmed.

## 2. What the field does (2025-2026)

### Capture (cheapest, biggest effect)
- Loops at two or three heights, slow movement, locked exposure, overlap, ending at the start. SkyeBrowse's public
  capture guide says the same; their whole product relies on it (see section 4). [Q]
- Visual-inertial tracking (ARCore) instead of image-only SfM on white walls. Our phone room (AR) reached 35.5 dB;
  the lobby (plain video, COLMAP split into 4 pieces) sits at ~20 dB in its middle. [M]

### Training that constrains geometry
- **3DGS-MCMC** (Kheradmand et al., NeurIPS 2024): densification as MCMC sampling, fewer floaters; one flag in gsplat. [Q]
- **DN-Splatter** (Turkulainen et al., WACV 2025): depth and normal losses from monocular priors, built for indoor
  phone captures. [Q]
- **2DGS / PGSR**: flat surfels that sit on surfaces; cleaner walls and usable meshes for collision. [Q]
- **Mip-Splatting**: 3D smoothing filter, removes shimmer when moving closer or further. [Q]
- All of these are in `gsplat_train.py` (default / mcmc / depth / dn / 2dgs) and wait for a GPU run. [P]

### Generative priors for the parts nobody filmed
- **Difix3D+** (NVIDIA, CVPR 2025): one-step diffusion cleans renders from novel cameras, which are fed back as
  training views. Reported off-trajectory gains are modest in PSNR (+0.85 dB Nerfbusters, +0.8 dB DL3DV for 3DGS)
  but large in perceptual quality (FID). [Q] ([paper](https://arxiv.org/abs/2503.01774))
- **SyncFix** (2026): makes the diffusion fix consistent across views (joint latent bridge matching); reports beating
  Difix3D+ on off-trajectory Nerfbusters views. Code status unclear. [Q] ([paper](https://arxiv.org/abs/2604.11797))
- **ViewExtrapolator**, **GenFusion**, **GSFixer**, **TRACE-GS** (2026, sparse views): same family; TRACE-GS reports
  +2.0 to +3.2 dB over plain 3DGS with 9 to 3 input views. Different protocols, not directly comparable. [Q]
- Our own earlier test (ICL): generative fill beats classical fill by +1.08 dB inside holes but is ~13 dB worse
  than really seeing the surface. Generative priors improve how it looks, not how true it is. [M]

### Feed-forward (no per-scene training)
- **AnySplat** (SIGGRAPH Asia 2025), **LongSplat** (AAAI 2026), **StreamSplat**, **AVSplat** (2026): frames to
  splats in seconds. Lower quality than per-scene training, but useful for a quick preview on modest hardware. [Q]

### Poses without sensors
- **GLOMAP / global mapper**: our measured 0.75 cm vs 82 cm for incremental COLMAP on the same video. [M]
- **VGGT**: 20.5 cm vs our 0.95 cm on ICL, so not for poses; good depth. [M]
- **MASt3R-SfM**: robust matching on low texture; the candidate for the lobby if the VGGT check finds pose errors. [Q]

## 3. What we are doing, in order

1. **Capture**: recorder now asks for 60 fps, widest lens, locks exposure and white balance, and shows a filming
   checklist (`app/static/record/`). Needs a real-phone test.
2. **Demo**: fly mode stays within 0.6 m of the capture path (`viewer/src/demo.js`, `data/<room>/cameras.json`),
   so visitors are not dropped into unfilmed views while training improves.
3. **Training bake-off** on the held-out stretch: CPU 3k runs locally now; GPU runs on Colab or Kaggle when someone
   starts them.
4. **Generative prior**: Difix3D+ round 2 after the bake-off picks a base recipe; SyncFix when its code is out.
5. **Lobby**: re-record in AR mode with the checklist; VGGT pose check to see whether the 20 dB middle is poses.

## 4. SkyeBrowse, for comparison

From their public showcase model and the files a browser receives: video, about 360 frames, classic photogrammetry,
a 12 M point Potree 2.0 cloud (converted from LAS) plus a ~300k triangle textured mesh (gltfpack, meshopt), an
orthophoto, and 43 walkthrough cameras; about 10 minutes of cloud processing. Viewer: Potree 1.8.2 on three.js
r109. No splats or learned reconstruction (their own blog: they are not a splat generator). Their edge is capture
discipline and delivery; their reconstruction is the classic approach our splats already beat on the camera path.
Worth copying: gltfpack/meshopt compression for meshes we ship, and a walkthrough mode from the real camera poses.
