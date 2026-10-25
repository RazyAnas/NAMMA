# Findings that must not be lost (newest first)

## 2026-10-25 (3): gsplat opacity reset never runs (confirmed in gsplat 1.5.3 source; found by the Mac session)
`DefaultStrategy.step_post_backward`: `if step % self.reset_every == 0 & step > 0:` is always False (`&` binds before `==`), so
the 3DGS opacity reset that clears floaters never happens. Every gsplat fit so far ran without it: a strong candidate for the
held-out fog. `train_splat.py` now resets opacities to 0.01 every 3000 steps while densifying. Training cap raised to 30 min for
the 600-view variants (the only capped run so far was A at step 10028; S1/D/V1 finished 15000 steps in 470-520 s).
Sanity check still worth one run: Nerfstudio splatfacto on the same SfM cameras (`scripts/ns_env.sh`), lobby gave 25-27 dB.

## 2026-10-25 (2): V1/V2 run
- SfM again 195/200, 1.14 px. DA3/SfM per-frame scale spread 47%.
- V1 (SfM cameras, no refinement): held-out 18.79 dB raw, SSIM 0.780, 841k splats. Owner-visible held-out renders are FOG: fan,
  window, door gone. PSNR on white-wall rooms hides this; always look at the held-out images.
- V2 (2M MCMC + SH3, fixed lr/opacity): 12.55 dB, train 16.0. Still broken. Found: we ran densify/relocate BEFORE the optimizer
  step; gsplat's trainer does it after. Fixed. MCMC paused until a plain fit is right.
- Floor up from mean image-up: biased by camera pitch when walking mostly one way (synthetic: 25 deg). Now from the camera x axes
  (always horizontal for a phone without roll): 0.4 deg in the same test.
- Next: 600 frames (SfM on all, DA3 on every 3rd), W1 vs W2 (stronger depth anchor), step-0 and final renders + splat stats.

## 2026-10-25: SfM cameras run (200 frames, Kaggle T4, 69 min)
- hloc SuperPoint+LightGlue (sequential 10 + NetVLAD 15 pairs): **195/200 registered, 8425 points, reprojection error 1.07 px**.
- Chunked DA3 depth scale wandered **32.8% between frames** (std of log DA3/SfM ratio): DA3 alone cannot give consistent geometry
  for a long walk; use it only for dense depth after per-frame alignment to SfM (`sfm_align.py`).
- Held-out (frames 3,11,19,..., 25 views) with camera refinement on, after aligning held-out cameras:
  SfM cameras 22.12 dB / SSIM 0.841 (722k splats); DA3 cameras 22.42 / 0.836 (645k). Raw cameras: 17.1 dB for both.
  -> camera refinement co-drifts with the scene (moved even SfM cameras ~10 cm / 2.8 deg). Turned off for SfM cameras.
- 2M MCMC + SH3 variant: 15.3 dB (train 16.3), clearly broken training, NOT evidence against more splats. Suspected cause: means
  learning rate scaled by point-cloud extent (~3x the camera extent) + init opacity 0.5. Fixed (camera extent, opacity 0.1).
- Floor "up" from surface normals was 25 deg off (SfM run) and 88 deg off on a synthetic room (it picked a wall): walkable area
  collapsed to 0.1-0.5 m2. Now taken from the upright phone cameras; synthetic walkable 4.5 -> 14.0 m2, robust to a 25 deg tilt.

## 2026-10-24: the SDXL-filled panorama invents the room (owner-verified)
`matrix3d/kaggle/run.py` builds `pano_img.jpg` from 22 Depth Anything 3 frames; ~55% of the sphere is unseen and SDXL inpaints it.
The owner (who was in the room) says the result shows a **yellow door that does not exist** and a **vent that is really a fan**; it does
not match the video. Earlier notes calling its contents "real" were reported, never verified. Treat that panorama as hallucinated.
- Do NOT feed it to Matrix-3D / Wan: the video model spreads the invention to the whole walk-through.
- Rule from the owner: generation may only fill gaps, never replace what was filmed, and must stay identifiable as generated.
- Safer order: (1) best splat from real frames (`final/run.py`, A/B with held-out PSNR), (2) find holes by rendering off-path views,
  (3) repair renders with a model that edits the render (Difix3D+) instead of inventing a whole panorama, (4) keep a per-pixel/
  per-Gaussian "generated" flag so the viewer can show what was not filmed.
- If a panorama is still wanted: render it from the 200-frame cloud only, leave the unseen pixels as an explicit mask, and have a human
  check it against the video before it goes into any generator.

## Measured so far (real video, 3:45, 4K, Kaggle T4)
- DA3 chunked (8 views, 3 shared, Sim3 merge): all frames placed; 60-frame splat looked better than the point cloud but blurry and broken
  (owner screenshot). Train-view PSNR 25.1 dB (training views, not a quality claim). A/B with 200 frames + camera correction + depth anchor
  is the next measurement (held-out PSNR/SSIM printed by `train_splat.py`).

## 2026-10-25: verified-camera list
- From the all-frame real-pixel check (median 34.3 dB): keep frames whose rebuild is >= 20 dB -> `good_frames.json`, 507/596.
  Dropped stretches 60-63, 89-108, 323-330, 444-447, 526-549 (corridor look-alikes linked by NetVLAD).
- Walk-order-only SfM (`namma-resfm`) was worse: 372/600 registered in 5 models, then OOM in sfm_align (lazy npz reads, fixed).
  So: keep the original SfM, drop the bad cameras. `ibr.py/ibr_check.py --keep`, training via env `KEEP_FRAMES`.
- The list is only valid for the current namma-prep cameras; re-run ibr_check if prep is re-run.
- Run 2 (507 kept): held-out real-pixel rebuild 27.0 -> **30.4 dB, SSIM 0.90**; camera-check median 34.9 dB. But 28 kept frames
  now fail (64-100, 274-294, 318-332, 439-443, 525-552): a misplaced stretch rebuilds itself fine and hides until its worst
  frames go. Frame 75 still shows a door where the photo has the window.
- Bug fixed in ibr.warp: np.round (half-to-even) left every other pixel empty when warped positions were near integers
  (synthetic test: coverage 0.30 -> 1.0). Real runs were less affected, but off-path renders had speckle from it.
- New `ibr_check --prune`: cut the walk where the camera jumps (step > 6x median and > 10 cm), drop short pieces whole, then
  drop < 20 dB and re-score the frames that used them, until clean. Synthetic test with 8 planted misplaced frames: drops exactly those 8.
