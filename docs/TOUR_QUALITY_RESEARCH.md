# Tour quality: owner's problems -> causes -> fixes (26 Oct 2026)

Owner on the first tour (DL3DV hotel room, run 3, tour `20261025-2336-...`): **no walk mode; details could be sharper; smears and
blur at some viewpoints; colours could be better.** Labels: **[M]** measured in this project, **[Q]** quoted from a paper,
**[C]** read in code. Earlier measurements: `docs/OFF_PATH_RESEARCH.md` (19 Oct), the YouTube lessons: `docs/VIDEO_METHOD_PLAN.md`.

## 1. No walk mode
- **Cause [C]**: the SuperSplat Viewer (`@playcanvas/supersplat-viewer@1.37.0`, `dist/viewer.js`) only allows walk mode with a
  collision model AND a scene footprint of at least `WALK_MIN_HORIZONTAL_RANGE = 5` (metres) in both x and z; eyes at
  `eyeHeight = 1.3`, capsule radius 0.2 / height 1.5, `hoverHeight = 0.2`. Run 3's tour had no collision file, and our model is in
  SfM units (room ~3.8 x 3.8 x 1.4 units), not metres.
- **Fix**: (a) metric scale from the camera path: scale so the median phone height above the floor is 1.35 m (handheld height;
  then the viewer's 1.3 m eye height walks exactly where the video was filmed = where the model is best); (b) collision: room
  box (floor + walls aligned to the room's main axes) now, the TSDF mesh (`final/mesh_tsdf.py`) from run 4 on; (c) lower the
  5 m minimum in the viewer file we ship (bedrooms are smaller than 5 x 5 m).

## 2. Smears and blur at some viewpoints (off the filmed path)
- **Measured [M]**: off the path quality drops fast (phone room: ~34 dB on the path, 15-17 dB 18 cm away). Post-hoc cleanups
  of a finished splat cost on-path quality and do not fix off-path views. Run 3: 15.8% needle splats (long in one direction);
  the tour filter removed 15,794 big ones.
- **What fixes it, best evidence first**
  1. **Stay near the filmed path** (what SplatTour does: guided walk, start spots, room crop) and walk at the filmed height
     (fix 1a). Free.
  2. **Capture coverage** (new `docs/CAPTURE_GUIDE.md`: 3 passes per room, ultra-wide). Biggest single lever [M][Q].
  3. **Generative clean-up of novel views, distilled back into the splat**: **Difix3D+** (NVIDIA, CVPR 2025,
     [paper](https://arxiv.org/abs/2503.01774), weights `nvidia/difix_ref`): render views beside the path, clean them with a
     one-step diffusion model (nearest real frame as reference), add them as training views, fine-tune. Measured here on
     19 Oct: +0.37 dB / SSIM +0.021 on our held-out stretch (best splat so far) [M]; paper: +0.8 dB DL3DV, large perceptual
     gains [Q]. Successor: **GSFixer** (ICML 2026, [paper](https://arxiv.org/abs/2508.09667), code github.com/gvclab/GSFixer):
     reference-guided VIDEO diffusion (consistent across frames), reports beating Difix3D+ on DL3DV [Q]; heavier (CogVideoX).
     **"Filling the Unseen"** (arXiv 2609.13262, Sep 2026): quality-aware mask down-weights generated content [Q].
  4. **Indoor floater suppression with depth priors**: **TIDI-GS** (Jan 2026, [paper](https://arxiv.org/abs/2601.09291)),
     StableGS, DN-Splatter, 2DGS/PGSR [Q]. Our gsplat run already uses SfM depth loss; monocular depth priors are the next step.
  5. **The YouTube (Mickmumpitz / Matrix-3D) lesson**: make the unknown explicit (render beside the path, holes = low alpha),
     fill with REAL re-projected pixels first (`final/ibr.py synth(kmax=40)`), generate only what is left with a video model,
     train again; "send in more drones" where it looks bad. = Difix3D+/GSFixer pattern with our real-pixel step first.

## 3. Details could be sharper
- **Cause**: DL3DV `images_4` frames are 960 x 540; a full-screen browser shows ~2x that, so every splat is magnified [M: run 3
  trained at 960 px]. The YouTube lesson: "real high-res pixels wherever they exist" was their biggest jump [Q].
- **Fix**: train from `images_2` (1920 x 1080) at 1600 px (owner's phone video: 4K source -> 1600-1920 px); both T4s
  data-parallel (`simple_trainer` multi-GPU, `--steps-scaler 0.5`) to keep the time near 1.5 h; viewer sharpness post effect
  (amount 0.2, range 0-1) as a light display-side boost.

## 4. Colours could be better
- **Cause**: the first tour shipped base colour only (SOG with `-H 0`), because SOG's spherical-harmonics k-means on CPU takes
  hours [M: 150k splats SH1 208 s, SH3 > 300 s, SH3 with one k-means pass > 5 min]. That drops the view-dependent part of the
  colour (sheen on floors, sheets, glass). Exposure: the bilateral grid trains each frame's exposure separately, so the base
  model has the average exposure [C].
- **Fix**: ship PlayCanvas compressed PLY with SH band 1 (25 bytes/splat, ~37 MB for 1.5M, converts in seconds on CPU [M]);
  later full SH3 SOG compressed on the Kaggle GPU (splat-transform's WebGPU k-means under Vulkan [Q], needs the NVIDIA Vulkan
  driver in the Kaggle container, unverified). No colour grading by default (true to the room).

## 5. Order
1. Tour v2 on run 3 (no retraining, ~15 min): metric scale, collision (room box), walk mode allowed, SH1 colour, sharpness.
2. Run 4: 1600 px from `images_2`, two GPUs, mesh at the end (collision with furniture), same QA.
3. Difix3D+ stage after training (`scripts/gsplat_difix.py` adapted to simple_trainer checkpoints): views 30-60 cm beside the
   path at walk height, real-pixel re-projection first, Difix for the rest, fine-tune; compare QA off-path before/after.
4. GSFixer if Difix3D+ leaves visible smears and the T4 fits it.

Sources: [Difix3D+](https://arxiv.org/abs/2503.01774), [GSFixer](https://arxiv.org/abs/2508.09667),
[TIDI-GS](https://arxiv.org/abs/2601.09291), [Filling the Unseen](https://arxiv.org/abs/2609.13262),
[EVPGS](https://arxiv.org/abs/2503.21816), [Robust Gaussian Splatting](https://arxiv.org/abs/2404.04211),
[splat-transform SOG speed-up](https://radiancefields.com/playcanvas-splat-transform-v3.9.0-speeds-up-sog-compression),
[SOG format](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/).
