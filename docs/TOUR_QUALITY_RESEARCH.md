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

## 6. Round 2 (26 Oct, after tour v2): what is still bad and what fixes it
Tour v2 checked on screenshots: walk mode works, the guided walk is photoreal, but **streaky smears on the floor, ceiling and
everything off the filmed path**, and the **floor-plan minimap is unreadable** (black with sparse dots).

**What the newest evidence says**
- **Off-path quality is a coverage problem first [Q]**: "Mind the Gap" (arXiv 2607.01556, Jul 2026): standard held-out tests
  (every 8th frame) only measure near-path interpolation; real off-path views are **3-12 dB worse**, the gap tracks the angle to
  the nearest training view, and **loss-side regularisation gives only marginal gains**. Our own 19 Oct numbers agree (34 dB on
  the path, 15-17 dB 18 cm away). So: more filming coverage and generated views fix off-path; regularisers fix shapes.
- **Floor/ceiling streaks = textureless planes + few views [Q]**: indoor 3DGS papers all add a geometric prior there:
  DN-Splatter (WACV 2025, depth + normal priors), GaussianRoom, **PlanarGS** (Oct 2025, MIT code, GroundedSAM + DUSt3R planes,
  1 h on an RTX 3090), **UncertainGS** (Feb 2026, Manhattan-world normals with uncertainty). The streaks themselves are needle
  splats (effective rank 1): **effective-rank regulariser** (Hyung et al., NeurIPS 2024) pushes them to discs.
- **Generative clean-up, licences checked [C]**: Difix3D+ (measured +0.37 dB here) is **NVIDIA licence = non-commercial only**,
  and it was **fine-tuned on DL3DV**, so on our DL3DV test scene it will look better than on the owner's own videos.
  **FreeFix** (Jan 2026, **MIT code** github.com/hyzhou404/FreeFix): no fine-tuning, any image diffusion model (SDXL or Flux),
  interleaved render -> fix -> retrain per view with a per-pixel confidence mask; reports beating Difix3D+ on LLFF/Mip-NeRF 360.
  Heavier (SDXL on a T4 fits in fp16; Flux does not). GSFixer: video diffusion, heaviest.
- Monocular priors if we go further: **MoGe-2** (MIT code, metric depth + normals), Depth Anything V2 **Small** (Apache; Base/
  Large are non-commercial), DSINE (restrictive Imperial licence).

**Done now (code ready, measured on synthetic data; real numbers come from run 5)**
1. **Room prior** (`final/room_prior.py`, variant kind `prior`): needle penalty (longest/middle > 6, = effective-rank idea in our
   QA's terms) + floor/ceiling planes (floor height from SfM points, up from the cameras; splats within ~5 cm of a plane are
   turned flat to it). No extra network. Synthetic room: needles 9.4 % -> 0 %, floor-splat tilt 61 deg -> 0 deg. Patched into
   gsplat's simple_trainer by run.py, off unless the variant asks for it, so run 5 trains `full` and `prior` side by side.
2. **Floor plan** (`make_tour.floorplan`): the model seen from above, ceiling and wall tops cut off, each pixel the highest splat
   in its own colour, walls dark on top; redrawable without a GPU (`tour-publish.yml` with `"floorplan": true`).

**Next, in order**: run 5 = run 4's settings + `prior` variant (A/B on the same frames, QA off-path views + needle %);
Difix3D+ stage for the research demo (licence!) and FreeFix + SDXL as the commercial-safe path; MoGe-2 depth/normal loss if the
cheap prior is not enough; the owner's own re-filmed room (coverage beats every loss term).

Sources: [Mind the Gap](https://arxiv.org/abs/2607.01556), [FreeFix](https://arxiv.org/abs/2601.20857),
[PlanarGS](https://arxiv.org/abs/2510.23930), [UncertainGS](https://www.sciencedirect.com/science/article/abs/pii/S0925231225028085),
[Effective rank](https://arxiv.org/abs/2406.11672), [DN-Splatter](https://arxiv.org/abs/2403.17822),
[MoGe-2](https://arxiv.org/abs/2507.02546).

## 7. "Use a ChatGPT/Gemini-class image AI to fix the smears" (owner's idea, 26 Oct)
**Problem [Q]**: an image editor fixes each picture on its own; two neighbouring views get two different "fixes", and training on
both makes the splat blurry again (FreeFix, 3DGS-Enhancer, GSFixer all name multi-view consistency as the main difficulty).
**What makes it work (combined from Difix3D+, FreeFix, GSFixer, ExploreGS)**:
1. Give the editor the **nearest real photos as references** (it copies real detail instead of inventing).
2. Only let it change the **unreliable pixels** (confidence mask from the render; real reprojected pixels kept), composite back.
3. Fix views **one at a time along a path and retrain in between** (each fixed view constrains the next = consistency), with
   low weight and a per-view colour correction (FreeFix).
4. Place the extra views where the walker goes but the video did not (ExploreGS: information-gain camera placement, 2025).
**Editors**:
- **Qwen-Image-Edit-2511** (Alibaba, **Apache 2.0, open weights**, ChatGPT/Gemini-class editing): takes several input images
  (render + real reference photos), new-viewpoint ability built in, 4-step Lightning LoRA. 20B parameters: on Kaggle's T4 only
  as a 4-bit/GGUF build; quality and speed on a T4 **unverified** (community reports of artefacts at very low bits) -> smoke test.
- **Gemini 2.5 Flash Image ("Nano Banana") / OpenAI gpt-image**: best quality, multi-image input, no GPU needed, but paid
  (~$0.04 per image from price aggregators, not verified on the official pages; free tier for images is unclear). ~60 views per
  room = ~$2-3. Would need an API key as a GitHub secret.
- Difix3D+ (baseline we measured, non-commercial), FreeFix (MIT, SDXL/Flux, no fine-tuning), GSFixer (video model, heavy),
  ArtifactWorld (Apr 2026, video model trained on 107k artefact clips; code "to be released").
**Proof**: hold out a contiguous stretch of the video (not every 8th frame: "Mind the Gap" shows that only tests the path) and
measure that stretch before/after; plus the off-path QA views.
**Plan (`final/fix_views.py`, after run 4)**: render ~60 views 30-60 cm beside the path at walk height -> mask -> editor with 2
nearest real photos -> composite -> 3 rounds of fine-tuning (weight 0.3, colour affine) -> QA on the held-out stretch.
Smoke test first: 5 views through Qwen-Image-Edit-2511 on a T4 (time per image, does it keep the layout).

Sources: [FreeFix](https://arxiv.org/abs/2601.20857), [GSFixer](https://github.com/GVCLab/GSFixer),
[ExploreGS](https://arxiv.org/abs/2508.06014), [ArtifactWorld](https://arxiv.org/abs/2604.12251),
[FixingGS](https://arxiv.org/abs/2509.18759), [3DGS-Enhancer](https://nips.cc/virtual/2024/poster/95333),
[GSFix3D](https://arxiv.org/abs/2508.14717), [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
[Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning).
