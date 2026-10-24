# Learning from "We Open Sourced World Generation" (Mickmumpitz) for an indoor, true-to-room model

Source: youtu.be/eJuYBNrD8HI (transcript read in full, 2026-10-24). Their goal: a whole 3D world from ONE image, free, local.
Our goal: a photoreal, walkable, true-to-room model from a phone video. Different goals, but their reasoning transfers.

## 1. What they tried, why each failed, and the lesson behind it

| Attempt | What went wrong (their words) | The lesson |
|---|---|---|
| Apple Sharp per view of a 360 pano, stitched | "everything generates individually, you get really ugly seams" | Independent per-view prediction is inconsistent. Never fill views one by one. |
| Sharp + MoGe geometry to align views | "looks good from the initial perspective, but once you start moving it's falling apart"; Sharp "preserves, doesn't generate" | A preserving model can't cover what wasn't seen; geometry alignment alone doesn't add content. |
| UniSharp (360 feed-forward) | "worlds fall apart really quickly once you start moving" | One viewpoint of feed-forward 3D only covers that viewpoint. |
| Hunyuan 2.0, Nvidia Lyra | 4 GPUs / 91 GB checkpoints | Pick tools that fit the hardware you actually have. |
| **Matrix-3D (winner)** | pano -> depth -> rough mesh; move a camera through it; holes are black; Wan 2.1 + LoRA fills the holes | **Geometry first, then generate only what's missing, with a VIDEO model so the fill is consistent across frames.** |
| Wan at 720p | "the resolution is too low ... really soft" | Generated pixels are low-res. |
| SeedVR upscaling | "so slow, didn't do too much" | Upscaling generated pixels doesn't add real detail. |
| **Re-project the original high-res image onto the generated video** | "we know exactly how every single pixel moves ... this worked so well" | **Real high-res pixels wherever they exist, generated pixels only as fallback.** This was their biggest quality jump. |
| Drone paths -> dataset -> any splat trainer | "areas where we sent our drone look better ... send in a few more drones" | **Splat quality = quality of the dataset (coverage x consistency x resolution).** They fixed the data, not the trainer. |

## 2. The thinking behind it (what to copy)
1. **Split structure from appearance.** Geometry comes from a reliable source; generation is only allowed to paint appearance where
   geometry says "unknown".
2. **Make the unknown explicit.** Rendering the reconstruction from a new camera turns "imagine a world" into "fill these exact black
   pixels". A masked inpainting task is far more controllable than free generation.
3. **Consistency beats per-frame quality.** A video model fills holes coherently along a path; per-image fills (what we did with SDXL on
   the panorama) produce seams and inventions.
4. **Best source wins, per pixel.** Real pixels > re-projected real pixels > generated pixels. Composite in that order.
5. **Engineer the dataset, not the trainer.** Choose camera paths that cover what matters; the trainer just fuses a good dataset.
6. **Drop an approach for a fundamental reason, fast** (seams, can't generate, hardware, licence), not after weeks of tuning.

## 3. How our indoor case differs
- We start with far more than one image: ~6,700 real 4K frames along a walk. Most of the room IS filmed. Generation should be the
  exception, at small holes (behind furniture, ceiling corners, door reveals, views 30-100 cm off the path).
- Truth matters (real estate). They don't care if the model invents; we must keep real pixels exact and **flag** every generated pixel.
- Indoors: near walls (big parallax, many occlusions), plain white walls (no texture), shiny floor tiles (view-dependent colour),
  auto-exposure drift in the 22 Oct video.
- Our splat so far: 332k splats, colour only (SH degree 0), trained at 1008 px, 60 frames, cameras from chained chunks. In their terms:
  **our dataset is thin, low-res and slightly inconsistent**, and our trainer can't show reflections.

### What we did wrong, in their terms
- Built the panorama from 22 frames and let SDXL fill 55% per image = Sharp's mistake (independent fill, invents). Owner saw a door
  and an AC that don't exist.
- Never re-projected real pixels; trained at 1008 px though we have 4K = threw away their biggest win.
- SH degree 0: no view-dependent colour, the floor tiles can't reflect (they explicitly kept SH for reflections).

## 4. The pipeline, adapted (each step has a file and a test)

| # | Step | Their equivalent | Ours | Status |
|---|---|---|---|---|
| 1 | Geometry first | pano depth (MoGe) -> mesh | Depth Anything 3 on 200 sharp frames, chunked + Sim3 merge; camera correction in training | running (A/B) |
| 2 | High-res real source | original 8K image | the 4K frames (currently used at 1008 px) | train at 2016 px: E6 |
| 3 | Viewing region + drone paths | path editor, 4 drones | auto paths from the walkable grid + real camera path: along path, +/-30/60/100 cm sideways, look-forward and look-around, 1.2-1.7 m eye height | `drone_paths.py` (to build) |
| 4 | Make holes explicit | render rough mesh, holes black | render splat v1 per drone view + alpha; hole = low alpha | gsplat render in Kaggle job (to build) |
| 5 | Re-project real pixels | 8K re-projection | for each drone view, warp the nearest real frames with their depth, occlusion test by depth, take best (closest view angle, sharpest) | `reproject.py` (to build; CPU-testable) |
| 6 | Fill only what's left | Wan 2.1 + Matrix-3D LoRA along a path | Wan video inpainting along each drone path, conditioned on the composited frames + mask; Difix3D+ for small artifacts | after E4 says how much is left |
| 7 | Composite again | high-res composite | real/re-projected pixels over generated, feathered seam; per-pixel "generated" flag saved | with step 6 |
| 8 | Dataset -> splat | COLMAP -> Brush | real frames (weight 1, cameras refined) + drone views (weight 0.3, masked, fixed cameras); gsplat MCMC 1-3M splats, SH degree 3 | extend `train_splat.py` |
| 9 | Walkable model | - | floor/obstacles from geometry -> walk grid, collision, viewer with "not filmed" overlay | `fuse_nav.py`, `build_viewer.py` (exist) |

Panorama route (Matrix-3D as-is) stays possible: `make_pano.py` renders 360s from the real cloud with holes kept; but Matrix-3D
re-estimates depth with MoGe and generates a whole pano video, so it only becomes useful after a human checks the pano against the
video. Perspective drone paths (rows 3-7) are the safer fit for a real room.

## 5. Experiments, each measured (no claim without a number)
Held-out metric: every 8th real frame (mild) AND a held-out **block** of the walk (off-path-like, harder; like the phone-room 448-516 test).

| ID | Question | Change | Pass if |
|---|---|---|---|
| E1 (running) | Are frame count and camera drift the blur? | 200 frames; B = camera correction + depth anchor | B held-out PSNR > A by >= 0.5 dB |
| E2 | Is splat count the limit? (owner: 2M looks real) | C = B + MCMC 2M | C > B on held-out, and visibly sharper |
| E3 | Reflections / tiles | SH degree 3 | held-out PSNR up, floor highlights move with view |
| E4 | **Their key trick, no generation**: does adding re-projected drone views help off-path? | add drone views built from real pixels only | held-out BLOCK PSNR up; report % of drone-view pixels covered by real pixels |
| E5 | Does generated fill help or hurt? | fill remaining holes (Wan / Difix3D+), flagged | block PSNR not worse, owner sees no invented objects |
| E6 | Resolution | train at 2016 px instead of 1008 | held-out PSNR up / visibly sharper; time cost noted |

Order: E1 -> E2 + E3 + E6 together (cheap, trainer-only) -> E4 (the core idea) -> E5 only if E4 leaves real holes.

## 6. Guard rails
- Generation never touches a pixel a real frame covers. Every generated pixel is flagged, and the viewer can show it.
- The owner checks any generated panorama or video against the real video before it enters training.
- Measured vs reported numbers kept apart; training-view PSNR is never a quality claim.
