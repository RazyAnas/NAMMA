# Why the lobby fails and the phone room works: measured inputs, our pipeline gaps, what the open-source pipelines do

Status 2026-10-21. Research only, no new training. Labels: **[M]** measured here, **[Q]** quoted, not reproduced.

## 1. The same pipeline already works

Phone room, AR recording, our pipeline: 35.5 dB on held-out frames, photo and render indistinguishable near the
filmed path (see the Photoreal Proof page). So reconstruction and training are not fundamentally broken. [M]

| Recording | Held-out PSNR | Poses |
|---|---|---|
| Phone room 2026-10-17, AR | 35.5 dB | ARCore, no resets, refined by BA |
| Lobby 2026-10-17, plain video | 23.2 dB | COLMAP, split into 4 pieces and merged |
| Lobby 2026-10-20, AR | 20.0-20.4 dB | ARCore with 16 origin resets, stitched; BA did not help on average |

## 2. The lobby inputs are measurably worse [M]

Every training frame, scored the same way at 432x960 (`docs` run of 2026-10-21):

| | Phone room (works) | Lobby 10-20 (fails) |
|---|---|---|
| Sharpness, median (variance of Laplacian) | 69.6 | **40.9** (41% blurrier) |
| Sharpness, worst 10% | 34.0 | **20.1** |
| SIFT keypoints per frame, median | 363 | **150** (2.4x fewer) |
| SIFT keypoints, worst 10% | 194 | **50** |
| Brightness range, 5th-95th percentile | 117-153 | **109-175** (exposure swings) |
| AR origin resets | 0 | **16**, with 3-20 m jumps |

Whole lobby video (9,735 frames): 13% of frames are below half the median sharpness; turning speed reaches
40 deg/s (95th percentile); our every-15th-frame sampling keeps frames at 84% of the sharpness of the best frame
in each 15-frame window; the recorder encodes at 8.5 Mbit/s H.264 Baseline. [M]

Blur, few features and exposure swings all hit the same two things: camera poses (less to match, so SfM splits and
AR loses tracking) and the splat itself (blurred and differently exposed training images).

## 3. What the open-source video-to-splat pipelines do that we do not

| Step | Open-source practice | Us |
|---|---|---|
| Frame selection | keep the sharpest frame per window (blur rejection) | every 15th frame, blind |
| Features and matching | SuperPoint + LightGlue via hloc (vid2scene, Apache-2.0) [Q]: many more matches on low texture than SIFT | SIFT, 150 per frame in the lobby |
| SfM | GLOMAP global mapper by default (vid2scene; its README calls VGGT "noticeably worse") [Q] | COLMAP global/incremental with SIFT |
| Pose refinement during training | gsplat `pose_opt` (off by default) [Q] | not used |
| Exposure / white balance per image | gsplat `post_processing="bilateral_grid"` or `app_opt` [Q] | not used |
| Blur and rolling shutter | modelled in training, phone video specifically (Seiskari et al., ECCV 2024) [Q] | not modelled |
| Capture | the phone's own camera app: high bitrate, sharp, locked exposure possible | WebXR camera image at 8.5 Mbit/s; AR mode cannot lock exposure |

Sources: vid2scene (github.com/samuelm2/vid2scene), gsplat simple_trainer.py, "Gaussian Splatting on the Move"
(arXiv 2403.13327), AirVis Studio (free, local; accepts COLMAP input, masks people first).

## 4. Our mistakes, ranked by how much they cost

1. **Trusting ARCore poses across resets.** 16 resets put the lobby in 16 frames; stitching recovered consistency
   to about 5-20 cm, but a photoreal splat needs about 1 cm.
2. **Input frames we chose and recorded**: no sharpest-frame selection; the WebXR camera stream is softer than the
   phone's own video (41% lower sharpness than the phone room at the same size); exposure is not locked in AR mode.
3. **SIFT on white walls**: 150 keypoints per frame is too few; this is why the image-only solve collapsed.
4. **Training without the two standard real-world switches**: per-image exposure correction and pose refinement.
5. **Coverage**: one walk at one height (1.03-1.19 m); off-path views fall apart (measured 15 vs 34 dB).

## 5. What would give the evidence before spending time

No training needed for the first two:
- Run SuperPoint + LightGlue + GLOMAP (the vid2scene recipe) on the existing lobby frames and count registered
  images, track lengths and reprojection error against our SIFT result. If it registers the whole lobby in one
  model, poses are solved without AR.
- Re-select frames by sharpness from the same video and re-measure sharpness and matches.
- Then one training run with pose_opt + bilateral grid on those poses, scored on the same 69 held-out frames.
