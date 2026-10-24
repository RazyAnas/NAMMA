# Findings that must not be lost (newest first)

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
