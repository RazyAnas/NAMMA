# Techfest Round 1 (due 31 Oct 2026): draft

Deliverables: abstract (PDF), hosted viewer, 3-5 min video, repo with sample data. Every number below is measured on our
22 Oct room video (see `colab/real_scenes/final/NOTES.md`); update them as results come in. Items marked PENDING are not done yet.

## Abstract (draft, ~250 words)

**NAMMA: true-to-room 3D walkthroughs from a single phone video**

Real-estate tours need two things that today's tools give only one of: the look of a photograph from anywhere you stand, and
the guarantee that what you see is really there. Capture rigs (Matterport) are accurate but expensive; generative world models
(World Labs Marble, Tencent HY-World) look stunning but invent what they did not see. NAMMA turns a one-minute phone video into a
walkable 3D model on free GPUs, under one rule: real pixels first, generation only where nothing was filmed, and every generated
pixel is flagged.

The pipeline extracts sharp frames, solves camera positions (SuperPoint + LightGlue SfM) and metric depth (Depth Anything 3), and
re-projects the real photos into any viewpoint. We found that indoor rooms break camera solving through look-alike places
(symmetric corridors, repeated windows), which silently places parts of the walk in the wrong spot. NAMMA detects this with a
real-pixel test (rebuild each frame only from other passes through the room) and repairs it by following the video itself across
the gaps, verifying every join against the whole model. On our test room this raised cross-pass agreement from 15.7 to 28.2 dB.
Views rebuilt from neighbouring real photos reach 30.7 dB PSNR / 0.91 SSIM on held-out frames, against 18.8 dB for a per-scene
Gaussian splat. The browser viewer shows the real photo itself wherever the walker stands on the filmed path and blends only real
photos in between; holes are shown, not hidden. A video diffusion model (Wan 2.1 VACE) fills only the remaining holes
[PENDING: result], and the final model is distilled into a 3D Gaussian splat with a floor plan and collision [PENDING].

## Demo video (3-5 min) storyboard
1. 0:00-0:20 The problem: a phone video of a room; "can this become a walkable 3D tour that never lies?"
2. 0:20-1:00 Side by side: per-scene splat (blurry fog, 18.8 dB) vs real-photo re-projection (photographic, 30.7 dB).
3. 1:00-1:50 The look-alike problem: viewer before the fix (walls popping in wrong places), the cross-pass test plot, after the fix.
4. 1:50-2:40 Walkthrough in the viewer: click-to-walk landing on real photos; press H to show what was never filmed (magenta).
5. 2:40-3:30 Hole filling: holes -> Wan fill -> composite with real pixels on top; the red "generated" flag overlay. [PENDING]
6. 3:30-4:00 Floor plan + collision, final splat, what's next (multi-room, multi-floor). [PENDING]

## Hosted viewer
The photo viewer is static files (index.html, three.min.js, cams.json, f/*.jpg + f/*.bin). Host on GitHub Pages from a separate
repo or branch (about 100 MB for 318 photos; each file well under GitHub's 100 MB limit). Use the best aligned group only.

## Repo with sample data
- `colab/real_scenes/final/`: the pipeline (each stage has a synthetic CPU test).
- Sample data: 40 frames + `da3_sfm.npz` of one pass (small), so judges can run `photo_viewer.py` and `ibr_check.py` on CPU.
- README: one command per stage, the measured results table, and `docs/CAPTURE_GUIDE.md`.

## Checklist
- [ ] Final numbers after the alignment + fill runs
- [ ] Viewer on GitHub Pages
- [ ] Screen recordings for the storyboard (viewer, plots, fill videos)
- [ ] Abstract as PDF
- [ ] Sample data folder + README
