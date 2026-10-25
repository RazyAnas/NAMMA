# Photoreal AND accurate: research and plan (2026-10-25)

Goal (owner): a model of the REAL room that looks like a photograph from anywhere you walk, with geometry for collision,
as beautiful as Marble / HY-World but never inventing what was filmed. Free GPUs only (Kaggle / Colab / HF Spaces).
Rule for every step: a proof that runs in 5-10 minutes before any long job.

## 1. What the strongest systems do (sources at the end)

| System | How it gets its look | Accurate to the real place? | Usable for us |
|---|---|---|---|
| **Matterport** (real estate leader) | The photoreal views ARE the captured photos (panoramas at scan points). The 3D mesh only carries you between them; the next photo fades in when you arrive (patent). | Yes: you are looking at real photos. | The idea, fully. |
| **World Labs Marble** | A generative model outputs Gaussian splats; accepts text, images, 2-8 photos or a short video. Its newer model Atlas claims faithful reconstruction from 2-3 photos (no paper/code). | Partly: independent paper reports floaters; geometry needs checking. | Closed, paid video input. Not usable; the look is the benchmark. |
| **Tencent HY-World 2.0** | 4 stages: panorama generation, trajectory planning (WorldNav), world expansion with a video model + memory (WorldStereo 2.0), and **WorldMirror 2.0**: one feed-forward pass from images/video to cameras, depth, normals, points AND 3D Gaussians. | Reconstruction branch (WorldMirror) = real; expansion = generated. | WorldMirror 2.0 weights open (Apr 2026); 1.2B params, est. 12-24 GB; v1.0 measured 7.6 GB for 16 views, 60 GB for 256. Free demo of v1.0: HF Space `tencent/HunyuanWorld-Mirror`. Full generation pipeline: 4-8 GPUs, out of reach. |
| Feed-forward splats (AnySplat, TokenGS, VGGT-based) | One network pass, no per-scene training. | Yes (from real views) but soft. | Best reported ~22-26 dB on DL3DV; a frozen VGGT + TokenGS gets within 0.5 dB of COLMAP. Good base, not "extreme photoreal" alone. |
| Generative repair (Difix3D+, GSFixer, GenFusion, ArtifactWorld) | A diffusion model cleans rendered views; Difix3D+ also distils the fixes back into the 3D model. | Mostly: it edits renders of the real model. | Difix3D+ demo: HF Space `georgeniu/difix`. Measured on our lobby: +0.37 dB, and visibly cleaner. |
| Splat Tour (phone video → tour startup) | Per-room splats in one shared SfM frame, strict capture rules, MCMC + bilateral grid, streamed by room. | Yes. | Already studied in `docs/SPLATTOUR_RESEARCH.md`. |

## 2. What we measured ourselves (repo docs + this session)
- On the walked path, per-scene splats reach **31-36 dB** (phone room, lobby): photographic.
- 18 cm off the path every method drops to **15-18 dB** (splats, meshes, Difix). Off-path is the whole problem.
- Our room video: SfM cameras 195/200 at 1.1 px; 600 frames removed the fog, but walls blotchy, needles off the path.
- Per-scene optimisation will not give the Marble look; tuning it cost a day.

## 3. The strategy: every pixel from the best available source
1. **Real pixels first** (Matterport principle, the video's "8K re-projection"): any view near the walk is built from the nearest
   real 4K frames, re-projected through the SfM cameras + depth. Exact photo where a frame saw the surface. Nothing invented.
2. **Reconstruction second**: splats/mesh give geometry, collision, floor plan and the in-between views where real pixels would
   stretch. Feed-forward (WorldMirror / DA3 Gaussians) for speed; Difix3D+ to clean it.
3. **Generation last, only for holes**: a video model (Matrix-3D on Wan, or WorldStereo-style) fills what no frame ever saw,
   flagged per pixel, checked by the owner against the video. Never over filmed pixels.
4. **Viewer like Matterport**: walking along the filmed path shows real frames (photo-exact), free roaming blends to the 3D
   model, and a toggle shows what was never filmed.

## 4. Quick proofs (each 5-10 min, free) — run these before any long job
| # | Question | How | Pass if |
|---|---|---|---|
| **P1** | Do re-projected real pixels look like photos, and how much of the room do they cover? | Kaggle CPU job `namma-ibr` on the prep cache: rebuild held-out real frames from neighbours; also views 30 cm off the path | held-out PSNR ≥ 25 dB and SSIM ≥ 0.85 (our splats: ~18.8 dB); report off-path coverage |
| **P2** | Does Tencent's reconstruction look Marble-clean on our room? | Browser: HF Space `tencent/HunyuanWorld-Mirror`, upload ~30-60 s of the video, Reconstruct, download `gaussians.ply` | owner: clean and the same room |
| **P3** | Does generative repair make off-path views photoreal without inventing? | Browser: HF Space `georgeniu/difix`, feed P1's off-path images | edges/holes cleaned, nothing new |
| P4 | Can a video model fill the holes consistently? | Only after P1-P3; holes from P1's off-path coverage | owner check against the video |

## 4b. Proof results (2026-10-25)
- **P1 PASSED**: 16 held-out real frames rebuilt only from neighbours: **27.0 dB, SSIM 0.86, 99% coverage** (splats: 18.8 dB);
  many views 32-42 dB (photographic). 30 cm off the path real pixels still cover 95%.
  **4/16 views failed (12-15 dB): the rebuild shows a window where the photo shows a door, with a source 2-7 cm away ->
  those SfM cameras are wrong** (repetitive corridor). Every splat so far trained on these wrong cameras.
- P2 (WorldMirror 1.0 demo, 4.5M splats): clearly the room, but streaky; not photoreal. Shape only.
- P3 (Difix demo): small effect on our images. Final polish only.
- Next: `ibr_check.py` scores every frame to list the wrong cameras (synthetic test: finds exactly the 3 planted ones).

- **Camera check on all 596 frames**: real-pixel rebuild median **34.3 dB**; ~75% of frames 30-47 dB. Wrong cameras: 20 isolated
  plus whole stretches (86-108 ~11 dB, 321-330 ~15, 525-549 ~17, 15-27 ~20). Worst frames: a door frame placed where the window
  frames are: NetVLAD retrieval linked the look-alike corridor ends. Next: re-solve with walk-order pairs only (`namma-resfm`).

## 5. Final pipeline (built from what passes)
prep (exists: frames, SfM, depth) → base 3D (WorldMirror 2.0 or DA3 Gaussians, per room) → Difix3D+ clean-up → hole fill (P4) →
viewer with real-frame mode + collision + floor plan. Long jobs only at this last step.

## 6. Stop doing
Per-scene splat tuning loops; panoramas filled by an image model (invented a door); judging by PSNR alone on white walls.

## Sources
- Matterport transitions: patent US 10,030,979 / 10,655,969 (photorealism restored at panoramas; mesh between), WGAN forum analysis.
- Marble: TechCrunch launch (2025-11-12); World API (2026-01-21); Atlas (radiancefields.com); floaters: arXiv 2603.03482.
- HY-World 2.0: arXiv 2604.14268, Tencent model card, HY-World-2.0-Playground (VRAM notes), WorldMirror 1.0 arXiv 2510.10726.
- Feed-forward: AnySplat (ACM TOG 2025, arXiv 2505.23716), TokenGS (CVPR 2026, arXiv 2604.15239), frame-consistent FF-3DGS (arXiv 2610.23039).
- Repair: Difix3D+ (arXiv 2503.01774), GSFixer (arXiv 2508.09667), GenFusion, ArtifactWorld (arXiv 2604.12251), G4Splat (arXiv 2510.12099).
- Indoor phone: DN-Splatter (WACV 2025), AGS-Mesh (3DV 2025). Splat Tour: `docs/SPLATTOUR_RESEARCH.md`.
