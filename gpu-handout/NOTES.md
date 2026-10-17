# GPU handout: notes for us (not for the friend)

- Send `namma_gpu_handout.zip` (26 MB, data included, no download script needed). Source of the zip: `package/`.
- When the results zip comes back: `python3 gpu-handout/ingest_results.py <namma_results_*.zip> <out dir>` (pass an out dir; GT images are read from the unzipped package next to the script).
- `builder/` rebuilds `package/data` from ICL-NUIM (degrade.py -> prep_track.py clean / degraded --clahe -> assemble_data.py).

## Data (measured here, 4-core CPU)
- Frames: train 1200..2216 step 16 (64), held-out test 1208..2216 step 16 (64); scored against clean test frames.
- Clean: COLMAP global SfM on train+test, ATE 0.68 cm RMSE (0.49 median), 4,321 sparse points. Monocular boot points AbsRel 3.3%, boot+multi-view 1.9%.
- Degraded (this thread's own simulation: 0-133 ms blur from averaging 30 fps frames, +-0.5 stop drifting exposure, white-balance drift, 10 Hz flicker, noise, JPEG 85; milder than the robustness thread's sim): CLAHE + DSP-SIFT global SfM, 125/128 posed, ATE 13.2 cm RMSE (6.5 median). Boot AbsRel 15.9%, boot+mv 3.6%. A stronger blur (up to 300 ms) gave 48 cm and was dropped.
- Point sets: sparse (SfM), random (100k in the room's box), boot (150k, 1 cm voxels), gtdepth (150k, true depth + true poses). MODP per-frame: sparse and boot_mv (<=4,000/frame).

## Runs on the friend's GPU
gsplat 1.5.3, 30,000 steps, full 640x480, SH 3, default densification; same recipe for all, only init points differ:
clean_sparse, clean_random, clean_boot, clean_gtdepth, degraded_sparse, degraded_boot (degraded ones with per-image colour transform).
MODP (commit 7d75169, paper config, pose refinement off, held-out views rendered with its renderer): modp_clean_sparse, modp_clean_boot_mv.

## Tested vs not tested
- Tested on CPU (40-60 steps at 40x30 px, CPU reference rasteriser in train_gs.py): data build, training incl. gsplat's densification strategy and the colour transform, rendering, eval (PSNR/SSIM/colour-matched PSNR), side-by-side, pack, ingest. Smoke-test output in example_cpu_smoke_test/ (numbers meaningless).
- MODP: our dataset + config load correctly in MODP's own loader and config merge (lietorch stubbed). Mapping and rendering not run.
- Not run at all (no NVIDIA GPU here): gsplat CUDA path, setup scripts (conda/Docker), LPIPS (weights host blocked here), MODP install.
