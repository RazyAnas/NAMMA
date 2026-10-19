# Local CPU off-path tests (MacBook Air M2, no CUDA)

- `scripts/render.py`: CPU Gaussian-splat renderer (torch, no CUDA) used to score OpenSplat .ply files. On the phone room it
  reproduces the training run's held-out scores within ~0.1-4 dB, so it is fine for A/B comparisons.
- `scripts/clean_splat.py`, `sweep.py`, `off.py`: post-hoc cleanup test. Every cleanup lowered on-path PSNR (needles 6x -1.19,
  4x -2.25, near-camera -4.21, faint haze -1.59 dB) and none improved the off-path renders
  (`results/cleanup_offpath_base_needle6_fog.jpg`, columns base / needle6 / fog). Dead end.
- `scripts/run_local.sh`: the held-out-stretch test (`make_block_split.py`, frames 448-516) with OpenSplat 3k on CPU, built with
  `opensplat_namma.patch`, `opensplat_aniso.patch` and `opensplat_finite.patch` (the CPU backend lets a few Gaussians go NaN).
  Results in `results/`: default 15.40 dB, anisotropy cap 10 15.52 dB (+0.11, CI +0.05 to +0.18, 20 of 35 views).
  Anisotropy cap 5 with weight 1.0: 15.48 dB (+0.08, CI -0.08 to +0.23, 26 of 35 views; SSIM 0.757 vs 0.752), not significant
  and less stable (more NaN steps). Keep cap 10 / weight 0.1 if any.
- `scripts/run_local4.sh`, same stretch: TSDF mesh from our depth 18.76 dB (SSIM 0.811); mesh textured from the 4 nearest
  frames 18.84 (0.824); splat initialised on that mesh, same 3k recipe, 14.59 (0.720), worse than the default 15.40. A good start
  alone does not hold: without a geometry loss, training pulls the Gaussians off the surface again.
