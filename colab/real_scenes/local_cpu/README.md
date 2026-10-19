# Local CPU off-path tests (MacBook Air M2, no CUDA)

- `scripts/render.py`: CPU Gaussian-splat renderer (torch, no CUDA) used to score OpenSplat .ply files. On the phone room it
  reproduces the training run's held-out scores within ~0.1-4 dB, so it is fine for A/B comparisons.
- `scripts/clean_splat.py`, `sweep.py`, `off.py`: post-hoc cleanup test. Every cleanup lowered on-path PSNR (needles 6x -1.19,
  4x -2.25, near-camera -4.21, faint haze -1.59 dB) and none improved the off-path renders
  (`results/cleanup_offpath_base_needle6_fog.jpg`, columns base / needle6 / fog). Dead end.
- `scripts/run_local.sh`: the held-out-stretch test (`make_block_split.py`, frames 448-516) with OpenSplat 3k on CPU, built with
  `opensplat_namma.patch`, `opensplat_aniso.patch` and `opensplat_finite.patch` (the CPU backend lets a few Gaussians go NaN).
  Results in `results/`: default 15.40 dB, anisotropy cap 10 15.52 dB (+0.11, CI +0.05 to +0.18, 20 of 35 views).
