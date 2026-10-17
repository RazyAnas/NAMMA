# NAMMA prototype backend

This repository now contains the first local pipeline for a laptop-only video-to-3D demonstration:

1. A FastAPI service with SQLite-backed project and job records.
2. Safe, size-limited video upload plus FFmpeg frame extraction.
3. An offline CUDA COLMAP pipeline for feature extraction, sequential matching and sparse mapping.
4. Reconstruction quality scoring with explicit coverage, point-count, reprojection and fragmentation gates.
5. Export of the strongest model as a portable COLMAP + `transforms.json` Gaussian-training dataset.
6. A configurable, quality-gated Gaussian trainer that exports and publishes a verified `.ply` model.

The photorealistic MODP mapper, dense collision geometry, NavMesh and viewer are later milestones.

## Run the API

```powershell
.\run.ps1
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The service deliberately binds only to localhost.

## API workflow

Upload a video as multipart form data:

```powershell
$form = @{ name = "Demo room"; video = Get-Item ".\room.mp4" }
$project = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/projects" -Form $form
```

Extract frames:

```powershell
$job = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/projects/$($project.id)/extract-frames"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/jobs/$($job.id)"
```

Run sparse reconstruction:

```powershell
$job = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/projects/$($project.id)/reconstruct"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/jobs/$($job.id)"
```

Inspect quality and the exported training dataset:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/projects/$($project.id)/quality"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/projects/$($project.id)/training-dataset"
```

Start Gaussian training after the quality report passes:

```powershell
$job = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/projects/$($project.id)/train-gaussian"
```

Configure the trainer in the private `.env` file. The command is executed without a shell and supports
`{dataset}` and `{output}` placeholders. For example:

```dotenv
GAUSSIAN_TRAINER_COMMAND=.\.venv-gsplat\Scripts\python.exe scripts\train_splatfacto.py --dataset {dataset} --output {output}
```

On an RTX 50-series laptop, install the NVIDIA CUDA Toolkit 12.8 first, then create the isolated
trainer environment with `scripts\install_gaussian_env.ps1`. The script uses Torch 2.7.1 CUDA 12.8
and compiles gsplat 1.5.3 for compute capability 12.0 because the published Windows wheels predate
Blackwell support.

On a Mac (or any machine without an NVIDIA GPU), use `scripts/train_opensplat.py` instead: it wraps
[OpenSplat](https://github.com/pierotofy/OpenSplat), which trains on Apple silicon through Metal or on the CPU.
See the script's docstring for the `.env` lines.

If quality fails, the backend returns HTTP 409 instead of spending GPU time on an unusable training run.

## Fast-motion and lighting robustness

- **Sharpest-frame selection** (`FRAME_SELECTION=sharpest`, default): FFmpeg decodes
  `FRAME_CANDIDATES_PER_KEYFRAME` (default 4) frames per keyframe and keeps the sharpest one, scored by
  Laplacian variance over intensity variance so exposure changes do not bias it. Results and capture
  warnings are written to `artifacts/frame-quality.json`.
- **Exposure / white-balance / flicker compensation** (`PHOTOMETRIC_COMPENSATION=true`, default): after
  SfM, per-frame RGB gains are solved from the colours of tracked 3D points and removed from the training
  images. Report: `artifacts/photometric-report.json`.
- **Global mapper** (`COLMAP_MAPPER=global`, default) instead of incremental mapping.
- `scripts/train_splatfacto.py` enables splatfacto's per-image bilateral grid and SO3xR3 pose refinement
  (`--no-bilateral-grid`, `--camera-optimizer off` to disable).

## Live capture from a phone browser

No app install: the phone opens a web page served by this backend and streams keyframes while the laptop
maps them.

```powershell
.\.venv\Scripts\python.exe scripts\serve_live.py
```

Open the printed `https://<laptop-ip>:8443/capture/` URL on a phone on the same Wi-Fi, accept the self-signed
certificate once, and press **Start**. Phone browsers only expose the camera and motion sensors on HTTPS pages,
which is why this mode uses its own certificate and listens on the LAN; the normal `run.ps1` server stays
localhost-only.

What happens during capture:

- **On the phone** (`app/static/capture/`): after auto-exposure settles, exposure, white balance and focus are
  locked where the browser allows it (Android Chrome), with exposure capped at 1/120 s and ISO raised to match,
  so lighting drift never starts and motion blur is about four times shorter. The gyroscope predicts blur in
  pixels and asks the user to slow down. Each 0.4 s window sends only its sharpest frame.
- **On the laptop** (`app/services/live_mapper.py`): every new keyframe is checked again for blur, then SIFT
  features are matched against recent keyframes and COLMAP's incremental mapper resumes from the current model,
  so the map grows within about a second per update on a CPU. The phone shows mapped keyframes and a top-down
  map. Gravity from the phone's accelerometer, and AR positions when a client sends them, are stored as COLMAP
  pose priors.
- **Finish & build** stops streaming and runs the full offline pass above (global SfM, quality gate, exposure
  compensation). Gaussian training is then started as usual.

## Laptop web viewer

The proof-of-concept viewer is served by the same FastAPI process at `http://127.0.0.1:8000/viewer/`.
It lists completed projects, streams the published Gaussian PLY, and provides pointer-lock free roaming with
W/A/S/D, Q/E, mouse look, Shift sprint, and mouse-wheel speed control. Build its static assets after cloning:

```powershell
Set-Location viewer
npm install
npm run build
Set-Location ..
.\run.ps1
```

The viewer is for visual roaming. Collision and obstacle blocking require a separate navigation mesh and are
not inferred from the rendered splats.

## Dense collision candidate

After a project passes reconstruction quality, start dense stereo with
`POST /api/v1/projects/{project_id}/build-collision`. The job runs CUDA COLMAP image undistortion,
PatchMatch stereo and fusion, then produces a filtered, simplified GLB candidate.
Its manifest and model are available at `collision-artifact` and `collision-model` under the same project API.

The checked-in laptop profile uses filtered photometric PatchMatch to avoid Windows GPU timeout resets on the
display-connected RTX 50-series GPU. Set `DENSE_GEOM_CONSISTENCY=true` for the higher-quality two-pass mode on
a dedicated GPU or a validated Linux environment.

This output is deliberately marked `safe_for_navigation: false`: monocular scale calibration, floor alignment,
unknown-space exclusion and a human geometry review are required before generating a NavMesh.

For the downloaded TUM RGB-D benchmark only, `scripts/build_tum_rgbd_collision.py` can fuse its metric sensor
depth and publish the same artifact format. This validates fusion/mesh/export independently and is not used as
a substitute for monocular depth on ordinary room videos.

For ordinary RGB video, configure the Apache-2.0 Depth Anything V2 Small subprocess shown in `.env.example`.
The backend predicts relative depth per registered frame, robustly chooses direct- or inverse-depth alignment
against visible COLMAP sparse points, fuses the aligned views, and publishes the same review-gated collision GLB.
The result remains in COLMAP's arbitrary units until an explicit metric scale measurement is supplied.

Navigation preparation is exposed through `POST /api/v1/projects/{project_id}/prepare-navigation`. It detects
candidate floor planes using fused points, registered camera centers and the estimated scene-up direction, then
writes a review artifact with an explicit `block_unknown` policy. Supply a known room measurement through
`POST /api/v1/projects/{project_id}/calibrate-scale` before NavMesh generation; calibration updates the GLB and
manifest atomically but does not mark the scene navigable.

Once scale and the floor candidate have been reviewed, start the conservative NavMesh build with
`POST /api/v1/projects/{project_id}/build-navmesh`. The generated artifact is available from
`GET /api/v1/projects/{project_id}/navigation-mesh`. It is a compressed, floor-aligned voxel grid: cells
without floor support, cells inside the inflated obstacle mask, and disconnected components are blocked.
The artifact remains `navmesh_ready: false` until a human reviewer approves the floor, scale and unknown-space
boundary. The laptop viewer reads this artifact and performs sub-stepped collision-constrained movement only
when that explicit approval flag is enabled; otherwise it remains visual-only. Record that review explicitly with
`POST /api/v1/projects/{project_id}/approve-navigation` and a JSON body containing `reviewer` and optional `notes`.
Approval now runs an integrity check first: compressed-grid length and hash, binary cell values, spawn validity,
single-component connectivity, metric scale and the `block_unknown` policy must all pass. Inspect the same check
without approving through `GET /api/v1/projects/{project_id}/navigation-validation`.

Use `GET /api/v1/projects/{project_id}/readiness` as the application integration contract. It reports every
ordered gate, whether visual roaming is ready, whether obstacle-aware roaming is ready, and the exact next action.
This keeps the separate roaming application from guessing readiness from the project status alone.

The metric TUM RGB-D benchmark utility now exercises collision generation, floor detection, navigation-grid
generation and validation. On the downloaded Freiburg desk sequence it integrated 39 frames, generated a
100,000-triangle collision mesh, blocked 1,302 obstacle cells and produced a connected 52-cell walkable region
whose integrity validation passed. Its intentionally smaller benchmark-agent profile is recorded in the report;
production API defaults remain a 0.30 m radius plus 0.10 m safety margin.

Artifacts are stored under `data/projects/<project-id>/`. Quality reports and trained outputs are under
`artifacts/`; exported images, camera poses and COLMAP data are under `training/dataset/`. The SQLite
database is `data/namma.db`.

After training, the roaming application can fetch integrity metadata and stream the model from:

```text
GET /api/v1/projects/{project_id}/gaussian-artifact
GET /api/v1/projects/{project_id}/gaussian-model
```

## COLMAP backend

The adapter uses the first available backend:

1. The executable configured by `COLMAP_BIN`.
2. `tools/colmap/COLMAP.bat` from the official portable Windows CUDA distribution.
3. The official `pycolmap` Python package, using CPU feature extraction and sequential matching.

For bounded scenes, the CLI pipeline performs GPU sequential matching followed by GPU exhaustive
loop-closure matching. Exhaustive matching is enabled for at most 500 extracted images by default;
configure `COLMAP_EXHAUSTIVE_LOOP_CLOSURE` and `COLMAP_EXHAUSTIVE_MAX_IMAGES` to change that guard.

For CUDA acceleration on Windows, install the official portable build into the ignored `tools/` directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_colmap_cuda.ps1
```

The installer downloads COLMAP 4.2.1 from the official GitHub release in verified byte ranges, validates the final archive size and ZIP structure, and extracts it to `tools/colmap`. The application detects `tools/colmap/COLMAP.bat` automatically. To make the selection explicit, set `COLMAP_BIN=./tools/colmap/COLMAP.bat` in the private `.env` file.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

The checks pass on the reference laptop. The official COLMAP 4.2.1 Windows CUDA build was validated with GPU SIFT feature extraction; PyCOLMAP remains a CPU-only fallback on Windows.
