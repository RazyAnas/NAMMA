# NAMMA prototype backend

This repository now contains the first local pipeline for a laptop-only video-to-3D demonstration:

1. A FastAPI service with SQLite-backed project and job records.
2. Safe, size-limited video upload plus FFmpeg frame extraction.
3. An offline CUDA COLMAP pipeline for feature extraction, sequential matching and sparse mapping.
4. Reconstruction quality scoring with explicit coverage, point-count, reprojection and fragmentation gates.
5. Export of the strongest model as a portable COLMAP + `transforms.json` Gaussian-training dataset.
6. A configurable, quality-gated Gaussian trainer job interface.

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
GAUSSIAN_TRAINER_COMMAND=ns-train splatfacto --data {dataset} --output-dir {output}
```

If quality fails, the backend returns HTTP 409 instead of spending GPU time on an unusable training run.

Artifacts are stored under `data/projects/<project-id>/`. Quality reports and trained outputs are under
`artifacts/`; exported images, camera poses and COLMAP data are under `training/dataset/`. The SQLite
database is `data/namma.db`.

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
