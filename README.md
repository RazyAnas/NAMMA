# NAMMA prototype backend

This repository now contains the first local pipeline for a laptop-only video-to-3D demonstration:

1. A FastAPI service with SQLite-backed project and job records.
2. Safe, size-limited video upload plus FFmpeg frame extraction.
3. A COLMAP sparse-reconstruction adapter that prefers the Windows CLI and falls back to PyCOLMAP.

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

Artifacts are stored under `data/projects/<project-id>/`. The SQLite database is `data/namma.db`.

## COLMAP backend

The adapter uses the first available backend:

1. The executable configured by `COLMAP_BIN`.
2. `tools/colmap/COLMAP.bat` from the official portable Windows CUDA distribution.
3. The official `pycolmap` Python package, using CPU feature extraction and sequential matching.

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
