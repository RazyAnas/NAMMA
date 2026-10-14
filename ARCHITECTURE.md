# NAMMA: video to photorealistic 3D with constrained navigation

## Scope and implementation status

This is a proposed architecture for a non-commercial student technology demonstration.
The initial deliverable is one static indoor scene reconstructed from a prerecorded
RGB video, viewed on a laptop with mouse/WASD navigation and obstacle collision.
Phone deployment, live mapping, dynamic scenes and DRS-VPT integration are deferred.

This repository initially contains planning, configuration and a local Python
environment. It does not yet contain a working reconstruction service or viewer.
Installing requirements.txt does not install MODP, ORB-SLAM3, COLMAP, Recast or a
CUDA toolchain. Those require separate native builds or distributions.

## Architectural decisions

Use MODP as the preferred research mapping backend, subject to reproducing its
results and making its dependencies work on the target GPU. Use COLMAP poses for
the first custom-video experiment; integrate ORB-SLAM3 afterwards for streaming.
The COLMAP-to-MODP adapter is proposed engineering work, not an upstream feature.
Its result cannot be assumed to reproduce the paper's online benchmarks.

Maintain three aligned representations:

| Representation | Responsibility |
| --- | --- |
| Gaussian splats | Photorealistic appearance and novel-view rendering |
| Dense mesh plus observed-space evidence | Surfaces, collision and confidence |
| NavMesh | Connected walkable regions and pathfinding |

MODP's MOHV structure manages Gaussian density. It is not a validated physical
occupancy map. A pleasing render does not prove geometric completeness.

## Data flow

1. Import video and create a project/job record.
2. Select sharp, overlapping keyframes and calibrate/undistort images.
3. Run COLMAP feature matching, reconstruction and bundle adjustment.
4. Convert registered cameras and their observed sparse points for MODP.
5. Run MODP mapping and optional post-refinement; preserve its final camera poses.
6. Generate dense geometry using images and mutually consistent camera poses.
7. Calibrate scene scale and floor orientation; apply the same transform to all assets.
8. Clean geometry, mark uncertain areas blocked and build collision/NavMesh assets.
9. Export a scene package and load it in a PlayCanvas laptop viewer.

If MODP adjusts poses significantly, geometry from the original COLMAP poses may
not align locally. Regenerate/refine geometry using compatible poses or reject the
package on alignment checks. A single global transform cannot fix non-rigid drift.

## Video preprocessing

Use OpenCV for frame checks and imageio-ffmpeg for a bundled FFmpeg executable.
The latter does not necessarily place `ffmpeg` on PATH; obtain its executable via
`imageio_ffmpeg.get_ffmpeg_exe()` or configure FFMPEG_BIN explicitly.

Start with a 30-90 second video of a static room. Preserve timestamps and camera
motion; use 2 FPS and a maximum image width of 1280 as adjustable starting settings.
These are memory controls, not guaranteed optimal capture settings. Avoid excessive
frame rejection that removes overlap. Downscaling requires scaling intrinsics.

Evaluate blur, overlap, reprojection error and registered-frame fraction. Mask
moving subjects where practical. Stop with an actionable error if tracking fails.
Mirrors, glass, blank surfaces and unseen furniture sides need conservative handling.

## Tracking and MODP integration

MODP consumes RGB images, camera intrinsics, poses and sparse tracked points.
The public stream loader leaves tracker initialization and stream construction
unfinished. Merely installing Python dependencies does not supply that tracker.

Implement adapters with a shared packet:

```text
frame_id, timestamp, image_path, width, height
intrinsics: fx, fy, cx, cy, distortion/undistortion metadata
T_world_camera: 4x4 float matrix
points_world: Nx3 sparse points observed by this frame
point_ids, confidence, tracking_status
```

COLMAP adapter: use image observations to select associated 3D points; do not feed
the entire cloud indiscriminately per frame. Invert world-to-camera poses where
needed. Test projection and positive depth before calling the mapper.

ORB-SLAM3 adapter (later): export frame poses, visible map points and timestamps
through a C++ exporter/process bridge. Stock trajectory export alone is insufficient.
Account for lost tracking, map resets, pose corrections and map-point updates.
Saving one final global cloud does not preserve per-frame observations.

Use a canonical right-handed world frame with Y up. Preserve the source camera
convention inside each adapter and explicitly convert at renderer boundaries.
Store transformations and units in scene metadata; test known projected points.
Monocular metric scale requires a known distance. Until calibrated, do not treat
agent dimensions as physical metres.

MODP contributes hierarchical Gaussian placement, density control and selection
of local/global optimization views. Benchmark quality and runtime on held-out
views. Its published speeds use a desktop RTX 4090; they are not laptop guarantees.

## Geometry, collision and navigation

Build dense multi-view depth with COLMAP and fuse consistent observations.
Use Open3D for inspection, filtering, mesh processing and optional TSDF fusion.
A TSDF stores truncated surface distances and weights; its sign alone is not a
global free-space classification. Maintain observed-free/occupied/unknown evidence
separately, using ray observations and confidence thresholds where implemented.

Avoid blindly closing every mesh hole: this can seal actual doors or invent floors.
Permit recorded manual mesh corrections for the demonstration. Keep the original
mesh and correction provenance alongside the final collision mesh.

Recast builds walkable polygons with agent radius, head clearance, slope and step
limits. Recast alone does not know what the camera failed to observe; apply an
explicit unknown/low-confidence exclusion mask and restrict to the spawn component.
Do not inflate obstacles twice accidentally: distinguish agent-radius erosion from
the additional safety margin.

Use a capsule sweep for continuous motion, sliding contacts and ground checks.
Then validate the swept movement and its connected NavMesh corridor. Testing only
the destination or snapping to the nearest polygon can teleport through a wall.
Click-to-move uses Detour pathfinding and the same collision controller.

The system can prevent penetration of its modeled obstacles. It cannot guarantee
that every real-world obstacle was recovered from RGB video. Unknown-space blocking,
manual review and specific collision tests are part of demo acceptance.

## Viewer and scene package

Use TypeScript and PlayCanvas/SuperSplat rendering components. A React UI is optional.
WebGPU support, browser/GPU combinations, asset formats and fallback rendering need
testing. Load collision/navigation before enabling walking. Keep orbit inspection
separate from collision-constrained walking.

```text
scene/
  scene.json             # schema version, units, transforms, spawn, asset hashes
  visual.ply             # Gaussian representation; compressed export later
  collision.glb          # invisible mesh
  navigation.bin         # versioned Recast/Detour format
  observed_space.bin     # confidence/unknown-region information
  cameras.json
  localization.ply       # optional future localization geometry
  thumbnail.webp
```

Export format conversion must preserve SH coefficients, opacity and transforms.
Start with bounded scenes; profile splat count, load time, VRAM and frame-time
percentiles before adding LOD or streaming. Target 30 FPS viewing, measured on the
actual laptop. Reconstruction speed and rendering FPS are separate metrics.

## Backend and persistence

Proposed backend: FastAPI, SQLAlchemy, SQLite and one background reconstruction
process. SQLite is included in Python; do not install an unrelated sqlite package.
Bind the development service to localhost. Use process-level jobs for heavy work;
request handlers must not run blocking GPU reconstruction directly.

Store project/video metadata, job stage, settings, error messages, scene versions,
artifact paths/hashes and metrics in SQLite. Store binary assets on disk. Implement
stage checkpoints and atomic publication of complete scene packages. Resume only
when input hashes, configuration and upstream artifacts still match.

```text
data/projects/<id>/
  source/
  frames/
  tracking/
  splats/
  geometry/
  navigation/
  exports/
logs/
```

Suggested stages: imported, preprocessing, tracking, mapping, refinement, geometry,
navigation, packaging, ready, failed, cancelled. The database is created by future
application initialization; .env alone does not create directories or tables.

## Dependency and environment boundaries

| Environment | Components | Setup |
| --- | --- | --- |
| Local Python 3.11 `.venv` | API, metadata, image/mesh utilities, tests | requirements.txt |
| Research mapper | MODP, PyTorch, custom gsplats/DepthCov extensions | Isolated Linux environment; upstream setup and hardware validation |
| Native tracking/geometry | COLMAP; later ORB-SLAM3 and Recast | Separate executables/builds |
| Viewer | Node.js, TypeScript, PlayCanvas, selected navigation/physics bindings | Future package.json and npm lockfile |

Do not assume pip `gsplat` replaces MODP's `gsplats` dependency. Do not mix current
PyTorch with upstream MODP pins without checking native extension compatibility.
Use a subprocess adapter to connect the research environment to the application.

Initially detected on 2026-10-08: Windows, Python 3.11 and 3.14, RTX 5060-series laptop GPU
with approximately 8 GB VRAM, NVIDIA driver 592.00. WSL is not installed. `nvcc`,
`colmap`, `ffmpeg` and `cmake` were not found on PATH; Node.js is available.
The CUDA version displayed by nvidia-smi is driver capability, not an installed
CUDA compiler. MODP documents Python 3.9/Linux/CUDA 11.8 and known CUDA 12 issues.
Its legacy setup is not a validated configuration for this newer GPU. First verify
a compatible PyTorch CUDA tensor operation, then compile/test each extension in
an isolated environment. Another compatible university Linux GPU machine is an
alternative if porting blocks progress. WSL alone does not solve GPU compatibility.

Updated on 2026-10-14: the official COLMAP 4.2.1 Windows CUDA distribution is
installed locally under the ignored `tools/colmap` directory. Its CLI reports a
CUDA-enabled build, and a GPU SIFT smoke test completed successfully on device 0.
The PyCOLMAP Windows wheel remains CPU-only and is retained only as a fallback.

Local installation (PowerShell, from project root):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

Activation is optional: `.\.venv\Scripts\Activate.ps1`. Direct use of the venv
Python works without changing the system PowerShell execution policy.
requirements.txt constrains compatible families; retain a resolved package snapshot
when a validated build is established for reproducibility.

## Implementation milestones and acceptance

1. Reproduce a supplied MODP sequence in a compatible research environment.
2. Track a custom video and validate the camera/point adapter by reprojection.
3. Produce splats and assess held-out PSNR/SSIM/LPIPS plus visible artifacts.
4. Generate scaled, aligned collision geometry; review doors, floors and furniture.
5. Build NavMesh and walking controls. Test thin walls, high-speed motion, corners,
   narrow passages, stairs, unknown space and disconnected regions.
6. Package a complete scene, demonstrate laptop viewing and report time/VRAM/FPS.
7. Add optional ORB-SLAM3 streaming and relocalization after the offline demo works.

DRS-VPT is an optional research extension, not needed for virtual camera movement.
It requires its own implementation/weights/license assessment and registration
validation before it can be considered a tracking-recovery component.

## Attribution and sources

Student status alone does not determine whether a use is non-commercial. For the
intended non-commercial demonstration, retain MODP attribution, licence and change
notices. Respect GPL requirements when distributing ORB-SLAM3 or combined software;
running separate processes does not automatically settle licensing obligations.
Audit dependency and dataset terms before publishing source or demo assets.

- [MODP project and paper](https://poiw.github.io/MODP/)
- [MODP repository and environment limitations](https://github.com/facebookresearch/Online-3DGS-Monocular)
- [MODP missing stream integration](https://github.com/facebookresearch/Online-3DGS-Monocular/blob/main/utils_new/stream_loader.py)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [COLMAP](https://github.com/colmap/colmap)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Recast/Detour](https://github.com/recastnavigation/recastnavigation)
- [SuperSplat components](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/)
