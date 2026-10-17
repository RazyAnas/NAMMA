# Unified pipeline API

One FastAPI app, one router per stage, every stage a background job with a status endpoint. The
original `/api/v1/projects/...` endpoints are unchanged; the pipeline lives under `/api/v1/pipeline`.
Open `/docs` for the interactive version; the stage tags are numbered in pipeline order.

```
phone /record/ bundle ─┐                                   ┌─> splats/train | splats/modp ──> export/splat-pack
                       ├─> capture ─> tracking ─> points ──┤
recorded video ────────┘   (frames)   (sfm | ar-poses)     └─> walk (scale, floor, navmesh) ─> export/walk-scene
```

Every POST that starts work returns a job (`202`). Poll `GET /api/v1/pipeline/jobs/{id}` for status,
progress, parameters, result and the last 40 log lines. Files for each job are in
`data/projects/<project>/jobs/<job>/` (`params.json`, `result.json`, `log.txt`).

## Hardware switch

`GET /api/v1/pipeline/backends` reports CPU, Apple silicon (MPS, Core ML) and CUDA, and which tools are
installed. Each stage declares what it needs; a step this machine can't run answers `503` with the
missing pieces (for example `{"missing": ["cuda"]}` for splatfacto) instead of failing halfway.
A `device` parameter (`cpu`, `mps`, `cuda`) adds that device to the check.

| Needs CUDA | Mac / CPU path |
|---|---|
| splatfacto training | OpenSplat (Metal or CPU): set `OPENSPLAT_BIN` |
| COLMAP PatchMatch dense stereo (`walk/collision` with `backend=colmap`) | `backend=monocular` (Depth Anything V2 via ONNX) |
| MODP (`splats/modp`) | `splats/train` with OpenSplat; or run MODP in the Colab handout |

## Endpoints

Status: **verified** = ran end to end through the API on the real phone AR recording from
2026-10-17 (or in `tests/test_pipeline_api.py`); **wired** = endpoint and job exist but have not run
here (the tool, GPU or input isn't in the cloud machine).

| Stage | Endpoint | What it wraps | Status |
|---|---|---|---|
| 0 | `GET /pipeline/backends`, `/stages`, `/findings` | machine report, stage catalog, measured results | verified |
| 0 | `POST /pipeline/projects/{id}/run` | runs stages in order in one job, skips what can't run | verified |
| 1 capture | `POST /pipeline/capture/sensor-bundle` | `sensor_bundle.ingest` (PR #2 recorder) | verified |
| 1 capture | `POST .../capture/frames` | sharpest-frame selection (`video.extract_frames`) | verified (unit tests) |
| 1 capture | `/api/v1/live/...` + `/capture/` | streaming mapper (unchanged) | existing |
| 2 tracking | `POST .../tracking/sfm` | COLMAP global mapper, CLAHE + DSP-SIFT, quality gate, exposure solve | verified |
| 2 tracking | `POST .../tracking/ar-poses` | triangulate from metric ARCore poses, optional BA | verified |
| 2 tracking | `POST .../tracking/evaluate` | ATE after Sim3 vs ARCore poses or a ground-truth file | verified |
| 2 tracking | `POST .../tracking/deblur` | Restormer (`research/deblur_restormer.py`), CPU or MPS | wired |
| 2 tracking | `POST .../tracking/live-tracker` | Depth-Anything pseudo-depth tracker (`research/live_track3.py`) | wired |
| 3 points | `POST .../points/monocular` | Depth Anything V2 ONNX + anchoring + fusion + collision mesh | verified |
| 3 points | `POST .../points/export-modp` | MODP "aria" dataset with monocular points | verified |
| 4 splats | `POST .../splats/train` | OpenSplat / splatfacto / custom command; `init=monocular` | wired |
| 4 splats | `POST .../splats/modp` | Meta's MODP, code unchanged, fed our poses (`sfm`/`ar`) and points (`sparse` = stock, `dense` = monocular); held-out PSNR/SSIM/LPIPS | wired |
| 4 splats | `POST .../splats/gpu-results` | `gpu-handout/ingest_results.py` | wired |
| 5 walking | `POST .../walk/collision` | monocular (CPU) or COLMAP dense (CUDA) | verified (monocular) |
| 5 walking | `POST .../walk/auto-scale` | metric scale from ARCore poses, no tape measure | verified |
| 5 walking | `POST .../walk/prepare`, `/navmesh`, `/approve` | floor RANSAC, NavMesh with movement + climb_top layers, human approval | prepare ran; navmesh blocked by floor (gap 1) |
| 6 export | `GET .../export/walk-scene` | zip of splat.ply, navmesh.json, collision.glb for the walk demo | verified (collision only) |
| 6 export | `POST .../export/splat-pack` | splat to PNG records for the browser viewer | wired |
| 6 export | `GET .../export/outputs` | every file a stage produced | verified |

## What ran on the real phone recording (cloud CPU, 4 cores, no GPU)

Input: `sensor-capture/ar-2026-10-17T11-28-46/bundle.zip` (21.6 s, ARCore tracked 94%), stride 10.

| Step | Result | Time |
|---|---|---|
| capture/sensor-bundle | 58 frames, all 58 with ARCore pose, IMU offset -129 ms | 2 s |
| tracking/sfm (global mapper) | 58/58 registered, 21,529 points, 0.90 px | 3.4 min |
| tracking/ar-poses (+BA) | 58 frames, 22,700 points, 1.66 px (0.60 px after BA) | 1.6 min |
| tracking/evaluate, SfM vs ARCore | ATE 8.81 cm RMSE, 6.04 cm median; 8.57 cm on a second run (gap 2) | <1 s |
| tracking/evaluate, AR+BA vs ARCore | ATE 3.65 cm RMSE, 2.07 cm median | <1 s |
| points/monocular (AR poses) | 58 frames, median alignment error 2.9%, 593k fused points, 100k-triangle collision mesh | 3.7 min (0.56 s/frame depth) |
| points/export-modp | 58 frames | 2 s |
| walk/auto-scale | `poses=ar`: metric already, 1.0; `poses=sfm`: 0.221 m per SfM unit from 58 frames | <1 s |
| pipeline/run (default plan, second project) | sfm 58/58 at 0.85 px, monocular points 2.9%, splat skipped (no OpenSplat) | 5.8 min |
| walk/prepare | floor not accepted: picked a wall, confidence 0.43 (gap 1) | <1 s |

ARCore is a reference here, not ground truth.

## Running on the Mac

```bash
pip install -r requirements.txt onnxruntime        # Core ML provider is used automatically when present
uvicorn app.main:app --port 8000
# optional: OPENSPLAT_BIN=/path/to/opensplat for splat training (Metal)
# optional: pip install torch; RESTORMER_DIR=... RESTORMER_WEIGHTS=... for deblurring (use device=mps)
curl -F name=room -F bundle=@namma-ar-....zip localhost:8000/api/v1/pipeline/capture/sensor-bundle
curl -X POST -H 'content-type: application/json' -d '{}' localhost:8000/api/v1/pipeline/projects/<id>/run
```

Depth Anything V2 Small weights download on first use from the GitHub release
`fabio-sim/Depth-Anything-ONNX` v2.0.0 into `tools/models/` (override with `NAMMA_MODELS_DIR`).

## MODP on a GPU machine (Linux + NVIDIA)

`splats/modp` runs Meta's MODP (facebookresearch/Online-3DGS-Monocular @ `7d75169`). MODP has no
trained scene model to download. It optimises each video online, and its only pretrained weights are
DepthCov `scannet.ckpt`, which ship in its repo. The job:
1. undistorts the frames, crops them to multiples of 8, and writes per-frame points (`prep_modp_inputs.py`);
2. writes MODP's own dataset layout with the paper config `orb_tracking/aria_base.yaml` (`make_modp_dataset.py`);
3. runs `SceneMapper.run()` and `eval_gaussians` in MODP's environment, then renders every 8th frame held out (`run_modp.py`);
4. scores the held-out frames (`score.py`). The model is `artifacts/modp/results/<poses>_<points>/point_cloud.ply`, a standard 3DGS PLY.

```bash
bash scripts/modp/setup_modp.sh tools      # micromamba env: Python 3.9, CUDA 11.8, MODP's setup_env.sh (20-35 min)
MODP_DIR=tools/Online-3DGS-Monocular MODP_PYTHON=tools/modp_env/bin/python uvicorn app.main:app
curl -X POST -H 'content-type: application/json' -d '{"poses":"ar","points":"dense","resize":0.5}' \
  localhost:8000/api/v1/pipeline/projects/<id>/splats/modp
```

The backend reports `modp: false` unless the checkout, `scannet.ckpt`, MODP's python and `nvidia-smi`
all exist, so on a Mac the endpoint answers `503` and every CPU stage is unchanged. RTX 50xx is too new
for CUDA 11.8. Without a GPU, the Colab handout (`modp-colab/` in the project files) runs the same
scripts on a free T4. Tested here: the job wiring end to end with a stand-in for MODP. MODP itself was not run.

## Known gaps for the fix-later pass

1. **Floor on phone captures.** RANSAC picked a wall on the phone recording, so the NavMesh can't be
   built. ARCore gives gravity, so the floor normal is known: use it to choose the plane.
2. **SfM vs ARCore.** Two API runs of the global mapper landed 8.81 and 8.57 cm from ARCore, while the
   sensor-capture thread measured about 3 cm on the same bundle. The API run self-calibrates a
   SIMPLE_RADIAL camera; seeding it with ARCore's intrinsics and adding ARCore positions as BA priors are
   the first things to try (inferred, not yet tested).
3. **Splat training not run here.** No OpenSplat binary in the cloud machine. The `init=monocular` dataset
   (fused points as the starting cloud, +3.7 dB at 2k steps on ICL) is untested through the API.
4. **Deblur** needs PyTorch and the Restormer clone + weights; not run through the API.
5. **Live tracker** is still the research script and reads the ICL-NUIM folder layout, not project frames;
   the streaming `/capture/` mapper is separate and has no tracking-error number yet.
6. **Real pose-prior BA** (the 1.53 cm result used simulated priors) is not implemented; `ar-poses` only
   triangulates and runs a free-gauge BA.
7. **Anchoring.** The service fits per-frame scale/shift. The better research variants (boot anchors 3.0%
   AbsRel, multi-view check 1.7%) are not ported.
8. **MODP export** writes the frames as-is; MODP expects undistorted frames. MODP itself needs CUDA.
9. **Lobby recipe.** The settings that worked on the white-wall lobby video (incremental mapper, SIFT peak
   0.002, overlap 25, matching the revisit) are not a preset yet; the global mapper smeared it.
10. **Jobs.** No cancel endpoint, one job per project at a time, parameters live in files, and a server
    restart marks running jobs as failed.
11. `export/splat-pack`, `splats/gpu-results` and the walk demo with a
    trained splat have not run yet.
12. Phone VR (stage 2 of walking) has a design only (`navigation/phone-vr-design.md` in the shared folder).
