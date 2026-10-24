# NAMMA Space: hand-off for parallel work (2026-10-24)

## Target
Phone video (no LiDAR, no special hardware) of a real indoor space becomes a **photoreal, walkable 3D model**:
clean from every angle, including views off the path the phone was carried along, with collision, floor plan and
navigation, produced fast and without hand work. Real-estate use, so the model must be **true to the real room**:
generation may only fill gaps, never replace what was filmed. Later: multi-room and multi-floor.

## Where everything is

| What | Where |
|---|---|
| Code (all work below) | github.com/RazyAnas/NAMMA, branch **`claude/project-thread-l70z4u`** |
| 4K test video (22 Oct, 3 min 45 s, 2160x3840, 30 fps, camera-only recording) | Mac: `/Users/mariyafatima/NAMMA/namma-camera-2026-10-22T10-13-20.zip` (`video.h264` inside, 1.2 GB) |
| Same video as MP4 on Kaggle (private dataset) | `anasrazy/namma-room-2026-10-22` → `/kaggle/input/namma-room-2026-10-22/video.mp4` |
| 30 s lobby clip used for the tests | repo `colab/real_scenes/videos/lobby_clip30.mp4` (+ 180 frames in `colab/real_scenes/frames/lobby_clip30`) |
| Room splat, Nerfstudio 30k (soft, see results) | Mac: `/Users/mariyafatima/NAMMA/room_2026-10-22_splat.ply` (= `~/Downloads/results (1).zip`) |
| Kaggle account | `anasrazy`; CLI token on the Mac at `~/.kaggle/access_token` (`kaggle kernels push/status/output`) |
| Recorder web app | repo `app/static/record/` (published copy: repo `RazyAnas/namma-space-pages`, edited by another session) |
| Render server job (Modal / Colab / laptop) | repo `app/pipeline/stages/phone.py` (`final_plan`, task `splats.final`) |
| Research notes | repo `docs/SPLATTOUR_RESEARCH.md`, `docs/WHY_LOBBY_FAILED.md` |
| Result pages | Nerfstudio runs: claude.ai/artifact/6Rj1BWscw6KqFbmqpTG2zr · walkable lobby: claude.ai/artifact/1fcs8dFCoa97yhuGjJYJVT |

## What is proven (measured)

| Run | Input | Held-out PSNR / SSIM / LPIPS | Note |
|---|---|---|---|
| Nerfstudio defaults | whole lobby, 1 frame/s | 21.0 / 0.887 / 0.48 | fog, floaters |
| Same, sharp 30 s clip, sequential matching | 101/180 frames placed | 22.8 / 0.904 / 0.28 | |
| Same clip, exhaustive SuperPoint+LightGlue | 173/180 | 24.95 / 0.914 / 0.28 | |
| Full pipeline, 30k steps | 173/180 | **26.9 / 0.923 / 0.24** | best so far; doors/fire-hose near photo |
| 4K room video, render-server mode | 574/578 cameras | (no hold-out) | geometry right, surfaces mush: only 317k splats for 3+ min of walking |
| **Depth Anything 3 (feed-forward, no training)** | 22 frames at 540p, HF Space | | **61 s**, clean coherent room, softer texture |

Lessons: sharp dense frames are the biggest lever; SuperPoint+LightGlue solves what SIFT/ARCore could not; deleting
"floaters" after training lowers quality; plain white walls fail with every trainer; off-path views break beyond
~1-3% of the path length; one model for a whole floor is too thin (Splat Tour trains one splat per room).

## Plan (each step is independent enough to run in parallel)

1. **Feed-forward 3D, no training** (in progress): WorldMirror 2.0 (Tencent HY-World 2.0, `tencent/HY-World-2.0`) and
   Depth Anything 3 on the 4K room frames: cameras + depth + Gaussians in one pass. Also MapAnything (Meta) for metric
   scale. Output: a clean, gap-free base model in minutes.
2. **Fill only the gaps with generation**: render views off the walked path from step 1, let a video/image model fill
   just the holes (NVIDIA Difix3D+, or Skywork Matrix-3D's Wan 2.1 LoRA), guided by the real geometry.
3. **Keep filmed pixels true**: re-project the real 4K frames onto those views (the "8K re-projection" trick from the
   Matrix-3D ComfyUI video) so everything filmed stays exact.
4. **Short splat fit** (5-10 min, not 2 h) on real + filled views, starting from step 1. Then splat → mesh/TSDF for collision,
   floor plan and walkable area (like Splat Tour's navigability grid).
5. **Per room**: split the walk into rooms (camera position + time), one model per room plus an overview, streamed by
   room in the viewer (PlayCanvas, SOG format), as Splat Tour does. Multi-floor = same per floor.
6. **Viewer**: walk limit around the filmed path, room streaming, 360° backdrop, keep the capture's aspect.

## How to run things

- Kaggle background job (no notebook babysitting): a folder with `run.py` + `kernel-metadata.json`
  (`enable_gpu`, `enable_internet`, `machine_shape: NvidiaTeslaT4`, `dataset_sources: ["anasrazy/namma-room-2026-10-22"]`),
  then `kaggle kernels push -p <folder>`, `kaggle kernels status anasrazy/<name>`, `kaggle kernels output anasrazy/<name> -p <dir>`.
  Max 2 GPU jobs at once; stop extra sessions at kaggle.com → View Active Events.
- Full Nerfstudio pipeline: `colab/real_scenes/scripts/full_pipeline.sh` (env: `VIDEO`/`FRAMES`, `FPS`, `STEPS`, `FINAL=1`,
  `METHOD`, `TRAIN_ARGS`, `DATA_ARGS`, `WORK`, `OUT`). Kernel templates: `colab/real_scenes/kaggle_full/`.
- Local fast checks without a GPU: `colab/real_scenes/scripts/splat_cpu.py` (CPU splat renderer, 0.5 s/view, matches gsplat).
- Feed-forward demo without any GPU: Hugging Face Space `depth-anything/depth-anything-3` via `gradio_client`
  (`/handle_uploads` then `/gradio_demo` with `infer_gs=True`, `num_max_points` ≤ 100000).

## Running now (2026-10-24)
- `anasrazy/namma-room-render-mcmc` v2: the room with Nerfstudio `splatfacto-big` at 1080x1920 (more, smaller splats).
- `anasrazy/namma-worldmirror2` v5 (queued): WorldMirror 2.0 on ~22 frames of the room.

## Gotchas already solved (don't repeat them)
- Kaggle's Python is 3.13 and uv targets it: create a 3.10/3.11 venv and pass `--python <venv>/bin/python` to every `uv pip install`.
- Nerfstudio 1.1.5: needs `setuptools<70` (pkg_resources), `apt-get install colmap` (ns-process-data checks it), hloc from a
  `--recursive` clone (SuperPoint submodule), prebuilt `gsplat==1.4.0+pt21cu118`; it has **no MCMC strategy** (use `splatfacto-big`).
- DN-Splatter on gsplat 1.4 needs `scripts/patch_dn_splatter.py`; its ZoeDepth script breaks on torch 2.1 (use `scripts/mono_depth_prior.py`).
- WorldMirror 2.0 on a T4: torch 2.7.1 cu128, install requirements one by one, FlashAttention does not run on T4 → stand-in using
  PyTorch SDPA in fp16 with the memory-efficient kernel (the math fallback needs 13 GB and OOMs).
- Web viewers: never crop the splat to the camera-path box (cuts far walls), keep the capture's portrait aspect, artifacts
  serve only web file types (ship .ply bytes as `.wasm`).
- Recorder: AR removed (poses drifted/reset, camera capped); 4K30 camera mode; exposure/colour locked by the user in a
  preview; focus automatic (tap to focus, hold or button to lock, never auto-lock: it blurred).
