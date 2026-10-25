# NAMMA hand-off: v3 pipeline (current, written 26 Oct 2026 ~00:30 IST)

**Start here.** This replaces `docs/HANDOFF.md` (24 Oct) and `docs/HANDOFF_CLOUD_2026-10-25.md` (25 Oct) as the entry point; they
describe the older custom pipeline and are kept for history. Measured findings, newest last: `colab/real_scenes/final/NOTES.md`.
Repo `github.com/RazyAnas/NAMMA` (public), branch **`claude/wonderful-cori-i3fn41`** (all work below is there, not on `main`).

## 1. Goal and owner rules
- **Goal**: phone video of a real room in → photoreal, walkable 3D model of *that* room out (real estate). Generation may only
  fill gaps, never replace or invent what was filmed, and generated content is flagged. Free GPUs only (Kaggle T4 x2).
  One command: video in, best possible 3D model out. Owner wants "extreme photorealism" (like Marble / Tencent HY-World) and
  quick proofs. The YouTube method (Mickmumpitz / Matrix-3D: geometry → drone views → re-project real pixels → video model fills
  only the rest → retrain) is planned as step 6 (below).
- **Owner**: Raiyan Razi (they/them), GitHub `RazyAnas`, Kaggle `anasrazy`, Hugging Face `anasrazy`. Time zone IST.
- **Rules** (keep them):
  - Don't ask questions, act. Short answers, plain words.
  - Check every result yourself (QA sheet, metrics) before handing anything over. Never hand over unchecked models.
  - Measure with proof. Keep measured numbers and claims apart. Say plainly when something failed.
  - Give an expected finish time (IST) with every job.
  - Only one AI drives Kaggle at a time.
  - Tokens and keys: never in git or chat. They live in GitHub Actions secrets and Kaggle Secrets.
  - Public / passive data only; never leaked data; never log in as anyone else.
  - The owner's Mac disk is nearly full: download only small files (QA sheet, metrics, `.splat`), never the whole output.

## 2. Why v3 (decided 25 Oct after namma-all failed)
Three days of custom code produced one decent partial model (corridor: 162 of 596 frames, 38.1 dB held-out, 38% needle
splats, no windows). Root causes: (1) the owner's 22 Oct capture (fast turns, blank walls, look-alike corridor ends,
auto-exposure) broke the camera solve into disagreeing passes; (2) custom trainer + repair layers (pass alignment, video bridges,
IBR checks, Wan filling) compensated for a bad capture instead of using good capture + standard tools; (3) namma-all (ARKitScenes
video) died because Depth Anything 3 ran out of GPU memory on landscape frames (5/300 merged) and every later stage needed depth.
**v3 = what everyone with good results uses**: good capture → SfM cameras (hloc) → gsplat's own `simple_trainer.py` → QA.
The custom pieces stay in the repo as options (`final/train_splat.py`, `align_passes.py`, `ibr*.py`, `fill_holes.py`, `densify.py`,
photo viewer) but are out of the default path.

## 3. The v3 pipeline (one Kaggle job: `colab/real_scenes/v3/run.py`, kernel `anasrazy/namma-v3`, 2x T4)
| Step | What | Code | Measured time |
|---|---|---|---|
| 1 frames | video: sharpest frame per time window (variance of Laplacian); folder: evenly spaced subset; long side ≤ 1280 | `final/frames.py` | < 1 min |
| 2 cameras | hloc SuperPoint+LightGlue, pairs = next 10 frames + NetVLAD top 15, COLMAP mapper, one shared SIMPLE_RADIAL camera → `sfm_report.json` | `final/sfm_hloc.py` (py3.10 uv venv, torch 2.1.2 cu118) | 411 frames: ~11 min incl. venv install |
| 3 splats | gsplat v1.5.3 `examples/simple_trainer.py mcmc`, bilateral grid (phone auto-exposure), SfM depth loss, held-out eval every 8th frame, `--save-ply`. GPU 1: 7k-step preview (cap 1M splats); GPU 0 at the same time: 30k-step full model (cap 1.5M) | upstream trainer, system Python 3.13 | preview ~15 min, full ~60 min (estimate, not yet measured) |
| 4 QA | `qa_sheet.jpg` (row 1: 3 held-out frames real ∣ model; row 2: two views ~50 cm beside the walk, one from outside above, one top-down), `metrics.json` (held-out PSNR/SSIM/LPIPS, exposure-fair `cc_` versions, needle fraction, off-path coverage, `pass`), levelled `model.splat` (floor at y=0, for SuperSplat, SH0 only), `model.ply` (full SH3), `walkthrough.mp4` (trainer fly-through, colour ∣ depth) | `final/v3_qa.py`, `final/to_splat.py` | ~1 min |

**Pass** = exposure-fair held-out PSNR ≥ 28 dB, needles < 10%, and the QA sheet shows the whole room (no missing walls / fog /
floaters in the off-path, outside and top-down views), model levelled.

Kaggle output layout: `/kaggle/working/{source.json, sfm_report.json, v3_summary.json, preview/, full/}`, each model folder:
`qa_sheet.jpg metrics.json nav.json model.splat model.ply walkthrough.mp4 train.log`.

`SOURCE` (set by `gh_run.py`, or edit the default in `run.py`): `kaggle:<dataset name>` (video, frame folder or .zip in that
attached dataset) · `kaggle` (first .mp4/.mov under /kaggle/input) · `dl3dv:<hash>|dl3dv:auto` (needs Kaggle secret `HF_TOKEN`
attached in the Kaggle editor) · a path. Other env knobs: `NFRAMES` (default 300), `LONG` (1280), `STEPS` (30000), `PREVIEW` (7000).

## 4. How runs are started and collected (GitHub Actions; no Mac needed)
The Claude cloud container **cannot reach kaggle.com** (proxy 403). GitHub's runners can, so:
- Workflow `.github/workflows/namma-v3.yml` runs `colab/real_scenes/v3/gh_run.py` on every push to the branch that changes
  **`colab/real_scenes/v3/RUN.json`**. (`workflow_dispatch` only works once the file is on `main`.)
- **To start a run**: edit `RUN.json`: bump `n`, set `source`, make sure `"collect": false` (or remove it), push.
  `source`: `dl3dv:<hash>` / `dl3dv:auto` (runner downloads the DL3DV-Benchmark scene with the HF token, converts to JPG, uploads
  it once as private Kaggle dataset `anasrazy/namma-dl3dv-<hash[:10]>`) or `kaggle:<owner>/<dataset>` (a video/frames dataset
  already on Kaggle, e.g. the owner's re-filmed room). `env` overrides run.py knobs.
- **`"collect": true`** = push nothing; wait for the Kaggle version already running and collect its outputs (use when a GitHub run
  was cancelled: cancelling GitHub does NOT stop the Kaggle job). **Set it back to false before the next real run.**
- gh_run then pushes the Kaggle job (run.py with the settings prepended), polls `kaggle kernels status` every minute, downloads all
  outputs, prints sfm_report / v3_summary / metrics / log tail into the GitHub log, and uploads everything (except the preview .ply)
  to the private HF dataset **`anasrazy/namma-v3-out/<YYYYMMDD-HHMM UTC of the GitHub run>/`**. Exits red if the Kaggle job
  failed or made no full model.
- **Secrets** (repo Settings → Secrets and variables → Actions; already set by the owner): `KAGGLE_TOKEN` (contents of
  `~/.kaggle/access_token`; `KAGGLE_API_TOKEN` also accepted) and `HUGGINGFACE_WRITE_TOKEN` (`HF_TOKEN` also accepted). The
  owner has clicked "Agree" on `DL3DV/DL3DV-Benchmark` with that HF account.
- **Watching**: run status is public JSON: `curl -s https://api.github.com/repos/RazyAnas/NAMMA/actions/runs/<run id>`
  (`status`, `conclusion`). Job logs are only downloadable after the job ends (GitHub MCP `get_job_logs`). Results: HF connector
  `hf_fs` (authenticated as anasrazy): `cat hf://datasets/anasrazy/namma-v3-out/<run>/full/metrics.json`,
  `attach .../full/qa_sheet.jpg` to look at the sheet. Owner opens `model.splat` (or `model.ply`) by drag-and-drop in SuperSplat
  (superspl.at/editor), downloaded from the HF dataset page while logged in.
- Old way (owner's Mac, still works for kernels that need no secrets): `bash colab/real_scenes/final/run_and_alarm.sh <kernel
  folder> anasrazy/<kernel> <out dir> '<file regex>'` (pushes, waits, downloads matching files, rings).

## 5. State right now (00:30 IST, 26 Oct)
- **Test scene**: DL3DV-Benchmark #44 `389a460ca1995e0658e85fe8e6b520b4e88b370cd6710dfe728b1564bba31aee` (Hotels-Motels, bounded,
  non-reflective, non-transparent): a hotel room filmed handheld on a phone, 411 frames 960x540 (bed, sofa, window, bathroom door,
  a mirror, a dog walks through). Chosen by looking at its frames. On Kaggle as `anasrazy/namma-dl3dv-389a460ca1`.
- **Run 1** (Kaggle version 1): cameras **411/411 registered, 1 model, 1.15 px reprojection, k1 = -0.0007, 13,964 points**.
  gsplat CUDA build reused from namma-prep (ready in 1 s), fused-ssim compiled, 2 GPUs. Both trainers died on
  `No module named 'datasets.colmap'` (Kaggle's Hugging Face `datasets` package hides gsplat's `examples/datasets`, which has no
  `__init__.py`). Fixed: run.py touches `examples/datasets/__init__.py`, imports the trainer once and pip-installs anything missing.
- **Run 2** (Kaggle version 2, pushed 22:46 IST, finished ~00:04 IST; its GitHub runner was cancelled by hand at 00:03 IST, the
  outputs were fetched by collect run `40221983672`). **Results: `hf://datasets/anasrazy/namma-v3-out/20261025-1835/`**
  (https://huggingface.co/datasets/anasrazy/namma-v3-out/tree/main/20261025-1835): cameras 411/411 again (2 models in the folder,
  411 in the main one, 1.15 px).
  | model | steps / time | splats | held-out PSNR raw / exposure-fair | SSIM | LPIPS |
  |---|---|---|---|---|---|
  | preview (GPU 1) | 7k / 10 min | 317k | 24.54 / 26.76 dB | 0.908 | 0.207 |
  | **full (GPU 0)** | 30k / 68 min (~6.3 it/s on a T4 while the preview shared the CPU) | 1.5M (cap) | **25.71 / 29.50 dB** | **0.939 / 0.945** | **0.132 / 0.120** |
  QA sheet checked by eye: held-out renders are near-identical to the real frames (bed, mirror, bathroom, sofa, the dog), and the
  two views 50 cm beside the walk are clean and photoreal. **First genuinely photoreal model.** `pass` was false only because
  of two QA-tool faults, both fixed after the run (not yet re-run): (1) the needle count flagged flat discs (longest/shortest > 10),
  76%; now needle = longest/middle > 10, flat discs are reported separately; (2) the outside and top-down overview shots looked
  into the shell of big splats that models the view through the window and showed only blur; they now render a dollhouse (room box
  from the SfM points, ceiling cut). Files: `full/model.splat` (48 MB, levelled, SH0), `full/model.ply` (354 MB, SH3),
  `full/walkthrough.mp4`, `full/qa_sheet.jpg`, `full/metrics.json`; same for `preview/` (no .ply on HF).
- `RUN.json` was reset to `n: 0, collect: false` after collecting (pushing RUN.json always starts the workflow; n = 0 exits at once).
- No reminders or jobs are pending (the old session's check-in was deleted once run 2 was collected).

## 6. Next steps (in order)
1. Show the owner run 2 (`full/model.splat` in SuperSplat). Optional: re-run the same scene (RUN.json `n` bump, `collect` false)
   to get the fixed QA sheet (dollhouse views, true needle count) before claiming "whole room OK"; the off-path / outside quality is
   still unproven. If smears or floaters show up: MCMC `--scale-reg` up / cap down, or more steps.
2. **Owner's own room**: owner re-films following `docs/CAPTURE_GUIDE.md` (landscape, locked exposure/focus, slow turns, one loop,
   1-2 min), uploads the video as a Kaggle dataset, then RUN.json `source: "kaggle:anasrazy/<dataset>"`, bump `n`, push.
3. gh_run: check whether the Kaggle dataset already exists *before* downloading DL3DV (wastes ~8 min now); optionally upload a
   progress file to HF while waiting (logs are invisible until the GitHub job ends).
4. **Camera fallback** when SfM breaks a video into several models (`sfm_report.models > 1` or registered < 85%): VGGT-Long
   (`facebook/VGGT-1B`, CC BY-NC) poses → triangulate + bundle adjust with the hloc matches (new `final/vggt_poses.py`).
5. **Step 6, the YouTube method** (only after the base model passes): virtual "drone" paths 30-100 cm off the walk, render from the
   model, holes = low alpha; fill holes with real pixels re-projected from all frames first (`final/ibr.py` `synth(..., kmax=40)`),
   Wan 2.1 VACE 1.3B fills only what is left (`final/fill_holes.py`, flagged), add these views to training at weight 0.3 with
   unfilmed pixels masked, +5k steps (new `final/drone_views.py`). Measured on the old data: Wan filled 0.8% of pixels for 20 GPU
   minutes, so real coverage matters more than generation.
6. Viewer integration (trained model between photo spots, collision from the floor plan: `final/fuse_nav.py`, `photo_viewer.*`),
   Techfest R1 on **31 Oct** (`docs/TECHFEST_R1.md`: numbers, screen recording, abstract, hosted viewer).
7. Old bugs (only if those paths are used again): DA3 OOM on landscape frames in `final/run.py`; a failed depth merge must not kill
   the camera solve (`sfm_align.py`); clear errors on 0 frames in `ibr_check.py` / `align_passes.py`.

## 7. Gotchas (each one cost a run or an hour)
- **Two Pythons on Kaggle**: hloc needs the official `pycolmap` (py3.10 venv at `/kaggle/temp/v3/sfmenv`); simple_trainer needs
  the `rmbrualla/pycolmap` fork (system Python). The fork overflows on numpy 2 at import (`np.uint64(-1)`): run.py sed-patches
  `scene_manager.py`, locating it with `importlib.util.find_spec` (importing it would fail).
- gsplat 1.5.3 from PyPI compiles CUDA on first use (~5 min). namma-prep's output (`gsplat_ext.tgz`, attached via
  `kernel_sources`) is restored into `~/.cache` and the gsplat sources are back-dated so ninja reuses it; `final/gsplat_warm.py`
  compiles/loads once before two GPUs start.
- fused-ssim is compiled with `TORCH_CUDA_ARCH_LIST=7.5`; if it fails, `final/fused_ssim_shim/` (pure torch) goes on PYTHONPATH.
- simple_trainer's depth loss crashes (KeyError) if any image has no SfM points; sfm_report counts `frames_without_points` and
  run.py drops `--depth-loss` then.
- Checkpoints are `ckpts/ckpt_{steps-1}_rank0.pt`, PLY `ply/point_cloud_{steps-1}.ply`, stats `stats/val_step{steps-1:04d}.json`.
  With the bilateral grid, `psnr` is raw and `cc_psnr` is after colour matching (exposure-fair; compare this with old numbers).
- The trainer's world is normalised (Parser `normalize=True`); v3_qa uses the same Parser, so cameras/points match the PLY.
  `nav.json` has `flipped: false` (simple_trainer PLYs are not y/z flipped; old train_splat PLYs were).
- Trained in classic mode (not `--antialiased`) so SuperSplat shows exactly what was measured. `.splat` keeps only SH0 colour;
  `.ply` keeps view-dependent SH3.
- After `kaggle kernels push`, status can still show the previous version's `complete`; gh_run waits to see running/queued first
  (or 15 min).
- Kaggle mounts datasets at `/kaggle/input/datasets/<owner>/<name>/` (new layout) or `/kaggle/input/<name>/`: always glob.
- DL3DV-Benchmark meta labels: `environment` bounded/unbounded, `ref_degree_binary` (non)reflection, `trans_degree_binary`
  (non)transparent, `scene` category (no "indoor"/"room" words). Frames: `<hash>/nerfstudio/images_4/frame_00001.png` (960x540).
- kaggle CLI 2.2.5 authenticates with the `KAGGLE_API_TOKEN` env var alone.

## 8. Earlier results worth knowing (details in NOTES.md)
- Lobby/room tests with Nerfstudio and our trainer: fog and floaters whenever cameras were wrong; camera quality was the lever.
- Corridor model (namma-model, 25 Oct): 162 frames of one consistent pass group, **38.1 dB / SSIM 0.957 held-out** (custom trainer,
  15 min), 37.8% needles; MCMC variant 34.2 dB with 0.1% needles. Levelled .splat opened by the owner: no windows, smears.
- Wan 2.1 VACE hole filling: honest only for small gaps (0.8% of pixels at 30 cm off path); in big holes it invented objects.
- `docs/CAPTURE_GUIDE.md` is the owner's filming guide; `docs/VIDEO_METHOD_PLAN.md`, `docs/PHOTOREAL_PLAN.md` older plans;
  `docs/TODO_2026-10-26.md` the to-do written before v3 (its "one-command job" item is now v3).
- Status page: `dash/status.json` (+ `dash/index.html`, `dash/watch.sh`).
