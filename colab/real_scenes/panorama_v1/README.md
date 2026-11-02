# NAMMA Panorama v1 — Street View panoramas to 3D

This directory is an independent reconstruction pipeline. It is designed to run beside NAMMA's existing video-to-3D flow without editing or invoking that flow.

## Isolation boundary

The panorama pipeline is contained in this directory plus one new GitHub Actions workflow:

- `.github/workflows/namma-panorama.yml`
- `colab/real_scenes/panorama_v1/**`

It does **not** modify or read the video pipeline's `colab/real_scenes/v3/RUN.json`, `v3/run.py`, `v3/gh_run.py`, `v3/inbox.py`, `.github/workflows/namma-v3.yml`, or the video inbox. Its Kaggle kernel ID, job files, status path and output dataset are separate.

The static tour exporter imports the existing read-only SuperSplat tour builder in `colab/real_scenes/final/make_tour.py` so panorama results open in the same NAMMA viewer. It only calls its generic PLY-to-static-viewer build function; it does not call any video capture, SfM, frame extraction or training stage. No file in `final/` is modified.

## What the pipeline does

1. Accepts a set of 360-degree equirectangular panoramas and a JSON manifest.
2. Generates 12 overlapping pinhole perspective views from each panorama.
3. Extracts ALIKED features, matches them with LightGlue, adds NetVLAD retrieval pairs, and runs COLMAP structure-from-motion.
4. Creates one shared COLMAP camera model and estimates camera poses/3D points across panorama positions.
5. Trains a separate gsplat 3D Gaussian model on Kaggle GPU.
6. Fits a route alignment to GPS coordinates when enough panorama positions have GPS metadata.
7. Exports a route-oriented static NAMMA tour, the raw PLY model, SfM diagnostics and a training log.

A panorama gives a complete view from one camera centre, not depth from every direction. Multiple spatially separated panoramas are essential: the reconstruction gets parallax from the movement between Street View camera positions, not from the 12 synthetic crops of one panorama. Synthetic crops from a single pano have the same camera centre.

## Panorama bundle format

Create a ZIP with this layout:

```text
bundle.zip
├── manifest.json
└── panoramas/
    ├── pano_00001.jpg
    ├── pano_00002.jpg
    └── ...
```

Example manifest:

```json
{
  "schema_version": 1,
  "scene_name": "Koramangala street route",
  "panoramas": [
    {
      "pano_id": "pano-id-1",
      "image": "panoramas/pano_00001.jpg",
      "sequence": 0,
      "lat": 12.9352,
      "lon": 77.6245,
      "heading_deg": 82.0
    },
    {
      "pano_id": "pano-id-2",
      "image": "panoramas/pano_00002.jpg",
      "sequence": 1,
      "lat": 12.9353,
      "lon": 77.6247,
      "heading_deg": 84.0
    }
  ]
}
```

The example has two entries only to explain the fields; a real job needs at least 8 panoramas and should generally use 20-100+ well-overlapping positions. Latitude and longitude are optional, but strongly recommended. Without GPS, the tour still renders but its world scale is arbitrary. Images should be equirectangular, approximately 2:1 aspect ratio, reasonably sharp, and taken from distinct positions along the same connected scene. Avoid large gaps, repeated identical views, and routes with long stretches of blank walls.

The selector/downloader should export its chosen panoramas into this bundle. The selector integration patch below adds per-panorama GPS coordinates and order to the JSON export, so the uploader can match the downloaded `<pano_id>.jpg` files and preserve route alignment. Without that metadata, the tour still renders but has arbitrary scale/orientation.

## Add the Start 3D Rendering action to the Street View Selector

The Street View Selector is hosted separately in the Hugging Face Space `anasrazy/namma-tour`. Deploy its integration from this branch:

```bash
hf auth login
python3 -m pip install "huggingface_hub>=0.28"
python3 colab/real_scenes/panorama_v1/deploy_selector_integration.py
```

The deployer verifies that the logged-in account is `anasrazy` and updates only these three Space files: `streetview-selector/index.html`, `streetview-selector/app.js`, and `streetview-selector/style.css`. It does not change the root tour viewer, existing tours, Space README/settings, or any video-pipeline file.

After deployment, select at least 8 panorama positions, download their JPEGs using the existing downloader, and click **Start 3D Rendering**. The browser downloads a local `.sh` launcher. Run it in Terminal; it asks for your Hugging Face and GitHub tokens without displaying them, uploads the selected JPEGs and metadata, and queues a panorama job on this branch. It does not store tokens in the browser or embed them in the generated script.

The launcher targets `feature/streetview-panorama-3d` while this is being tested. After this branch is merged into `main`, redeploy the selector integration with:

```bash
NAMMA_PANORAMA_BRANCH=main python3 colab/real_scenes/panorama_v1/deploy_selector_integration.py
```

This updates the generated launcher to fetch the submission helper from `main` and queue jobs on `main`.

## Submit a job

Set repository secrets in **RazyAnas/NAMMA → Settings → Secrets and variables → Actions**:

- `KAGGLE_API_TOKEN`: Kaggle API token with permission to create datasets and private GPU kernels.
- `HF_TOKEN`: Hugging Face write token. It needs write access to the private input/output datasets and the `anasrazy/namma-tour` Space. A legacy `HUGGINGFACE_WRITE_TOKEN` secret is also accepted.

For the first run, create a local Hugging Face write token and a GitHub fine-grained token with **Contents: Read and write** on `RazyAnas/NAMMA`. Do not commit or share either token.

On macOS/Linux, from the repository checkout:

```bash
export HF_TOKEN='your-hugging-face-write-token'
export GITHUB_TOKEN='your-github-contents-token'
python colab/real_scenes/panorama_v1/submit_job.py \
  --input /path/to/downloaded/panoramas \
  --scene-name "My Street View route" \
  --steps 30000 \
  --max-panoramas 100
```

The default target is `feature/streetview-panorama-3d`. After this branch is merged, submit future jobs with `--branch main`. The helper uploads the ZIP to `anasrazy/namma-panorama-inbox`, then creates a single job manifest under `panorama_v1/jobs/`. Only that job-manifest path triggers the panorama workflow.

The workflow creates its own private Kaggle input dataset and kernel, waits for the GPU job, publishes the static tour under:

`https://anasrazy-namma-tour.static.hf.space/panorama/<job_id>/index.html`

It stores the raw PLY model, logs and diagnostics under the private dataset `anasrazy/namma-panorama-out/<job_id>`. A job status JSON is also written to `<job_id>/status.json`.

## Initial reconstruction settings

| Setting | Default | Meaning |
|---|---:|---|
| Panoramas | up to 200 | Maximum input positions consumed |
| Perspective views | 12 per panorama | Overlapping horizontal views around 360° |
| View image size | 768 × 576 | Resolution supplied to SfM and training |
| Horizontal field of view | 60° | Perspective crop field of view |
| SfM sequence window | 18 | Nearby views paired for matching |
| Retrieval neighbours | 10 | Additional long-range matching pairs |
| Gaussian training | 30,000 steps | Dedicated gsplat MCMC run |
| Splat cap | 1.5 million | Upper limit for the Gaussian representation |

The Kaggle runner requires an NVIDIA GPU and stops if Kaggle assigns CPU only. The first run compiles/warms the gsplat CUDA extension. Runtime depends on panorama count, feature matching, Kaggle queue and GPU availability; no completion time is guaranteed.

## What the first release does not claim

- One panorama cannot produce a complete 3D reconstruction by itself.
- The 12 perspective images from one panorama are not 12 independent camera positions.
- GPS helps position and scale the route, but noisy or mismatched coordinates can make the metric fit inaccurate. Check `tour.json` and `gps_alignment_rms_m`.
- A Gaussian appearance model is not a watertight mesh. This first panorama release publishes orbit/route viewing, not collision-safe walking or a certified geometry model.
- Moving vehicles, pedestrians, changing storefronts, glass, repetitive facades and sparse coverage can cause reconstruction artifacts.
- The external Street View downloader is an image source, not part of the reconstruction engine. Its availability, output quality and service rules remain separate constraints.

## Tests

The panorama test workflow installs only panorama test dependencies, compiles the panorama modules, and runs `pytest -q --confcutdir=tests/panorama_v1 tests/panorama_v1`. The `confcutdir` setting prevents the repository-wide application/video test fixture from loading. It does not execute the video tests or video pipeline.
