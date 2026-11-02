# NAMMA Panorama v1 — online Street View to 3D

This is a separate 360°-panorama-to-3D pipeline for NAMMA. Its user-facing entry point is the existing online Street View Selector. There is no Terminal, local launcher, manual file download, or pasted API token in the normal user flow.

## Isolation from the video pipeline

All panorama reconstruction code lives under `colab/real_scenes/panorama_v1/`. Its jobs, Kaggle datasets, GPU kernels, Hugging Face inbox/output, and GitHub Actions workflows are separate from the video-to-3D system.

The panorama code does not import or execute the video capture, frame extraction, SfM, or training pipeline under `colab/real_scenes/v3/`. The video workflow `.github/workflows/namma-v3.yml` and every existing file under `v3/` remain untouched. The static tour exporter calls the existing read-only generic PLY-to-viewer builder in `colab/real_scenes/final/make_tour.py`; no file in `final/` is modified.

## Online user flow

Open the Street View Selector:

<https://anasrazy-namma-tour.static.hf.space/streetview-selector/>

1. Connect the map with the Google Maps key, draw the road, and scan it for Street View panorama positions.
2. Select at least 8 panorama positions in route order. The pipeline accepts up to 200.
3. Keep the default capture quality at **Zoom 4, 44 × 22 tiles (968 tiles per panorama)**. The downloader's limit is 1,000 tiles per panorama.
4. Click **Start 3D Rendering online** and sign in with the Hugging Face account that owns the NAMMA private inbox.
5. The browser calls the Street View downloader, validates each returned 2:1 panorama, and uploads it directly into the private Hugging Face inbox. It uploads the route manifest and publishes the job request last.
6. The cloud worker stages the uploaded images into a private Kaggle dataset, runs the independent GPU reconstruction, then publishes a static tour and private diagnostics.

The selector uses Hugging Face's browser OAuth flow with the Space's existing read/write repository scopes. The access token stays in memory in the page and is not written into browser storage, a source file, or a downloadable script. The browser checks that the authenticated owner is `anasrazy` before accepting a submission.

The live Space deployer updates only these four files in `anasrazy/namma-tour`:

- `streetview-selector/index.html`
- `streetview-selector/app.js`
- `streetview-selector/style.css`
- `streetview-selector/panorama-client.js`

It does not change the main tour viewer, existing published tours, Space OAuth settings, or video code.

## Queue activation

The browser writes to the private inbox `anasrazy/namma-panorama-inbox`. The default-branch GitHub Actions worker polls this inbox every five minutes and processes one pending panorama job at a time. The scheduled worker becomes active once this feature branch is merged into `main`, because GitHub scheduled workflows run from the default branch. The draft PR is not merged yet.

The private result dataset is `anasrazy/namma-panorama-out`. A job status file is published at `<job_id>/status.json`. After a successful run, the public static tour is published at:

`https://anasrazy-namma-tour.static.hf.space/panorama/<job_id>/index.html`

The selector polls the job status and shows **Open 3D tour** when the published link is ready. A queued or failed job does not get a made-up viewer URL.

## Reconstruction pipeline

1. Read the uploaded 360° equirectangular panoramas and their GPS coordinates/order.
2. Generate 12 overlapping perspective views per panorama. Each view is 1280 × 960 pixels.
3. Extract ALIKED features, match with LightGlue, add NetVLAD retrieval pairs, and run COLMAP structure-from-motion.
4. Estimate camera poses and 3D points across the spatially separated panorama positions.
5. Train a Gaussian-splat model on an NVIDIA GPU in Kaggle.
6. Fit route alignment against GPS when enough reliable coordinates are available.
7. Export the raw PLY model, held-out quality metrics, SfM diagnostics, logs, and a route-oriented NAMMA static tour.

The synthetic perspective views created from one panorama all share the same camera centre. They do not create depth by themselves. Actual 3D depth comes from parallax between the different Street View camera positions, so choose a route with many distinct, overlapping panoramas rather than widely separated points.

## Video-pipeline quality target

The panorama trainer uses the same core full-run training budget as NAMMA's video pipeline:

| Setting | Panorama target |
|---|---:|
| Input perspective-view resolution | 1280 × 960 |
| Perspective views per panorama | 12 |
| Training iterations | 30,000 |
| Maximum Gaussian splats | 1.5 million |
| Appearance correction | Bilateral grid |
| Depth loss | Enabled when every registered training view has 3D points |
| Validation | Held-out PSNR, SSIM and LPIPS when supported by the run |

This matches the video pipeline's core quality settings; it does **not** guarantee identical visual quality. A quality claim requires a completed GPU run, its held-out metrics, and an actual inspection of the published tour. Weak coverage, poor overlap, motion, reflections, repetitive buildings, or inaccurate GPS can still reduce the result.

The Kaggle runner requires an NVIDIA GPU and stops if it receives CPU-only hardware. A model is not considered complete until the held-out metrics and the exported `full/model.ply` exist and the static NAMMA viewer can open the scene.

## Panorama input requirements

- 8–200 positions per route; 20–100 well-overlapping viewpoints is a better target for most routes.
- Full 2:1 equirectangular images, preferably at least 2048 × 1024 pixels.
- Preserve sequence order and latitude/longitude for route alignment.
- Avoid large spatial gaps, duplicated images, blurred captures, and unrelated scenes.
- The online Street View downloader uses an unofficial external tile service. Its output, availability, and rate limits are separate from the reconstruction pipeline and may change.

## What this first release does not claim

- One panorama cannot provide a full 3D reconstruction.
- A Gaussian appearance model is not a watertight mesh. This release supports orbit/route viewing, not collision-safe walking or engineering-grade geometry.
- GPS-based scale and orientation can be inaccurate when the pano metadata is noisy or mismatched.
- Quality parity with a specific video tour is only established after held-out metrics and visual inspection, not from configuration alone.

## Tests and isolation checks

The panorama test workflow compiles and tests only `panorama_v1`. It uses a test configuration boundary so repository-wide video/application fixtures are not imported. The GPU quality benchmark is separate from unit tests: passing unit tests does not mean that a 3D model has been trained.
