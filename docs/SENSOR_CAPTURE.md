# Phone sensor capture: video + AR pose + IMU from Android Chrome

`/record/` is a web page (no app install) that records, on one clock, the camera video, the ARCore
pose and intrinsics of every frame (through WebXR), and raw gyroscope/accelerometer samples, and
saves them as one `.zip` bundle. `scripts/ingest_sensor_bundle.py` turns the bundle into inputs the
existing pipeline already takes.

Labels used below: **measured** means run and checked in this repo (headless Chromium or Python);
**claimed** means a spec, vendor or forum statement not yet checked on a phone; **inferred** means
reasoning, not a test.

## Does the brief allow it?

Quoted from the Namma Space problem statement:

- Inputs: "reconstructs an interactive 3D model of a real indoor space from data users can
  realistically provide (images, video, floor plans, sensor data)".
- "teams are free to utilize any technology currently available in the market."
- SOP: "specifying camera angles, corner shots,floor plan or utilizing sensors".
- Round 3: "Teams will collect data on-site using their required hardware/sensors according to their SOP".
- The non-technical user must capture "using only a smartphone"; a phone's own IMU and ARCore are
  part of the smartphone, so this fits (inferred).
- AR bonus is judged on "Accuracy of real-time camera tracking and live directional arrow overlays on-site."

## Where IMU / AR data helps, most valuable first

| Use | How | Status |
|---|---|---|
| Round 3 turnaround | ARCore poses (camera + IMU fusion) replace structure-from-motion: extract features, then `colmap point_triangulator` on `colmap_known_poses/` and a bundle adjustment, instead of a full mapper run | inferred; time saving not measured yet |
| Metric scale and "up" | Poses are in metres in a gravity-aligned frame (`local-floor`, y up), so the manual scale calibration step and floor-direction guessing go away; navmesh and paths come out in metres | claimed (WebXR spec), checked only in simulation |
| AR navigation bonus | The map is built in ARCore's own frame, the same tracking the on-site AR overlay runs on; a new session still needs aligning to the map, because every WebXR session starts a new origin (lesson from sixthsense-lab) | inferred |
| Blur and fast motion | Gyro rate at each frame flags blurred frames before they reach SfM; gyro rotation bridges short ARCore tracking losses (`emulatedPosition`) | inferred |
| Loop closure | Pose proximity proposes revisit pairs to match (already used by `live_mapper.py`); ARCore drift itself is not closed by WebXR, so the final bundle adjustment still has to | inferred |
| Offline visual-inertial SLAM from raw IMU | Weak: VIO systems (ORB-SLAM3, OpenVINS) are normally fed 200 Hz IMU, and Chrome caps web sensors at 60 Hz | cap measured (below) |
| Scale from double-integrated accelerometer | Not useful: drift grows with time squared | not pursued |

## What Android Chrome can deliver

| Stream | API | What we know |
|---|---|---|
| 6-DoF pose per camera frame | WebXR `immersive-ar` (ARCore), `local-floor` space | One pose per XR frame; tracking loss shows as `pose.emulatedPosition`, not a missing pose (sixthsense-lab). Needs an ARCore-certified phone |
| Camera image during AR | WebXR Raw Camera Access (`camera-access` feature, `XRWebGLBinding.getCameraImage`) | Image and pose come from the same ARCore frame, so they are synchronised by construction. **Resolution is not documented anywhere I found; the page prints it.** `getUserMedia` cannot be used meanwhile because ARCore holds the camera (claimed in forums) |
| Intrinsics | `view.projectionMatrix` | Converted to fx, fy, cx, cy for the camera image (formula checked against an OpenGL projection, measured in tests); assumes the camera image is aligned with the view, which needs a phone check |
| Video encoding | WebCodecs `VideoEncoder` | Baseline H.264 Annex B (no B-frames, so decode order = frame order) with VP8/VP9 in IVF as fallback. Avoids `readPixels`, which a Babylon.js user measured dropping a Pixel 6 to about 20 fps |
| IMU | Generic Sensor API (`Gyroscope`, `Accelerometer`, `GravitySensor`) plus `devicemotion` | **Measured:** Chromium rejects requests above 60 Hz ("Maximum allowed frequency value for this sensor type is 60 Hz.") and delivered about 61 Hz. Whether sensors keep firing inside an AR session must be checked on the phone |
| ARCore depth (optional) | WebXR `depth-sensing`, CPU buffer | Low-resolution metric depth on phones without LiDAR; saved every 6th frame when ticked |
| Clock | `performance.now()` | XR frame times, sensor timestamps and event times share it. The IMU-to-camera offset is estimated from the data, not assumed |

## Bundle format (`namma-sensor-bundle` v1)

Uncompressed zip:

- `meta.json`: mode (`ar` or `camera`), device, camera size, codec, frame/drop counts, measured IMU rates, errors.
- `video.h264` (raw Annex B) or `video.ivf` (VP8/VP9).
- `video.csv`: one row per encoded chunk in stream order: `chunk, ts_us, key, bytes`. The k-th decoded frame is chunk k.
- `frames.csv`: one row per XR or camera frame: `t_ms`, `video_ts_us` (empty if the frame was dropped from video), `tracking` (0 tracked, 1 position lost, 2 no pose, 3 camera-only), screen angle, `m0..m15` (column-major world_from_camera, OpenGL camera axes), `p0 p5 p8 p9` (projection terms), image size.
- `imu.csv`: `t_ms, sensor, x, y, z` in device axes; gyro rad/s, accelerometer m/s² including gravity.
- `depth.csv` + `depth/NNNNNN.bin` when depth is on.

## Ingest on the laptop

```bash
python scripts/ingest_sensor_bundle.py namma-ar-....zip data/bundles/room1 --stride 10
```

Writes `frames/` (every 10th frame), `colmap_known_poses/` (COLMAP text model, PINHOLE camera),
`pose_priors.jsonl` (same fields as live-capture keyframes: position in metres, gravity in camera
axes), `video.mp4` that the normal project upload accepts, and `report.json` with frame rate and
gaps, tracked fraction and longest loss, path length, intrinsics, IMU rates and the estimated
IMU-to-camera time offset and rotation.

## What was tested here, and what was not

Measured in this container (no phone):

- `tests/browser/record_e2e.mjs` drives the real page in headless Chromium with a scripted WebXR
  session (analytic walk-and-pan trajectory, a 0.3 s tracking loss, a 640x480 camera texture that
  paints its frame number) and a gyro deliberately 42 ms late. The ingest recovered the offset as
  42.0 ms and 42.1 ms in two runs, the rotation within 0.01 degrees, the intrinsics exactly, and
  every decoded frame carries the image of the XR frame its pose came from.
- Camera-only mode with Chromium's fake camera: 1920x1080 frames and IMU recorded and ingested;
  only about 7 fps were encoded because this container encodes VP8 in software.
- H.264 Annex B ingest and MP4 remux with an FFmpeg-made stream.

Not tested, needs your phone: whether `camera-access` is granted and at what resolution, whether
the frames come out upright, whether H.264 is offered and keeps up at full frame rate, whether
motion sensors fire during an AR session, and the real IMU-to-camera offset.

## First real phone recording (2026-10-17, Android Chrome 155, AR mode with depth)

Measured from the bundle (`/mnt/project-files/sensor-capture/ar-2026-10-17T11-28-46/`):

- `camera-access` and `depth-sensing` were both granted. The AR camera image is **867 x 1920**, a
  portrait crop matching the screen shape (412 x 912 css px), not the full sensor; fx = fy = 1331 px,
  so about 36 x 72 degrees field of view. Frames come out upright.
- H.264 Baseline (`avc1.42E028`) hardware encoding kept up: 576 frames in 21.6 s, **30.5 fps**, 5 dropped.
- ARCore tracked **94%** of frames; one loss of 1.4 s. Walk of 5.1 m over a 1.3 x 2.0 m area.
- IMU kept firing during the AR session: Generic Sensors about 59 Hz, `devicemotion` 60 Hz.
- Two surprises, both fixed in the recorder and ingest: Generic Sensor `timestamp` is on a different
  clock from `performance.now()` (about 230,000 s apart), and Chrome reports `rotationRate` alpha,
  beta, gamma about device x, y, z (the spec says z, x, y).
- **IMU-to-camera offset: -129 ms** (gyro motion appears 129 ms before the XR frame time of the same
  motion), residual 6.4 deg/s against a 25.7 deg/s signal.
- **Pose accuracy check:** 58 frames (every 10th) triangulated with the ARCore poses as-is gave 8,827
  points at **1.69 px** mean reprojection error. A free bundle adjustment then moved the camera
  centres by a median of **2.8 cm** (max 9.2 cm) after removing the similarity gauge. So the AR poses
  are usable directly as priors; triangulation took 2 s after 52 s of feature extraction and matching
  on this container's CPU.

## Phone test

1. On the Mac: `python scripts/serve_live.py`, then on the Android phone (same Wi-Fi) open
   `https://<mac-ip>:8443/record/` in Chrome and accept the certificate warning once.
2. Screenshot the setup screen (five capability rows).
3. Tick "Also record ARCore depth", tap **Record with AR**, walk slowly around one room for about
   30 seconds, tap **Stop and save**, screenshot the summary, then **Save bundle**.
4. Tap **Record again**, then **Record camera only**, and record the same walk.
5. Upload both zips to the project.

Sources: [Babylon.js forum on recording WebXR camera video](https://forum.babylonjs.com/t/webxr-immersive-session-screen-recording-potential-solution-optimization-help/56567),
[Babylon.js forum on camera-access context loss](https://forum.babylonjs.com/t/using-camera-access-feature-in-webxr-causes-context-loss-and-flickering-screen-issues/51610),
[W3C TAG review of WebXR Raw Camera Access](https://tag-github-bot.w3.org/review/652),
[Chromium sensor frequency cap](https://crrev.com/9d5b8269d088402d1d9c261d1c12585d81a4ac78),
[ARCore camera configs](https://developers.google.com/ar/develop/unity-arf/camera-configs?hl=en).
