"""Results gathered from the project's research threads, served at /api/v1/pipeline/findings.

M = measured in this project on a 4-core CPU without a GPU. Q = quoted from a paper, not reproduced.
Every measured number is on Augmented ICL-NUIM living room 1 unless the entry says otherwise.
Full logs: namma-robustness/RESEARCH.md, monocular-points/point_quality.md, benchmark/final_report.md,
navigation/README.md and docs/SENSOR_CAPTURE.md in the project's shared folder.
"""

FINDINGS = {
    "capture": [
        {"claim": "Phone AR bundle: 867x1920 H.264 at 30.5 fps, 94% of frames tracked, sensors about 59 Hz during AR", "label": "M", "data": "real phone, 2026-10-17"},
        {"claim": "IMU-to-camera offset -129 ms (dm_gyro), recovered from the bundle", "label": "M", "data": "real phone"},
        {"claim": "Live /capture/ path: 28 of 30 keyframes mapped live, about 0.6 s per update", "label": "M", "data": "simulated stream"},
    ],
    "tracking": [
        {"claim": "Full video ATE: COLMAP incremental 82.3 cm (331/359 frames) -> global mapper 0.75 cm (354/359)", "label": "M"},
        {"claim": "ORB-SLAM3 mono (MODP's tracker): 3.3-6.5 cm on the part it tracks, loses about half the video", "label": "M"},
        {"claim": "MODP on TUM: 1.87 cm at 11 FPS on an RTX 4090", "label": "Q"},
        {"claim": "Blurred, auto-exposed footage: 68 cm -> sharpest frame 22 -> +CLAHE/DSP-SIFT 7.5 -> +Restormer 3.3 cm", "label": "M"},
        {"claim": "+ pose-prior BA with simulated AR positions: 1.53 cm (fair clean ceiling 0.77 cm)", "label": "M", "data": "simulated priors, not a real phone"},
        {"claim": "Real phone bundle: SfM and AR-triangulated poses agree with ARCore to about 3 cm; AR poses triangulate at 1.66-1.69 px", "label": "M", "data": "real phone"},
        {"claim": "Depth-Anything pseudo-depth tracker with windowed BA: 7.9 cm ATE (57 cm without the depth grid)", "label": "M"},
        {"claim": "Global mapper can break scale on a near-static stretch (131 cm vs 2.45 cm incremental on a 64-frame subset)", "label": "M", "data": "known failure"},
    ],
    "points": [
        {"claim": "Depth Anything V2 Small anchored to SfM: 98.8% pixels with depth vs 0.05% for SfM points; AbsRel 3.0% (boot) / 4.4% (global)", "label": "M"},
        {"claim": "MODP uses tracker sparse points (ORB-SLAM3 / Aria), not LiDAR", "label": "M", "data": "read from MODP code"},
        {"claim": "DA-V2 Small runs 0.51 s/frame on CPU; Base 2.0 s/frame for little gain", "label": "M"},
    ],
    "splats": [
        {"claim": "Held-out PSNR by init (2k steps): random 28.07, SfM sparse 30.99, monocular 34.7, true depth 34.44 dB", "label": "M"},
        {"claim": "At 7k steps: SfM sparse 34.2 vs monocular 35.9 dB", "label": "M"},
        {"claim": "Poses: global mapper 28.86 vs old NAMMA 28.35 dB (+0.51 dB, significant); vs ORB-SLAM3 28.64 (not significant)", "label": "M"},
        {"claim": "Exposure solve: 22.6 -> 26.5 dB render PSNR under lighting changes (clean 28.3)", "label": "M"},
        {"claim": "Real lobby video: OpenSplat CPU 4000 steps, held-out PSNR 27.7 mean / 29.7 median", "label": "M", "data": "real video"},
        {"claim": "MODP 25.45 dB TUM / 35.85 dB Replica (different data, GPU)", "label": "Q"},
    ],
    "walking": [
        {"claim": "Synthetic room floor reachable 91% vs 71% before (2% before with stray points)", "label": "M"},
        {"claim": "ICL noisy depth: 73-74% of free floor walkable vs 17% before", "label": "M"},
        {"claim": "Jumping onto furniture: landings 0% -> 87%", "label": "M"},
        {"claim": "Glass and mirrors have no depth and are not handled", "label": "M", "data": "known gap"},
    ],
}

DEAD_ENDS = [
    "LK keypoint refinement to the sharpest view (no gain over BA)",
    "Stronger Restormer settings (flip ensemble, second pass)",
    "Trajectory smoothing at keyframe stride",
    "Dropping loop matches by temporal vote (worse)",
    "KLT tracks through all frames (global mapper broke, 138 cm)",
    "Lucky-frame dynamic programming (no gain)",
    "Cepstrum + Richardson-Lucy deblur (ringing)",
    "Extra keyframes in the hard stretch (worse)",
    "Joint multi-view depth alignment (ties boot anchoring)",
]
