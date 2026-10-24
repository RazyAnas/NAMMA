#!/bin/bash
# Full video -> walkable 3D pipeline on a Kaggle 2x T4 machine, in one run. Each stage logs to $W/<stage>.log and prints a
# one-line result; a failed optional stage does not stop the others.
#   1 frames   sharpest frame in every window of 30/FPS video frames (blur rejection, as AirVis does), full resolution
#   2 cameras  hloc SuperPoint + LightGlue through ns-process-data; exhaustive matching up to 300 frames, vocab tree above
#   3 priors   DSINE monocular normals (DN-Splatter script) + Depth Anything V2 depth fitted to the SfM points
#   4 train    GPU0 dn-splatter (depth + normal losses: keeps blank walls flat and solid)
#              GPU1 splatfacto (Nerfstudio default), same data, as the reference
#   5 eval     held-out every 8th frame: PSNR/SSIM/LPIPS (ns-eval) + off-path strips (offpath_render.py) for both
#   6 export   splat .ply (ns-export) for each model and a TSDF mesh from the dn-splatter depths (gs-mesh o3dtsdf) as .glb
# Results: /kaggle/working/full_pipeline.zip and /kaggle/working/summary.json
# usage: VIDEO=<mp4> [FPS=6] [STEPS=30000] bash full_pipeline.sh
#   FRAMES=<dir of already-picked frames> skips stage 1; WORK=/OUT= move the work and result folders.
#   FINAL=1 is the render-server mode (phone "Send for render"): one GPU, Nerfstudio on every frame (nothing held out),
#   no evaluation, just the splat at $OUT/splat.ply.
set -o pipefail
R=$(cd $(dirname $0)/..; pwd); S=$R/scripts; W=${WORK:-/content/full}; O=${OUT:-/kaggle/working}; mkdir -p $W $O
VIDEO=${VIDEO:-$R/videos/lobby_clip30.mp4}; FPS=${FPS:-6}; STEPS=${STEPS:-30000}
LOG() { echo "== $(date +%H:%M:%S) $*"; }
LOG "start video=$VIDEO fps=$FPS steps=$STEPS"; nvidia-smi -L

LOG env
source $S/ns_env.sh || { echo ENV_FAILED; exit 1; }
# DN-Splatter (pins nerfstudio 1.1.3 / gsplat 1.0, but only uses APIs that 1.1.5 / 1.4 still have): no-deps install, then
# the extra packages it needs. ZoeDepth's torch.hub code needs timm 0.6.x.
if [ "$FINAL" != 1 ] && ! python -c "import dn_splatter" 2>/dev/null; then
  rm -rf /content/dn-splatter && git clone -q --depth 1 https://github.com/maturk/dn-splatter /content/dn-splatter
  python $S/patch_dn_splatter.py /content/dn-splatter   # gsplat 1.4 has no legacy rasterizer
  uv pip install -q --python /content/nsenv/bin/python --no-deps -e /content/dn-splatter > $W/dn_pip.log 2>&1
  # one at a time, so one package without a wheel (PyMCubes 0.1.2 does not build here) cannot block the rest
  for p in natsort omnidata-tools pytorch-lightning geffnet "timm==0.6.7" open3d trimesh PyMCubes pymeshlab vdbfusion "transformers==4.46.3" "setuptools<70"; do
    uv pip install -q --python /content/nsenv/bin/python "$p" >> $W/dn_pip.log 2>&1 || echo "pip failed: $p"
  done
fi
# dn_splatter registers Nerfstudio plugins; if it cannot import, every ns-* command fails, so take it out again
[ "$FINAL" = 1 ] || python -c "import dn_splatter, open3d, trimesh; print('dn-splatter ok')" || { echo DN_INSTALL_FAILED; tail -20 $W/dn_pip.log; uv pip uninstall -q --python /content/nsenv/bin/python dn-splatter; }

LOG frames
if [ -n "$FRAMES" ]; then mkdir -p $W/frames && cp $FRAMES/*.jpg $W/frames/ && echo "frames given: $(ls $W/frames | wc -l)"; else
python - "$VIDEO" $W/frames $FPS > $W/frames.log 2>&1 <<'EOF'
# keep the sharpest frame (variance of Laplacian at 1/3 size) in each window of round(src_fps / fps) frames
import cv2, sys, os, numpy as np
src, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3]); os.makedirs(out, exist_ok=True)
c = cv2.VideoCapture(src); vf = c.get(cv2.CAP_PROP_FPS) or 30; win = max(1, round(vf / fps))
best, n, i, kept, size = None, 0, 0, [], None
while True:
    ok, f = c.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 3, f.shape[0] // 3)), cv2.COLOR_BGR2GRAY)
    s = cv2.Laplacian(g, cv2.CV_64F).var()
    if best is None or s > best[0]: best = (s, f)
    size = f.shape
    i += 1
    if i % win == 0:
        cv2.imwrite(f'{out}/{n:05d}.jpg', best[1], [cv2.IMWRITE_JPEG_QUALITY, 95]); kept.append(best[0]); n += 1; best = None
print('video fps', vf, 'window', win, 'frames', n, 'size', size[1], 'x', size[0], 'sharpness median', round(float(np.median(kept)), 1))
EOF
tail -1 $W/frames.log; fi; NF=$(ls $W/frames | wc -l); [ $NF -gt 10 ] || { echo FRAMES_FAILED; cat $W/frames.log; exit 1; }

LOG cameras
which colmap > /dev/null || { apt-get -qq update > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get -qq install -y colmap > /tmp/colmap_apt.log 2>&1; }
M=exhaustive; [ $NF -gt 300 ] && M=vocab_tree
ns-process-data images --data $W/frames --output-dir $W/data --sfm-tool hloc --feature-type superpoint \
  --matcher-type superpoint+lightglue --matching-method $M --num-downscales 2 > $W/cameras.log 2>&1 || { echo CAMERAS_FAILED; tail -30 $W/cameras.log; exit 1; }
NP=$(python -c "import json; print(len(json.load(open('$W/data/transforms.json'))['frames']))"); echo "cameras solved: $NP of $NF ($M matching)"

if [ "$FINAL" = 1 ]; then
  LOG train
  ns-train splatfacto --max-num-iterations $STEPS --vis tensorboard --output-dir $W/out --experiment-name full --timestamp final \
    --viewer.quit-on-train-completion True $TRAIN_ARGS nerfstudio-data --data $W/data --eval-mode all $DATA_ARGS > $W/train_final.log 2>&1 \
    || { echo TRAIN_FAILED; tail -25 $W/train_final.log; exit 1; }
  LOG export
  ns-export gaussian-splat --load-config $(ls $W/out/full/*/final/config.yml | head -1) --output-dir $W/export > $W/export.log 2>&1 \
    && cp $W/export/splat.ply $W/data/transforms.json $O/ || { echo EXPORT_FAILED; tail -10 $W/export.log; exit 1; }
  LOG FULL_PIPELINE_DONE; exit 0
fi

LOG priors
(cd /content/dn-splatter && python dn_splatter/scripts/normals_from_pretrain.py --data-dir $W/data --normal-format dsine) > $W/normals.log 2>&1 \
  && echo "normals: $(ls $W/data/normals_from_pretrain | wc -l)" || { echo NORMALS_FAILED; tail -15 $W/normals.log; }
# Depth Anything V2 fitted to the SfM points (DN-Splatter's ZoeDepth script fails on torch 2.1)
python $S/mono_depth_prior.py $W/data > $W/depth.log 2>&1 && tail -1 $W/depth.log || { echo DEPTH_FAILED; tail -15 $W/depth.log; }
HAVE_N=$(ls $W/data/normals_from_pretrain 2>/dev/null | wc -l); HAVE_D=$(ls $W/data/mono_depth/*_aligned.npy 2>/dev/null | wc -l)

LOG train
(cd /tmp && python -c "from gsplat.cuda._backend import _C; assert _C is not None") > $W/kernels.log 2>&1 || { echo KERNELS_FAILED; tail -20 $W/kernels.log; }
EV="--eval-mode interval --eval-interval 8"
finish() {   # name gpu: eval, off-path strips, splat export
  n=$1; g=$2; C=$(ls $W/out/full/*/$n/config.yml 2>/dev/null | head -1); [ -n "$C" ] || { echo "NO_MODEL $n"; return 1; }
  CUDA_VISIBLE_DEVICES=$g ns-eval --load-config $C --output-path $W/eval_$n.json --render-output-path $W/renders_$n > $W/eval_$n.log 2>&1 || tail -5 $W/eval_$n.log
  CUDA_VISIBLE_DEVICES=$g python $S/offpath_render.py $C $W/offpath_$n > $W/offpath_$n.log 2>&1 || tail -5 $W/offpath_$n.log
  CUDA_VISIBLE_DEVICES=$g ns-export gaussian-splat --load-config $C --output-dir $W/export_$n > $W/export_$n.log 2>&1 || tail -5 $W/export_$n.log
  python -c "import json; r = json.load(open('$W/eval_$n.json'))['results']; print('$n', {k: round(v, 3) for k, v in r.items() if k in ('psnr', 'ssim', 'lpips')})"
}
dn() {
  if [ $HAVE_N -gt 0 ] && [ $HAVE_D -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=0 ns-train dn-splatter --max-num-iterations $STEPS --vis tensorboard --output-dir $W/out --experiment-name full \
      --timestamp dn --viewer.quit-on-train-completion True --pipeline.model.use-depth-loss True --pipeline.model.depth-lambda 0.2 \
      --pipeline.model.use-normal-loss True --pipeline.model.use-normal-tv-loss True --pipeline.model.normal-supervision mono \
      normal-nerfstudio --data $W/data --normal-format dsine --depth-unit-scale-factor 1.0 $EV > $W/train_dn.log 2>&1 || { echo TRAIN_FAILED dn; tail -25 $W/train_dn.log; return 1; }
    finish dn 0
    C=$(ls $W/out/full/*/dn/config.yml | head -1)
    CUDA_VISIBLE_DEVICES=0 gs-mesh o3dtsdf --load-config $C --output-dir $W/mesh > $W/mesh.log 2>&1 || { echo MESH_FAILED; tail -10 $W/mesh.log; }
  else echo "SKIP dn-splatter: normals $HAVE_N, depths $HAVE_D"; fi
}
base() {
  CUDA_VISIBLE_DEVICES=1 ns-train splatfacto --max-num-iterations $STEPS --vis tensorboard --output-dir $W/out --experiment-name full \
    --timestamp base --viewer.quit-on-train-completion True nerfstudio-data --data $W/data $EV > $W/train_base.log 2>&1 \
    || { echo TRAIN_FAILED base; tail -25 $W/train_base.log; return 1; }
  finish base 1
}
dn & base & wait

LOG export
python - $W > $W/glb.log 2>&1 <<'EOF'
# TSDF mesh -> decimated .glb for the web viewer
import sys, glob, open3d as o3d, trimesh, numpy as np
W = sys.argv[1]; fs = sorted(glob.glob(f'{W}/mesh/*.ply'))
if not fs: print('no mesh'); sys.exit(0)
m = o3d.io.read_triangle_mesh(fs[0]); print('mesh', fs[0], len(m.triangles), 'triangles')
if len(m.triangles) > 400000: m = m.simplify_quadric_decimation(400000)
m.remove_unreferenced_vertices()
t = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8) if m.has_vertex_colors() else None)
t.export(f'{W}/room_mesh.glb'); print('glb triangles', len(t.faces))
EOF
tail -2 $W/glb.log
python - $W $NF $NP > $O/summary.json <<'EOF'
import json, sys, os
W, nf, np_ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); out = {'frames': nf, 'cameras_solved': np_}
for n in ('dn', 'base'):
    p = f'{W}/eval_{n}.json'
    if os.path.exists(p): out[n] = {k: round(v, 4) for k, v in json.load(open(p))['results'].items() if k in ('psnr', 'ssim', 'lpips')}
print(json.dumps(out, indent=1))
EOF
cat $O/summary.json
cd $W && zip -qr $O/full_pipeline.zip *.log eval_*.json offpath_* room_mesh.glb export_*/*.ply data/transforms.json 2>/dev/null
cp offpath_*/sheet.jpg $O/ 2>/dev/null; for n in dn base; do [ -f offpath_$n/sheet.jpg ] && cp offpath_$n/sheet.jpg $O/sheet_$n.jpg; done
ls -la $O; LOG FULL_PIPELINE_DONE
