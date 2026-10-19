#!/bin/bash
# Round 3 of the off-path test, in the same Kaggle session after kaggle_bakeoff.sh (needs /content/block/phone_sfm with the
# block split and the MODP init). Tests the survey's top fix step by step, each scored on the same 35 held-out views:
#   mesh_boot     TSDF mesh from our per-frame anchored depth (CPU reference, 18.76 dB locally)
#   mesh_vda      TSDF mesh from Video Depth Anything (temporally consistent, one scale for the clip)
#   gs_app        gsplat depth variant + per-image affine colour + SH degree 1, MODP init      (isolates APP / SH)
#   gs_app_mesh   same, initialised on the boot mesh surface                                    (adds mesh init)
#   gs_vda_mesh   same, VDA depth loss and VDA mesh init                                        (adds consistent depth)
# Every run (old ones too) is rescored with score_whole.py, which now also reports psnr_cc (after an affine colour fit).
D=$(cd $(dirname $0); pwd); W=/content/block/phone_sfm; O=/kaggle/working; S=${STEPS:-30000}; cd /content
[ -f $W/split.txt ] || { echo "no block split: run kaggle_bakeoff.sh first"; exit 1; }
pip -q install open3d plyfile scikit-image > /tmp/pip_r3.log 2>&1
python3 $D/mesh_offpath.py $W $W/mesh_boot 0.01 > $W/mesh_boot.log 2>&1; echo mesh_boot $(tail -1 $W/mesh_boot.log | cut -c1-260)
python3 $D/mesh_points.py $W/mesh_boot/mesh.ply $W/init_mesh_boot.ply 300000
if [ ! -d /content/vda ]; then
  git clone -q --depth 1 https://github.com/DepthAnything/Video-Depth-Anything /content/vda && mkdir -p /content/vda/checkpoints
  pip -q install einops easydict decord imageio-ffmpeg >> /tmp/pip_r3.log 2>&1
  wget -q -O /content/vda/checkpoints/metric_video_depth_anything_vitl.pth \
    https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth || true
  wget -q -O /content/vda/checkpoints/video_depth_anything_vitl.pth \
    https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth || true
fi
MET=1; [ -s /content/vda/checkpoints/metric_video_depth_anything_vitl.pth ] || MET=0
CUDA_VISIBLE_DEVICES=0 python3 $D/vda_depth.py $W /content/vda vitl $MET > $W/vda.log 2>&1; echo vda $(tail -1 $W/vda.log | cut -c1-300)
if [ ! -f $W/clean/vda_depth.json ]; then   # Large did not fit or download: the Small relative model (Apache-2.0)
  wget -q -O /content/vda/checkpoints/video_depth_anything_vits.pth \
    https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
  CUDA_VISIBLE_DEVICES=0 python3 $D/vda_depth.py $W /content/vda vits 0 > $W/vda_s.log 2>&1; echo vda_small $(tail -1 $W/vda_s.log | cut -c1-300)
fi
VDA=0; [ -f $W/clean/vda_depth.json ] && VDA=1
if [ $VDA = 1 ]; then
  DEPTH_DIR=vda_depth python3 $D/mesh_offpath.py $W $W/mesh_vda 0.01 > $W/mesh_vda.log 2>&1; echo mesh_vda $(tail -1 $W/mesh_vda.log | cut -c1-260)
  python3 $D/mesh_points.py $W/mesh_vda/mesh.ply $W/init_mesh_vda.ply 300000
fi
run() {   # name init gpu [env...]
  n=$1; i=$2; g=$3; shift 3; o=$W/$n
  env "$@" CUDA_VISIBLE_DEVICES=$g python3 $D/gsplat_train.py $W $i $o depth $S > $W/$n.log 2>&1 && \
    python3 $D/score_whole.py $o $W/data/color_raw $o/scores.json > $o/score.txt; echo "$n $(cut -c1-260 $o/score.txt 2>/dev/null || tail -3 $W/$n.log)"; }
run gs_app $W/init_k5_xyzrgb.ply 0 APP=1 SH_MAX=1 &
if [ $VDA = 1 ]; then run gs_vda_mesh $W/init_mesh_vda.ply 1 APP=1 SH_MAX=1 DEPTH_DIR=vda_depth & fi
wait
run gs_app_mesh $W/init_mesh_boot.ply 0 APP=1 SH_MAX=1
for r in b3k b30k b30k_aniso gs_mcmc gs_depth gs_dn gs_2dgs gs_default mesh_boot mesh_vda; do
  [ -d $W/$r ] && ls $W/$r/test_*.png > /dev/null 2>&1 && python3 $D/score_whole.py $W/$r $W/data/color_raw $W/$r/scores_cc.json > /dev/null 2>&1
done
python3 - $W <<'EOF'
import json, os, sys
W = sys.argv[1]
for r in 'b3k b30k b30k_aniso gs_mcmc gs_depth gs_dn gs_2dgs gs_default mesh_boot mesh_vda gs_app gs_app_mesh gs_vda_mesh'.split():
    p = next((f'{W}/{r}/{n}' for n in ('scores_cc.json', 'scores.json') if os.path.exists(f'{W}/{r}/{n}')), None)
    if p: s = json.load(open(p)); print(f"{r:12s} psnr {s['psnr_mean']:.2f} median {s['psnr_median']:.2f} cc {s.get('psnr_cc_mean', float('nan')):.2f} ssim {s['ssim_mean']:.3f}")
EOF
cd $W && zip -qr $O/round3_results.zip *.log clean/vda_depth.json */score.txt */scores*.json */train_stats.json */test_*.png gs_app*/splat.ply gs_vda_mesh/splat.ply mesh_*/mesh.ply 2>/dev/null
ls -la $O; echo ROUND3_DONE
