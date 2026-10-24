#!/bin/bash
# The whole current NAMMA pipeline on one scene, GPU build of OpenSplat (same steps and flags as the CPU run that
# reached 32.26 dB on ICL-NUIM; only --cpu is dropped):
#  1. MODP placement (feature + grid pixels, MOHV, k-feat 5 / k-grid 20) from the anchored depth, as xyzrgb points
#  2. train -n 3000 with paper densification, stop after splat_1000.ply
#  3. render every training view, SSIM < 0.5 error masks (stride 8), add error-compensation Gaussians via the MOHV
#  4. resume to 3000, render the held-out views, score PSNR / SSIM
# usage: run_scene.sh <work dir>   (needs OS_BIN; the work dir comes from prep_sfm.py + prep_depth.py)
set -e
D=$(cd $(dirname $0); pwd); W=$1; out=$W/${RUN:-run}; mkdir -p $out
export OPENSPLAT_PAPER_ADC=1; EXTRA=${EXTRA:-"--refine-every 100 --densify-from 500 --densify-until 2500"}
FLAGS="-n ${TOTAL:-3000} -d 1 --num-downscales 2 --resolution-schedule 500 $EXTRA"
date +%s > $out/start.txt
[ -f $W/proj/sparse/0/points3D.bin ] || { python3 $D/modp_init.py $W $W/init_k5.ply --k-feat 5 --k-grid 20 --stats $out/init_stats.json | cut -c1-300
python3 $D/gauss_to_xyzrgb.py $W/init_k5.ply $W/init_k5_xyzrgb.ply
python3 $D/make_whole_proj.py $W $W/init_k5_xyzrgb.ply $W/proj; }
OPENSPLAT_TEST_PREFIX=test_ $OS_BIN $W/proj $FLAGS --save-every 1000 -o $out/splat.ply > $out/log_a.txt 2>&1 &
P=$!
until [ -f $out/splat_1000.ply ] && grep -q "^Step 1010:" $out/log_a.txt; do
  sleep 2; kill -0 $P 2>/dev/null || { tail -20 $out/log_a.txt; echo "training stopped early"; exit 1; }
done
kill $P; wait $P 2>/dev/null || true
date +%s > $out/t_stage1.txt; rm -f $out/test_*.png
bash $D/render_train.sh $W $out/splat_1000.ply $out/train_renders 30 > $out/log_render.txt 2>&1
python3 $D/ec_masks.py $out/train_renders $W/data/color_raw $out/ec_masks 0.5 --stride 8 | tee $out/ec_masks.json
python3 $D/modp_init.py $W $out/splat_1000_ec.ply --ec $out/ec_masks --ec-only --base $out/splat_1000.ply \
  --base-step 1000 --k-ec 1.5 --stats $out/ec_stats.json | cut -c1-300
date +%s > $out/t_stage2.txt
VAL=$(python3 -c "import json; print('test_%05d.jpg' % json.load(open('$W/clean/stats.json'))['test_frames'][0])")
OPENSPLAT_TEST_PREFIX=test_ $OS_BIN $W/proj $FLAGS --resume $out/splat_1000_ec.ply --val-image $VAL --val-render $out \
  -o $out/splat.ply > $out/log.txt 2>&1
date +%s > $out/end.txt
python3 $D/score_whole.py $out $W/data/color_raw $out/scores.json | tee $out/score.txt
