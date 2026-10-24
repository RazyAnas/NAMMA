#!/bin/bash
# The current pipeline plus camera refinement (our OpenSplat patch, MODP Sec. 3.4 idea) in the post-error-compensation
# stage, then held-out views re-rendered with interpolated corrections (pose_fix.py). Reuses stage 1 + error
# compensation from an existing run dir. usage: run_posecorr.sh <work> <base run dir name> <new run dir name> [pose lr]
set -e
D=$(cd $(dirname $0); pwd); W=$1; B=$W/$2; out=$W/$3; LR=${4:-1e-3}; mkdir -p $out
export OPENSPLAT_PAPER_ADC=1; EXTRA=${EXTRA:-"--refine-every 100 --densify-from 500 --densify-until 2500"}
N=${TOTAL:-3000}; FLAGS="-n $N -d 1 --num-downscales 2 --resolution-schedule 500 $EXTRA"
date +%s > $out/start.txt
VAL=$(python3 -c "import json; print('test_%05d.jpg' % json.load(open('$W/clean/stats.json'))['test_frames'][0])")
OPENSPLAT_POSE_LR=$LR OPENSPLAT_POSE_OUT=$out/poses.txt OPENSPLAT_TEST_PREFIX=test_ $OS_BIN $W/proj $FLAGS \
  --resume $B/splat_1000_ec.ply --val-image $VAL --val-render $out/raw -o $out/splat.ply > $out/log.txt 2>&1
python3 $D/score_whole.py $out/raw $W/data/color_raw $out/scores_uncorrected.json | tee $out/score_uncorrected.txt
python3 $D/pose_fix.py $W $out/poses.txt $out/poses_fixed.json | tee $out/pose_fix.json
python3 $D/make_whole_proj.py $W $W/init_k5_xyzrgb.ply $W/proj_fixed $out/poses_fixed.json > /dev/null
OPENSPLAT_TEST_PREFIX=test_ $OS_BIN $W/proj_fixed $FLAGS --resume $out/splat.ply --val-image $VAL --val-render $out -o $out/x.ply > $out/log_render.txt 2>&1
rm -f $out/x.ply; date +%s > $out/end.txt
python3 $D/score_whole.py $out $W/data/color_raw $out/scores.json | tee $out/score.txt
