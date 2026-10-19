#!/bin/bash
# Round 2 of the off-path bake-off, after colab_bakeoff.sh on the same runtime (it leaves /content/block/phone_sfm/gs_*):
#  1. Difix3D / Difix3D+ (gsplat_difix.py) on the best round-1 gsplat splat, scored on the same held-out stretch
#  2. VGGT second opinion on the lobby poses (vggt_pose_check.py), to see if its ~20 dB middle is a pose problem
# One GPU job at a time. Difix's pinned diffusers / transformers go in their own folder so round 1's packages stay as they are.
# usage: nohup bash colab_round2.sh [scene=phone_sfm] [base variant=best] > /content/round2.log 2>&1 &
#        progress: grep -a "DONE\|psnr\|worst\|round " /content/round2.log
R=$(cd $(dirname $0)/..; pwd); s=${1:-phone_sfm}; W=/content/block/$s; cd /content
BASE=${2:-$(python3 -c "
import json, glob
r = [(json.load(open(p))['psnr_mean'], p.split('/')[-2][3:]) for p in glob.glob('$W/gs_*/scores.json')]
print(max(r)[1] if r else '')")}
[ -f $W/gs_$BASE/splat.ply ] || { echo "no round-1 gsplat splat in $W (run colab_bakeoff.sh first)"; exit 1; }
echo "base: gs_$BASE $(cut -c1-160 $W/gs_$BASE/score.txt)"

[ -d /content/Difix3D ] || git clone -q https://github.com/nv-tlabs/Difix3D /content/Difix3D
python3 -c "import sys; sys.path.insert(0, '/content/difix_pkgs'); import diffusers; assert diffusers.__version__ == '0.25.1'" 2>/dev/null || \
  pip -q install --target /content/difix_pkgs --no-deps diffusers==0.25.1 transformers==4.38.0 huggingface-hub==0.25.1 \
    peft==0.9.0 tokenizers==0.15.2 > /tmp/pip_difix.log 2>&1 || { echo DIFIX_INSTALL_FAILED; tail /tmp/pip_difix.log; }
O=$W/difix_$BASE
DIFIX_PKGS=/content/difix_pkgs DIFIX_SRC=/content/Difix3D/src python3 $R/scripts/gsplat_difix.py $W $W/gs_$BASE/splat.ply $O 3 1500 > $O.log 2>&1 \
  || { echo DIFIX_FAILED; tail -15 $O.log; }
for m in base difix3d plus; do
  [ -d $O/$m ] && python3 $R/scripts/score_whole.py $O/$m $W/data/color_raw $O/$m/scores.json > $O/$m/score.txt && echo "difix $m $(cut -c1-200 $O/$m/score.txt)"
done
echo DIFIX_DONE

python3 -c "import vggt" 2>/dev/null || pip -q install git+https://github.com/facebookresearch/vggt.git > /tmp/pip_vggt.log 2>&1
python3 $R/scripts/vggt_pose_check.py $R/frames/lobby_frames $R/scenes/lobby_ref/clean/poses_w2c.json /content/lobby_pose_check.json 24 16 \
  > /content/lobby_pose_check.log 2>&1 || { echo POSE_CHECK_FAILED; tail -15 /content/lobby_pose_check.log; }
grep -A6 "^worst" /content/lobby_pose_check.log; echo POSE_CHECK_DONE

cd /content && zip -qr /content/round2_results.zip $O/*/test_*.png $O/*/scores.json $O/*/score.txt $O/difix_stats.json $O/difix3d/splat.ply \
  /content/lobby_pose_check.json $W/gs_*/scores.json 2>/dev/null; ls -la /content/round2_results.zip; echo ROUND2_DONE
