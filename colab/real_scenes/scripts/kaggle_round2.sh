#!/bin/bash
# Round 2 (generative prior + lobby pose check) in the same Kaggle session after kaggle_bakeoff.sh: runs colab_round2.sh
# (Difix3D / Difix3D+ on the best gsplat splat of the held-out test, then the VGGT check of the lobby poses) and copies its
# results to Output. Optional first argument: the gsplat variant to start from (default: best psnr_mean of gs_*).
# usage: bash kaggle_round2.sh [variant]      results: /kaggle/working/round2_results.zip
D=$(cd $(dirname $0); pwd); O=/kaggle/working
[ -f /content/block/phone_sfm/split.txt ] || { echo "no block split: run kaggle_bakeoff.sh first"; exit 1; }
bash $D/colab_round2.sh phone_sfm $1
cp /content/round2_results.zip $O/ 2>/dev/null; cp /content/lobby_pose_check.json $O/ 2>/dev/null
ls -la $O; echo KAGGLE_ROUND2_DONE
