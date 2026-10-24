#!/bin/bash
# The bake-off on a Kaggle GPU notebook (run as "Save & Run All": it keeps running with the browser closed, up to 12 h).
# Kaggle has 30 GB RAM and usually two T4s, so after the 3k run (which builds OpenSplat and the shared init) the gsplat
# recipes train on GPU 1 while OpenSplat 30k / 30k + cap train on GPU 0. 30k steps for every recipe, so all are compared
# on the same schedule. Results: /kaggle/working/bakeoff_results.zip (scores, held-out renders, splats, logs).
D=$(cd $(dirname $0); pwd); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; S=${STEPS:-30000}
mkdir -p /content; W=/content/block/phone_sfm; O=/kaggle/working
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2
ONLY_3K=1 bash $D/colab_block_test.sh phone_sfm
if [ $(nvidia-smi -L | wc -l) -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=1 bash $D/colab_gsplat_test.sh phone_sfm $S "mcmc depth dn 2dgs default" > /content/gsplat.out 2>&1 &
  CUDA_VISIBLE_DEVICES=0 bash $D/colab_block_test.sh phone_sfm; wait; cat /content/gsplat.out
else
  bash $D/colab_gsplat_test.sh phone_sfm $S "mcmc depth dn 2dgs default"; bash $D/colab_block_test.sh phone_sfm
fi
bash $D/colab_vggt_floor.sh lobby_ref
cd $W && zip -qr $O/bakeoff_results.zip split.txt *.log */score.txt */scores.json */train_stats.json */test_*.png */splat.ply 2>/dev/null
[ -f /content/vggt_floors.zip ] && cp /content/vggt_floors.zip $O/
ls -la $O; for r in b3k b30k b30k_aniso gs_mcmc gs_depth gs_dn gs_2dgs gs_default; do echo $r $(cut -c1-200 $W/$r/score.txt 2>/dev/null); done
echo BAKEOFF_DONE
