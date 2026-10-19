#!/bin/bash
# Everything on one T4, one GPU job at a time (VGGT next to a build filled the 12.7 GB RAM and froze the runtime):
# 3k OpenSplat on the held-out stretch (also builds the init), VGGT floors, gsplat recipes, then OpenSplat 30k and 30k + cap.
# usage: nohup bash colab_bakeoff.sh > /content/bakeoff.log 2>&1 &     progress: grep -a "DONE\|dB\|psnr" /content/bakeoff.log
D=$(cd $(dirname $0); pwd); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ONLY_3K=1 bash $D/colab_block_test.sh phone_sfm
bash $D/colab_vggt_floor.sh "phone_sfm lobby_ref"
bash $D/colab_gsplat_test.sh phone_sfm 30000 "mcmc depth dn 2dgs default"
bash $D/colab_block_test.sh phone_sfm
echo BAKEOFF_DONE
