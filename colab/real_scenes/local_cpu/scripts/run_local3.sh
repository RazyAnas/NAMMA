#!/bin/bash
export OS_BIN=$SCRATCH/os/opensplat-cpu; R=$SCRATCH/wt/colab/real_scenes; W=$SCRATCH/block/phone_sfm
L="--refine-every 100 --densify-from 500 --densify-until 2500"
RUN=b3k_aniso5 TOTAL=3000 EXTRA="$L" OPENSPLAT_ANISO_MAX=5 OPENSPLAT_ANISO_W=1.0 bash $R/scripts/run_scene.sh $W > $W/b3k_aniso5.log 2>&1; echo "b3k_aniso5 $(cut -c1-220 $W/b3k_aniso5/score.txt)"
echo LOCAL_BLOCK3_DONE
