#!/bin/bash
# Local CPU off-path test (same as colab_block_test.sh, 3k steps): hold out a stretch, train default vs anisotropy cap.
export OS_BIN=$SCRATCH/os/opensplat-cpu; R=$SCRATCH/wt/colab/real_scenes; W=$SCRATCH/block/phone_sfm
cd $SCRATCH/block
python3 $R/scripts/make_block_split.py $W | tee $W/split.txt
L="--refine-every 100 --densify-from 500 --densify-until 2500"
RUN=b3k TOTAL=3000 EXTRA="$L" bash $R/scripts/run_scene.sh $W > $W/b3k.log 2>&1; echo "b3k $(cut -c1-220 $W/b3k/score.txt)"
RUN=b3k_aniso TOTAL=3000 EXTRA="$L" OPENSPLAT_ANISO_MAX=10 bash $R/scripts/run_scene.sh $W > $W/b3k_aniso.log 2>&1; echo "b3k_aniso $(cut -c1-220 $W/b3k_aniso/score.txt)"
echo LOCAL_BLOCK_DONE
