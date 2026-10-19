#!/bin/bash
# Off-path test on a Colab GPU: hold out a stretch of the walk (make_block_split.py), then train the same pipeline
# 3k vs 30k steps, and 30k with the anisotropy cap (OPENSPLAT_ANISO_MAX), and score each on that stretch.
# usage: bash colab_block_test.sh [scene=phone_sfm]   (OpenSplat is built with opensplat_namma + opensplat_aniso)
R=$(cd $(dirname $0)/..; pwd); s=${1:-phone_sfm}; cd /content
pip -q install pycolmap==4.2.1 plyfile scikit-image > /tmp/pip.log 2>&1
if [ ! -x /content/OpenSplat/build/opensplat ]; then
  apt-get -qq update > /dev/null; apt-get -qq install -y libopencv-dev > /dev/null
  rm -rf OpenSplat && git clone -q https://github.com/pierotofy/OpenSplat && cd OpenSplat && git checkout -q 688944f01291a1939f688b49feef6936620b47b8
  git apply $R/opensplat_namma.patch && git apply $R/opensplat_aniso.patch; CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_RUNTIME=CUDA -DCMAKE_CUDA_ARCHITECTURES=$CC \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') > /tmp/cmake.log 2>&1 && make -j$(nproc) > /tmp/make.log 2>&1
  cd /content; [ -x /content/OpenSplat/build/opensplat ] || { echo BUILD_FAILED; exit 1; }
fi
echo BUILD_OK; export OS_BIN=/content/OpenSplat/build/opensplat
f=phone_frames; [ $s = lobby_ref ] && f=lobby_frames; W=/content/block/$s
mkdir -p $W/data/color_raw $W/clean; cp $R/frames/$f/*.jpg $W/data/color_raw/; cp -r $R/scenes/$s/clean/. $W/clean/
python3 $R/scripts/make_block_split.py $W | tee $W/split.txt
L="--refine-every 100 --densify-from 500"
[ -f $W/b3k/score.txt ] || RUN=b3k TOTAL=3000 EXTRA="$L --densify-until 2500" bash $R/scripts/run_scene.sh $W > $W/b3k.log 2>&1   # also builds the shared init project
[ -n "$ONLY_3K" ] && { echo b3k $(cut -c1-200 $W/b3k/score.txt); echo BLOCK_3K_DONE; exit 0; }
RUN=b30k TOTAL=30000 EXTRA="$L --densify-until 15000" bash $R/scripts/run_scene.sh $W > $W/b30k.log 2>&1   # one at a time: two OpenSplat runs OOM a T4
RUN=b30k_aniso TOTAL=30000 EXTRA="$L --densify-until 15000" OPENSPLAT_ANISO_MAX=10 bash $R/scripts/run_scene.sh $W > $W/b30k_aniso.log 2>&1
for r in b3k b30k b30k_aniso; do echo $r $(cat $W/$r/score.txt 2>/dev/null | cut -c1-200); done; echo BLOCK_DONE
