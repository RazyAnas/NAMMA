#!/bin/bash
# One-shot runner for a fresh Colab GPU runtime (any Google account, no Drive needed): build OpenSplat (CUDA, NAMMA
# patch), stage the frames and prepared inputs from the repo, train the given scenes with the 3DGS paper schedule
# (30,000 steps, densify every 100 from 500 to 15,000, error compensation at 1,000), score and collect the results.
# usage: bash colab_resume.sh "phone_sfm lobby_ref"   (from the cloned NAMMA repo; logs in /content/resume.log)
R=$(cd $(dirname $0)/..; pwd); SCENES=${1:-"phone_sfm lobby_ref"}
cd /content
pip -q install pycolmap==4.2.1 plyfile scikit-image > /tmp/pip.log 2>&1
if [ ! -x /content/OpenSplat/build/opensplat ]; then
  apt-get -qq update > /dev/null; apt-get -qq install -y libopencv-dev > /dev/null
  rm -rf OpenSplat && git clone -q https://github.com/pierotofy/OpenSplat && cd OpenSplat && git checkout -q 688944f01291a1939f688b49feef6936620b47b8
  git apply $R/opensplat_namma.patch; CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_RUNTIME=CUDA -DCMAKE_CUDA_ARCHITECTURES=$CC \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') > /tmp/cmake.log 2>&1 && make -j$(nproc) > /tmp/make.log 2>&1
  cd /content; [ -x /content/OpenSplat/build/opensplat ] || { echo BUILD_FAILED; exit 1; }
fi
echo BUILD_OK
export OS_BIN=/content/OpenSplat/build/opensplat
for s in $SCENES; do
  f=phone_frames; [ $s = lobby_ref ] && f=lobby_frames
  mkdir -p /content/work/$s/data/color_raw /content/work/$s/clean
  cp $R/frames/$f/*.jpg /content/work/$s/data/color_raw/; cp -r $R/scenes/$s/clean/. /content/work/$s/clean/
  (RUN=run30k TOTAL=30000 EXTRA="--refine-every 100 --densify-from 500 --densify-until 15000" \
     bash $R/scripts/run_scene.sh /content/work/$s > /content/work/$s.run30k.log 2>&1;
   mkdir -p /content/results; python3 $R/scripts/save_results.py /content/work/$s run30k /content/results >> /content/work/$s.run30k.log 2>&1) &
done
wait; echo ALL_DONE
