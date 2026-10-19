#!/bin/bash
# Open-source training recipes on the same off-path test as colab_block_test.sh (run that first: it builds the block split
# and the MODP init in /content/block/<scene>). Each gsplat variant (gsplat_train.py) trains on our poses + init, renders
# the held-out stretch, and is scored with score_whole.py. Two runs share the GPU at a time.
# usage: bash colab_gsplat_test.sh [scene=phone_sfm] [steps=30000] [variants="mcmc depth dn 2dgs default"]
R=$(cd $(dirname $0)/..; pwd); s=${1:-phone_sfm}; N=${2:-30000}; V=${3:-"mcmc depth dn 2dgs default"}; W=/content/block/$s
[ -f $W/init_k5_xyzrgb.ply ] || { echo "no block init in $W: run colab_block_test.sh first"; exit 1; }
if ! python -c "import gsplat, gsplat.cuda._wrapper" 2>/dev/null; then
  TV=$(python -c "import torch;v=torch.__version__.split('+')[0].split('.');print('pt'+v[0]+v[1])"); CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
  pip -q install ninja jaxtyping rich > /tmp/pip_gs.log 2>&1
  pip -q install gsplat --index-url https://docs.gsplat.studio/whl/${TV}${CU} >> /tmp/pip_gs.log 2>&1 || pip -q install gsplat==1.5.3 >> /tmp/pip_gs.log 2>&1
fi
python -c "import gsplat; print('gsplat', gsplat.__version__)" || { echo GSPLAT_INSTALL_FAILED; tail /tmp/pip_gs.log; exit 1; }
run() { o=$W/gs_$1; python3 $R/scripts/gsplat_train.py $W $W/init_k5_xyzrgb.ply $o $1 $N > $W/gs_$1.log 2>&1 && \
  python3 $R/scripts/score_whole.py $o $W/data/color_raw $o/scores.json > $o/score.txt; echo "$1 $(cut -c1-200 $o/score.txt 2>/dev/null || tail -3 $W/gs_$1.log)"; }
set -- $V
while [ $# -gt 0 ]; do run $1 & [ $# -gt 1 ] && run $2 & wait; shift; shift 2>/dev/null; done
echo GSPLAT_DONE
