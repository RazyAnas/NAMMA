#!/bin/bash
# Train the lobby recorded 2026-10-20 on Kaggle: ARCore poses stitched across 16 origin resets (stitch_ar_segments.py),
# 552 frames (every 15th), init = the stitched quick model. gsplat default and MCMC side by side (two T4s), scored on
# every 8th frame held out; writes /kaggle/working/lobby_stitched_results.zip (scores, test renders, splats).
# usage: bash kaggle_lobby_stitched.sh [steps=30000]
R=$(cd $(dirname $0)/..; pwd); S=${1:-30000}; W=/content/lobby_stitched; O=/kaggle/working
mkdir -p $W/data/color_raw $W/clean; cp $R/frames/lobby_stitched_frames/*.jpg $W/data/color_raw/; cp -r $R/scenes/lobby_stitched/clean/. $W/clean/
pip -q install plyfile scikit-image > /tmp/pip_ls.log 2>&1
if ! python -c "import gsplat, gsplat.cuda._wrapper" 2>/dev/null; then
  TV=$(python -c "import torch;v=torch.__version__.split('+')[0].split('.');print('pt'+v[0]+v[1])"); CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
  pip -q install ninja jaxtyping rich >> /tmp/pip_ls.log 2>&1
  pip -q install gsplat --index-url https://docs.gsplat.studio/whl/${TV}${CU} >> /tmp/pip_ls.log 2>&1 || pip -q install gsplat==1.5.3 >> /tmp/pip_ls.log 2>&1
fi
python -c "import gsplat; print('gsplat', gsplat.__version__)" || { echo GSPLAT_INSTALL_FAILED; tail /tmp/pip_ls.log; exit 1; }
# A pip-built gsplat compiles its CUDA kernels on first use; do it once here, or two parallel runs race on the build folder.
python -c "from gsplat.cuda._backend import _C; print('gsplat kernels ready', _C is not None)" 2>&1 | tail -1
run() { o=$W/gs_$1; CUDA_VISIBLE_DEVICES=$2 python3 $R/scripts/gsplat_train.py $W $W/clean/init_quick_xyzrgb.ply $o $1 $S > $W/gs_$1.log 2>&1 && \
  python3 $R/scripts/score_whole.py $o $W/data/color_raw $o/scores.json > $o/score.txt; echo "$1 $(cut -c1-220 $o/score.txt 2>/dev/null || tail -3 $W/gs_$1.log)"; }
G=$(nvidia-smi -L | wc -l); run default 0 & if [ $G -gt 1 ]; then run mcmc 1 & else wait; run mcmc 0; fi; wait
cd $W && zip -qr $O/lobby_stitched_results.zip gs_*.log gs_*/score.txt gs_*/scores.json gs_*/train_stats.json gs_*/test_*.png gs_*/splat.ply
ls -la $O/lobby_stitched_results.zip; echo LOBBY_STITCHED_DONE
