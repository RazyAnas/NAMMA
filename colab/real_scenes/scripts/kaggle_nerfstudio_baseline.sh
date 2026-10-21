#!/bin/bash
# Reference run: Nerfstudio's own video-to-splat pipeline, unchanged, on the 2026-10-20 lobby recording
# (325 full-resolution frames, every 30th). Then the same with two of Nerfstudio's own switches. Fast settings so the
# answer comes in well under an hour on a Kaggle T4:
#   base    ns-process-data images (hloc: SuperPoint + LightGlue) -> ns-train splatfacto, defaults, 7k steps
#   tuned   same data, + camera-optimizer SO3xR3 (pose refinement) + bilateral grid (per-image exposure / colour)
# Every 8th frame is held out (eval-mode interval). Off-path views: each of 4 held-out cameras moved sideways and
# backwards, rendered next to the real photo (offpath_render.py). Results: /kaggle/working/nerfstudio_lobby.zip
# usage: bash kaggle_nerfstudio_baseline.sh [steps=7000]
set -o pipefail
R=$(cd $(dirname $0)/..; pwd); N=${1:-7000}; W=/content/ns_lobby; O=/kaggle/working; mkdir -p $W
LOG() { echo "== $(date +%H:%M:%S) $*"; }
LOG env
pip -q install uv > /tmp/uv.log 2>&1
# Kaggle's image is Python 3.13 and sets uv to its system Python; nerfstudio 1.1.5 needs <=3.10 wheels, so name the venv's python everywhere
unset UV_SYSTEM_PYTHON UV_PYTHON PIP_TARGET
uv venv -q -p 3.10 /content/nsenv && source /content/nsenv/bin/activate || { echo ENV_FAILED; exit 1; }
P="--python /content/nsenv/bin/python"
uv pip install -q $P torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 > /tmp/ns_pip.log 2>&1
uv pip install -q $P "numpy<2" setuptools wheel wrapt ninja >> /tmp/ns_pip.log 2>&1
uv pip install -q $P "numpy<2" nerfstudio==1.1.5 --no-build-isolation-package fpsample >> /tmp/ns_pip.log 2>&1
uv pip install -q $P gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt21cu118 >> /tmp/ns_pip.log 2>&1 || uv pip install -q $P gsplat==1.4.0 >> /tmp/ns_pip.log 2>&1
uv pip install -q $P "git+https://github.com/cvg/Hierarchical-Localization.git@v1.5" "lightglue @ git+https://github.com/cvg/LightGlue" >> /tmp/ns_pip.log 2>&1
which python ns-train
python -c "import nerfstudio, gsplat, hloc, torch; print('nerfstudio ok, gsplat', gsplat.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo NS_INSTALL_FAILED; tail -30 /tmp/ns_pip.log; exit 1; }

LOG process
ns-process-data images --data $R/frames/lobby1020_full --output-dir $W/data --sfm-tool hloc --feature-type superpoint \
  --matcher-type superpoint+lightglue --matching-method vocab_tree --num-downscales 2 > $W/process.log 2>&1 || { echo PROCESS_FAILED; tail -30 $W/process.log; exit 1; }
grep -iE "registered|Colmap matched|images|poses" $W/process.log | tail -5
python - $W/data <<'EOF'
import json, sys; t = json.load(open(sys.argv[1] + '/transforms.json')); print('frames with poses:', len(t['frames']), 'of 325')
EOF

train() {   # name gpu [extra args]
  n=$1; g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g ns-train splatfacto --max-num-iterations $N --vis tensorboard --output-dir $W/out --experiment-name lobby \
    --timestamp $n --viewer.quit-on-train-completion True "$@" nerfstudio-data --data $W/data --eval-mode interval --eval-interval 8 \
    > $W/train_$n.log 2>&1 || { echo "TRAIN_FAILED $n"; tail -20 $W/train_$n.log; return 1; }
  C=$(ls $W/out/lobby/splatfacto/$n/config.yml)
  CUDA_VISIBLE_DEVICES=$g ns-eval --load-config $C --output-path $W/eval_$n.json --render-output-path $W/renders_$n > $W/eval_$n.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python $R/scripts/offpath_render.py $C $W/offpath_$n > $W/offpath_$n.log 2>&1 || tail -5 $W/offpath_$n.log
  python - $W/eval_$n.json $n <<'EOF'
import json, sys; r = json.load(open(sys.argv[1]))['results']; print(sys.argv[2], {k: round(v, 3) for k, v in r.items() if k in ('psnr', 'ssim', 'lpips')})
EOF
}
LOG train
G=$(nvidia-smi -L | wc -l)
train base 0 &
if [ $G -gt 1 ]; then train tuned 1 --pipeline.model.camera-optimizer.mode SO3xR3 --pipeline.model.use-bilateral-grid True & wait
else wait; train tuned 0 --pipeline.model.camera-optimizer.mode SO3xR3 --pipeline.model.use-bilateral-grid True; fi

cd $W && zip -qr $O/nerfstudio_lobby.zip process.log train_*.log eval_*.json renders_* offpath_* data/transforms.json 2>/dev/null
ls -la $O/nerfstudio_lobby.zip; LOG NERFSTUDIO_DONE
