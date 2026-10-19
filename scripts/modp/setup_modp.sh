#!/usr/bin/env bash
# Install Meta's MODP (facebookresearch/Online-3DGS-Monocular) on Linux + NVIDIA (Colab, a cloud GPU, a desktop), as its README asks:
# Python 3.9, CUDA 11.8, PyTorch 2.5.1, its setup_env.sh. Everything goes in a separate micromamba environment at $ENV
# (MODP's pins break on Python 3.12, DepthCov has known CUDA 12 build issues) and MODP runs as a subprocess of that python.
#   bash scripts/modp/setup_modp.sh tools      -> then set MODP_DIR=tools/Online-3DGS-Monocular MODP_PYTHON=tools/modp_env/bin/python
set -eo pipefail
trap 'echo; echo "!! MODP SETUP FAILED at line $LINENO: $BASH_COMMAND"; echo "!! Send these last lines to the project."' ERR
BASE=$(realpath -m "${1:-/content}"); mkdir -p "$BASE"; ENV=$BASE/modp_env; MM=$BASE/micromamba; REPO=$BASE/Online-3DGS-Monocular
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); SM=${CAP/./}
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute capability $CAP"
if [ "${CAP%%.*}" -ge 10 ]; then echo "This GPU is newer than CUDA 11.8 supports; MODP can't run on it."; exit 2; fi

# 1. MODP code at the commit we tested against, with its DepthCov submodule (holds the only pretrained weights it uses)
[ -d $REPO ] || git clone --recursive https://github.com/facebookresearch/Online-3DGS-Monocular $REPO
cd $REPO && git checkout -q 7d75169074e4177e29284bc881a69f4866226ae1 && git submodule update --init --recursive
ls -la DepthCov-Modified/models/scannet.ckpt
# DepthCov builds only for sm_86 (RTX 30xx); build it for this GPU instead
sed -i "s/arch=compute_86,code=sm_86/arch=compute_${SM},code=sm_${SM}/" DepthCov-Modified/setup.py
# gsplat is unpinned in MODP's installer; pin the version we checked its API against
sed -i 's/^python -m pip install gsplat$/python -m pip install gsplat==1.5.3/' setup_env.sh

# 2. micromamba + Python 3.9 + CUDA 11.8 compiler + gcc 11 (accepted by nvcc 11.8)
MM_URL=https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64   # release tags end in -0
if ! $MM/bin/micromamba --version >/dev/null 2>&1; then
  mkdir -p $MM/bin && rm -f $MM/bin/micromamba
  curl -fsSL --retry 3 -o $MM/bin/micromamba "$MM_URL" || { echo "!! Could not download micromamba from $MM_URL"; exit 3; }
  chmod +x $MM/bin/micromamba
  $MM/bin/micromamba --version >/dev/null 2>&1 || { echo "!! The micromamba download is not a working program ($(head -c 60 $MM/bin/micromamba))"; exit 3; }
fi
echo "micromamba $($MM/bin/micromamba --version)"
export MAMBA_ROOT_PREFIX=$MM
[ -x $ENV/bin/python ] || $MM/bin/micromamba create -y -p $ENV -c conda-forge python=3.9
$MM/bin/micromamba install -y -p $ENV -c "nvidia/label/cuda-11.8.0" -c conda-forge cuda-toolkit=11.8 gxx_linux-64=11 gcc_linux-64=11
cat > $ENV/modp_vars.sh <<EOF
export PATH=$ENV/bin:\$PATH CUDA_HOME=$ENV TORCH_CUDA_ARCH_LIST=$CAP MAX_JOBS=4
export CPATH=$ENV/targets/x86_64-linux/include:$ENV/include LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib LD_LIBRARY_PATH=$ENV/lib:\$LD_LIBRARY_PATH
export CC=$(ls $ENV/bin/x86_64-conda-linux-gnu-gcc) CXX=$(ls $ENV/bin/x86_64-conda-linux-gnu-g++)
EOF
source $ENV/modp_vars.sh

# 3. MODP's own installer (PyTorch 2.5.1 cu118, gsplat, fused-ssim, DepthCov, lietorch). The CUDA builds take 10-20 min.
bash setup_env.sh
python -m pip install pyyaml
# 4. compile gsplat's CUDA kernels now (otherwise the first MODP run does it silently) and check everything imports
python - <<'EOF'
import torch, lietorch, depth_cov, fused_ssim, gsplat, lpips
from gsplat.rendering import rasterization
m = torch.randn(10, 3, device='cuda') + torch.tensor([0, 0, 3.], device='cuda')
q = torch.nn.functional.normalize(torch.randn(10, 4, device='cuda'), dim=-1); s = torch.rand(10, 3, device='cuda') * .1
o = torch.rand(10, device='cuda'); c = torch.rand(10, 3, device='cuda'); K = torch.tensor([[[100, 0, 32], [0, 100, 32], [0, 0, 1.]]], device='cuda')
img, _, _ = rasterization(m, q, s, o, c, torch.eye(4, device='cuda')[None], K, 64, 64)
from depth_cov_backends import *   # noqa: the compiled DepthCov kernels
print('MODP env OK | torch', torch.__version__, '| CUDA', torch.version.cuda, '| gsplat', gsplat.__version__, '| render', tuple(img.shape))
EOF
touch $ENV/.modp_ok && echo "SETUP COMPLETE: MODP is ready"
