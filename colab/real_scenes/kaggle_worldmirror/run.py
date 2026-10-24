# WorldMirror 2.0 (Tencent HY-World 2.0): one forward pass, no per-scene training, on 48 frames of the 2026-10-22 room video.
import subprocess, sys, glob, os
def sh(c): print('$', c[:160], flush=True); return subprocess.call(c, shell=True)
video = (glob.glob('/kaggle/input/**/video.mp4', recursive=True) or [''])[0]
W = '/kaggle/temp/wm'; os.makedirs(W + '/frames', exist_ok=True)
sh(f'ffmpeg -loglevel error -i {video} -vf "fps=32/225,scale=-2:720" -q:v 2 {W}/frames/f_%03d.jpg && ls {W}/frames | wc -l')
sh(f'git clone -q --depth 1 https://github.com/Tencent-Hunyuan/HY-World-2.0 {W}/repo')
sh('pip -q install uv')
# Kaggle's uv targets its system Python 3.13 unless told otherwise: name the venv's python on every install
P = f'--python {W}/env/bin/python'
# torch 2.7.1 cu128 as the repo pins (matches Kaggle's CUDA 12.8 nvcc); then each requirement on its own, cupy as the
# prebuilt cuda12x wheel, so one package that fails to build cannot block the rest (it did: nothing installed).
sh(f'unset UV_SYSTEM_PYTHON UV_PYTHON; cd {W} && uv venv -q -p 3.11 env && '
   f'uv pip install -q {P} torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 > pip.log 2>&1; '
   f'grep -vE "^#|^$|^torch|^cupy" repo/requirements.txt | while read r; do uv pip install -q {P} "$r" >> pip.log 2>&1 || echo "FAILED $r"; done; '
   f'uv pip install -q {P} cupy-cuda12x >> pip.log 2>&1 || echo FAILED cupy; uv pip install -q {P} gsplat >> pip.log 2>&1 || echo FAILED gsplat')
# FlashAttention does not run on a T4 (needs Ampere or newer). The model only calls flash_attn_func(q, k, v) with
# (batch, seq, heads, dim) tensors, so a stand-in package routes that to PyTorch's own attention.
shim = f'{W}/env/lib/python3.11/site-packages/flash_attn'
os.makedirs(shim, exist_ok=True)
open(shim + '/__init__.py', 'w').write('')
# Half precision and the memory-efficient kernel only: the plain fallback builds the whole attention matrix (13 GB).
open(shim + '/flash_attn_interface.py', 'w').write('''import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
    dtype = q.dtype
    q, k, v = (t.transpose(1, 2).half() for t in (q, k, v))
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2).to(dtype)
''')
code = sh(f'cd {W}/repo && {W}/env/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())" && '
          f'{W}/env/bin/python -m hyworld2.worldrecon.pipeline --input_path {W}/frames --output_path /kaggle/working/worldmirror 2>&1 | tail -40')
sh('ls -laR /kaggle/working | head -60')
sys.exit(code)
