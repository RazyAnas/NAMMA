#!/bin/bash
# Nerfstudio 1.1.5 + gsplat 1.4 (prebuilt) + hloc in a Python 3.10 venv at /content/nsenv; source it. Shared by the Kaggle runners.
pip -q install uv > /tmp/uv.log 2>&1
# Kaggle's image is Python 3.13 and sets uv to its system Python; nerfstudio 1.1.5 needs <=3.10 wheels, so name the venv's python everywhere
unset UV_SYSTEM_PYTHON UV_PYTHON PIP_TARGET
if ! /content/nsenv/bin/python -c "import nerfstudio, gsplat, hloc" 2>/dev/null; then   # reuse an earlier install in this session
[ -x /content/nsenv/bin/python ] || uv venv -q -p 3.10 /content/nsenv || { echo ENV_FAILED; return 1; }
P="--python /content/nsenv/bin/python"
uv pip install -q $P torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 > /tmp/ns_pip.log 2>&1
uv pip install -q $P "numpy<2" setuptools wheel wrapt ninja scikit-build-core cython >> /tmp/ns_pip.log 2>&1
uv pip install -q $P --no-build-isolation "numpy<2" fpsample >> /tmp/ns_pip.log 2>&1
uv pip install -q $P "numpy<2" nerfstudio==1.1.5 --no-build-isolation-package fpsample >> /tmp/ns_pip.log 2>&1
# prebuilt kernels (saves a ~13 min CUDA build); falls back to the source package, which builds on first use
uv pip install -q $P "gsplat==1.4.0+pt21cu118" --extra-index-url https://docs.gsplat.studio/whl/pt21cu118 --index-strategy unsafe-best-match >> /tmp/ns_pip.log 2>&1 || uv pip install -q $P gsplat==1.4.0 >> /tmp/ns_pip.log 2>&1
fi
source /content/nsenv/bin/activate
# hloc's documented install: recursive clone (SuperPoint lives in a git submodule that a plain pip install leaves out)
if ! python -c "import hloc.extractors.superpoint, lightglue" 2>/dev/null; then
  rm -rf /content/hloc && git clone -q --recursive https://github.com/cvg/Hierarchical-Localization /content/hloc
  uv pip install -q --python /content/nsenv/bin/python -e /content/hloc "lightglue @ git+https://github.com/cvg/LightGlue" --no-build-isolation-package fpsample >> /tmp/ns_pip.log 2>&1
fi
# torch 2.1 cpp_extension still imports pkg_resources, which setuptools 81 removed
uv pip install -q --python /content/nsenv/bin/python "setuptools<70" >> /tmp/ns_pip.log 2>&1
which python ns-train
