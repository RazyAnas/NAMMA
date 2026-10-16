[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$basePython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainerPython = Join-Path $projectRoot ".venv-gsplat\Scripts\python.exe"

if (-not (Test-Path $basePython)) {
    throw "Create the main Python 3.11 .venv before installing the Gaussian environment."
}
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    throw "CUDA Toolkit 12.8 with nvcc is required to compile current gsplat for an RTX 50-series GPU. Install it from NVIDIA, then rerun this script."
}

& $basePython -m venv (Join-Path $projectRoot ".venv-gsplat")
& $trainerPython -m pip install --upgrade pip "setuptools<82" wheel ninja
& $trainerPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $trainerPython -m pip install nerfstudio==1.1.5
& $trainerPython -m pip uninstall --yes gsplat
& $trainerPython -m pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git"
& $trainerPython (Join-Path $PSScriptRoot "verify_gaussian_gpu.py")
