[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$basePython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainerPython = Join-Path $projectRoot ".venv-gsplat\Scripts\python.exe"

if (-not (Test-Path $basePython)) {
    throw "Create the main Python 3.11 .venv before installing the Gaussian environment."
}
$nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if (-not $nvcc) {
    $defaultCudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    $defaultNvcc = Join-Path $defaultCudaRoot "bin\nvcc.exe"
    if (Test-Path $defaultNvcc) {
        $env:CUDA_HOME = $defaultCudaRoot
        $env:CUDA_PATH = $defaultCudaRoot
        $env:PATH = "$(Join-Path $defaultCudaRoot 'bin');$env:PATH"
        $nvcc = Get-Item $defaultNvcc
    }
}
if (-not $nvcc) {
    throw "CUDA Toolkit 12.8 with nvcc is required to compile current gsplat for an RTX 50-series GPU. Install it from NVIDIA, then rerun this script."
}

& $basePython -m venv (Join-Path $projectRoot ".venv-gsplat")
& $trainerPython -m pip install --upgrade pip "setuptools<82" wheel ninja
& $trainerPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $trainerPython -m pip install nerfstudio==1.1.5
& $trainerPython -m pip uninstall --yes gsplat

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvars)) {
    throw "Visual Studio Build Tools with the C++ workload is required to compile gsplat."
}
$developerEnvironment = cmd.exe /d /s /c ('"' + $vcvars + '" x64 -vcvars_ver=14.44 >nul && set')
foreach ($line in $developerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2]
    }
}
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:DISTUTILS_USE_SDK = "1"
$env:MAX_JOBS = "1"
& $trainerPython -m pip install --no-deps --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
& $trainerPython (Join-Path $PSScriptRoot "verify_gaussian_gpu.py")
