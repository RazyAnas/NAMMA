# NammaSpace GPU run: thanks for helping!

This folder trains a few 3D models of a living room (from a public test video) on your NVIDIA graphics card, scores how
realistic they look, and packs the scores into one small zip for you to send back. You don't need to understand the
research. You copy and paste the commands below, wait, and send one file back.

**What you send back:** one file named `namma_results_<date>.zip` (a few MB). It holds only scores, a few small images
and short logs. Nothing personal: your home folder and user name are scrubbed from the logs.

## What you need

| | Minimum | Notes |
|---|---|---|
| GPU | NVIDIA, 8 GB memory | 6 GB works with a smaller setting (see Troubleshooting). Any RTX 20xx/30xx/40xx/50xx is fine. |
| System | Linux (Ubuntu 22.04 or 24.04 tested by design) | Windows works through WSL2, see below. macOS can't run this (no NVIDIA). |
| NVIDIA driver | 525 or newer (570+ for RTX 50xx) | Check with `nvidia-smi`. |
| Disk | 15 GB free (30 GB if you also do the optional MODP step) | Most of it is the PyTorch and CUDA downloads. |
| Internet | Yes, during setup | About 5 GB of downloads. |
| Time | 1 to 3 hours, mostly unattended | See the time table below. |

## Steps (copy and paste)

### 1. Basic tools and Miniconda (skip what you already have)

```bash
sudo apt update && sudo apt install -y build-essential git zip unzip wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash
```

### 2. Unzip and set up (about 15 to 30 minutes)

```bash
cd ~ && unzip namma_gpu_handout.zip && cd namma_gpu_handout
bash setup/setup_gs.sh
```

The setup ends with a health check. The first time, it compiles some GPU code, which takes 5 to 10 minutes with no output.
It should finish with **ALL GOOD**. If it prints FAIL, see Troubleshooting.

### 3. Run everything (about 1 to 2.5 hours)

```bash
conda activate namma-gs
bash run_all.sh
```

You can leave it running. To be safe from a closed terminal, run it as `nohup bash run_all.sh > run.log 2>&1 &` and check
progress with `tail -f run.log`. If it stops halfway, run the same command again: finished parts are skipped.

### 4. Optional: the comparison method (MODP), only on RTX 20xx/30xx/40xx

This installs Meta's published method so we can compare against it on the same data. It is fiddly to install, so skip it if
it fails: everything else still works. RTX 50xx cards are too new for it (the script says so and stops).

```bash
bash setup/setup_modp.sh          # about 20 to 40 minutes, compiles a lot
conda activate namma-gs
bash run_all.sh                   # runs only the two MODP parts, the rest is already done
```

### 5. Pack and send

```bash
bash pack_results.sh
```

It prints the path of `namma_results_<date>.zip`. Send that one file back (email, WhatsApp, Google Drive, anything).
If something failed, send the zip anyway: it includes the error logs we need.

## Windows (WSL2)

1. Update the normal Windows NVIDIA driver (GeForce Experience or nvidia.com). Do **not** install a driver inside Linux.
2. Open PowerShell as administrator, run `wsl --install -d Ubuntu-22.04`, reboot, and finish the Ubuntu user setup.
3. Open "Ubuntu" from the Start menu and check that `nvidia-smi` shows your card.
4. Copy the zip into Linux (faster than working on the C: drive):
   `cp /mnt/c/Users/<your Windows name>/Downloads/namma_gpu_handout.zip ~`
5. Follow steps 1 to 5 above. To get the result zip back to Windows: `cp namma_results_*.zip /mnt/c/Users/<your Windows name>/Downloads/`

## How long it takes

Estimates for 640x480 images; we could not time them ourselves because we have no NVIDIA GPU.

| Part | RTX 3060 class | RTX 4090 class |
|---|---|---|
| Setup (`setup_gs.sh`, incl. first compile) | 15-30 min | 10-20 min |
| 6 training runs of 30,000 steps each (`run_all.sh`) | 1.5-2.5 h | 40-70 min |
| Scoring | 2-5 min | 1-2 min |
| Optional MODP setup + 2 runs | 40-70 min | 30-50 min |

Each run uses roughly 3 to 7 GB of GPU memory and about 1 GB of disk in `runs/`.

## Troubleshooting

| What you see | What to do |
|---|---|
| `nvidia-smi: command not found` or it shows no GPU | Install or update the NVIDIA driver (`sudo ubuntu-drivers autoinstall`, then reboot). On WSL, update the Windows driver instead. |
| `conda: command not found` | Run `exec bash` or open a new terminal after step 1. |
| Health check says `PyTorch cannot see the GPU` | The driver is older than the CUDA version picked. Update the driver, then `conda env remove -n namma-gs` and rerun `bash setup/setup_gs.sh`. |
| `gsplat could not compile` / `nvcc` / `unsupported GNU version` | Make sure `build-essential` is installed (step 1), then rerun `python scripts/check_env.py` inside `conda activate namma-gs`. If it still fails, use Docker (below). |
| `CUDA out of memory` | Run with half-size images: `rm -rf runs && DOWNSCALE=2 bash run_all.sh`. Tell us you did, the result zip records it anyway. |
| A run says `FAILED` | The other runs continue. Send the zip anyway, the log is inside. |
| MODP setup fails | Skip it: `SKIP_MODP=1 bash run_all.sh` then `bash pack_results.sh`. |
| `CondaToSNonInteractiveError` or terms-of-service message | Run the `conda tos accept ...` command it prints, then rerun the setup. |
| Want a 2-minute test before the long run | `STEPS=300 bash run_all.sh`, then `rm -rf runs results` and run the real one. |

**Docker instead of conda** (needs Docker and the NVIDIA Container Toolkit; driver 570+, or see the note at the top of the Dockerfile):

```bash
docker build -t namma-gs -f setup/Dockerfile .
docker run --gpus all --rm -it -v "$PWD":/work -v namma-cache:/root/.cache -w /work namma-gs bash -c "python scripts/check_env.py && SKIP_MODP=1 bash run_all.sh && bash pack_results.sh"
```

## What is in here (for the curious)

- `data/` (26 MB): 64 training frames and 64 held-out frames of the ICL-NUIM "living room 1" video (a public research
  dataset), a blurred and flickering copy of the training frames, camera positions, and the starting point sets.
- `scripts/train_gs.py`: trains one Gaussian-splat 3D model with [gsplat](https://github.com/nerfstudio-project/gsplat).
  Every run uses the same images, camera positions and recipe (30,000 steps); only the starting points differ:
  sparse feature points (what a normal tracker gives), random points, our depth-network points, and true depth.
  Two runs use the degraded copy, with a per-image colour correction switched on.
- `scripts/run_modp.py`, `scripts/make_modp_dataset.py`: feed the same frames, camera positions and points to MODP
  (Meta, SIGGRAPH 2025), unchanged except that its camera fine-tuning is off so its renders line up with the held-out frames.
- `scripts/eval.py`: renders the 64 held-out frames from each model and scores them (PSNR, SSIM, LPIPS) against the real frames.
- `pack_results.sh`: makes the zip to send back.
