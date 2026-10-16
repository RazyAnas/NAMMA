"""Fail fast unless Torch and gsplat can execute on the configured NVIDIA GPU."""

import torch
import gsplat


if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access CUDA")
device = torch.device("cuda:0")
sample = torch.ones(16, device=device)
if float((sample * 2).sum().cpu()) != 32.0:
    raise SystemExit("CUDA tensor smoke test failed")
capability = torch.cuda.get_device_capability(device)
print(f"torch={torch.__version__}")
print(f"gsplat={gsplat.__version__}")
print(f"gpu={torch.cuda.get_device_name(device)} capability={capability[0]}.{capability[1]}")
