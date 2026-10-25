"""Load (or build, the first time) gsplat's CUDA extension and rasterize one splat. Prints how long it took:
~300 s means it compiled, a few seconds means the cached build was reused."""
import time, torch
t = time.time()
from gsplat import rasterization
d = 'cuda'; m = torch.zeros(1, 3, device=d); q = torch.tensor([[1., 0, 0, 0]], device=d)
r, _, _ = rasterization(m, q, torch.ones(1, 3, device=d) * .1, torch.ones(1, device=d), torch.ones(1, 3, device=d),
                        torch.eye(4, device=d)[None], torch.tensor([[[100., 0, 50], [0, 100., 50], [0, 0, 1]]], device=d), 100, 100)
print('gsplat ready in', round(time.time() - t), 's', flush=True)
