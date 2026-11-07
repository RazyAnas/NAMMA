"""TSDF fusion + marching cubes, copied read-only from colab/real_scenes/final/mesh_tsdf.py (video pipeline, main)."""
import numpy as np
import torch

def fuse(frames, lo, hi, R, vox, trunc=4, device='cuda', chunk=1 << 22):
    """TSDF fusion. frames: list of (depth [H,W] tensor, 0 = no data; K; c2w; rgb [H,W,3] uint8 tensor) on `device`.
    The grid lives in the rotated frame whose axes are the rows of R (orthonormal), box lo..hi, voxel size vox.
    Returns tsdf, weight, colour as numpy arrays of shape dims (+3)."""
    dims = np.ceil((hi - lo) / vox).astype(int); n = int(dims.prod()); tr = trunc * vox
    tsdf = torch.ones(n, device=device); wt = torch.zeros(n, device=device); col = torch.zeros(n, 3, device=device); cw = torch.zeros(n, device=device)
    Rt, lo_t = torch.tensor(R, dtype=torch.float32, device=device), torch.tensor(lo, dtype=torch.float32, device=device)
    cams = [(d, torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device), torch.tensor(K, dtype=torch.float32, device=device), im)
            for d, K, c2w, im in frames]
    for a in range(0, n, chunk):
        i = torch.arange(a, min(n, a + chunk), device=device)
        ijk = torch.stack([i // (dims[1] * dims[2]), (i // dims[2]) % dims[1], i % dims[2]], 1).float()
        p = (lo_t + (ijk + 0.5) * vox) @ Rt                                               # voxel centres, world
        for d, w2c, K, im in cams:
            pc = p @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]; H, W = d.shape
            u = torch.round(pc[:, 0] / z.clamp(min=1e-6) * K[0, 0] + K[0, 2]).long(); v = torch.round(pc[:, 1] / z.clamp(min=1e-6) * K[1, 1] + K[1, 2]).long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any(): continue
            k = ok.nonzero()[:, 0]; dd = d[v[k], u[k]]; sdf = dd - z[k]
            m = (dd > 0) & (sdf > -tr); k, sdf = k[m], sdf[m]
            if not len(k): continue
            g = a + k; s = (sdf / tr).clamp(max=1.0); w0 = wt[g]
            tsdf[g] = (tsdf[g] * w0 + s) / (w0 + 1); wt[g] = w0 + 1
            if im is not None:
                near = sdf.abs() < tr; gn = g[near]                                        # colour only from close to the surface
                col[gn] += im[v[k][near], u[k][near]].float(); cw[gn] += 1
    col = col / cw.clamp(min=1)[:, None]
    return tsdf.view(*dims).cpu().numpy(), wt.view(*dims).cpu().numpy(), col.view(*dims, 3).cpu().numpy()

def extract(tsdf, wt, col, lo, R, vox, cams_world, min_faces=500):
    """Marching cubes on observed voxels -> (verts world, faces, colours uint8), loose bits dropped, faces turned to the cameras."""
    from skimage.measure import marching_cubes
    import trimesh
    v, f, _, _ = marching_cubes(tsdf, level=0.0, spacing=(vox,) * 3, mask=wt > 0)
    ijk = np.clip(np.round(v / vox - 0.5).astype(int), 0, np.array(tsdf.shape) - 1)
    c = col[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    m = trimesh.Trimesh((lo + v + 0.5 * vox) @ R, f, vertex_colors=np.clip(c, 0, 255).astype(np.uint8), process=False)
    parts = m.split(only_watertight=False)
    m = trimesh.util.concatenate([p for p in parts if len(p.faces) >= min_faces]) if len(parts) > 1 else m
    fc = m.triangles_center; s = np.random.default_rng(0).choice(len(fc), min(20000, len(fc)), replace=False)
    near = cams_world[np.argmin(((fc[s, None] - cams_world[None]) ** 2).sum(-1), 1)]
    if (np.einsum('ij,ij->i', m.face_normals[s], near - fc[s]) > 0).mean() < 0.5: m.invert()   # normals face the free space
    return m

