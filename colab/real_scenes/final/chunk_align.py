"""Merge feed-forward predictions made on overlapping view groups into one world frame (Sim3 fitted on shared frames)."""
import numpy as np

def backproject(depth, K, c2w, stride=4):
    H, W = depth.shape; v, u = np.mgrid[0:H:stride, 0:W:stride]; z = depth[v, u]
    p = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1).reshape(-1, 3)
    return p @ c2w[:3, :3].T + c2w[:3, 3], (z > 0).reshape(-1)

def umeyama(src, dst):
    """Least-squares s, R, t with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src)); d = np.sign(np.linalg.det(U @ Vt)); E = np.diag([1, 1, d])
    R = U @ E @ Vt; s = (sig * np.diag(E)).sum() / (S ** 2).sum(1).mean() * 1.0
    return s, R, mu_d - s * R @ mu_s

def merge(chunks):
    """chunks: list of dicts(idx=[global frame ids], depth (n,H,W), K (n,3,3), c2w (n,4,4)); consecutive chunks share frames.
    Returns depth (N,H,W), K (N,3,3), c2w (N,4,4) in the first chunk's frame, indexed by global frame id."""
    N = max(max(c["idx"]) for c in chunks) + 1
    H, W = chunks[0]["depth"].shape[1:]
    D = np.zeros((N, H, W), np.float32); Ks = np.zeros((N, 3, 3)); C = np.tile(np.eye(4), (N, 1, 1)); have = np.zeros(N, bool)
    for ci, c in enumerate(chunks):
        idx = list(c["idx"]); dep, K, c2w = c["depth"].astype(np.float64), c["K"], c["c2w"].copy()
        if ci > 0:
            shared = [j for j, g in enumerate(idx) if have[g]]
            if not shared: continue                      # a skipped chunk broke the chain: drop this one rather than guess
            A, B = [], []
            for j in shared:
                pl, ml = backproject(dep[j], K[j], c2w[j]); pg, mg = backproject(D[idx[j]], Ks[idx[j]], C[idx[j]])
                m = ml & mg; A.append(pl[m]); B.append(pg[m])
            s, R, t = umeyama(np.concatenate(A), np.concatenate(B))
            dep = dep * s
            for j in range(len(idx)):
                c2w[j][:3, :3] = R @ c2w[j][:3, :3]; c2w[j][:3, 3] = s * R @ c2w[j][:3, 3] + t
        for j, g in enumerate(idx):
            if not have[g]: D[g], Ks[g], C[g], have[g] = dep[j], K[j], c2w[j], True
    return D, Ks, C, have
