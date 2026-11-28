"""Seed a flat road layer under each panorama.

Why: the capture car is cut out of every panorama (vehicle_mask_erp: everything below about -46 deg), so no
camera sees the road within ~2.5 m of where it stood. The dense depth init has a hole there, training never
puts splats in it, and the game shows the sky dome through the road as a pale blob with shards (H14 fit sheet,
2026-11-08). Neighbouring panoramas do see that road at a grazing angle, so seeding opaque flat splats there
gives them something to colour.

ground_seeds(centres, points, rgb, up) works in the trainer's normalised frame and returns
(xyz [N,3], rgb [N,3], scale [N,3] log-free, quat [N,4] wxyz).
"""
from __future__ import annotations

import numpy as np


def _basis(up: np.ndarray):
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return up, e1, e2


def quat_z_to(up: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) rotating local +z onto `up` (the splat's thin axis becomes vertical)."""
    u = up / np.linalg.norm(up)
    z = np.array([0.0, 0, 1])
    c = float(np.dot(z, u))
    if c < -0.999999:
        return np.array([0.0, 1, 0, 0])
    axis = np.cross(z, u)
    q = np.array([1 + c, *axis])
    return q / np.linalg.norm(q)


def ground_seeds(centres: np.ndarray, points: np.ndarray, rgb: np.ndarray, up: np.ndarray,
                 radius_frac: float = 0.35, step_frac: float = 0.012, default_height_frac: float = 0.25):
    """centres [P,3] panorama centres, points [M,3] / rgb [M,3] (0..1) initial cloud, all in one frame.

    Units come from the median gap between neighbouring panoramas (~10 m on Street View), so the same
    fractions work whatever the normalisation. Per panorama: the road height is the median height of cloud
    points 0.15..0.6 spacings away horizontally and 0.08..0.5 spacings below the camera; its colour is their
    median colour. A disk of radius radius_frac spacings is filled on a grid of step_frac spacings.
    """
    up, e1, e2 = _basis(np.asarray(up, np.float64))
    C = np.asarray(centres, np.float64)
    if len(C) < 2:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)))
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    sp = float(np.median(d.min(1)))
    P = np.asarray(points, np.float64)
    cols = np.asarray(rgb, np.float64)
    r = radius_frac * sp
    st = step_frac * sp
    g = np.arange(-r, r + 1e-12, st)
    gx, gy = np.meshgrid(g, g)
    disk = (gx ** 2 + gy ** 2) <= r * r
    gx, gy = gx[disk], gy[disk]
    xyz, col = [], []
    fallback = np.median(cols, 0) if len(cols) else np.array([0.4, 0.4, 0.4])
    for c in C:
        rel = P - c
        h = rel @ up
        horiz = np.linalg.norm(rel - h[:, None] * up, axis=1)
        sel = (horiz > 0.15 * sp) & (horiz < 0.6 * sp) & (h < -0.08 * sp) & (h > -0.5 * sp)
        if sel.sum() >= 20:
            height = float(np.median(h[sel]))
            colour = np.median(cols[sel], 0)
        else:
            height = -default_height_frac * sp
            colour = fallback
        base = c + height * up
        xyz.append(base + gx[:, None] * e1 + gy[:, None] * e2)
        col.append(np.repeat(colour[None], len(gx), 0))
    xyz = np.concatenate(xyz)
    col = np.concatenate(col)
    scale = np.tile([st, st, 0.15 * st], (len(xyz), 1))
    quat = np.tile(quat_z_to(up), (len(xyz), 1))
    return xyz, col, scale, quat
