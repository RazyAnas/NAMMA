"""Make the learned sky dome a sky-only backdrop.

The trainer's environment texture is shown wherever splats are thin. Before 2026-11-08 it learned from every
see-through pixel, so buildings got baked into it and showed up as blocky facades in the game (paris-h13).

clean_sky(img, observed) keeps the texels that were really seen as sky and fills the rest per column from the
lowest sky texel above, fading to a haze colour at and below the horizon, then softens the seams.
When `observed` is unknown (old exports), sky texels are guessed: bright, smooth, connected to the zenith.

CLI (used by the publish workflow for exports made before the trainer fix):
    python sky_clean.py sky.jpg [--out sky.jpg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Box filter over rows/cols (wraps horizontally like an equirect)."""
    k = 2 * r + 1
    p = np.pad(a, ((r, r), (0, 0)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    p = np.concatenate([p[:, -r:], p, p[:, :r]], 1)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def guess_sky(img: np.ndarray, std_max: float = 0.045, bright_min: float = 0.45) -> np.ndarray:
    """Bool [H, W]: texels that look like open sky and connect to the top row through sky-like texels."""
    g = img.mean(-1)
    std = np.sqrt(np.clip(_box(g * g, 3) - _box(g, 3) ** 2, 0, None))
    sat = img.max(-1) - img.min(-1)
    blueish = img[..., 2] >= img[..., 0] - 0.02
    like = (std < std_max) & (g > bright_min) & ((sat < 0.12) | blueish)
    H = img.shape[0]
    like[H // 2:] = False  # never below the horizon
    sky = np.zeros_like(like)
    sky[0] = like[0] | True  # zenith row is sky by definition
    for r in range(1, H // 2):
        sky[r] = like[r] & sky[r - 1]
    return sky


def clean_sky(img: np.ndarray, observed: np.ndarray | None = None) -> np.ndarray:
    """img float [H, W, 3] in 0..1, row 0 = straight up, row H/2 = horizon. Returns a sky-only image."""
    img = img.astype(np.float32)
    H, W = img.shape[:2]
    sky = guess_sky(img) if observed is None else observed.astype(bool).copy()
    sky[H // 2:] = False
    col_has = sky.any(0)
    # colour of the lowest sky texel in each column, spread to columns with no sky and smoothed sideways
    # (a per-column fill alone leaves vertical streaks)
    last_row = np.where(col_has, H - 1 - np.argmax(sky[::-1], 0), 0)
    low = img[last_row, np.arange(W)]
    haze = np.median(low[col_has], 0) if col_has.any() else np.array([0.80, 0.85, 0.90], np.float32)
    if col_has.any() and not col_has.all():
        xs = np.where(col_has)[0]
        idx = np.searchsorted(xs, np.arange(W)) % len(xs)
        idx_l = (idx - 1) % len(xs)
        near = np.where(np.abs(xs[idx] - np.arange(W)) % W <= np.abs(np.arange(W) - xs[idx_l]) % W, xs[idx], xs[idx_l])
        low = low[near]
    elif not col_has.any():
        low = np.tile(haze, (W, 1))
    k = max(1, W // 48)
    low_s = _box(np.repeat(low[None], 2 * k + 1, 0), k)[k]
    lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0  # +90 up .. -90 down
    t = np.where(lat > 0, np.clip(1 - lat / 25.0, 0, 1), 1.0)[:, None, None]  # fade to haze near/below horizon
    fill = low_s[None] * (1 - t) + haze[None, None] * t
    out = np.where(sky[..., None], img, fill)
    # soften the fill seams (keep observed sky texels crisp)
    soft = _box(out, 6)
    out = np.where(sky[..., None], out, soft)
    return np.clip(out, 0, 1)


def main():
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("sky", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    img = np.asarray(Image.open(a.sky).convert("RGB"), np.float32) / 255
    out = clean_sky(img)
    Image.fromarray((out * 255).round().astype(np.uint8)).save(a.out or a.sky, quality=92)
    print(f"[sky_clean] {a.sky} -> {a.out or a.sky}")


if __name__ == "__main__":
    main()
