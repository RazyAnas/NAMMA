"""Experiment 1 (18 Nov): can a phone find itself in a scanned room from the room's SHAPE alone, in real time, with no server?

Idea ("shape lock"): an AR phone already knows which way is down, how far it has moved (metres), and, through the depth API,
how far away the surfaces in front of it are. Collected over a few seconds of looking around, those depth points are a rough
3D sketch of the part of the room the person has looked at, in the phone's own coordinates. The scan gives the same room as a
mesh. Only four numbers are unknown: one turn about the vertical, a shift on the floor, and the scale between the scan's
guessed metres and real ones. That is a 2D scan-matching problem (as robots do with a laser scanner and a floor plan), small
enough for a phone's CPU:
  map    three height bands of the mesh (0.25-0.9, 0.9-1.6, 1.6-2.3 m) as "how close is a surface" fields on a 5 cm grid,
         plus a free-space field (where a person can stand).
  live   the depth points in the same three bands, and the phone's own path.
  match  for every turn (5 deg steps) and scale: one FFT cross-correlation per band gives the best shift; then a local polish.
         The path must lie in free space (a strong cue: it rules out the room turned half round when furniture differs).
What this file measures, on the real mesh of a published tour (its ar_map.bin, read into memory) and the real filmed walk, with
SIMULATED depth (z-buffer of the mesh from each pose + noise + dropouts + wrong pixels): how often and how exactly the join is
found after 2 / 5 / 10 / 20 s, for depth noise of 2-10 %, and how long one match takes. What it cannot measure: how good
ARCore's depth really is on white walls. That needs the phone.

    python exp_geo.py <tour folder link ending in /> [trials, default 20]
"""
import json, sys, time, urllib.request
import numpy as np
from scipy import ndimage

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'colab', 'real_scenes', 'final'))
import tour_cloud

from ar_shape import BANDS, CELL, N, ShapeMap, prepare, locate, ry, peak


# ------------------------------------------------------------------ the simulated phone
def camera(pos, fwd):
    fwd = fwd / np.linalg.norm(fwd); x = np.cross(fwd, [0, 1.0, 0]); x /= np.linalg.norm(x); y = np.cross(fwd, x)   # x right, y down, z forward
    return np.stack([x, y, fwd], 1), pos


def depth_points(P, R, pos, W=27, H=60, noise=0.05, drop=0.3, wrong=0.05, far=4.5, rng=None):
    """What the phone's depth image gives for one view: surface points in the room's coordinates, with noise that grows with
    distance, missing pixels and some wrong ones. Field of view 36 x 72 degrees (the test phone's AR camera, SENSOR_CAPTURE.md)."""
    f = 1331 / 867 * W; q = (P - pos) @ R; m = q[:, 2] > 0.15; q = q[m]
    u = np.floor(f * q[:, 0] / q[:, 2] + W / 2).astype(int); v = np.floor(f * q[:, 1] / q[:, 2] + H / 2).astype(int); ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    D = np.full(W * H, np.inf); np.minimum.at(D, v[ok] * W + u[ok], q[ok, 2]); D = D.reshape(H, W)
    vv, uu = np.nonzero(np.isfinite(D) & (D < far)); d = D[vv, uu]
    keep = rng.random(len(d)) > drop; vv, uu, d = vv[keep], uu[keep], d[keep]
    d = d * (1 + noise * rng.standard_normal(len(d))); bad = rng.random(len(d)) < wrong; d[bad] = rng.uniform(0.3, far, bad.sum())
    c = np.stack([(uu + 0.5 - W / 2) / f * d, (vv + 0.5 - H / 2) / f * d, d], 1)
    return c @ R.T + pos


def session(P, walk, T, kind, noise, rng, hz=5):
    """T seconds of a visit: 'walk' follows the filmed walk from a random moment; 'look' stands at a random spot of it and turns.
    Returns depth points and the phone's path in the phone's own coordinates, and the true join."""
    pos, tgt, dur = walk; n = max(2, int(T * hz)); t0 = rng.uniform(0, dur - T - 1); k = lambda t: np.clip(t / dur * (len(pos) - 1), 0, len(pos) - 1.001)
    at = lambda A, t: A[int(k(t))] * (1 - k(t) % 1) + A[int(k(t)) + 1] * (k(t) % 1); pts, path = [], []
    turn = rng.choice([-1, 1]) * np.radians(rng.uniform(25, 45)) * T / 1.0 / n               # 25-45 degrees per second
    for i in range(n):
        tw = max(0.0, i / hz - 5) if kind == 'look+walk' else i / hz                         # 'look+walk': 5 s of looking round, then walking on
        if kind == 'walk' or (kind == 'look+walk' and i / hz >= 5): p = at(pos, t0 + tw); f = at(tgt, t0 + tw) - p
        else: p = at(pos, t0); f0 = at(tgt, t0) - p; f = ry(turn * i) @ np.r_[f0[0], -0.15 * np.hypot(f0[0], f0[2]), f0[2]]
        R, c = camera(p, f); pts.append(depth_points(P, R, c, noise=noise, rng=rng)); path.append(c)
    per = [len(x) for x in pts]; pts, path = np.concatenate(pts), np.array(path)
    yaw, s, t = rng.uniform(0, 2 * np.pi), rng.uniform(0.88, 1.12), np.r_[rng.uniform(-3, 3), rng.uniform(-1.5, -1.0), rng.uniform(-3, 3)]
    to_phone = lambda x: (x @ ry(yaw).T) * s + t                                              # the phone's world: turned, real metres, own origin
    if kind.endswith('*'): return to_phone(pts), to_phone(path), (yaw, s, t), per
    return to_phone(pts), to_phone(path), (yaw, s, t)


def replay(P, M, walk, kind, noise, rng, corners, T=14, hz=5, agree=0.15):
    """A visit second by second, as the page would run it: match on everything seen so far once a second, and believe the
    position only when three matches in a row put a fixed set of points within 15 cm of each other. Returns (second it was
    believed or None, how far the room's corners were off then, cm, and at the end)."""
    pts, path, (yaw, s, t), per = session(P, walk, T, kind + '*', noise, rng, hz); cum = np.cumsum(per)
    ref = np.array([[2, 0, 2], [-2, 0, 2], [-2, 0, -2], [2, 0, -2.]]) + path[0]              # any fixed points in the phone's world
    true_c = (corners @ ry(yaw).T) * s + t; hist, when, at_lock, e = [], None, None, None
    for sec in range(2, T + 1):
        k = sec * hz; L = prepare(pts[:cum[k - 1]], path[:k]); J = locate(M, L)
        to_tour = lambda x: ((x - L['c']) @ ry(J['yaw']).T / J['s'])[:, [0, 2]] + J['t']
        hist.append(to_tour(ref)); e = float(np.linalg.norm(to_tour(true_c) - corners[:, [0, 2]], axis=1).max())
        if when is None and len(hist) >= 3 and all(np.linalg.norm(hist[-1] - h, axis=1).max() < agree for h in hist[-3:-1]): when, at_lock = sec, e
    return when, at_lock, e


def main(base, trials=20):
    get = lambda f: urllib.request.urlopen(base + f, timeout=120).read()
    P = tour_cloud.read_map(get("ar_map.bin")).astype(np.float64)[::3]; S = json.loads(get('settings.json')); W = json.loads(get('walls.json'))
    kv = S['animTracks'][0]['keyframes']['values']; walk = (np.array(kv['position']).reshape(-1, 3), np.array(kv['target']).reshape(-1, 3), S['animTracks'][0]['duration'])
    corners = np.array([[x, 0, z] for x, z in W['outline']]); t0 = time.time(); M = ShapeMap(P)
    print(f'map: {len(P)} mesh points -> {len(BANDS)} bands + free space on a {N}x{N} grid of {CELL * 100:.0f} cm, built in {time.time() - t0:.1f} s', flush=True)
    rng = np.random.default_rng(0); ALL = []
    print('visit      seconds  depth noise   found (corners < 20 cm, turn < 3 deg)   corners off, median   scale error   match time   clear winner')
    for kind in ('look', 'walk'):
        for T in (10,):
            for noise in (0.05,):
                ok, err, serr, ms, clear, hs = [], [], [], [], [], 0
                for _ in range(trials):
                    pts, path, (yaw, s, t) = session(P, walk, T, kind, noise, rng); L = prepare(pts, path); hs += bool(L['heights']); t1 = time.time(); J = locate(M, L); ms.append(time.time() - t1)
                    # a room corner, taken to the phone's world by the truth and brought back by the join that was found
                    cp = (corners @ ry(yaw).T) * s + t - L['c']; back = cp @ ry(J['yaw']).T / J['s']; back = back[:, [0, 2]] + J['t']
                    e = float(np.linalg.norm(back - corners[:, [0, 2]], axis=1).max()); dy = abs(np.degrees(np.angle(np.exp(1j * (J['yaw'] + yaw)))))
                    ok.append(e < 0.20 and dy < 3); err.append(e); serr.append(abs(J['s'] / s - 1)); clear.append(J['margin']); ALL.append((bool(ok[-1]), J['margin'], J['score']))
                print(f'{kind:9s} {T:6d}   {noise * 100:5.0f} %      {np.mean(ok) * 100:5.0f} %  ({int(np.sum(ok))} of {trials})                  {np.median(err) * 100:7.1f} cm        '
                      f'{np.median(serr) * 100:5.1f} %      {np.median(ms):5.2f} s     {np.median(clear):.3f}   ceiling seen in {hs} of {trials}', flush=True)
    print('\nsecond by second, believing a position only when three matches in a row agree (depth noise 5 %):')
    for kind in ('look', 'look+walk'):
        r = [replay(P, M, walk, kind, 0.05, rng, corners) for _ in range(trials)]; got = [x for x in r if x[0] is not None]
        print(f'  {kind:10s} believed in {len(got)} of {trials} visits, after {np.median([x[0] for x in got]) if got else 0:.0f} s (median), right (< 20 cm) in {sum(x[1] < 0.2 for x in got)} of {len(got)}; '
              f'corners off when believed {np.median([x[1] for x in got]) * 100 if got else 0:.1f} cm; never believed and wrong at the end: {sum(x[0] is None and x[2] >= 0.2 for x in r)}', flush=True)
    A = np.array(ALL, float); print('\nwhen is a match to be believed? margin = how much better the best place fits than the next different one')
    for m in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2):
        acc = A[:, 1] >= m; print(f'  margin >= {m:.2f}: shown in {acc.mean() * 100:3.0f} % of tries, right in {A[acc, 0].mean() * 100 if acc.any() else 0:3.0f} % of those ({int(A[acc, 0].sum())} of {int(acc.sum())})')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
