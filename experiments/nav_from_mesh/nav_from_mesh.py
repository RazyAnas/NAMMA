"""Quick CPU experiment: collision mesh -> floor height, walkable grid, floor plan, A* path.

Usage: python nav_from_mesh.py <collision.glb> <out_prefix> [--radius 0.25] [--cell 0.05]
No GPU, no training. Generic: the up axis and floor height are estimated from the mesh itself.
"""
import argparse, heapq, json, time
import numpy as np, trimesh
from scipy import ndimage

def estimate_up(m):
    n, a = m.face_normals, m.area_faces
    w, v = np.linalg.eigh((n * a[:, None]).T @ n)   # dominant normal axis = floor/ceiling
    up = v[:, -1]
    # floor+ceiling are antiparallel; vertical walls are orthogonal. Pick the sign with the lower big mass (floor is lowest)
    return up

def floor_frame(up):
    t = np.cross(up, [1, 0, 0] if abs(up[0]) < .9 else [0, 1, 0]); t /= np.linalg.norm(t)
    return t, np.cross(up, t)

def analyse(path, radius=0.25, cell=0.05, head=1.7, step=0.15):
    t0 = time.time()
    m = trimesh.load(path, force="mesh")
    up = estimate_up(m)
    pts, fi = trimesh.sample.sample_surface(m, 400000, seed=0)
    horiz = np.abs(m.face_normals[fi] @ up) > .85
    def floor_of(sign):
        h = sign * (pts @ up); hh = h[horiz]
        lo, hi = hh.min(), hh.max()
        hist, edges = np.histogram(hh, bins=np.arange(lo, hi + .05, .05))
        low = edges[:-1] < lo + .4 * (hi - lo)          # floor sits in the lower 40% of the height range
        k = int(np.argmax(np.where(low, hist, 0)))
        return hist[k], edges[k] + .025
    (c1, f1), (c2, f2) = floor_of(1), floor_of(-1)
    if c2 > c1: up = -up; floor = f2
    else: floor = f1
    h = pts @ up
    t, b = floor_frame(up)
    u, v = pts @ t, pts @ b
    rel = h - floor
    u0, v0 = u.min() - .5, v.min() - .5
    W, H = int((u.max() - u0 + .5) / cell) + 1, int((v.max() - v0 + .5) / cell) + 1
    iu, iv = ((u - u0) / cell).astype(int), ((v - v0) / cell).astype(int)
    floor_hit = np.zeros((W, H), bool); obst = np.zeros((W, H), bool)
    ok = (np.abs(rel) < step)
    floor_hit[iu[ok], iv[ok]] = True
    occ = (rel > step) & (rel < head)
    obst[iu[occ], iv[occ]] = True
    obst = ndimage.binary_closing(obst, iterations=1)
    floor_hit = ndimage.binary_closing(floor_hit, iterations=2)
    free = floor_hit & ~obst
    r = int(np.ceil(radius / cell))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    walk = ndimage.binary_erosion(free | (ndimage.binary_dilation(obst, structure=(xx**2 + yy**2 <= r*r)) & False),
                                  structure=None, iterations=0) if False else \
           free & ~ndimage.binary_dilation(obst, structure=(xx**2 + yy**2 <= r * r))
    lab, n = ndimage.label(walk)
    if n:
        sizes = ndimage.sum(walk, lab, range(1, n + 1)); walk = lab == (1 + int(np.argmax(sizes)))
    return dict(mesh=m, up=up, floor=floor, grid=(W, H, u0, v0, cell), floor_hit=floor_hit, obst=obst, walk=walk,
                seconds=time.time() - t0)

def astar(walk, s, g):
    W, H = walk.shape; dist = np.full(walk.shape, np.inf); dist[s] = 0; prev = {}
    pq = [(0.0, 0.0, s)]
    nb = [(1,0,1),(-1,0,1),(0,1,1),(0,-1,1),(1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]
    while pq:
        _, d, c = heapq.heappop(pq)
        if c == g: break
        if d > dist[c]: continue
        for dx, dy, w in nb:
            n = (c[0]+dx, c[1]+dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and walk[n] and d + w < dist[n]:
                dist[n] = d + w; prev[n] = c; heapq.heappush(pq, (d + w + np.hypot(g[0]-n[0], g[1]-n[1]), d + w, n))
    if g not in prev and s != g: return None
    p = [g]
    while p[-1] != s: p.append(prev[p[-1]])
    return p[::-1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("glb"); ap.add_argument("out")
    ap.add_argument("--radius", type=float, default=.25); ap.add_argument("--cell", type=float, default=.05)
    a = ap.parse_args()
    R = analyse(a.glb, a.radius, a.cell)
    W, H, u0, v0, cell = R["grid"]; walk = R["walk"]
    idx = np.argwhere(walk)
    s = tuple(idx[np.argmin(idx.sum(1))]); g = tuple(idx[np.argmax(idx.sum(1))])
    t1 = time.time(); path = astar(walk, s, g); tp = time.time() - t1
    plen = None if path is None else float(sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1)) * cell)
    stats = dict(scene=a.glb, floor_height_along_up=round(float(R["floor"]), 3), up=np.round(R["up"], 3).tolist(),
                 grid=[W, H], cell_m=cell, radius_m=a.radius, floor_seen_m2=round(float(R["floor_hit"].sum()*cell*cell), 2),
                 walkable_m2=round(float(walk.sum()*cell*cell), 2), analyse_s=round(R["seconds"], 2),
                 astar_s=round(tp, 3), path_len_m=None if plen is None else round(plen, 2))
    json.dump(stats, open(a.out + ".json", "w"), indent=1); print(json.dumps(stats))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = np.ones((H, W, 3)) * .08
    img[R["floor_hit"].T] = [.25, .25, .3]; img[walk.T] = [.2, .55, .35]; img[R["obst"].T] = [.9, .9, .9]
    fig, ax = plt.subplots(figsize=(7, 7 * H / W + .3)); ax.imshow(img, origin="lower")
    if path: ax.plot([p[0] for p in path], [p[1] for p in path], "r-", lw=1.5)
    ax.set_title(f"{a.out.split('/')[-1]}: walkable {stats['walkable_m2']} m2, path {stats['path_len_m']} m"); ax.axis("off")
    fig.savefig(a.out + ".png", dpi=110, bbox_inches="tight")
