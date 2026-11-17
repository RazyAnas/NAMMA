"""How well does DRS-VPT (github.com/ori-drs/drs-vpt: one picture + a point cloud in metres -> where the camera is) find
a camera inside one of our finished scans? Runs on a CPU machine (GitHub Actions: .github/workflows/ar-check.yml), nothing
on the laptop, nothing on Kaggle.

    python check_drsvpt.py <run folder> <gsplat examples dir> <out dir> [views, default 16]

<run folder> holds what the results dataset keeps of a run: sparse/0, full/model.ply, full/level.json, tour/plan.json,
tour/scene.compressed.ply, tour/thumbs/spot*.jpg, mesh/mesh.glb (optional).

Test pictures, each with its true camera (the scan's own camera solve):
  render   the scan drawn from the exact pose and lens of evenly spaced filmed frames (cpu_render; the frames themselves are
           not kept after a job). Same geometry as the map, so this shows the method and our axis conventions work; it is
           kinder than a photo taken on another day.
  photo    the tour's six spot thumbnails: real frames of the video, but only 240 px wide.
Maps tried: the published splat's centres (what a server without the private data can use) and the room mesh, each at two
densities; the room's own axes (y up) or turned z up, whichever works better on six views (the README names no convention).
Every answer is also compared with the camera turned half round about the room's middle ("twin"): a plain box-shaped room
looks the same from there to a model that sees only shape.
Result: <out>/report.json and one line per setting: median error in cm and degrees, share of pictures within 5 cm / 5 deg,
10 cm / 5 deg, 25 cm / 10 deg, seconds per picture. Weights are CC BY-NC 4.0: a test, not a product part."""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); FINAL = os.path.join(HERE, '..', '..', 'colab', 'real_scenes', 'final')
sys.path.insert(0, HERE); sys.path.insert(0, FINAL)
ZUP = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])                                         # room (x, y up, z) -> (x, -z, y up)


def errors(T_w2c, c2w):
    """(cm, degrees) between a predicted world-to-camera matrix and the true camera-to-world one."""
    R, t = np.asarray(T_w2c, float)[:3, :3], np.asarray(T_w2c, float)[:3, 3]; c = -R.T @ t
    cos = (np.trace(R @ c2w[:3, :3]) - 1) / 2                                               # world->cam (found) x cam->world (true)
    return float(np.linalg.norm(c - c2w[:3, 3]) * 100), float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def summary(rows):
    e = np.array([[r['cm'], r['deg']] for r in rows]); within = lambda cm, deg: round(float(np.mean((e[:, 0] <= cm) & (e[:, 1] <= deg))), 2)
    return dict(n=len(rows), median_cm=round(float(np.median(e[:, 0])), 1), median_deg=round(float(np.median(e[:, 1])), 2),
                within_5cm_5deg=within(5, 5), within_10cm_5deg=within(10, 5), within_25cm_10deg=within(25, 10),
                seconds=round(float(np.median([r['s'] for r in rows])), 1))


def lock(fixes, corners, tol=0.15, tol_deg=5.0, tilt=4.0):
    """What the AR page would do with these answers (ar_core.js solve, scale 1), with the phone's own tracking taken as exact.
    fixes = [(T_w2c found, true c2w)], room axes. Returns (answers left after the gravity test, answers that agree, how far
    the room's corners end up from where they are, cm) or (n upright, 0, None) when nothing is left."""
    ry = lambda a, p: np.array([np.cos(a) * p[0] + np.sin(a) * p[2], p[1], -np.sin(a) * p[0] + np.cos(a) * p[2]]); F = []
    for T, gt in fixes:
        S = gt @ T                                                                          # room as found -> room as it is
        if np.degrees(np.arccos(np.clip(S[1, 1], -1, 1))) > tilt: continue                  # would tilt the room: gravity says no
        F.append((np.arctan2(S[0, 2] - S[2, 0], S[0, 0] + S[2, 2]), -T[:3, :3].T @ T[:3, 3], gt[:3, 3]))
    best = None
    for yaw, ct, cx in F:
        t = cx - ry(yaw, ct); a = [f for f in F if np.linalg.norm(ry(yaw, f[1]) + t - f[2]) <= tol and abs(np.angle(np.exp(1j * (f[0] - yaw)))) <= np.radians(tol_deg)]
        if best is None or len(a) > len(best): best = a
    if not best: return len(F), 0, None
    yaw = float(np.angle(np.sum([np.exp(1j * f[0]) for f in best]))); t = np.mean([f[2] - ry(yaw, f[1]) for f in best], 0)
    return len(F), len(best), round(float(max(np.linalg.norm(ry(yaw, np.array([x, 0, z])) + t - [x, 0, z]) for x, z in corners)) * 100, 1)


def main(run, examples, out, views=16):
    import torch
    from PIL import Image
    import make_tour, thin_model, tour_cloud, cpu_render
    os.makedirs(out, exist_ok=True); t0 = time.time(); say = lambda *a: print(f'[{time.time() - t0:5.0f}s]', *a, flush=True)
    plan = json.load(open(f'{run}/tour/plan.json')); s = s_ = plan['scale']; M = np.array(json.load(open(f'{run}/full/level.json'))['matrix'])
    C, _ = make_tour.colmap_cameras(f'{run}/sparse/0', examples); n = len(C)                # camera-to-world, model frame, walk order
    cams, ims, _ = thin_model.read_model(f'{run}/sparse/0'); _, W0, H0, prm = cams[next(iter(ims.values()))['cam']]
    mid = cams[next(iter(ims.values()))['cam']][0]                                          # COLMAP model id: 0, 2, 3 have one focal length
    fx0, fy0, cx0, cy0 = (prm[0], prm[0], prm[1], prm[2]) if mid in (0, 2, 3) else (prm[0], prm[1], prm[2], prm[3]); f0 = fx0
    K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1.]])
    room = lambda c2w: np.block([[M[:3, :3] @ c2w[:3, :3], ((M[:3, :3] @ c2w[:3, 3] + M[:3, 3]) * s)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    say(f'{n} cameras, lens {W0}x{H0} f {f0:.0f}, scale {s:.3f} m per unit')
    # test pictures
    tests = []
    spl = cpu_render.load_ply(f'{run}/full/model.ply'); Wr = 518; Hr = int(round(Wr * H0 / W0)); Kr = K0 * (Wr / W0); Kr[2, 2] = 1
    for i in np.linspace(0, n - 1, views + 2).astype(int)[1:-1]:
        with torch.no_grad(): im = cpu_render.render(spl, C[i], Kr, Wr, Hr).clamp(0, 1).numpy()
        tests.append(dict(kind='render', frame=int(i), img=(im * 255).astype(np.uint8), K=Kr.copy(), c2w=room(C[i]), bright=round(float(im.mean()), 3)))
    del spl
    spots = plan.get('spots') or [int(x) for x in np.linspace(0, n - 1, 8).astype(int)[1:-1]]
    for k, i in enumerate(spots):
        p = f'{run}/tour/thumbs/spot{k + 1}.jpg'
        if not os.path.exists(p): continue
        im = np.asarray(Image.open(p).convert('RGB')); Kt = K0 * (im.shape[1] / W0); Kt[2, 2] = 1
        tests.append(dict(kind='photo', frame=int(i), img=im, K=Kt, c2w=room(C[i]), bright=round(float(im.mean() / 255), 3)))
    say(len(tests), 'test pictures; render brightness', [t['bright'] for t in tests if t['kind'] == 'render'])
    # maps (room metres)
    wj = json.load(open(f'{run}/tour/walls.json')) if os.path.exists(f'{run}/tour/walls.json') else {}; box = wj.get('box')
    scene = f'{run}/tour/scene.compressed.ply'
    maps = {}                                                                               # round 2: splat centres 12-25 % right, mesh 25-31 %; y up, not z up
    import trimesh
    for name, path, k in (('mesh', f'{run}/mesh/mesh.glb', s_), ('walk-collision mesh (public)', f'{run}/tour/collision.glb', 1.0)):
        if not os.path.exists(path): continue
        try:
            m = trimesh.load(path, force='mesh'); m.apply_scale(k); p = np.asarray(m.sample(2500000), float)
            if box: p = p[np.all((p >= np.array(box[:3]) - 0.3) & (p <= np.array(box[3:]) + 0.3), 1)]
            key = np.floor(p / 0.01).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True); maps[f'{name} 1 cm'] = (p[first].astype(np.float32), 600000)
        except Exception as e: say(name, 'map skipped:', type(e).__name__, str(e)[:120])
    if not maps: maps['splat 1 cm, faint ones too'] = (tour_cloud.cloud(scene, grid=0.01, opacity=0.1, box=box), 600000)
    say('maps:', {k: len(v[0]) for k, v in maps.items()})
    # the room turned half round about its own middle: a plain box looks the same from there to a model that sees only shape
    mid = np.mean(wj['outline'], 0) if wj.get('outline') else (np.array(box)[[0, 2]] + np.array(box)[[3, 5]]) / 2 if box else np.zeros(2)
    TWIN = np.array([[-1., 0, 0, 2 * mid[0]], [0, 1, 0, 0], [0, 0, -1, 2 * mid[1]], [0, 0, 0, 1]])
    corners = wj.get('outline') or ([[box[0], box[2]], [box[3], box[2]], [box[3], box[5]], [box[0], box[5]]] if box else [[-2, -2], [2, -2], [2, 2], [-2, 2]])
    from drs_vpt.hub import load_model
    from drs_vpt.inference import encode_map, localize
    model = load_model(device='cpu'); say('model loaded')
    report = dict(run=os.path.basename(os.path.normpath(run)), settings={})

    def trial(name, cloud, cap, A, subset):
        try: enc = encode_map(model, np.ascontiguousarray(cloud @ A.T, np.float32), grid_size=0.01, max_points=cap)
        except Exception as e: say(name, 'failed:', type(e).__name__, str(e)[:160]); return None
        rows, got = [], []
        for t in subset:
            t1 = time.time(); o = localize(model, t['img'], enc, t['K']); T = np.asarray(o['T_w2c'], float).copy(); T[:3, :3] = T[:3, :3] @ A   # -> room axes
            cm, deg = errors(T, t['c2w']); tcm, tdeg = errors(np.linalg.inv(TWIN @ np.linalg.inv(T)), t['c2w'])
            got.append((T, t['c2w'], t['kind'])); c = -T[:3, :3].T @ T[:3, 3]; up_cam = -T[1, :3]                                 # where it thinks it is; the camera's "up" in the room
            rows.append(dict(kind=t['kind'], frame=t['frame'], cm=round(cm, 1), deg=round(deg, 1), twin_cm=round(tcm, 1), twin_deg=round(tdeg, 1),
                             height_cm=round(float(c[1] - t['c2w'][1, 3]) * 100, 1), upright=round(float(up_cam[1]), 2), s=round(time.time() - t1, 1)))
        res = {}
        for k in ('render', 'photo'):
            rk = [r for r in rows if r['kind'] == k]
            if rk: res[k] = dict(summary(rk), twin_within_25cm_10deg=round(float(np.mean([(r['twin_cm'] <= 25) & (r['twin_deg'] <= 10) for r in rk])), 2))
        # would the AR page lock on? all pictures, the photos alone, and after how many pictures (walk order) it is right for good
        order = sorted(range(len(got)), key=lambda i: subset[i]['frame']); fx = [got[i][:2] for i in order]
        ok_from = next((k for k in range(1, len(fx) + 1) if all((lock(fx[:j], corners)[2] or 1e9) <= 15 for j in range(k, len(fx) + 1))), None)
        res['lock'] = dict(zip(('upright', 'agree', 'corners_off_cm'), lock(fx, corners)), photos_only=dict(zip(('upright', 'agree', 'corners_off_cm'), lock([g[:2] for g in got if g[2] == 'photo'], corners))), right_for_good_after=ok_from, pictures=len(fx))
        report['settings'][name] = dict(points=len(cloud), **res, rows=rows); say(name, json.dumps(res))
        say('   ' + '  '.join(f"{r['kind'][0]}{r['frame']}: {r['cm']:.0f} cm {r['deg']:.0f} deg (twin {r['twin_cm']:.0f} cm {r['twin_deg']:.0f} deg, up {r['upright']})" for r in rows))
        json.dump(report, open(f'{out}/report.json', 'w'), indent=1); return res

    best = 'y up'; A = np.eye(3); report['up'] = best                                         # measured in round 2 (z up: 0 of 6)
    for k, (cloud, cap) in maps.items(): trial(f'{k}, {best}', cloud, cap, A, tests)
    json.dump(report, open(f'{out}/report.json', 'w'), indent=1); say('done')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16)
