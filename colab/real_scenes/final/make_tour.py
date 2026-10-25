"""v3 tour: the trained room as a web tour like splattour.com, built on the official SuperSplat Viewer (PlayCanvas, MIT).
  scene.sog      the full model levelled (floor at y = 0), cropped to the room box (no blurry shell from the window view, no
                 floaters outside), SOG-compressed (~10x smaller than the .ply) by @playcanvas/splat-transform
  settings.json  start view inside the room at the height the phone was held, a guided walk along the real camera path (where the
                 model is at its best: it plays first), hotspots at evenly spaced spots of the walk, all in room coordinates
  collision.glb  the room mesh (mesh_tsdf.py), if there is one: walk mode stays inside the room on the floor
Coordinates: the viewer turns every splat file 180 deg about z (3DGS convention), so the file gets F (R p + t) with F = diag(-1,-1,1)
while cameras/hotspots are written in room coordinates R p + t (measured with a test scene, see HANDOFF_V3.md section 9).
Two steps (the camera reader needs gsplat's examples + torch, the converter needs Node):
  python make_tour.py plan  <examples dir> <data dir (images/ + sparse/0)> <run out dir (full/, mesh/)> <tour dir>
  python make_tour.py build <run out dir> <tour dir> <viewer public dir> [splat-transform command]"""
import json, os, shutil, subprocess, sys, numpy as np
F = np.diag([-1., -1, 1])

def smooth(x, r):
    """Gaussian moving average along the walk (handheld jitter out), edges kept."""
    k = np.exp(-0.5 * (np.arange(-2 * r, 2 * r + 1) / max(r, 1e-6)) ** 2); k /= k.sum()
    xp = np.pad(x, ((2 * r, 2 * r), (0, 0)), mode='edge'); return np.stack([np.convolve(xp[:, j], k, 'valid') for j in range(x.shape[1])], 1)

def euler_xyz(M):
    """Extrinsic x-y-z Euler angles in degrees of a rotation matrix (splat-transform -r)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(M).as_euler('xyz', degrees=True)

def plan(examples, data, run, tour):
    sys.path.insert(0, examples); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets.colmap import Parser
    from plyfile import PlyData
    import to_splat
    os.makedirs(tour, exist_ok=True)
    p = Parser(data, factor=1, normalize=True, test_every=8); C = p.camtoworlds; nav = f'{run}/full/nav.json'
    if os.path.exists(f'{run}/full/level.json'): M = np.array(json.load(open(f'{run}/full/level.json'))['matrix'])
    else:
        v = PlyData.read(f'{run}/full/model.ply')['vertex'].data; M = to_splat.level_matrix(nav, np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64))
    R, t = M[:3, :3], M[:3, 3]; L = lambda x: x @ R.T + t                                  # model world -> room coordinates
    cam = L(C[:, :3, 3]); fwd = C[:, :3, 2] @ R.T; pts = L(p.points)
    # room box: SfM points, 3rd-97th percentile horizontally (+12 %), floor - 5 % to the 99th percentile height (+5 %)
    lo, hi = np.percentile(pts, 3, 0), np.percentile(pts, 97, 0); pad = 0.12 * (hi - lo); lo, hi = lo - pad, hi + pad
    h = np.percentile(pts[:, 1], 99); lo[1], hi[1] = -0.05 * h, 1.05 * h
    E = euler_xyz(R @ F)                                                                   # file' = F E F p + F t  ->  E = R F
    json.dump(dict(rotate=E.tolist(), translate=t.tolist(), box=lo.tolist() + hi.tolist()), open(f'{tour}/plan.json', 'w'))
    # cameras in room coordinates, in walk order (Parser sorts the frame names = video order)
    K = p.Ks_dict[p.camera_ids[0]]; Wd, Hd = p.imsize_dict[p.camera_ids[0]]; fov = float(np.clip(np.degrees(2 * np.arctan(Hd / 2 / K[1, 1])), 10, 120))
    r = max(2, len(cam) // 60); cs, fs = smooth(cam, r), smooth(fwd, r); fs /= np.linalg.norm(fs, axis=1, keepdims=True)
    look = 0.25 * float(np.linalg.norm(hi[[0, 2]] - lo[[0, 2]]))                            # look-at distance ~ quarter of the room
    pose = lambda i: dict(position=[round(float(x), 4) for x in cs[i]], target=[round(float(x), 4) for x in cs[i] + look * fs[i]], fov=round(fov, 1))
    centre = (lo + hi) / 2; start = int(np.argmin(np.linalg.norm((cam - centre)[:, [0, 2]], axis=1)))   # nearest to the middle
    keys = np.unique(np.linspace(0, len(cam) - 1, min(len(cam), 120)).astype(int)); dur = round(len(cam) / 6.0, 1)
    frames = [int(round(k / (len(cam) - 1) * dur * 30)) for k in keys]                        # keyframe times are frame numbers
    track = dict(name='Walk', duration=dur, frameRate=30, loopMode='repeat', interpolation='spline', smoothness=1,
                 keyframes=dict(times=frames,
                                values=dict(position=[float(x) for k in keys for x in pose(k)['position']],
                                            target=[float(x) for k in keys for x in pose(k)['target']], fov=[fov] * len(keys))))
    spots = np.linspace(0, len(cam) - 1, 8).astype(int)[1:-1]
    ann = [dict(position=pose(i)['target'], title=f'Spot {n + 1}', text='Real filmed view', camera=dict(initial=pose(i))) for n, i in enumerate(spots)]
    settings = dict(version=2, tonemapping='none', highPrecisionRendering=False, background=dict(color=[0.08, 0.08, 0.09]),
                    postEffectSettings=dict(sharpness=dict(enabled=False, amount=0), bloom=dict(enabled=False, intensity=0.1, blurLevel=2),
                                            grading=dict(enabled=False, brightness=1, contrast=1, saturation=1, tint=[1, 1, 1]),
                                            vignette=dict(enabled=False, intensity=0.5, inner=0.3, outer=0.75, curvature=1),
                                            fringing=dict(enabled=False, intensity=0.5)),
                    animTracks=[track], cameras=[dict(initial=pose(start))], annotations=ann, startMode='animTrack')
    json.dump(settings, open(f'{tour}/settings.json', 'w'))
    info = dict(frames=len(cam), start_frame=start, fov=round(fov, 1), room_box=[lo.round(3).tolist(), hi.round(3).tolist()], walk_seconds=dur, spots=len(ann))
    json.dump(info, open(f'{tour}/tour.json', 'w'), indent=1); print('TOUR PLAN', json.dumps(info), flush=True)
    return info

def build(run, tour, public, st='npx --no-install splat-transform'):
    """plan.json + settings.json (from plan) + the model -> a static tour folder."""
    pl = json.load(open(f'{tour}/plan.json')); E, t, b = pl['rotate'], pl['translate'], pl['box']
    cmd = (f'{st} -w -g cpu {run}/full/model.ply -r {E[0]:.6f},{E[1]:.6f},{E[2]:.6f} -t {t[0]:.6f},{t[1]:.6f},{t[2]:.6f} '
           f'-B {",".join(f"{x:.4f}" for x in b)} -N {tour}/scene.sog')
    print(cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
    boot = dict(contentUrl='./scene.sog', settingsUrl='./settings.json')
    if os.path.exists(f'{run}/mesh/mesh.glb'): shutil.copy(f'{run}/mesh/mesh.glb', f'{tour}/collision.glb'); boot['collisionUrl'] = './collision.glb'
    for f in ('index.js', 'index.css'): shutil.copy(f'{public}/{f}', tour)
    html = open(f'{public}/index.html').read(); a = html.index('id="sse-bootstrap">') + len('id="sse-bootstrap">'); c = html.index('</script>', a)
    html = (html[:a] + '\n' + json.dumps(boot) + '\n        ' + html[c:]).replace('<title>SuperSplat Viewer</title>', '<title>NAMMA 3D tour</title>')
    open(f'{tour}/index.html', 'w').write(html)
    info = json.load(open(f'{tour}/tour.json')); info.update(collision='collisionUrl' in boot, sog_mb=round(os.path.getsize(f'{tour}/scene.sog') / 1e6, 1))
    json.dump(info, open(f'{tour}/tour.json', 'w'), indent=1); print('TOUR', json.dumps(info), flush=True)
    return info

if __name__ == '__main__':
    {'plan': plan, 'build': build}[sys.argv[1]](*sys.argv[2:])
