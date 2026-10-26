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

# Floor plan minimap (live position dot, clickable spots) + strip of real-photo thumbnails; clicking goes to that spot.
OVERLAY = """<style>
  #nm-plan{position:fixed;left:16px;top:16px;width:min(30vw,260px);background:#111c;border:1px solid #fff2;border-radius:10px;padding:6px;z-index:10;font:12px system-ui,sans-serif;color:#ddd}
  #nm-plan .wrap{position:relative} #nm-plan img{width:100%;display:block;border-radius:6px}
  #nm-plan .dot{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:#e8a96b;border:2px solid #fff;box-shadow:0 0 6px #000}
  #nm-plan .spot{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;background:#fff9;border:1px solid #000;cursor:pointer}
  #nm-strip{position:fixed;right:16px;top:16px;display:flex;gap:8px;z-index:10;max-width:calc(100vw - 320px);overflow-x:auto}
  #nm-strip button{border:2px solid #fff3;border-radius:8px;padding:0;background:#000;cursor:pointer;position:relative}
  #nm-strip img{width:96px;height:64px;object-fit:cover;display:block;border-radius:6px}
  #nm-strip span{position:absolute;left:4px;bottom:2px;color:#fff;font:11px system-ui,sans-serif;text-shadow:0 0 3px #000}
  @media (max-width:640px){#nm-plan{width:38vw}#nm-strip{display:none}}
</style>
<div id="nm-plan"><div class="wrap"><img src="./floorplan.png" alt="floor plan"><div class="dot" id="nm-dot"></div></div><div>FLOOR PLAN</div></div>
<div id="nm-strip"></div>
<script type="module">
  const o = await (await fetch('./overlay.json')).json(); const f = o.floor; const wrap = document.querySelector('#nm-plan .wrap');
  const pos = (x, z) => [((x - f.x0) * f.px) / f.w * 100, ((z - f.z0) * f.px) / f.h * 100];
  const go = (i) => { try { window.viewer.selectAnnotation(i); } catch (e) {} };
  o.spots.forEach((s, i) => {
    const d = document.createElement('div'); d.className = 'spot'; d.title = s.title; [d.style.left, d.style.top] = pos(s.x, s.z).map(v => v + '%'); d.onclick = () => go(i); wrap.appendChild(d);
    const b = document.createElement('button'); b.innerHTML = `<img src="./${s.thumb}" alt=""><span>${s.title}</span>`; b.onclick = () => go(i); document.getElementById('nm-strip').appendChild(b);
  });
  const dot = document.getElementById('nm-dot');
  setInterval(() => { const app = window.app; if (!app) return; let c; app.root.forEach(e => { if (e.camera) c = e.getPosition(); });
    if (c) { const [l, t] = pos(c.x, c.z); dot.style.left = Math.max(0, Math.min(100, l)) + '%'; dot.style.top = Math.max(0, Math.min(100, t)) + '%'; } }, 200);
</script>"""

def room_collision(pts, h, path):
    """Walk-mode collision when there is no mesh yet: a floor slab at y = 0 and four walls along the room's main horizontal axes
    (2nd-98th percentile of the SfM points), 3 m high. Room coordinates in metres."""
    import trimesh
    flat = pts[:, [0, 2]]; m0 = flat.mean(0); ax = np.linalg.svd(flat - m0, full_matrices=False)[2]
    if np.linalg.det(ax) < 0: ax[1] *= -1
    q = (flat - m0) @ ax.T; lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0); pad = 0.02 * (hi - lo); lo, hi = lo - pad, hi + pad
    def box(cu, cv, cy, eu, ev, ey):
        T = np.eye(4); T[[0, 2], 0] = ax[0]; T[[0, 2], 2] = ax[1]; T[0, 3], T[2, 3] = m0 + cu * ax[0] + cv * ax[1]; T[1, 3] = cy
        return trimesh.creation.box(extents=(eu, ey, ev), transform=T)
    (u0, v0), (u1, v1) = lo, hi; cu, cv, wu, wv, th, hh = (u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0, 0.1, 3.0
    parts = [box(cu, cv, -0.05, wu + 1, wv + 1, 0.1),                                      # floor (top at y = 0)
             box(u0 - th / 2, cv, hh / 2, th, wv + 2 * th, hh), box(u1 + th / 2, cv, hh / 2, th, wv + 2 * th, hh),
             box(cu, v0 - th / 2, hh / 2, wu + 2 * th, th, hh), box(cu, v1 + th / 2, hh / 2, wu + 2 * th, th, hh)]
    trimesh.util.concatenate(parts).export(path)

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
    R, t = M[:3, :3], M[:3, 3]
    # metres: SfM has no scale; the phone was held ~1.35 m above the floor, so scale the median camera height to that. Then the
    # viewer's walk mode (eyes at 1.3 m) walks exactly where the video was filmed, where the model is at its best
    s = float(1.35 / max(np.median(C[:, :3, 3] @ R[1] + t[1]), 1e-6))
    L = lambda x: (x @ R.T + t) * s                                                        # model world -> room coordinates, metres
    cam = L(C[:, :3, 3]); fwd = C[:, :3, 2] @ R.T; pts = L(p.points)
    # room box: SfM points, 3rd-97th percentile horizontally (+12 %), floor - 5 % to the 99th percentile height (+5 %)
    lo, hi = np.percentile(pts, 3, 0), np.percentile(pts, 97, 0); pad = 0.12 * (hi - lo); lo, hi = lo - pad, hi + pad
    h = np.percentile(pts[:, 1], 99); lo[1], hi[1] = -0.05 * h, 1.05 * h
    E = euler_xyz(R @ F)                                                                   # file' = F E F p + F t  ->  E = R F
    json.dump(dict(rotate=E.tolist(), translate=t.tolist(), scale=s, box=lo.tolist() + hi.tolist()), open(f'{tour}/plan.json', 'w'))
    room_collision(pts, h, f'{tour}/room_collision.glb')
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
                    postEffectSettings=dict(sharpness=dict(enabled=True, amount=0.2), bloom=dict(enabled=False, intensity=0.1, blurLevel=2),
                                            grading=dict(enabled=False, brightness=1, contrast=1, saturation=1, tint=[1, 1, 1]),
                                            vignette=dict(enabled=False, intensity=0.5, inner=0.3, outer=0.75, curvature=1),
                                            fringing=dict(enabled=False, intensity=0.5)),
                    animTracks=[track], cameras=[dict(initial=pose(start))], annotations=ann, startMode='animTrack')
    json.dump(settings, open(f'{tour}/settings.json', 'w'))
    # floor plan like splattour's: top-down density of the model's splats between knee and head height (walls light up), real
    # photos of the spots as thumbnails; both drive the overlay in index.html (minimap with a live "you are here" dot)
    from PIL import Image
    v = PlyData.read(f'{run}/full/model.ply')['vertex'].data; Q = L(np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64))
    a = 1 / (1 + np.exp(-v['opacity'].astype(np.float64))); m = np.all((Q >= lo) & (Q <= hi), 1) & (Q[:, 1] > 0.15 * h) & (Q[:, 1] < 0.85 * h) & (a > 0.3)
    span = hi[[0, 2]] - lo[[0, 2]]; px = 900 / span.max(); Wp, Hp = (span * px).astype(int) + 1
    H2, _, _ = np.histogram2d((Q[m, 2] - lo[2]) * px, (Q[m, 0] - lo[0]) * px, bins=(Hp, Wp), range=((0, Hp), (0, Wp)), weights=a[m])
    g = np.log1p(H2); g = np.clip(g / max(np.percentile(g[g > 0], 99), 1e-6), 0, 1) ** 0.7 if (g > 0).any() else g
    img = (np.stack([g * 245, g * 232, g * 215], -1) + 8).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'{tour}/floorplan.png')
    os.makedirs(f'{tour}/thumbs', exist_ok=True)
    for n, i in enumerate(spots):
        Image.open(p.image_paths[i]).convert('RGB').resize((240, int(240 * Hd / Wd))).save(f'{tour}/thumbs/spot{n + 1}.jpg', quality=82)
    overlay = dict(floor=dict(x0=float(lo[0]), z0=float(lo[2]), px=float(px), w=int(Wp), h=int(Hp)),
                   spots=[dict(title=a_['title'], x=float(cs[i][0]), z=float(cs[i][2]), thumb=f'thumbs/spot{n + 1}.jpg') for n, (i, a_) in enumerate(zip(spots, ann))])
    json.dump(overlay, open(f'{tour}/overlay.json', 'w'))
    info = dict(metres_per_unit=round(s, 4), frames=len(cam), start_frame=start, fov=round(fov, 1), room_box=[lo.round(3).tolist(), hi.round(3).tolist()], walk_seconds=dur, spots=len(ann))
    json.dump(info, open(f'{tour}/tour.json', 'w'), indent=1); print('TOUR PLAN', json.dumps(info), flush=True)
    return info

def build(run, tour, public, st='npx --no-install splat-transform'):
    """plan.json + settings.json (from plan) + the model -> a static tour folder."""
    pl = json.load(open(f'{tour}/plan.json')); E, t, b, s = pl['rotate'], pl['translate'], pl['box'], pl.get('scale', 1.0)
    # smears out: splats long in ONE direction (longest > 8x middle axis) that are also big (> 1 % of the room) are the spiky
    # streaks seen off the walk (run 3: 16 % needles); thin flat discs on surfaces stay
    from plyfile import PlyData, PlyElement
    v = PlyData.read(f'{run}/full/model.ply')['vertex'].data; sc = np.sort(np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)), 1)
    room = float(np.linalg.norm(np.array(b[3:]) - np.array(b[:3]))) / s; bad = (sc[:, 2] > 8 * sc[:, 1]) & (sc[:, 2] > 0.01 * room)
    PlyData([PlyElement.describe(v[~bad], 'vertex')]).write(f'{tour}/clean.ply'); print(f'smears removed: {bad.sum()} of {len(v)}', flush=True)
    # colour: SOG's spherical-harmonics clustering takes hours on a CPU (150k splats: SH1 208 s, SH3 > 300 s, SH0 2 s), so the
    # default is PlayCanvas compressed PLY with SH band 1 (view-dependent colour, 25 bytes/splat, seconds); TOUR_FORMAT=sog gives
    # base colour only at ~11 bytes/splat. model.ply keeps all bands.
    fmt = os.environ.get('TOUR_FORMAT', 'cply')
    out, sh = (f'{tour}/scene.compressed.ply', 1) if fmt == 'cply' else (f'{tour}/scene.sog', 0)
    cmd = (f'{st} -w -g cpu {tour}/clean.ply -H {sh} -r {E[0]:.6f},{E[1]:.6f},{E[2]:.6f} -t {t[0]:.6f},{t[1]:.6f},{t[2]:.6f} -s {s:.6f} '
           f'-B {",".join(f"{x:.4f}" for x in b)} -N {out}')
    print(cmd, flush=True); subprocess.run(cmd, shell=True, check=True); os.remove(f'{tour}/clean.ply')
    boot = dict(contentUrl='./' + os.path.basename(out), settingsUrl='./settings.json')
    # collision for walk mode: the room mesh (mesh_tsdf.py, room coordinates before scaling) if there is one, else the room box
    import trimesh
    if os.path.exists(f'{run}/mesh/mesh.glb'):
        mesh = trimesh.load(f'{run}/mesh/mesh.glb', force='mesh'); mesh.apply_scale(s); mesh.export(f'{tour}/collision.glb')
    elif os.path.exists(f'{tour}/room_collision.glb'): shutil.copy(f'{tour}/room_collision.glb', f'{tour}/collision.glb')
    if os.path.exists(f'{tour}/collision.glb'): boot['collisionUrl'] = './collision.glb'
    shutil.copy(f'{public}/index.css', tour)
    js = open(f'{public}/index.js').read()                                                 # walk mode also in rooms < 5 x 5 m
    assert 'WALK_MIN_HORIZONTAL_RANGE = 5' in js; open(f'{tour}/index.js', 'w').write(js.replace('WALK_MIN_HORIZONTAL_RANGE = 5', 'WALK_MIN_HORIZONTAL_RANGE = 2'))
    html = open(f'{public}/index.html').read(); a = html.index('id="sse-bootstrap">') + len('id="sse-bootstrap">'); c = html.index('</script>', a)
    html = (html[:a] + '\n' + json.dumps(boot) + '\n        ' + html[c:]).replace('<title>SuperSplat Viewer</title>', '<title>NAMMA 3D tour</title>')
    html = html.replace("createViewer({ container: document.body, settings, ...options });",
                        "window.viewer = await createViewer({ container: document.body, settings, ...options });")
    if os.path.exists(f'{tour}/overlay.json'): html = html.replace('</body>', OVERLAY + '\n    </body>')
    open(f'{tour}/index.html', 'w').write(html)
    info = json.load(open(f'{tour}/tour.json')); info.update(smears_removed=int(bad.sum()), collision='collisionUrl' in boot, scene_mb=round(os.path.getsize(out) / 1e6, 1), format=fmt)
    json.dump(info, open(f'{tour}/tour.json', 'w'), indent=1); print('TOUR', json.dumps(info), flush=True)
    return info

if __name__ == '__main__':
    {'plan': plan, 'build': build}[sys.argv[1]](*sys.argv[2:])
