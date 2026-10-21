"""Copy a finished scene's results to the shared folder: results.json, strips, score previews, mesh (.glb, gltfpacked if
available), splat (.ply) when under 150 MB. Usage: python publish.py <work_dir> <dest_dir> [gltfpack_binary]"""
import sys, os, json, shutil, subprocess

W, D = sys.argv[1:3]; gp = sys.argv[3] if len(sys.argv) > 3 else ''
os.makedirs(D, exist_ok=True)
res = dict(args=json.load(open(f'{W}/args.json')))
for st in ['frames', 'sfm', 'undist', 'depth', 'mesh', 'splat', 'score', 'strips']:
    p = f'{W}/.done_{st}'
    if os.path.exists(p): res[st] = json.load(open(p))
if os.path.exists(f'{W}/score/results.json'):
    res['score_per_frame'] = json.load(open(f'{W}/score/results.json'))
json.dump(res, open(f'{D}/results.json', 'w'), indent=1)
for sub in ['strips', 'score']:
    if os.path.isdir(f'{W}/{sub}'):
        os.makedirs(f'{D}/{sub}', exist_ok=True)
        for f in os.listdir(f'{W}/{sub}'):
            if f.endswith('.jpg'): shutil.copy(f'{W}/{sub}/{f}', f'{D}/{sub}/{f}')
glb = f'{W}/mesh/mesh.glb'
if os.path.exists(glb):
    out = f'{D}/mesh.glb'
    if gp and subprocess.run([gp, '-i', glb, '-o', out, '-cc', '-kn'], capture_output=True).returncode == 0:
        pass
    else:
        shutil.copy(glb, out)
ply = f'{W}/splat/splat.ply'
if os.path.exists(ply) and os.path.getsize(ply) < 150e6:
    shutil.copy(ply, f'{D}/splat.ply')
print(json.dumps({k: v for k, v in res.items() if k not in ('score_per_frame', 'args')}, indent=1)[:3000])
