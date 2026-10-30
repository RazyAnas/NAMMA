"""NVIDIA Fixer smoke test (.github/workflows/fixer.yml, settings in v3/FIXER.json; plan of 30 Oct, run 4 only).
GitHub CPU runner: render the run's cleaned model (<run>/solid/clean.ply) from tour-like views off the filmed path (1 m to the side,
turned up to 50 deg, raised; final/cpu_render.py) plus 2 held-out filmed views -> private Kaggle dataset namma-fixer-in ->
Kaggle T4 job namma-fixer (final/fixer_smoke.py) fixes them -> before/after sheet -> hf://datasets/<user>/namma-v3-out/fixer/<time>/."""
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/FIXER.json')); OUT = os.path.abspath('out'); IN = os.path.abspath('in'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('FIXER.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok and os.environ.get('KAGGLE_API_TOKEN'), 'needs HF_TOKEN and KAGGLE_API_TOKEN secrets'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok, allow_patterns=[f'{R}/solid/clean.ply', f'{R}/tour/tour.json', f'{R}/sparse/**'])
mpu = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'
for p in (data, OUT, IN): os.makedirs(p, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy '
    'typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))

import numpy as np, torch, cpu_render, tour_fidelity
from PIL import Image
os.chdir(EX); sys.path.insert(0, EX)
s = cpu_render.load_ply(f'{d}/{R}/solid/clean.ply'); X = s['means'].double().numpy(); op = s['opacities'].numpy()
K, W, H, real, off = tour_fidelity.cameras(data, 1024, 2, X, op, mpu); say('view size', W, H)
from datasets.colmap import Parser
C = Parser(data, factor=1, normalize=True, test_every=8).camtoworlds; up = -np.median(C[:, :3, 1], 0); up /= np.linalg.norm(up)

def turn(deg):
    kx = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]]); a = np.radians(deg)
    return np.cos(a) * np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * np.outer(up, up)

views = [(f'held{i}', c) for i, (_, c) in enumerate(real)]                                  # filmed views: Fixer must not spoil them
for i, j in enumerate(np.linspace(0, len(C) - 1, int(cfg.get('views', 10)) + 2).astype(int)[1:-1]):
    c = C[j].copy(); side = (1 if i % 2 else -1) * float(cfg.get('side_m', 1.0)) / mpu        # off the walk, like the tour viewer goes
    c[:3, 3] += c[:3, 0] * side + up * 0.15 / mpu; c[:3, :3] = turn((-1 if i % 2 else 1) * (25 + 25 * (i % 3) / 2)) @ c[:3, :3]
    views.append((f'off{i}', c))
for n, c in views:
    im = cpu_render.render(s, c, K, W, H); Image.fromarray((im.clamp(0, 1) * 255).byte().numpy()).save(f'{IN}/{n}.png')
real_imgs = {f'held{i}': (img.clamp(0, 1) * 255).byte().numpy() for i, (img, _) in enumerate(real)}
say('rendered', len(views), 'views')

import kaggle_io
ku = kaggle_io.account(cfg); ds, kid = f'{ku}/namma-fixer-in', f'{ku}/namma-fixer'
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tf: tf.add(f'{FIN}/fixer_shim', arcname='.')
jcfg = dict(cfg, shim=base64.b64encode(buf.getvalue()).decode())
kaggle_io.push_dataset(IN, ds, 'namma-fixer-in', f'{R} {STAMP}', say)
kaggle_io.push_kernel(kid, 'namma-fixer', f'{FIN}/fixer_smoke.py', jcfg, [ds], say)
st = kaggle_io.wait_kernel(kid, OUT, say, minutes=int(cfg.get('minutes', 120)))
rep = json.load(open(f'{OUT}/fixer_report.json')) if os.path.exists(f'{OUT}/fixer_report.json') else {}
say('kaggle', st[-100:], json.dumps(rep)[:800])
for f in ('fixer.log',):
    if os.path.exists(f'{OUT}/{f}'): print(''.join(open(f'{OUT}/{f}', errors='ignore').readlines()[-60:]), flush=True)

fx = next((f'{OUT}/fixed_{p}' for p in ('bfloat16', 'float16', 'float32') if os.path.isdir(f'{OUT}/fixed_{p}')), None)
if fx:                                                                                   # sheet: [real photo |] render | Fixer, per view
    rows = []; tw = 512; th = round(H * tw / W)
    for n, _ in views:
        a = Image.open(f'{IN}/{n}.png').resize((tw, th)); b = Image.open(f'{fx}/{n}.png').resize((tw, th))
        r = Image.new('RGB', (tw * 3, th)); r.paste(Image.fromarray(real_imgs[n]).resize((tw, th)) if n in real_imgs else Image.new('RGB', (tw, th)), (0, 0))
        r.paste(a, (tw, 0)); r.paste(b, (2 * tw, 0)); rows.append(r)
        if n in real_imgs:                                                                 # does Fixer move a filmed view away from the photo?
            ref = real_imgs[n].astype(np.float32); f = lambda im: 10 * np.log10(255 ** 2 / max(((np.asarray(im.resize((W, H)), np.float32) - ref) ** 2).mean(), 1e-6))
            rep.setdefault('held_psnr', {})[n] = dict(render=round(f(Image.open(f'{IN}/{n}.png')), 2), fixed=round(f(Image.open(f'{fx}/{n}.png')), 2))
    sheet = Image.new('RGB', (tw * 3, th * len(rows)))
    for i, r in enumerate(rows): sheet.paste(r, (0, i * th))
    sheet.save(f'{OUT}/fixer_sheet.jpg', quality=88); json.dump(rep, open(f'{OUT}/fixer_report.json', 'w'), indent=1)
    say('held-out (photo vs render / vs Fixer):', rep.get('held_psnr'))
shutil.copytree(IN, f'{OUT}/renders', dirs_exist_ok=True)
dst = f'fixer/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0 if fx else 'Fixer produced no images (see fixer.log)')
