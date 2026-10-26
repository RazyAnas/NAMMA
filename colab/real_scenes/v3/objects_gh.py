"""GitHub Actions side of the object layer (.github/workflows/objects.yml, settings in v3/OBJECTS.json).
The run's model.ply + sparse/ (private HF results dataset) -> private Kaggle dataset namma-objects-in -> Kaggle job
anasrazy/namma-objects (final/objects_job.py -> final/objects3d.py, with the run's frames dataset attached; the run's level matrix
and metres-per-unit from its tour go in JOB_CFG) -> outputs -> hf://datasets/<user>/namma-v3-out/objects/<time>/.
Only one Kaggle GPU job at a time."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/OBJECTS.json')); KU = cfg.get('kaggle_user', 'anasrazy'); KID = f'{KU}/namma-objects'; DS = f'{KU}/namma-objects-in'
OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

def kg(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True); out = (r.stdout + r.stderr).strip()
    if check and r.returncode: sys.exit(f'kaggle {" ".join(args[:2])} failed: {out[-800:]}')
    return r.returncode, out

if not cfg.get('n'): say('OBJECTS.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok and os.environ.get('KAGGLE_API_TOKEN'), 'needs HF_TOKEN and KAGGLE_API_TOKEN secrets'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok,
                      allow_patterns=[f'{R}/full/model.ply', f'{R}/full/level.json', f'{R}/tour/tour.json', f'{R}/sparse/**'])
cfg['level'] = json.load(open(f'{d}/{R}/full/level.json'))['matrix']; cfg['metres_per_unit'] = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
if not cfg.get('collect_only'):
    os.makedirs('in', exist_ok=True); shutil.copy(f'{d}/{R}/full/model.ply', 'in/model.ply'); shutil.make_archive('in/sparse', 'zip', f'{d}/{R}', 'sparse')
    json.dump(dict(title='namma-objects-in', id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
    rc, st = kg('datasets', 'status', DS, check=False)
    say(kg('datasets', 'version', '-p', 'in', '-m', f'{R} {STAMP}', '-q')[1][-300:] if rc == 0 and 'ready' in st.lower()
        else kg('datasets', 'create', '-p', 'in', '-q')[1][-300:])
    for _ in range(90):
        time.sleep(20); st = kg('datasets', 'status', DS, check=False)[1]
        if 'ready' in st.lower() or 'error' in st.lower(): break
    say('dataset', DS, st[-120:])
os.makedirs('kern', exist_ok=True)
json.dump(dict(id=KID, title='namma-objects', code_file='objects_job.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4', kernel_sources=[f'{KU}/namma-prep'],
               dataset_sources=[DS, cfg['frames_dataset']]), open('kern/kernel-metadata.json', 'w'))
code = open(f'{FIN}/objects_job.py').read(); first = 'import gc, glob, json, os, shutil, subprocess, sys, time\n'; assert code.startswith(first)
open('kern/objects_job.py', 'w').write(code.replace(first, first + f'os.environ["JOB_CFG"] = {json.dumps(json.dumps(cfg))}  # set by objects_gh.py\n', 1))
if cfg.get('collect_only'): say('collect only: no new Kaggle run, waiting for the last one'); t_push = time.time() - 3600  # poller ran out
else: say(kg('kernels', 'push', '-p', 'kern')[1][-400:]); t_push = time.time()
seen, last, s = False, '', ''
while time.time() - T0 < 280 * 60:
    time.sleep(60); rc, s = kg('kernels', 'status', KID, check=False); s = s.lower()
    if rc: say('status check failed:', s[-150:]); continue
    if s != last: say('status:', s[-100:]); last = s
    if 'running' in s or 'queued' in s: seen = True
    if 'error' in s or 'cancel' in s: break
    if 'complete' in s and (seen or time.time() - t_push > 900): break
os.makedirs(OUT, exist_ok=True); kg('kernels', 'output', KID, '-p', OUT, '-o', check=False)
for f in [f'{OUT}/objects.json'] + glob.glob(f'{OUT}/*.log'):
    if os.path.exists(f): print(f'==== {os.path.basename(f)}\n' + ''.join(open(f, errors='ignore').readlines()[-150:]), flush=True)
dst = f'objects/{STAMP}'
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}')
if not os.path.exists(f'{OUT}/objects.json'): sys.exit(f'Kaggle job ended with: {s[-200:]} (no objects.json)')
say('done')
