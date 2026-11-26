# Kaggle job anasrazy/namma-v3 (2x T4; 1.3 h for a 2-min room, ~5 h for a house, never past Kaggle's 12 h: final/budget.py):
# video in -> photoreal 3D model out, built only from the standard tools that
# Polycam / Postshot / nerfstudio users get good rooms with. Our own trainer and repair layers are out of the default path.
#   1 frames   video: frames follow the camera's motion (more in turns, fewer in pauses), the sharpest of each stretch
#              (frames.from_video_smart; FRAME_PICK=time = the old equal time windows); frame folder (DL3DV): evenly spaced
#   2 cameras  hloc ALIKED + LightGlue -> COLMAP model (one shared SIMPLE_RADIAL camera) + sfm_report.json, with a time
#              limit: out of time -> fewer frames; broken apart -> more frames where it broke, once
#   3 splats   gsplat's own examples/simple_trainer.py: MCMC densification, bilateral grid (phone auto-exposure), SfM depth loss,
#              held-out eval on every 8th frame. GPU 1 trains a 7k-step preview while GPU 0 trains the full 30k-step model
#   time       every stage has a share of the session (final/budget.py -> budget.json): the splat recipe is the best one that
#              fits (run 4's, else fewer steps / pixels / frames, never nothing), the trainer stops itself at its deadline
#              and keeps what it has (final/budget_patch.py), a snapshot of the model goes to HF at every checkpoint
#   4 QA       per model: qa_sheet.jpg (held-out real | model, 50 cm off the walk, outside, top-down), metrics.json (pass/fail),
#              levelled model.splat (SuperSplat), model.ply (full colour detail), walkthrough.mp4 (trainer's fly-through: colour | depth)
# SOURCE  dl3dv:auto | dl3dv:<scene hash>   handheld scene from HF DL3DV/DL3DV-Benchmark (Kaggle secret HF_TOKEN; "Agree" on HF once)
#         kaggle:<dataset name>             video or frames (or a .zip of them) in that attached Kaggle dataset
#         kaggle                            first .mp4/.mov under /kaggle/input (attach the video as a Kaggle dataset)
#         <path>                            a video file or a folder of frames
# Usually started from GitHub (.github/workflows/namma-v3.yml -> gh_run.py), which fetches DL3DV itself and attaches it as a dataset.
# With a write HF_TOKEN each result is also uploaded as soon as it exists to the private HF dataset <user>/namma-v3-out, so the
# preview can be checked while the full model still trains.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time(); W = '/kaggle/temp/v3'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
SOURCE = os.environ.get('SOURCE', 'dl3dv:auto')
NF = os.environ.get('NFRAMES', '300'); NFRAMES, LONG = (int(NF) if NF.isdigit() else 300), int(os.environ.get('LONG', 1280))   # what a T4 trains in ~1 h
PREP_ONLY = os.environ.get('PREP_ONLY') == '1'                                            # whole-house pre-check: frames, cameras, gate; no training
FULL, PREVIEW = int(os.environ.get('STEPS', 30000)), int(os.environ.get('PREVIEW', 7000))
CAP_FULL, CAP_PREVIEW = 1_500_000, 1_000_000                                               # MCMC splat budget
# models to train, [name, steps, splat cap(, '2dgs')]; the first two run at once on the two T4s, the rest after. 'full' is the
# photoreal deliverable; a '2dgs' model (flat surfels, normal + distortion losses) is the geometry source for the mesh.
VARIANTS = json.loads(os.environ.get('VARIANTS', json.dumps([['preview', PREVIEW, CAP_PREVIEW], ['full', FULL, CAP_FULL]])))
RUN = time.strftime('%Y%m%d-%H%M')
D = f'{W}/data'                                                                            # COLMAP layout: images/ + sparse/0

def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r

def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

# no GPU = hours of CPU work that can never finish (30 Oct: a new account's run got no accelerator and ran 6 h on CPU): stop now
_g = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True, capture_output=True, text=True)
say('GPUs:', _g.stdout.strip().replace('\n', ' | ') or 'none')
if _g.returncode or not _g.stdout.strip():
    raise SystemExit('NO GPU in this Kaggle session (account not phone-verified, or no GPU quota left): stopping before any work')
REF = os.environ.get('NAMMA_REF', 'claude/wonderful-cori-i3fn41')                           # a test run can take the code of another branch
sh(f'git clone -q --depth 1 -b {REF} https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F)
import budget
CLK = budget.Clock(T0, path=f'{O}/budget.json'); say(f'session budget: done by {CLK.deadline_h:.2f} h (Kaggle stops at {budget.HARD_H:.0f} h)')
MESH = os.environ.get('NO_MESH') != '1'
sh('ls /kaggle/input /kaggle/input/* | head -30')
EX = f'{W}/gsplat/examples'

# training dependencies install in the background while the cameras are solved (system Python; the SfM venv is separate because
# simple_trainer needs a pycolmap fork that clashes with the official pycolmap hloc uses)
PKGS = 'ninja gsplat==1.5.3 viser tyro "torchmetrics[image]" tensorboard tensorly scikit-learn "imageio[ffmpeg]" pyyaml matplotlib splines plyfile scikit-image trimesh'
deps = (f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat; '
        # a regular package, else Kaggle's preinstalled Hugging Face `datasets` hides examples/datasets (run 1 died on this)
        f'touch {W}/gsplat/examples/datasets/__init__.py; '
        f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; '
        f'pip install -q -c {W}/pin.txt {PKGS} || pip install -q {PKGS}; '
        'pip install -q --no-deps "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" '
        '"git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
        # the fork predates numpy 2: np.uint64(-1) overflows there
        # (located without importing it, since the import itself is what overflows)
        'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py; '
        'TORCH_CUDA_ARCH_LIST=7.5 pip install -q --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" '
        '|| echo "fused-ssim did not build"')
if os.environ.get('FAST_DEPS') == '1':      # fast recipe (26 Nov): the fused colour grid (MIT; the plain grid's maths in one CUDA kernel)
    deps += ('; TORCH_CUDA_ARCH_LIST=7.5 pip install -q --no-build-isolation '
             '"git+https://github.com/harry7557558/fused-bilagrid@3bd774b644c3194225a5a0ddf46157092511ba3d" || echo "fused-bilagrid did not build"')
if os.environ.get('FAST') == '1' and not glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True):
    # no stored build of gsplat's CUDA code (speed run 1: 322 s of compiling AFTER the camera solve): compile it here, behind the
    # installs, while the frames are read and the cameras solved. The same call further down then takes seconds.
    deps += f'; python {F}/gsplat_warm.py'
dep_proc = subprocess.Popen(deps, shell=True, executable='/bin/bash', stdout=open(f'{W}/deps.log', 'w'), stderr=subprocess.STDOUT)
# FAST=1 (preset 'fast', plan of 26 Nov): a room in minutes. At most FAST_FRAMES frames for a 5-minute walk, one short
# training with the fast recipe (final/fast_patch.py), a mesh from 100 cameras, mesh and object layer side by side.
FAST = os.environ.get('FAST') == '1'
if FAST: os.environ.setdefault('MESH_CAMS', '100'); os.environ.setdefault('MESH_VOXELS', '384')
E = f'{W}/sfmenv/bin/python'; P = f'--python {E}'                                           # the camera solve's own Python (hloc)
SFM_SETUP = (f'pip -q install uv 2>&1 | tail -1; unset UV_SYSTEM_PYTHON UV_PYTHON PIP_TARGET; uv venv -q -p 3.10 {W}/sfmenv && '
             f'uv pip install -q {P} torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 > {W}/sfm_pip.log 2>&1; '
             f'git clone -q --recursive https://github.com/cvg/Hierarchical-Localization {W}/hloc && '
             f'uv pip install -q {P} "numpy<2" -e {W}/hloc "lightglue @ git+https://github.com/cvg/LightGlue" >> {W}/sfm_pip.log 2>&1; tail -3 {W}/sfm_pip.log')
SFM_ENV = None                                   # FAST: installed while the frames are read (it was ~8 min of waiting before each solve)
if FAST and os.environ.get('REUSE') != '1':
    SFM_ENV = subprocess.Popen(SFM_SETUP, shell=True, executable='/bin/bash', stdout=open(f'{W}/sfm_env.log', 'w'), stderr=subprocess.STDOUT)

def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient; return UserSecretsClient().get_secret('HF_TOKEN')
    except Exception as e: say('Kaggle secret HF_TOKEN not available:', type(e).__name__, str(e)[:120])
    return os.environ.get('HF_TOKEN')

UP = None                                                                                  # (HfApi, repo) once uploads work
def hf_setup(tok):
    global UP
    if not tok: return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"
        api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True); UP = (api, repo); say(f'results also go to hf.co/datasets/{repo}/{RUN}')
    except Exception as e: say('results stay on Kaggle only (the token cannot create a repo):', str(e)[:150])

def push(path, name):
    if not UP: return
    try:
        if os.path.isdir(path): UP[0].upload_folder(folder_path=path, path_in_repo=f'{RUN}/{name}', repo_id=UP[1], repo_type='dataset', ignore_patterns=['*.ply'])
        else: UP[0].upload_file(path_or_fileobj=path, path_in_repo=f'{RUN}/{name}', repo_id=UP[1], repo_type='dataset')
        say('uploaded', name)
    except Exception as e: say('upload failed:', str(e)[:150])

sys.path.insert(0, f'{W}/repo/colab/real_scenes/v3'); import dl3dv

def source():
    if SOURCE.startswith('dl3dv:'):
        tok = hf_token()
        assert tok, ('DL3DV needs a Hugging Face token: on huggingface.co open datasets/DL3DV/DL3DV-Benchmark and click Agree, create a token '
                     '(huggingface.co/settings/tokens), then in this Kaggle notebook: Add-ons -> Secrets -> add HF_TOKEN and attach it')
        hf_setup(tok); scene, folder = dl3dv.fetch(SOURCE[6:], f'{W}/dl3dv', tok)
        json.dump(dict(source=f'{dl3dv.REPO}/{scene}', frames='nerfstudio/images_4 (960 x 540)'), open(f'{O}/source.json', 'w'))
        return folder
    hf_setup(hf_token())                         # also for kaggle: sources (every run started from GitHub): without it nothing was
    if SOURCE.startswith('kaggle:'): return attached(SOURCE[7:])   # sent to HF while the job ran, the snapshots included (test 1, 16 Nov)
    if SOURCE == 'kaggle':
        v = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mov', recursive=True)
                   + glob.glob('/kaggle/input/**/*.MOV', recursive=True))
        assert v, 'SOURCE=kaggle but no .mp4/.mov under /kaggle/input: attach the video as a dataset'
        json.dump(dict(source=v[0]), open(f'{O}/source.json', 'w')); return v[0]
    return SOURCE

def attached(name):
    """Video, frame folder or .zip of frames inside the attached Kaggle dataset `name` (its mount path differs between Kaggle layouts)."""
    roots = [d for d in glob.glob('/kaggle/input/**/', recursive=True) if os.path.basename(d.rstrip('/')) == name]
    assert roots, f'dataset {name} is not attached (see ls /kaggle/input above)'
    root = roots[0]; files = glob.glob(f'{root}/**/*', recursive=True)
    vid = sorted(f for f in files if f.lower().endswith(('.mp4', '.mov')))
    if vid: return vid[0]
    for z in [f for f in files if f.lower().endswith('.zip')]: shutil.unpack_archive(z, f'{W}/unzipped')
    if os.path.exists(f'{W}/unzipped'): files = glob.glob(f'{W}/unzipped/**/*', recursive=True)
    count = {}
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')): count[os.path.dirname(f)] = count.get(os.path.dirname(f), 0) + 1
    assert count, f'no video or frames in dataset {name}'
    return max(count, key=count.get)                                                       # the folder with the most frames

# 1 frames
import frames
REUSE = os.environ.get('REUSE') == '1'      # plan of 31 Oct: an earlier run's frames + camera model (its Kaggle output, attached by
                                            # gh_run as a kernel source): no decode, no camera solve; THIN = keep every k-th frame
R0 = None

def take(k):
    """The reused camera model thinned to every k-th frame (final/thin_model.py, COLMAP's own file format) + those frames."""
    shutil.rmtree(f'{W}/sfm/sfm', ignore_errors=True); shutil.rmtree(f'{D}/images', ignore_errors=True); os.makedirs(f'{W}/sfm', exist_ok=True)
    r = subprocess.run([sys.executable, f'{F}/thin_model.py', f'{R0}/sparse/0', f'{W}/sfm/sfm', str(k), f'{W}/kept.txt'], capture_output=True, text=True)
    assert r.returncode == 0, 'thin_model failed: ' + (r.stderr or r.stdout)[-800:]
    info = json.loads(r.stdout.strip().splitlines()[-1]); os.makedirs(f'{D}/images', exist_ok=True)
    for nm in open(f'{W}/kept.txt').read().split(): shutil.copy(f'{R0}/images/{nm}', f'{D}/images/{nm}')
    rep_ = dict(frames=info['images'], registered=info['images'], points=info['points'], models=1, frames_without_points=info['frames_without_points'],
                reused=R0, thin=k, images_in=info['images_in'], points_in=info['points_in'])
    json.dump(rep_, open(f'{W}/sfm/sfm_report.json', 'w')); say('reused cameras, every', k, 'th frame:', rep_); return rep_

if REUSE:
    hits = glob.glob('/kaggle/input/**/sparse/0/images.bin', recursive=True); assert hits, 'REUSE: no sparse/0/images.bin under /kaggle/input'
    R0 = os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))); assert os.path.isdir(f'{R0}/images'), f'REUSE: no images/ in {R0}'
    t_fr = t_sfm = time.time(); VTOT = 0; frames_min = sfm_min = 0.0; n = take(int(os.environ.get('THIN', '1')))['frames']
else:
    src = source(); say('source', src)
    VTOT = 0

    def plan_frames(v):
        """NFRAMES=auto: run 4's density (411 frames from 7398 video frames, one per 18) for any video length."""
        global NFRAMES, VTOT
        import cv2; cap_ = cv2.VideoCapture(v); VTOT = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap_.get(cv2.CAP_PROP_FPS) or 30
        # run 4's density by walking time (411 frames in 123 s), never more than the camera solve handles in its proven range
        # (31 Oct: 3,470 frames took 7.6 h of SfM and broke into 4 pieces; runs 4 / 6 solved 411 / 900 in minutes)
        # and never more than the solve is predicted to finish in its share of the session (budget.sfm_minutes)
        global FPS, CAP_N
        FPS = fps; CAP_N = min(int(os.environ.get('SFM_MAX', budget.SFM_MAX)), budget.sfm_max_frames(budget.sfm_budget_min(CLK.left_h())))
        if FAST and VTOT > 0: CAP_N = min(CAP_N, budget.fast_frame_cap(VTOT / fps, int(os.environ.get('FAST_FRAMES', 300))))
        if NF.startswith('auto') and VTOT > 0: NFRAMES = budget.frame_count(VTOT / fps, CAP_N)
        say('video frames', VTOT, '-> frames to extract', NFRAMES, f'(camera solve: at most {CAP_N} frames)')

    t_fr = time.time(); FPS = 30.0; CAP_N = NFRAMES; PICK = None; CAND = f'{W}/cand'
    SMART = NF.startswith('auto') and os.environ.get('FRAME_PICK', 'smart') == 'smart'
    if not os.path.isdir(src): plan_frames(src)
    if not os.path.isdir(src) and os.environ.get('FIX_BANDS') != '0':                          # a squeezed picture with smeared edges
        try:                                                                                   # (frames.find_bands): repaired in every reader
            if frames.find_bands(src): say('smeared edges in the video:', frames.BANDS['bands'], '-> each frame cut to', frames.BANDS['crop'], 'and resized to', frames.BANDS['out'])
        except Exception as e: say('edge check failed:', str(e)[:160])
    if os.path.isdir(src): n = frames.from_images(src, f'{D}/images', NFRAMES, LONG)
    else:                                                                                      # ffmpeg on all cores; OpenCV if that fails
        n = 0
        if SMART:                                                                              # frames by motion; any trouble: by time
            try: n, PICK = frames.from_video_smart(src, f'{D}/images', CAP_N, LONG, VTOT, FPS, cand=CAND)
            except Exception as e: say('motion-aware frame reader failed:', str(e)[:200])
            if n < 20: shutil.rmtree(f'{D}/images', ignore_errors=True); PICK = None; n = 0
            else: say('frames by motion:', {k: v for k, v in PICK.items() if k != 'kept'})
        if n < 20:
            try: n = frames.from_video_fast(src, f'{D}/images', NFRAMES, LONG, VTOT)
            except Exception as e: say('fast frame reader failed:', str(e)[:200]); n = 0
        if n < 20: shutil.rmtree(f'{D}/images', ignore_errors=True); n = frames.from_video(src, f'{D}/images', NFRAMES, LONG)
    if n < 20 and not os.path.isdir(src):                                                       # e.g. 4K HEVC OpenCV cannot decode:
        say('only', n, 'frames decoded; converting the video with ffmpeg first')                 # H.264 at the training size
        sh(f'ffmpeg -v error -y -i "{src}" -vf "scale=\'if(gt(iw,ih),{LONG},-2)\':\'if(gt(iw,ih),-2,{LONG})\'" -c:v libx264 -crf 14 -preset veryfast {W}/video.mp4')
        plan_frames(f'{W}/video.mp4'); shutil.rmtree(f'{D}/images', ignore_errors=True); n = frames.from_video(f'{W}/video.mp4', f'{D}/images', NFRAMES, LONG)
    say(n, 'frames ->', f'{D}/images'); assert n >= 20, f'only {n} frames from {src}'
    t_sfm = time.time(); frames_min = round((t_sfm - t_fr) / 60, 1)
    CLK.mark('frames', t_fr, frames=n, video_frames=VTOT, pick={k: v for k, v in (PICK or dict(mode='time')).items() if k != 'kept'}, bands=frames.BANDS)

    # 2 cameras (same hloc venv that solved 574/578 frames of the 22 Oct video)
    if SFM_ENV is not None: say('camera-solve environment, installed while the frames were read: exit', SFM_ENV.wait()); sh(f'tail -3 {W}/sfm_env.log')
    else: sh(SFM_SETUP)
    sh(f'{E} -c "import hloc, lightglue, pycolmap; print(\'hloc ok, pycolmap\', pycolmap.__version__)"', check=True)
    def solve(limit_min):
        """One camera solve with a time limit (30 Oct: 3,470 frames ran 7.6 h with nothing to stop it). hloc keeps its features
        and matches in {W}/sfm, so a second solve only works on what is new. True if a COLMAP model came out."""
        shutil.rmtree(f'{W}/sfm/sfm', ignore_errors=True)
        for x in ('sfm_report.json', 'sfm.npz'):
            if os.path.exists(f'{W}/sfm/{x}'): os.remove(f'{W}/sfm/{x}')
        # live in the Kaggle log (a `| tail` here held every line back until the end: the 30 Oct house run looked stuck for hours)
        sh(f'cd {W} && PYTHONUNBUFFERED=1 timeout -k 30 {int(limit_min * 60)} {E} {F}/sfm_hloc.py {D}/images {W}/sfm 10 15 SIMPLE_RADIAL 2>&1 '
           f'| tee -a {W}/sfm.log | grep --line-buffered -v -E "it/s|s/it" | while IFS= read -r l; do echo "[$(date -u +%H:%M:%S)] $l"; done')
        return os.path.exists(f'{W}/sfm/sfm/images.bin') and os.path.exists(f'{W}/sfm/sfm_report.json')

    count = lambda: len(glob.glob(f'{D}/images/*.jpg'))
    lim = min(budget.sfm_timeout_min(n), max(30.0, CLK.left_h() * 60 * 0.4)); say(f'camera solve: {n} frames, predicted {budget.sfm_minutes(n):.0f} min, limit {lim:.0f} min')
    ok = solve(lim); sfm_tries = [dict(frames=n, ok=ok)]
    if not ok and n > 80:                                                                        # out of time: 60 % of the frames
        n = frames.thin_dir(f'{D}/images', int(n * 0.6)); lim = min(budget.sfm_timeout_min(n), max(30.0, CLK.left_h() * 60 * 0.4))
        say(f'camera solve gave no model in its time: again with {n} frames, limit {lim:.0f} min'); ok = solve(lim); sfm_tries.append(dict(frames=n, ok=ok))
    assert ok, 'camera solve failed: no COLMAP model (see the hloc log above)'
    r1 = json.load(open(f'{W}/sfm/sfm_report.json'))
    if (PICK and (r1['registered'] < 0.85 * r1['frames'] or r1['models'] > 1) and os.path.isdir(CAND)
            and budget.sfm_minutes(n * 1.3) * 2.2 < CLK.left_h() * 60 * 0.25):                   # broke apart, and there is time:
        import numpy as np                                                                       # more frames where it lost the video
        reg = set(str(x) for x in np.load(f'{W}/sfm/sfm.npz', allow_pickle=True)['names'])
        lost = sorted(set(os.path.basename(f) for f in glob.glob(f'{D}/images/*.jpg')) - reg)
        room = int(CAP_N * 1.2) - n; added = frames.add_candidates(CAND, f'{D}/images', lost, around=2, most=room) if room > 0 else []
        if added:
            shutil.copytree(f'{W}/sfm/sfm', f'{W}/sfm_first', dirs_exist_ok=True)
            for x in ('sfm_report.json', 'sfm.npz'): shutil.copy(f'{W}/sfm/{x}', f'{W}/sfm_first_{x}')
            say(f"camera solve lost {len(lost)} of {r1['frames']} frames ({r1['models']} models): {len(added)} more frames around them, solving again")
            ok2 = solve(min(budget.sfm_timeout_min(count()), max(30.0, CLK.left_h() * 60 * 0.3)))
            r2 = json.load(open(f'{W}/sfm/sfm_report.json')) if ok2 else dict(registered=0)
            sfm_tries.append(dict(frames=count(), ok=ok2, added=len(added), registered=r2['registered'], was=r1['registered']))
            if r2['registered'] <= r1['registered']:                                             # no better: the first solve, the first frames
                say('the denser solve is no better: keeping the first one'); shutil.rmtree(f'{W}/sfm/sfm', ignore_errors=True)
                shutil.copytree(f'{W}/sfm_first', f'{W}/sfm/sfm')
                for x in ('sfm_report.json', 'sfm.npz'): shutil.copy(f'{W}/sfm_first_{x}', f'{W}/sfm/{x}')
                for a in added: os.remove(f'{D}/images/{a}')
            n = count()
    shutil.rmtree(CAND, ignore_errors=True)
    sfm_min = round((time.time() - t_sfm) / 60, 1); CLK.mark('cameras', t_sfm, tries=sfm_tries, predicted_min=round(budget.sfm_minutes(n), 1))
    try: shutil.copytree(f'{W}/sfm', f'{O}/sfm_work', dirs_exist_ok=True); shutil.copy(f'{W}/sfm.log', O)   # a retry continues from here
    except Exception as e: say('could not keep the SfM work:', str(e)[:150])
rep = json.load(open(f'{W}/sfm/sfm_report.json')); shutil.copy(f'{W}/sfm/sfm_report.json', O); push(f'{O}/sfm_report.json', 'sfm_report.json')
if rep['registered'] < 0.85 * rep['frames'] or rep['models'] > 1:
    say(f"WARNING: only {rep['registered']} of {rep['frames']} frames in the main model ({rep['models']} models): the capture broke "
        "apart (fast turns / blank walls); the model covers only the registered part")
assert rep['registered'] >= 20, 'fewer than 20 frames have cameras: nothing to train on'
DEPTH = '--depth-loss' if not rep.get('frames_without_points') else ''                     # its loader needs points in every frame
os.makedirs(f'{D}/sparse/0', exist_ok=True)
for x in ('cameras.bin', 'images.bin', 'points3D.bin'): shutil.copy(f'{W}/sfm/sfm/{x}', f'{D}/sparse/0/{x}')
# frames + cameras into the job's output NOW, in the layout REUSE reads (images/ + sparse/0): until 16 Nov they were copied
# only after training, so a run that died while training had to decode and solve again (sfm_work has no frames)
try: shutil.copytree(f'{D}/sparse', f'{O}/sparse', dirs_exist_ok=True); shutil.copytree(f'{D}/images', f'{O}/images', dirs_exist_ok=True)
except Exception as e: say('could not keep frames + cameras early:', str(e)[:150])
if not REUSE: R0 = O                                                                       # take(k) thins from the full set

# 3 splats: wait for the dependencies, reuse namma-prep's compiled gsplat if it matches (else it compiles, ~5 min)
dep_proc.wait(); sh(f'grep -v "^\\s*$" {W}/deps.log | tail -8')
tgz = glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True)
if tgz:
    sh(f'mkdir -p ~/.cache && tar xzf {tgz[0]} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'python {F}/gsplat_warm.py', check=True)                                              # compile once, before two GPUs load it
PP = '' if sh('python -c "import fused_ssim"') == 0 else f'{F}/fused_ssim_shim'           # pure-torch SSIM if the CUDA one failed
if PP: say('using the pure-torch SSIM shim')
tried = set()
for _ in range(6):                                                                         # import the trainer once; install what it lacks
    r = subprocess.run(f'cd {EX} && PYTHONPATH={PP}:$PYTHONPATH python -c "import simple_trainer, simple_trainer_2dgs"', shell=True, capture_output=True, text=True)
    if r.returncode == 0: say('simple_trainer imports ok'); break
    print(r.stderr[-1500:], flush=True); import re; m = re.search(r"No module named '([\w.]+)'", r.stderr)
    if not m: break
    pkg = {'cv2': 'opencv-python-headless', 'sklearn': 'scikit-learn', 'yaml': 'pyyaml', 'PIL': 'pillow'}.get(m.group(1).split('.')[0], m.group(1).split('.')[0])
    if pkg in tried: break                                                                  # installing it did not help
    tried.add(pkg); sh(f'pip -q install {pkg} 2>&1 | tail -1')
import room_prior                                                                       # floor/ceiling + needle prior (kind 'prior')
say('room prior patched in:', room_prior.patch(f'{EX}/simple_trainer.py'))
import geo_prior                                                                        # dense depth + normal guide (kind 'geo')
say('geo prior patched in:', geo_prior.patch(f'{EX}/simple_trainer.py'))
import budget_patch                                                                     # the trainer stops itself at NM_DEADLINE
TIMED = budget_patch.patch(f'{EX}/simple_trainer.py'); say('deadline stop patched in:', TIMED)
import fast_patch                                                                       # fast training (kinds fbg, thin<k>, res<s>_<f>, dense, cache)
FASTP = fast_patch.patch(f'{EX}/simple_trainer.py'); say('fast training patched in:', FASTP)
FBG = os.environ.get('FAST_DEPS') == '1' and sh('python -c "from fused_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss"') == 0
if os.environ.get('FAST_DEPS') == '1': say('fused colour grid:', 'installed' if FBG else 'NOT installed (the plain grid is used)')
SPEED = os.environ.get('SPEED') == '1'; DENSE_NPZ = None; BENCH_OUT = {}; TIMES = {}      # SPEED: the speed check (v3/SPEED.json)
if FAST and 'VARIANTS' not in os.environ:        # the fast recipe as the only model: it is done before a preview would be
    VARIANTS = [['full', int(os.environ.get('FAST_STEPS', 15000)), int(os.environ.get('FAST_CAP', 1_000_000)), os.environ.get('FAST_KIND', 'fbg+res0.5_0.6+dense')]]
if any(len(v) > 3 and 'geo' in v[3] for v in VARIANTS):                                  # DA3 depth per frame, before training
    sh(f'cd {EX} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/geo_prior.py {D} {W}/geo 2>&1 | grep -v "it/s" | tail -15')
    if os.path.exists(f'{W}/geo/geo_report.json'): shutil.copy(f'{W}/geo/geo_report.json', O); push(f'{O}/geo_report.json', 'geo_report.json')
    sh('nvidia-smi --query-gpu=memory.used --format=csv')
import torch
NGPU = torch.cuda.device_count(); say('GPUs:', NGPU)

if PREP_ONLY:                                    # plan of 30 Oct: prove a whole-house run fits before any splat training (final/precheck.py)
    tim = dict(frames=n, frames_planned=NFRAMES, video_frames=VTOT, frames_min=frames_min, sfm_min=sfm_min,
               hours_before_precheck=round((time.time() - T0) / 3600, 3))
    sh(f"cd {EX} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/precheck.py {EX} {D} {W} {O} '{json.dumps(tim)}' {PP}")
    shutil.copytree(f'{D}/sparse', f'{O}/sparse', dirs_exist_ok=True); shutil.copytree(f'{D}/images', f'{O}/images', dirs_exist_ok=True)
    for b in glob.glob(f'{W}/bench_*.log'): shutil.copy(b, O)
    for f in ('precheck.json', 'rooms.json', 'rooms.png', 'chunks.json'):
        if os.path.exists(f'{O}/{f}'): push(f'{O}/{f}', f)
    assert os.path.exists(f'{O}/precheck.json'), 'pre-check failed (see its log above)'
    say('PRE-CHECK DONE', json.load(open(f'{O}/precheck.json')).get('decision')); raise SystemExit(0)

if FASTP and (any(len(v) > 3 and 'dense' in v[3] for v in VARIANTS) or os.environ.get('DENSE') == '1'):   # dense start points, once
    t_d = time.time()
    sh(f"cd {EX} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/dense_init.py {D} {W}/dense {int(os.environ.get('DENSE_EVERY', 4))} 2>&1 | grep -v 'it/s' | tail -12")
    if os.path.exists(f'{W}/dense/dense.npz'):
        DENSE_NPZ = f'{W}/dense/dense.npz'; shutil.copy(f'{W}/dense/dense_report.json', O); push(f'{O}/dense_report.json', 'dense_report.json')
    else: say('dense start: no points were made, variants that ask for them start sparse')
    CLK.mark('dense start', t_d, ok=bool(DENSE_NPZ))

if SPEED and FASTP:                              # speed per step of each change alone (precheck.bench: 700 steps, splats grown to
    import math, threading, precheck             # the cap by step 500, s/step over the last 200), two at a time on the two T4s
    precheck.setup(EX, D, W, O, {}, PP); precheck.DEPTH = DEPTH; t_b = time.time()
    def bench_one(b, gpu):
        name, kind, cap = b; two = '2gpu' in kind
        fr = fast_patch.recipe(kind, 30000, FBG, DENSE_NPZ) or dict(env='', flags='', grid='--use-bilateral-grid', notes=[])
        extra = fr['flags'] + (' --packed' if two or 'packed' in kind else '') + (' --visible-adam' if 'vadam' in kind else '')
        try: r = precheck.bench(name, '0,1' if two else str(gpu), math.ceil(cap / 2) if two else cap, extra, env=fr['env'], grid=fr['grid'])
        except Exception as e: r = dict(ok=False, exit=repr(e)[:200])
        r.update(kind=kind, cap=cap, notes=fr['notes'], gpu='0,1' if two else gpu); BENCH_OUT[name] = r
    todo = [b for b in json.loads(os.environ.get('BENCH', '[]')) if '2gpu' not in b[1]]
    while todo:
        batch, todo = todo[:max(1, min(NGPU, 2))], todo[max(1, min(NGPU, 2)):]
        ths = [threading.Thread(target=bench_one, args=(b, g)) for g, b in enumerate(batch)]
        [t.start() for t in ths]; [t.join() for t in ths]
    for b in [b for b in json.loads(os.environ.get('BENCH', '[]')) if '2gpu' in b[1]]:
        if NGPU >= 2: bench_one(b, 0)
    json.dump(BENCH_OUT, open(f'{O}/speed_bench.json', 'w'), indent=1); push(f'{O}/speed_bench.json', 'speed_bench.json')
    for b in glob.glob(f'{W}/bench_*.log'): shutil.copy(b, O)
    CLK.mark('speed checks', t_b, n=len(BENCH_OUT)); say('SPEED CHECKS', json.dumps({k: v.get('s_per_step') for k, v in BENCH_OUT.items()}))

SPS = {}                                                                                   # model name -> predicted s/step
if os.environ.get('AUTO_STEPS') == '1':     # run 4's recipe for any size (~83.6 steps per training view, 1.5 M splats per 37 m2)
    # if it fits the session; else the best rung of budget.fit's ladder that does. Until 16 Nov a video that did not fit ended
    # here with NO-GO and nothing trained, after decode, cameras and the dry run had been paid for.
    import precheck, rooms, cv2
    precheck.setup(EX, D, W, O, {}, PP); precheck.DEPTH = DEPTH; t_plan = time.time()
    nm, C_, R_, P_, ob_ = rooms.load(f'{D}/sparse/0'); area = sum(r_['area_m2'] for r_ in rooms.split(nm, C_, R_, P_, ob_)[1])
    im0 = cv2.imread(sorted(glob.glob(f'{D}/images/*.jpg'))[0]); long0 = int(max(im0.shape[:2])); cap = budget.cap_for(area)
    dry = precheck.bench('dryrun', '0', cap)
    for _ in range(2):                                                                      # too much memory: fewer splats set it
        if not dry['ok'] or max(dry['peak_mem_gb'].values() or [0]) <= budget.MEM_GB: break
        cap = int(cap * 0.7); say('dry run peak memory over', budget.MEM_GB, 'GB: splat cap ->', cap); dry = precheck.bench('dryrun', '0', cap)
    if not dry['ok']: raise SystemExit(f"trainer dry run failed (exit {dry['exit']}, see bench_dryrun.log): nothing trained")
    sps0 = dry['s_per_step'] or 0.3                                                         # no speed read: a slow guess; the trainer's
    left = CLK.train_left_h(mesh=MESH)                                                      # own deadline is the safety either way
    plan = budget.fit(rep['registered'], area, long0, sps0, left, cap); dry2 = None
    if plan['thin'] > 1:
        rep = take(rep.get('thin', 1) * plan['thin']); n = rep['frames']; DEPTH = '--depth-loss' if not rep.get('frames_without_points') else ''; precheck.DEPTH = DEPTH
        for x in ('cameras.bin', 'images.bin', 'points3D.bin'): shutil.copy(f'{W}/sfm/sfm/{x}', f'{D}/sparse/0/{x}')
    if plan['long'] < long0: say('frames resized to', plan['long'], 'px:', frames.resize_dir(f'{D}/images', plan['long']))
    if plan['long'] < long0 or plan['thin'] > 1:                                            # another size: measure it, do not trust the scaling
        dry2 = precheck.bench('dryrun2', '0', plan['cap'])
        if dry2['ok'] and dry2['s_per_step']:
            plan['s_per_step'] = dry2['s_per_step']; left = CLK.train_left_h(mesh=MESH)
            fits = int(left * 3600 / budget.MARGIN / dry2['s_per_step'])
            if fits < plan['steps']: say(f"measured {dry2['s_per_step']} s/step: {fits} of {plan['steps']} steps fit"); plan['steps'] = max(500, fits)
            plan['hours'] = round(plan['steps'] * dry2['s_per_step'] * budget.MARGIN / 3600, 2)
    steps, cap = plan['steps'], plan['cap']; SPS['full'] = plan['s_per_step']
    say(f"PLAN: {n} frames, {plan['views']} train views, {area:.1f} m2, {long0} -> {plan['long']} px, cap {cap}, {steps} steps, "
        f"{plan['s_per_step']} s/step, peak {dry['peak_mem_gb']} GB -> {plan['hours']:.1f} h of {left:.1f} h for training: {plan['why']}")
    json.dump(dict(frames=n, train_views=plan['views'], area_m2=round(area, 1), cap=cap, steps=steps, dry=dry, dry2=dry2, hours=plan['hours'],
                   left=round(left, 2), plan=plan), open(f'{O}/go_nogo.json', 'w'), indent=1)       # the name older tools read
    CLK.mark('plan', t_plan, plan=plan, dry_s_per_step=dry['s_per_step'], peak_mem_gb=dry['peak_mem_gb']); push(f'{O}/budget.json', 'budget.json')
    VARIANTS = [['full', steps, cap]]
    if NGPU >= 2 and plan['kind'] == 'full':                                                # the other T4: a look at the room ~30 min in,
        VARIANTS.append(['preview', min(PREVIEW, steps), min(cap, CAP_PREVIEW)])            # then (CHALLENGER=<kind>) one alternative
        ch = os.environ.get('CHALLENGER', '')                                               # recipe, kept only if it scores better
        if ch: VARIANTS.append(['challenger', steps, cap, ch]); SPS['challenger'] = plan['s_per_step']
    say('training', VARIANTS)

TRAIN_END = CLK.train_deadline(mesh=MESH)                                                  # trainers stop themselves here (NM_DEADLINE)

def start(name, steps, cap, gpu, kind='gs'):
    res = f'{W}/{name}'
    if kind == '2dgs':                                                                    # geometry model: flat surfels
        cmd = (f'cd {EX} && CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={PP}:$PYTHONPATH python simple_trainer_2dgs.py --data-dir {D} --data-factor 1 '
               f'--result-dir {res} --max-steps {steps} --eval-steps {steps} --save-steps {steps} --disable-viewer --normal-loss --dist-loss {DEPTH}')
    else:                                                                                  # 3DGS (+ room prior / pose refinement)
        env = ('ROOM_PRIOR=1 ' if 'prior' in kind else '') + (f'GEO_PRIOR=1 GEO_DIR={W}/geo ' if 'geo' in kind else '')
        pr = f'{env}PYTHONPATH={F}:' if env else 'PYTHONPATH='                              # kinds: gs, prior, pose, geo and + of them
        po = ' --pose-opt' if 'pose' in kind else ''                                           # refine the SfM cameras while training
        ss = ' '.join(str(x) for x in budget.save_steps(steps, SPS.get(name, 0.2)))            # a checkpoint about every 20 min
        # MCMC refreshes splats until step 25k whatever the length: keep run 4's share (25k of 30k) on longer runs (the house's
        # 71.8k steps ran 47k without a refresh). Shorter runs keep the stock setting they were measured with.
        rs = f' --strategy.refine-stop-iter {budget.refine_stop(steps)}' if steps > 30000 else ''
        fr = fast_patch.recipe(kind, steps, FBG, DENSE_NPZ) if FASTP else None
        cmd = (f'cd {EX} && NM_DEADLINE={TRAIN_END:.0f} CUDA_VISIBLE_DEVICES={gpu} {pr}{PP}:$PYTHONPATH python simple_trainer.py mcmc --data-dir {D} '
               f'--data-factor 1 --result-dir {res} --max-steps {steps} --eval-steps {steps} --save-steps {ss} --ply-steps {steps} --save-ply '
               f'--disable-viewer --use-bilateral-grid {DEPTH} --strategy.cap-max {cap}{po}{rs}')
        if fr:                                 # fast recipe: gsplat's 30k-step schedules (colour detail, splat refresh, eval, save)
            N = fast_patch.NOMINAL             # shrunk together to `steps` by --steps-scaler; one save at the end (short runs)
            cmd = (f'cd {EX} && NM_DEADLINE={TRAIN_END:.0f} {fr["env"]} CUDA_VISIBLE_DEVICES={gpu} {pr}{PP}:$PYTHONPATH python simple_trainer.py mcmc --data-dir {D} '
                   f'--data-factor 1 --result-dir {res} --max-steps {N} --eval-steps {N} --save-steps {N} --ply-steps {N} --save-ply '
                   f'--steps-scaler {fr["scaler"]:.8f} --disable-viewer {fr["grid"]} {DEPTH} --strategy.cap-max {cap}{po} {fr["flags"]}' + (' --disable-video' if SPEED or FAST else ''))   # no fly-through video
            if fr['notes']: say(name, 'recipe notes:', fr['notes'])
    say(cmd); return dict(name=name, gpu=gpu, res=res, kind=kind, t=time.time(), steps=steps, snap=-1,
                          p=subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=open(f'{res}.log', 'w'), stderr=subprocess.STDOUT,
                                             start_new_session=True))                      # its own group: stop() ends the python too

def progress(j):
    try: return [x for x in open(j['res'] + '.log', errors='ignore').read()[-3000:].replace('\r', '\n').split('\n') if x.strip()][-1][:150]
    except Exception: return ''

def ckpts(j): return sorted(glob.glob(f"{j['res']}/ckpts/ckpt_*_rank0.pt"), key=lambda f: int(f.split('_')[-2]))

def snapshot(j):
    """The newest checkpoint as a small .splat in the job's output and on HF (snapshot/<model>.splat), older checkpoints
    deleted (350 MB - 1 GB each): if the session dies, the room as trained so far is still there."""
    if j['kind'] == '2dgs': return
    c = [f for f in ckpts(j) if time.time() - os.path.getmtime(f) > 20]                    # not one still being written
    if not c: return
    step = int(c[-1].split('_')[-2])
    if step <= j['snap'] or step >= j['steps'] - 1: return                                  # the last one is finish()'s
    try:
        os.makedirs(f'{O}/snapshot', exist_ok=True); dst = f"{O}/snapshot/{j['name']}.splat"; k = budget.ckpt_to_splat(c[-1], dst)
        json.dump(dict(step=step + 1, of=j['steps'], splats=k, minutes=round((time.time() - j['t']) / 60)), open(f"{O}/snapshot/{j['name']}.json", 'w'))
        say(f"snapshot {j['name']}: step {step + 1} of {j['steps']}, {k} splats"); push(dst, f"snapshot/{j['name']}.splat")
        push(f"{O}/snapshot/{j['name']}.json", f"snapshot/{j['name']}.json")
        for f in c[:-1]: os.remove(f)
    except Exception as e: say('snapshot failed:', str(e)[:150])
    j['snap'] = step

def stop(j, why):
    """A trainer that is still running 10 min after its deadline (the patch did not apply, or it hangs): end it. finish()
    then builds the model from its newest checkpoint."""
    import signal
    say(f"stopping {j['name']}: {why}"); j['stopped'] = why
    try: os.killpg(os.getpgid(j['p'].pid), signal.SIGTERM); j['p'].wait(timeout=60)
    except Exception:
        try: os.killpg(os.getpgid(j['p'].pid), signal.SIGKILL); j['p'].wait(timeout=30)
        except Exception as e: say('could not stop it:', str(e)[:100])

def salvage(j):
    """A trainer that ended without its .ply (stopped, crashed, out of memory late): the .ply from its newest checkpoint,
    so QA, the tour and the mesh still get a model. Its held-out numbers are missing (the trainer's own eval did not run)."""
    if j['kind'] == '2dgs' or glob.glob(f"{j['res']}/ply/point_cloud_*.ply"): return
    c = ckpts(j)
    if not c: return
    try:
        step = int(c[-1].split('_')[-2]); os.makedirs(f"{j['res']}/ply", exist_ok=True)
        k = budget.ckpt_to_ply(c[-1], f"{j['res']}/ply/point_cloud_{step}.ply"); j['salvaged'] = step + 1
        say(f"{j['name']}: no .ply from the trainer; built from the checkpoint of step {step + 1} ({k} splats)")
    except Exception as e: say('salvage failed:', str(e)[:150])

def finish(j):
    out = f"{O}/{j['name']}"; os.makedirs(out, exist_ok=True)
    lines = open(j['res'] + '.log', errors='ignore').read().replace('\r', '\n').split('\n')
    keep = [x for x in lines if x.strip() and '%|' not in x]                              # drop the progress bar
    open(f'{out}/train.log', 'w').write('\n'.join(keep)); print('\n'.join(keep[-25:]), flush=True)
    say(f"{j['name']} exit {j['p'].returncode} after {(time.time() - j['t']) / 60:.0f} min"); salvage(j)
    early = [x for x in keep if 'NM_DEADLINE reached' in x]
    CLK.mark('train ' + j['name'], j['t'], steps=j['steps'], exit=j['p'].returncode, stopped_early=(early[-1][:90] if early else j.get('stopped')),
             salvaged_step=j.get('salvaged'), predicted_s_per_step=SPS.get(j['name']))
    if j['kind'] == '2dgs':                                                              # its mesh is made at the end (needs full's level)
        st = sorted(glob.glob(f"{j['res']}/stats/val_step*.json"))
        if st: m = json.load(open(st[-1])); json.dump({k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, open(f'{out}/metrics.json', 'w'))
    else: sh(f"cd {EX} && CUDA_VISIBLE_DEVICES={j['gpu']} python {F}/v3_qa.py {EX} {D} {j['res']} {out}")
    st_ = sorted(glob.glob(f"{j['res']}/stats/train_step*_rank0.json"))                     # the trainer's own clock: training only,
    try: tr = json.load(open(st_[-1])) if st_ else {}                                       # without loading, eval and the fly-through
    except Exception: tr = {}
    TIMES[j['name']] = dict(kind=j['kind'], steps=j['steps'], train_min=round(tr['ellipse_time'] / 60, 2) if tr.get('ellipse_time') else None,
                            wall_min=round((time.time() - j['t']) / 60, 1), splats=tr.get('num_GS'), train_mem_gb=round(tr['mem'], 2) if tr.get('mem') else None)
    if SPEED and os.path.exists(f'{out}/model.ply'): os.remove(f'{out}/model.ply')          # the speed check keeps the small .splat only
    v = sorted(glob.glob(f"{j['res']}/videos/traj_*.mp4"))
    if v: shutil.copy(v[-1], f'{out}/walkthrough.mp4')
    push(out, j['name'])

queue = [tuple(v) for v in VARIANTS]; free = list(range(max(1, min(NGPU, 2))))           # GPUs not training right now
running = [start(*v[:3], free.pop(0), *v[3:]) for v in [queue.pop(0) for _ in range(min(len(queue), len(free)))]]
last = time.time()
while running:
    time.sleep(30)
    for j in [j for j in running if j['p'].poll() is not None]:
        running.remove(j); finish(j)
        if queue and time.time() < TRAIN_END - 900: v = queue.pop(0); running.append(start(*v[:3], j['gpu'], *v[3:]))
        elif queue: say('not started (under 15 min to the training deadline):', [v[0] for v in queue]); queue.clear()
    for j in running:
        snapshot(j)
        if time.time() > TRAIN_END + 600: stop(j, 'still training 10 min after its deadline')
    if time.time() - last > 300: last = time.time(); [say(j['name'], progress(j), f'| {CLK.left_h():.1f} h left') for j in running]

S = {n: json.load(open(f'{O}/{n}/metrics.json')) for n in [v[0] for v in VARIANTS] if os.path.exists(f'{O}/{n}/metrics.json')}
if TIMES:                                        # minutes of pure training per model, beside its held-out scores
    keys = ('heldout_psnr_exposure_fair', 'heldout_ssim_exposure_fair', 'heldout_psnr', 'heldout_ssim', 'heldout_lpips', 'needle_fraction', 'pass')
    runs = {n: dict(t, **{k: S.get(n, {}).get(k) for k in keys}) for n, t in TIMES.items()}
    json.dump(dict(frames=len(os.listdir(f'{D}/images')), fused_grid=FBG, dense=bool(DENSE_NPZ), bench=BENCH_OUT, runs=runs,
                   reference=os.environ.get('SPEED_REF', '')), open(f'{O}/speed_report.json', 'w'), indent=1)
    push(f'{O}/speed_report.json', 'speed_report.json'); say('SPEED REPORT', json.dumps(runs))
BEST = 'full'                                    # CHALLENGER: the alternative replaces full only if it is clearly better on the
if 'challenger' in S and 'full' in S:            # held-out frames (exposure-fair PSNR + 0.2 dB, LPIPS not worse); both stay on HF
    q = lambda m: (m.get('heldout_psnr_exposure_fair') or m.get('heldout_psnr') or 0, -(m.get('heldout_lpips') or 9))
    a, b = q(S['full']), q(S['challenger']); win = b[0] >= a[0] + 0.2 and b[1] >= a[1]
    say(f"challenger ({os.environ.get('CHALLENGER')}): {b[0]} dB vs full {a[0]} dB ->", 'it replaces full' if win else 'full stays')
    S['challenger_won'] = bool(win)
    if win:
        os.rename(f'{O}/full', f'{O}/full_base'); os.rename(f'{O}/challenger', f'{O}/full'); BEST = 'challenger'
        S['full_base'], S['full'] = S['full'], S.pop('challenger')
# what still fits before the deadline: QA is done; the mesh on the finest grid its predicted time allows (budget.mesh_minutes),
# after the object layer's share if that was asked for (else the tour walks on the room box, make_tour.room_collision)
NIMG = len(os.listdir(f'{D}/images'))
MCAM = min(NIMG, int(os.environ.get('MESH_CAMS') or NIMG))                                # cameras the mesh looks through (mesh_tsdf MESH_CAMS)
TL = budget.tail(CLK.left_h() * 60, mesh=MESH, objects=os.environ.get('OBJECTS') == '1', frames=MCAM, voxels=int(os.environ.get('MESH_VOXELS', 512)))
say(f'{CLK.left_h() * 60:.0f} min left, {NIMG} frames ->', TL)
# the mould: a solid mesh in the same coordinates as full/model.splat, from the 2DGS model if one was trained, else the full model
geo = next(([v[0], '2dgs'] for v in VARIANTS if len(v) > 3 and v[3] == '2dgs'), [BEST, '3dgs'])
def do_mesh():
    t_mesh = time.time()
    sh(f"cd {EX} && timeout -k 30 {int(TL['mesh_timeout_min'] * 60)} python {F}/mesh_tsdf.py {EX} {D} {W}/{geo[0]} {O}/mesh {O}/full/level.json {geo[1]} {TL['mesh_voxels']}")
    if os.path.exists(f'{O}/mesh/mesh.json'): S['mesh'] = json.load(open(f'{O}/mesh/mesh.json'))
    CLK.mark('mesh', t_mesh, ok='mesh' in S, voxels=TL['mesh_voxels'], cameras=MCAM, limit_min=TL['mesh_timeout_min'], predicted_min=round(budget.mesh_minutes(MCAM, TL['mesh_voxels']), 1))
MESH_TH = None                                   # FAST with two T4s: the mesh runs on GPU 0 while the object layer runs on GPU 1
if os.path.isdir(f'{W}/{geo[0]}/ckpts') and TL['mesh']:
    if FAST and NGPU >= 2 and TL['objects']:
        import threading; MESH_TH = threading.Thread(target=do_mesh); MESH_TH.start()
    else: do_mesh()
elif MESH: say('mesh skipped: not enough time left (the tour uses the room box for walking)'); CLK.note(mesh_skipped=True)
# the object layer + the search index for the tour's "find anything" (final/objects3d.py, ai_find.region_index; 14.5 min on a
# T4, 27 Oct), only with OBJECTS=1 and 33 min still left, with a time limit, and never failing the run: the tour works without
if os.environ.get('OBJECTS') == '1' and os.path.exists(f'{O}/full/model.ply') and os.path.exists(f'{O}/full/level.json'):
    if budget.tail(CLK.left_h() * 60, mesh=False, objects=True)['objects']:
        t_ob = time.time(); lim = int((min(budget.TAIL_MIN['objects'] + 15, CLK.left_h() * 60 - budget.TAIL_MIN['base'])) * 60); OB = f'{W}/objects'
        sh(f'pip install -q -c {W}/pin.txt plyfile open3d scipy accelerate 2>&1 | tail -2; '
           f'SAM2_BUILD_CUDA=0 pip install -q -c {W}/pin.txt "git+https://github.com/facebookresearch/sam2.git" 2>&1 | tail -2')
        g_ob = 'CUDA_VISIBLE_DEVICES=1 ' if MESH_TH is not None else ''
        r_ob = sh(f"cd {EX} && {g_ob}PYTHONPATH={EX}:{F}:$PYTHONPATH timeout -k 30 {lim} python {F}/objects3d.py {D} {O}/full/model.ply {O}/full/level.json 0 {OB} "
                  f"{int(os.environ.get('OBJECT_FRAMES', 40))} 2>&1 | grep -v 'it/s' | tail -60")
        if os.path.exists(f'{OB}/objects.json'):                                            # not its tinted full/model.ply
            for x in ('objects.json', 'objects_sheet.jpg', 'regions.json', 'regions.f16'):
                if os.path.exists(f'{OB}/{x}'): shutil.copy(f'{OB}/{x}', f'{O}/{x}'); push(f'{O}/{x}', x)
            shutil.copytree(f'{OB}/crops', f'{O}/crops', dirs_exist_ok=True); S['objects'] = len(json.load(open(f'{O}/objects.json'))['objects'])
        CLK.mark('objects', t_ob, ok=os.path.exists(f'{O}/objects.json'), objects=S.get('objects'), regions=os.path.exists(f'{O}/regions.f16'))
    else: say('object layer skipped: under 33 min left'); CLK.note(objects_skipped=True)
if MESH_TH is not None: MESH_TH.join()
shutil.copytree(f'{D}/sparse', f'{O}/sparse', dirs_exist_ok=True)                        # cameras, so QA can be redone later
shutil.copytree(f'{D}/images', f'{O}/images', dirs_exist_ok=True)                        # frames, for the tour (cameras need them)
json.dump(dict(source=SOURCE, sfm=rep, models=S, minutes=round((time.time() - T0) / 60)), open(f'{O}/v3_summary.json', 'w'), indent=1)
push(f'{O}/v3_summary.json', 'v3_summary.json')
CLK.note(done=True, minutes=round((time.time() - T0) / 60)); push(f'{O}/budget.json', 'budget.json')
say(f'time: {CLK.elapsed_h():.2f} h used, {CLK.left_h():.2f} h before the deadline, {budget.HARD_H - CLK.elapsed_h():.2f} h before Kaggle stops the session')
say('V3 DONE', json.dumps(S)); sh(f'ls -la {O} {O}/*/ | head -40')
if 'full' not in S and not SPEED: raise SystemExit('V3 FAILED: no full model (see the trainer logs above)')
