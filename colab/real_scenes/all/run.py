# Kaggle job anasrazy/namma-all (2x T4, ~75 min): ONE command from a room video to a checked, levelled 3D model.
#   1 video: a real room filmed handheld (default: Apple ARKitScenes raw iPad video, CC BY-NC-SA 4.0, no signup), or a video
#     attached as a Kaggle dataset (SOURCE = 'kaggle');
#   2 frames -> Depth Anything 3 depth on every frame -> SfM cameras (hloc) -> fused cloud -> gsplat build   (final/run.py prep)
#   3 camera check with real pixels (drop misplaced stretches)                                              (ibr_check --prune)
#   4 align walk passes (video bridges, whole-group check, ICP, split at camera jumps)                     (align_passes)
#   5 largest agreeing group -> floor plan, photo viewer, two splat fits (no needles), levelled .splat     (build_model)
# Output (/kaggle/working): model/*.splat + held-out renders + logs, viewer/, align_report.json, summary.json.
import csv, glob, io, json, os, subprocess, sys, time, urllib.request
SOURCE = 'arkit'          # 'arkit' | 'kaggle'
ARKIT_ID = ''             # empty = pick a Validation video automatically (upright, medium length)
NFRAMES = 300
T0 = time.time(); W = '/kaggle/temp/all'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
def sh(c, check=False, env=None):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash', env=env)
    if check and r: raise RuntimeError(f'failed: {c[:100]}')
sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F)

# 1 video
if SOURCE == 'arkit':
    base = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw'
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(f'{base}/metadata.csv', timeout=60).read().decode())))
    print('ARKitScenes raw videos:', len(rows), '| columns:', list(rows[0]), flush=True)
    cand = [r for r in rows if r.get('fold', '') == 'Validation' and r.get('sky_direction', 'Up') == 'Up']
    if ARKIT_ID: cand = [r for r in rows if r['video_id'] == ARKIT_ID]
    sized = []
    for r in cand[:40]:                                                              # size of the .mov ~ length of the scan
        url = f"{base}/{r['fold']}/{r['video_id']}/{r['video_id']}.mov"
        try: sized.append((int(urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30).headers['Content-Length']), url, r['video_id']))
        except Exception as e: print('no mov for', r['video_id'], e)
    ok = sorted([s for s in sized if 150e6 <= s[0] <= 900e6] or sized)
    size, url, vid = ok[len(ok) // 2] if not ARKIT_ID else ok[0]
    VIDEO = f'{W}/{vid}.mov'; print(f'chosen ARKitScenes video {vid}: {size/1e6:.0f} MB', flush=True)
    sh(f'curl -sS --fail -o {VIDEO} {url}', check=True)
else:
    v = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mov', recursive=True) + glob.glob('/kaggle/input/**/*.MOV', recursive=True))
    assert v, 'attach the room video as a Kaggle dataset'; VIDEO = v[0]; vid = os.path.basename(VIDEO)
print('VIDEO', VIDEO, f'{os.path.getsize(VIDEO)/1e6:.0f} MB', flush=True)

# 2 prep: frames, depth on every frame, SfM cameras, fused cloud, gsplat build -> /kaggle/working/cache
sh(f'{sys.executable} {F}/run.py', check=True, env={**os.environ, 'MODE': 'prep', 'VIDEO': VIDEO, 'NFRAMES': str(NFRAMES), 'DA3_EVERY': '1'})
cache = f'{O}/cache'; assert os.path.exists(f'{cache}/da3_sfm.npz'), 'prep produced no cameras'

# 3 camera check, 4 pass alignment (CPU)
sh(f'cd {F} && {sys.executable} ibr_check.py {cache} {W}/check --workers 4 --prune 6 2>&1 | grep -v Warning', check=True)
sh(f'cd {F} && {sys.executable} align_passes.py {cache} {W}/check/good_frames.json {W}/al --workers 4 --bridge {cache} 2>&1 | grep -v Warning', check=True)
sh(f'cp {W}/al/align_report.json {W}/check/pose_check.png {O}/ 2>/dev/null; true')

# 5 model
import build_model, numpy as np
build_model.build(f'{W}/al/da3_sfm.npz', f'{cache}/frames', f'{cache}/gsplat_ext.tgz', O)
z = np.load(f'{W}/al/da3_sfm.npz'); grp = z['group']
summary = dict(video=vid, frames_extracted=len(glob.glob(f'{cache}/frames/*.jpg')), frames_kept_by_check=int(len(grp)),
               frames_in_model=int((grp == 0).sum()), groups=int(grp.max() + 1), minutes=round((time.time() - T0) / 60))
json.dump(summary, open(f'{O}/summary.json', 'w'), indent=1); print('SUMMARY', json.dumps(summary), flush=True)
