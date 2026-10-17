# Copy images, poses and point sets from the prep work dir into package/data in the layout the handout scripts read.
# Usage: python assemble_data.py <work> <package dir>
import sys, os, json, shutil
W, P = sys.argv[1], sys.argv[2]; D = f'{P}/data'
shutil.rmtree(D, ignore_errors=True)
K = [[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1]]
clean = json.load(open(f'{W}/clean/stats.json'))
for track, src in [('clean', 'color_raw'), ('degraded', 'degraded')]:
    st = json.load(open(f'{W}/{track}/stats.json')); poses = json.load(open(f'{W}/{track}/poses_w2c.json'))
    os.makedirs(f'{D}/images/{track}', exist_ok=True)
    for f in st['train_frames']: shutil.copy(f'{W}/data/{src}/{f:05d}.jpg', f'{D}/images/{track}/{f:05d}.jpg')
    for f in clean['test_frames']:  # ground truth for scoring is always the clean frame
        shutil.copy(f'{W}/data/color_raw/{f:05d}.jpg', f'{D}/images/clean/{f:05d}.jpg')
    cams = dict(width=640, height=480, K=K, note='w2c = world-to-camera 4x4 (OpenCV axes), metres, ICL-NUIM ground-truth world frame (Sim3-aligned SfM poses)',
                train=[dict(frame=f, image=f'images/{track}/{f:05d}.jpg', w2c=poses[str(f)]) for f in st['train_frames']],
                test=[dict(frame=f, image=f'images/clean/{f:05d}.jpg', w2c=poses[str(f)]) for f in st['test_frames']])
    os.makedirs(f'{D}/{track}', exist_ok=True); json.dump(cams, open(f'{D}/{track}/cameras.json', 'w'))
    shutil.copytree(f'{W}/{track}/points', f'{D}/{track}/points')
    shutil.copytree(f'{W}/{track}/modp_points', f'{D}/{track}/modp_points')
    json.dump(st, open(f'{D}/{track}/prep_stats.json', 'w'), indent=1)
print('ok')
