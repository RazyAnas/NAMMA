# Copy the frames of one scene out of namma_modp_colab.zip's data folder into <work>/data/color_raw/<frame>.jpg,
# the naming the pipeline uses (frame number = digits of the original name).
# Usage: python link_frames.py <work> <images dir> <cameras.json>
import sys, os, re, json, shutil
W, IMG, CJ = sys.argv[1:4]; cam = json.load(open(CJ)); num = lambda s: int(re.sub(r'\D', '', s))
st = json.load(open(f'{W}/clean/stats.json')); need = set(st['train_frames'] + st['test_frames'])
os.makedirs(f'{W}/data/color_raw', exist_ok=True); n = 0
for e in cam['train'] + cam['test']:
    f = num(e['frame'])
    if f in need: shutil.copy(f"{IMG}/{os.path.basename(e['image'])}", f'{W}/data/color_raw/{f:05d}.jpg'); n += 1
assert n == len(need), (n, len(need)); print(n, 'frames')
