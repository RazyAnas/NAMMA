# VGGT-1B depth for all 359 whole-room frames (every 8th of ICL-NUIM lr1, 0..2864), chunks of consecutive frames,
# saved as one float16 npz at VGGT's input size (392x518). Run with exec() inside a Colab kernel; reuses `model` if loaded.
import os, zipfile, time, urllib.request, numpy as np, torch
from vggt.utils.load_fn import load_and_preprocess_images
f = 'livingroom1-color.zip'
if os.path.exists(f) and not zipfile.is_zipfile(f): os.remove(f)
if not os.path.exists(f): urllib.request.urlretrieve('https://github.com/isl-org/open3d_downloads/releases/download/augmented-icl-nuim/' + f, f)
FRAMES = list(range(0, 2870, 8)); CHUNK = 30
os.makedirs('all', exist_ok=True); z = zipfile.ZipFile(f)
for fr in FRAMES: open(f'all/{fr:05d}.jpg', 'wb').write(z.read(f'{fr:05d}.jpg'))
if 'model' not in globals():
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval()
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
D, C, chunk_of = [], [], []; t0 = time.time()
for c in range(0, len(FRAMES), CHUNK):
    fs = FRAMES[c:c + CHUNK]
    imgs = load_and_preprocess_images([f'all/{fr:05d}.jpg' for fr in fs]).to('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        p = model(imgs)
    D.append(p['depth'][0, ..., 0].float().cpu().numpy().astype(np.float16)); C.append(p['depth_conf'][0].float().cpu().numpy().astype(np.float16))
    chunk_of += [c // CHUNK] * len(fs); print(c, tuple(imgs.shape), f'{time.time() - t0:.0f} s', flush=True)
    del p, imgs; torch.cuda.empty_cache()
D, C = np.concatenate(D), np.concatenate(C)
np.savez_compressed('vggt_depth_all.npz', frames=np.array(FRAMES), depth=D, conf=C, chunk=np.array(chunk_of))
print('SAVED', D.shape, f'{os.path.getsize("vggt_depth_all.npz") / 1e6:.0f} MB', f'{time.time() - t0:.0f} s total')
