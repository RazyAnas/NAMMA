# Builds VGGT_all_frames.ipynb: VGGT-1B depth for all 359 whole-room frames (every 8th of ICL-NUIM lr1, 0..2864),
# run in chunks of consecutive frames, saved as one float16 npz (392x518 per frame, VGGT's input size) and downloaded.
import json
cells = []
md = lambda s: cells.append(dict(cell_type='markdown', metadata={}, source=s))
code = lambda s: cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s))
md("# NammaSpace: VGGT depth for all 359 whole-room frames (T4, Run all)")
code("""!pip -q install git+https://github.com/facebookresearch/vggt.git opencv-python-headless
import os, zipfile, urllib.request
f = 'livingroom1-color.zip'
if os.path.exists(f) and not zipfile.is_zipfile(f): os.remove(f)  # half-downloaded
if not os.path.exists(f): urllib.request.urlretrieve('https://github.com/isl-org/open3d_downloads/releases/download/augmented-icl-nuim/' + f, f)
print('data ok')""")
code("""import os, zipfile, time, json, numpy as np, torch
FRAMES = list(range(0, 2870, 8)); CHUNK = 30
os.makedirs('all', exist_ok=True); z = zipfile.ZipFile('livingroom1-color.zip')
for f in FRAMES: open(f'all/{f:05d}.jpg', 'wb').write(z.read(f'{f:05d}.jpg'))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval()
D, C, chunk_of = [], [], []; t0 = time.time()
for c in range(0, len(FRAMES), CHUNK):
    fs = FRAMES[c:c + CHUNK]
    imgs = load_and_preprocess_images([f'all/{f:05d}.jpg' for f in fs]).to('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        p = model(imgs)
    D.append(p['depth'][0, ..., 0].float().cpu().numpy().astype(np.float16)); C.append(p['depth_conf'][0].float().cpu().numpy().astype(np.float16))
    chunk_of += [c // CHUNK] * len(fs); print(c, tuple(imgs.shape), f'{time.time() - t0:.0f} s', flush=True)
    del p, imgs; torch.cuda.empty_cache()
D, C = np.concatenate(D), np.concatenate(C)
np.savez_compressed('vggt_depth_all.npz', frames=np.array(FRAMES), depth=D, conf=C, chunk=np.array(chunk_of))
print('SAVED', D.shape, f'{os.path.getsize("vggt_depth_all.npz") / 1e6:.0f} MB', f'{time.time() - t0:.0f} s total')""")
code("""from google.colab import files; files.download('vggt_depth_all.npz')""")
nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'), kernelspec=dict(name='python3', display_name='Python 3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=0)
json.dump(nb, open('VGGT_all_frames.ipynb', 'w'), indent=1); print('ok')
