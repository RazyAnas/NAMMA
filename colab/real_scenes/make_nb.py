import json, sys
def code(src): return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=src.strip('\n'))
def md(src): return dict(cell_type='markdown', metadata={}, source=src.strip('\n'))
BR = 'claude/project-thread-eopuhz'
cells = [
md("""# NAMMA on your own videos: phone AR room + lobby (GPU)
The same pipeline that reached 32.26 dB on the ICL-NUIM room (global SfM poses, Depth Anything V2 anchored depth,
MODP placement + hash voxels, error compensation at step 1,000, 3DGS paper densification, 3,000 steps), with OpenSplat built for CUDA.
Every 8th frame is held out and scored. Frames come from `namma_modp_colab.zip` in Google Drive (My Drive). Scenes: phone room with our global-SfM poses (`phone_sfm`), the same room with ARCore poses refined by bundle adjustment (`phone_arba`), and the lobby with the earlier merged COLMAP poses (`lobby_ref`; our global SfM broke on its white walls).
Run all. Results go to `My Drive/namma_real_scenes/`."""),
code("""#@title 1. GPU check
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv"""),
code("""#@title 2. Frames from Drive
from google.colab import drive
drive.mount('/content/drive')
import os, glob, subprocess
z = [p for p in glob.glob('/content/drive/MyDrive/**/namma_modp_colab.zip', recursive=True)]
print(z); assert z, 'namma_modp_colab.zip not found in My Drive'
subprocess.run(['unzip', '-q', '-o', z[0], 'namma_modp_colab/data/*/images/*', 'namma_modp_colab/data/*/*/cameras.json', '-d', '/content'], check=True)
print(len(os.listdir('/content/namma_modp_colab/data/lobby/images')), len(os.listdir('/content/namma_modp_colab/data/phone_ar/images')))"""),
code(f"""#@title 3. Code, libraries, OpenSplat (CUDA) with the NAMMA patch (about 10 minutes)
%%bash
set -e
cd /content
rm -rf NAMMA && git clone -q --depth 1 -b {BR} https://github.com/RazyAnas/NAMMA
pip -q install pycolmap==4.2.1 onnxruntime plyfile scikit-image > /tmp/pip.log 2>&1 || (tail -20 /tmp/pip.log; exit 1)
apt-get -qq update > /dev/null && apt-get -qq install -y libopencv-dev > /dev/null
mkdir -p models && [ -f models/depth_anything_v2_vits.onnx ] || wget -q -O models/depth_anything_v2_vits.onnx https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx
md5sum models/depth_anything_v2_vits.onnx
if [ ! -x OpenSplat/build/opensplat ]; then
  rm -rf OpenSplat && git clone -q https://github.com/pierotofy/OpenSplat && cd OpenSplat && git checkout -q 688944f01291a1939f688b49feef6936620b47b8
  git apply /content/NAMMA/colab/real_scenes/opensplat_namma.patch
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_RUNTIME=CUDA -DCMAKE_CUDA_ARCHITECTURES=$CC \\
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') > /tmp/cmake.log 2>&1 || (tail -30 /tmp/cmake.log; exit 1)
  make -j$(nproc) > /tmp/make.log 2>&1 || (grep -m5 -B2 -A5 error /tmp/make.log; tail -20 /tmp/make.log; exit 1)
fi
/content/OpenSplat/build/opensplat --help | head -3
echo SETUP_OK"""),
code("""#@title 4. Run the scenes in the background (depth for the next scene runs while one trains)
%%bash
cd /content && R=/content/NAMMA/colab/real_scenes && Z=/content/namma_modp_colab/data
cat > /content/run_all.sh <<EOS
set -x
export OS_BIN=/content/OpenSplat/build/opensplat
SCENES="phone_sfm:phone_ar/images:phone_ar/ar_poses/cameras.json phone_arba:phone_ar/images:phone_ar/ar_poses/cameras.json lobby_ref:lobby/images:lobby/sfm/cameras.json"
for e in \\$SCENES; do IFS=: read s img cj <<< "\\$e"
  mkdir -p /content/work/\\$s/clean; cp -r $R/data/\\$s/clean/. /content/work/\\$s/clean/
  python3 $R/scripts/link_frames.py /content/work/\\$s $Z/\\$img $Z/\\$cj; done
(for e in \\$SCENES; do s=\\${e%%:*}; python3 $R/scripts/prep_depth.py /content/work/\\$s /content/models/depth_anything_v2_vits.onnx > /content/work/\\$s.depth.log 2>&1; touch /content/work/\\$s.depth.done; done) &
for e in \\$SCENES; do s=\\${e%%:*}
  while [ ! -f /content/work/\\$s.depth.done ]; do sleep 10; done
  bash $R/scripts/run_scene.sh /content/work/\\$s || echo "FAILED \\$s"
done
echo ALL_DONE
EOS
nohup bash /content/run_all.sh > /content/run_all.log 2>&1 &
echo started"""),
code("""#@title 5. Progress (re-run this cell to refresh)
!tail -5 /content/run_all.log; echo; for s in phone_sfm phone_arba lobby_ref; do echo == $s; tail -1 /content/work/$s.depth.log 2>/dev/null; ls /content/work/$s/run 2>/dev/null | tr '\\n' ' '; echo; tail -c 300 /content/work/$s/run/log_a.txt 2>/dev/null; echo; tail -c 300 /content/work/$s/run/log.txt 2>/dev/null; echo; cat /content/work/$s/run/score.txt 2>/dev/null; done"""),
code("""#@title 6. Check results: scores, timings, held-out renders next to the real frames; save to Drive
import json, os, glob, shutil, numpy as np, cv2
out = '/content/drive/MyDrive/namma_real_scenes'; os.makedirs(out, exist_ok=True); summary = {}
for s in ['phone_sfm', 'phone_arba', 'lobby_ref']:
    W = f'/content/work/{s}'; R = f'{W}/run'
    if not os.path.exists(f'{R}/scores.json'): print(s, 'not finished'); continue
    sc = json.load(open(f'{R}/scores.json')); t = lambda n: int(open(f'{R}/{n}.txt').read())
    st = json.load(open(f'{W}/clean/stats.json')); ec = json.load(open(f'{R}/ec_stats.json'))['counts']
    hdr = open(f'{R}/splat.ply', 'rb').read(400).decode(errors='ignore'); n = int(hdr.split('element vertex ')[1].split()[0])
    summary[s] = dict(psnr=round(sc['psnr_mean'], 2), ssim=round(sc['ssim_mean'], 3), psnr_min=round(sc['psnr_min'], 2), views=sc['n_views'],
                      resolution=sc['resolution'], gaussians=n, ec_added=ec['ec'], minutes=round((t('end') - t('start')) / 60, 1),
                      sfm_vs_reference=st['vs_reference'], registered=f"{st['registered']}/{st['of']}",
                      depth=json.load(open(f'{W}/clean/depth_stats.json')))
    pf = sorted(sc['per_frame'], key=lambda r: r['psnr']); pick = [pf[0], pf[len(pf) // 2], pf[-1]]
    rows = []
    for r in pick:
        a = cv2.imread(f"{R}/test_{r['frame']:05d}.png"); b = cv2.imread(f"{W}/data/color_raw/{r['frame']:05d}.jpg")
        b = cv2.resize(b, (a.shape[1], a.shape[0])); row = np.hstack([b, a])
        cv2.putText(row, f"frame {r['frame']}  photo | render  {r['psnr']:.1f} dB", (10, 30), 0, 0.8, (0, 255, 255), 2); rows.append(row)
    h = min(x.shape[0] for x in rows); grid = np.vstack([cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in rows])
    if grid.shape[1] > 1800: grid = cv2.resize(grid, (1800, int(grid.shape[0] * 1800 / grid.shape[1])))
    cv2.imwrite(f'{out}/{s}_heldout_photo_vs_render.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    shutil.copy(f'{R}/splat.ply', f'{out}/{s}_splat.ply'); shutil.copy(f'{R}/scores.json', f'{out}/{s}_scores.json')
json.dump(summary, open(f'{out}/summary.json', 'w'), indent=1); print(json.dumps(summary, indent=1))
from IPython.display import Image, display
for s in summary: display(Image(f'{out}/{s}_heldout_photo_vs_render.jpg', width=1100))"""),
]
nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'),
          kernelspec=dict(name='python3', display_name='Python 3'), language_info=dict(name='python')), cells=cells)
json.dump(nb, open(sys.argv[1], 'w'), indent=1)
