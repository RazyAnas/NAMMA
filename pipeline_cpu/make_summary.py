"""Collect finished scenes into summary.json (one row per scene/variant/test split). Usage:
python make_summary.py <out summary.json> name=<work_dir> [name=<work_dir> ...]"""
import sys, json, os

out = sys.argv[1]; rows = []; scenes = {}
for arg in sys.argv[2:]:
    name, W = arg.split('=', 1)
    info = {s: json.load(open(f'{W}/.done_{s}')) for s in ['frames', 'sfm', 'undist', 'depth', 'mesh', 'splat', 'score', 'strips']
            if os.path.exists(f'{W}/.done_{s}')}
    scenes[name] = dict(work=W, stages=info, args=json.load(open(f'{W}/args.json')) if os.path.exists(f'{W}/args.json') else {})
    sc = info.get('score', {})
    for split in ['test_offpath', 'test_near']:
        if split not in sc: continue
        m = sc[split]
        for v in ['mesh', 'splat', 'hybrid', 'flat']:
            rows.append(dict(scene=name, split=split, n_views=m['n'], variant=v, psnr=m.get(v), psnr_cc_halves=m.get(f'{v}_cc'),
                             ssim=m.get(f'{v}_ssim'), lpips=None))
json.dump(dict(rows=rows, scenes=scenes,
               notes=dict(lpips='not measured: LPIPS needs torchvision AlexNet/VGG ImageNet weights, only hosted on download.pytorch.org / '
                                'Hugging Face (blocked here); no GitHub-hosted copy found',
                          psnr_cc_halves='affine colour fit on one half of the test image, scored on the other half, both ways',
                          flat='mean training colour everywhere (blank-guess baseline)')), open(out, 'w'), indent=1)
for r in rows: print(r)
