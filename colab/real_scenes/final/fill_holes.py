"""PROOF 4 (step 6 of the video method): can a video model fill ONLY the holes of our real-pixel views, consistently, without
inventing? One walk pass whose cameras are already right (no dependence on the cross-pass fix):
  1. drone path: follow the pass's cameras, drifting up to --offset metres sideways (off-path views have holes);
  2. render every view from that pass's REAL photos (ibr.synth); holes = pixels no photo covers;
  3. honesty strip: also hide a vertical strip of REAL pixels from the model, so its guess can be scored against the truth;
  4. Wan 2.1 VACE 1.3B (diffusers) fills the hidden pixels of the whole clip at once (consistent across frames);
  5. composite: real pixels everywhere they exist, generated only in holes, every generated pixel flagged.
Outputs (out dir): holes.mp4 (magenta = never filmed), generated.mp4, composite.mp4, flags.mp4, sheet.jpg, fill_report.json.
Run 1 (one pass as source, 50 cm off): holes 18% (max 40%); strip 35.7 dB, but the big hole behind the pillar got an invented
glass door, a screen and a small figure. So now: (a) sources = every pass aligned with this one (real pixels first), (b) the model
only fills SMALL holes (thin slivers / small patches where a wall or floor continues), (c) large unfilmed regions are never invented:
shown as 'not filmed' (neutral grey, flagged blue), (d) a sweep measures holes vs distance from the path.
Usage: python fill_holes.py <dir with da3_sfm.npz + frames> <out> [--pass 453 488] [--offset 0.3] [--dry]  (--dry: no model, CPU)"""
import argparse, gc, json, os, time, numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import ibr

H, W = 832, 480                                                                   # Wan 480p, portrait like the phone video
PROMPT = ('Photorealistic handheld phone video inside an empty indoor room. Continue the existing walls, floor tiles, ceiling and '
          'window frames exactly as they are. Static scene, nothing new, sharp, natural indoor light.')
NEG = 'new objects, furniture, people, text, watermark, cartoon, painting, blurry, distorted, warped, oversaturated'

def path(d, sel, n, offset):
    """n camera poses along the pass (position lerp, rotation slerp), drifting sideways up to offset m in the middle."""
    C = d['c2w'][sel]; t = np.linspace(0, len(sel) - 1, n); i0 = np.clip(t.astype(int), 0, len(sel) - 2); f = t - i0
    sl = Slerp(np.arange(len(sel)), Rotation.from_matrix(C[:, :3, :3])); R = sl(t).as_matrix()
    p = C[i0, :3, 3] * (1 - f)[:, None] + C[i0 + 1, :3, 3] * f[:, None]
    side = offset * np.sin(np.linspace(0, np.pi, n)) ** 2                          # 0 at both ends (exact photos), max mid-way
    out = np.tile(np.eye(4), (n, 1, 1)); out[:, :3, :3] = R; out[:, :3, 3] = p + side[:, None] * R[:, :, 0]
    return out, side

def split_holes(h, max_area=0.006, max_radius=10):
    """Holes the model may fill: thin slivers (no point farther than max_radius px from the hole's edge) or small patches
    (< max_area of the frame). Everything else is a region no photo saw: never invented."""
    from scipy.ndimage import label, distance_transform_edt
    lab, n = label(h); small = np.zeros_like(h); dt = distance_transform_edt(h)
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < max_area * h.size or dt[m].max() <= max_radius: small |= m
    return small, h & ~small

def sweep(src, sel, only, fid, out, offsets=(0.1, 0.2, 0.3, 0.4, 0.5), n=9):
    """How far from the walked path can a viewer go? Hole % (and 'not filmed' %) vs sideways offset, pass-only vs group sources."""
    d, files, K, has, Hs, Ws = ibr.load(src, scale=1); pas = np.zeros_like(has); pas[sel] = True; rows = []
    for o in offsets:
        poses, _ = path(d, sel, n, o); poses = poses[n // 2 - 2:n // 2 + 3]             # the 5 views nearest full offset
        for lab, srcs in (('pass', pas), ('group', only)):
            hs, ls = [], []
            for c in poses:
                _, m, _, _ = ibr.synth(d, files, K, srcs, Hs, Ws, K[sel[0]], c, exclude=set(), k=6, kmax=40)
                hm = to_wan(np.repeat((~m)[..., None].astype(np.float32), 3, 2))[..., 0] > 0.5; sm, lg = split_holes(hm)
                hs.append(hm.mean()); ls.append(lg.mean())
            rows.append(dict(offset_m=o, sources=lab, holes_pct=round(100 * float(np.mean(hs)), 1), not_filmed_pct=round(100 * float(np.mean(ls)), 1)))
            print('SWEEP', json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f'{out}/sweep.json', 'w'), indent=1)

def to_wan(img):
    """Render (1008 x 560, aspect 0.556) -> 832 x 480 (0.577): crop height, resize."""
    h, w = img.shape[:2]; ch = int(round(w * H / W))
    if ch <= h: y0 = (h - ch) // 2; img = img[y0:y0 + ch]
    else: cw = int(round(h * W / H)); x0 = (w - cw) // 2; img = img[:, x0:x0 + cw]
    return np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS), np.float32) / 255

def mp4(frames, path_, fps=12):
    import imageio.v2 as iio
    iio.mimwrite(path_, [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames], fps=fps, codec='libx264', quality=8, macro_block_size=16)

def wan_fill(video, mask, steps, seed=0):
    """video: list of HxWx3 in [0,1] with hidden pixels set to grey; mask: list of HxW bool (True = generate)."""
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from transformers import UMT5EncoderModel
    mid = 'Wan-AI/Wan2.1-VACE-1.3B-diffusers'
    te = UMT5EncoderModel.from_pretrained(mid, subfolder='text_encoder', torch_dtype=torch.bfloat16)   # 11 GB: encode on CPU (T4 has 16 GB)
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder='vae', torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(mid, vae=vae, text_encoder=te, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)       # 3.0 for 480p (model card)
    t0 = time.time()
    with torch.no_grad():
        pe, npe = pipe.encode_prompt(prompt=PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True, device=torch.device('cpu'), dtype=torch.float32)
    print(f'prompt encoded on CPU in {time.time()-t0:.0f}s', flush=True)
    pipe.text_encoder = None; del te; gc.collect()
    pipe.vae.to('cuda')
    try: pipe.vae.enable_tiling()
    except Exception: pass
    vid = [Image.fromarray((v * 255).astype(np.uint8)) for v in video]; msk = [Image.fromarray((m * 255).astype(np.uint8)) for m in mask]
    for dt in (torch.float16, torch.float32):                                       # fp16 is fast on a T4; fp32 only if fp16 overflows
        pipe.transformer.to('cuda') if dt == torch.float16 else pipe.transformer.to('cuda', dt); t0 = time.time()   # fp16 load keeps norms in fp32
        out = pipe(video=vid, mask=msk, prompt_embeds=pe.to('cuda', dt), negative_prompt_embeds=npe.to('cuda', dt),
                   height=H, width=W, num_frames=len(vid), num_inference_steps=steps, guidance_scale=5.0,
                   generator=torch.Generator('cuda').manual_seed(seed), output_type='np').frames[0]
        out = np.asarray(out, np.float32); bad = bool(np.isnan(out).any()) or float(out.std()) < 1e-3
        print(f'Wan VACE ({dt}): {len(vid)} frames in {time.time()-t0:.0f}s, broken output: {bad}', flush=True)
        if not bad: break
        torch.cuda.empty_cache()
    return np.nan_to_num(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--pass', dest='pas', nargs=2, type=int, default=[453, 488])
    ap.add_argument('--offset', type=float, default=0.3); ap.add_argument('--sources', default='group', choices=['group', 'pass']); ap.add_argument('--frames', type=int, default=49); ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--dry', action='store_true'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    d, files, K, has, Hr, Wr = ibr.load(a.src, scale=2)
    sel = [j for j in range(len(d['idx'])) if a.pas[0] <= d['idx'][j] <= a.pas[1] and has[j]]; assert len(sel) >= 4, f'pass {a.pas}: {len(sel)} frames'
    only = np.zeros_like(has); only[sel] = True                                       # sources: this pass only (its cameras agree)
    if a.sources == 'group' and 'group' in d:                                         # + every pass aligned with it (align_passes groups)
        only = has & (d['group'] == d['group'][sel[0]])
    print(f'sources: {int(only.sum())} real photos ({a.sources})', flush=True)
    sweep(a.src, sel, only, d['idx'], a.out)
    poses, side = path(d, sel, a.frames, a.offset); Kt = K[sel[0]]
    real, hole = [], []
    for c in poses:
        img, m, _, _ = ibr.synth(d, files, K, only, Hr, Wr, Kt, c, exclude=set(), k=4, kmax=40)
        real.append(to_wan(img)); hole.append(to_wan(np.repeat((~m)[..., None].astype(np.float32), 3, 2))[..., 0] > 0.5)
    from scipy.ndimage import binary_dilation
    small, large = zip(*[split_holes(h) for h in hole])
    hole = [binary_dilation(h, iterations=3) for h in hole]                           # soft edges of the warp go to the model too
    large = [binary_dilation(l, iterations=3) & h for l, h in zip(large, hole)]; small = [h & ~l for h, l in zip(hole, large)]
    strip = np.zeros((H, W), bool); strip[:, int(0.42 * W):int(0.58 * W)] = True     # honesty strip: real pixels hidden from the model
    hide = [h | (strip & ~h) for h in hole]
    video = [np.where(m[..., None], 0.5, r) for r, m in zip(real, hide)]
    print(f'{a.frames} views rendered from {len(sel)} real photos of pass {a.pas} [{time.time()-t0:.0f}s]; holes: '
          f'{100*np.mean([h.mean() for h in hole]):.1f}% of pixels (max {100*max(h.mean() for h in hole):.1f}% at {a.offset} m off the path)', flush=True)
    gen = video if a.dry else wan_fill(video, hide, a.steps)
    NF = np.array([.42, .44, .48])                                                    # 'not filmed' grey: never invented
    comp = [np.where(l[..., None], NF, np.where(sm[..., None], g, r)) for r, g, sm, l in zip(real, gen, small, large)]
    sm = [strip & ~h for h in hole]; truth = [ibr.psnr_gain(g, r, m) for g, r, m in zip(gen, real, sm) if m.sum() > 500]
    mag = [np.where(h[..., None], np.array([1, 0, 1.]), r) for r, h in zip(real, hole)]
    flags = [np.where(l[..., None], np.array([.2, .4, 1]), np.where(sm[..., None], np.array([1, .2, .2]), r * 0.35))
             for r, sm, l in zip(real, small, large)]                                 # red = generated (small gaps), blue = not filmed
    for name, fr in (('holes', mag), ('generated', gen), ('composite', comp), ('flags', flags)): mp4(fr, f'{a.out}/{name}.mp4')
    pick = np.linspace(0, a.frames - 1, 5).astype(int)
    sheet = np.concatenate([np.concatenate([mag[i], gen[i], comp[i]], 1) for i in pick], 0)
    Image.fromarray((sheet * 255).astype(np.uint8)).resize((3 * W // 2, len(pick) * H // 2)).save(f'{a.out}/sheet.jpg', quality=90)
    rep = dict(pass_=a.pas, frames=a.frames, offset_m=a.offset, sources=a.sources, real_photos=int(only.sum()), holes_pct=round(100 * float(np.mean([h.mean() for h in hole])), 2),
               generated_pct=round(100 * float(np.mean([m.mean() for m in small])), 2), not_filmed_pct=round(100 * float(np.mean([m.mean() for m in large])), 2),
               honesty_strip_psnr=round(float(np.median(truth)), 2) if truth else None, dry=a.dry, seconds=round(time.time() - t0))
    json.dump(rep, open(f'{a.out}/fill_report.json', 'w'), indent=1); print('FILL', json.dumps(rep), flush=True)

if __name__ == '__main__':
    main()
