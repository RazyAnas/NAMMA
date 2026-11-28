"""The "where is this picture" server for the tour's AR page (ar.html): DRS-VPT against one published tour.

    python serve_ar.py --tour <the tour's link, .../index.html> [--port 8300] [--device cuda|cpu] [--tunnel]

Runs wherever the model can run (a GPU session, a CPU box with ~8 GB free memory): it needs torch + the drs-vpt package
(github.com/ori-drs/drs-vpt; weights CC BY-NC 4.0, so a test, not a product part). It reads only the tour's public files:
ar_map.bin -> the map (points on the room mesh; a tour without one: its splat's centres), walls.json -> the room box. Nothing private, no token.
  POST /locate   form: image (jpeg), K (3x3 json, pixels of that image)  ->  {T_w2c: 4x4 in the tour's room metres, seconds}
  POST /report   json from the page's "Save report"                      ->  kept as ar_reports/<time>.json next to this file
  GET  /health   {ready, points, device}
Every request needs the header x-namma-key (printed at start, part of the phone link). One picture is located at a time.
--tunnel starts a cloudflared quick tunnel and prints the phone link (and a QR code when `qrcode` is installed); the tour
link stays on this machine and in that printed link only. --fake answers with a fixed pose (no model): plumbing tests."""
import argparse, io, json, os, secrets, sys, threading, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ZUP = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def make_app(locate, key, info, origin='*', reports=os.path.join(HERE, 'ar_reports')):
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI(); lock = threading.Lock()
    app.add_middleware(CORSMiddleware, allow_origins=[origin], allow_methods=['GET', 'POST'], allow_headers=['x-namma-key', 'content-type'])
    def check(k):
        if not k or not secrets.compare_digest(k, key): raise HTTPException(401, 'wrong key')

    @app.get('/health')
    def health(): return dict(ready=True, **info)

    @app.post('/locate')
    def where(image: UploadFile = File(...), K: str = Form(...), x_namma_key: str = Header(None)):
        check(x_namma_key); raw = image.file.read()
        if len(raw) > 4_000_000: raise HTTPException(413, 'picture too large')
        k = np.array(json.loads(K), float)
        if k.shape != (3, 3): raise HTTPException(400, 'K must be 3x3')
        if not lock.acquire(blocking=False): raise HTTPException(429, 'busy with another picture')
        try:
            from PIL import Image
            t0 = time.time(); T = locate(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')), k)
            return dict(T_w2c=np.asarray(T, float).round(6).tolist(), seconds=round(time.time() - t0, 2))
        finally: lock.release()

    @app.post('/report')
    async def report(req: Request, x_namma_key: str = Header(None)):
        check(x_namma_key); body = await req.body()
        if len(body) > 2_000_000: raise HTTPException(413, 'report too large')
        os.makedirs(reports, exist_ok=True); path = os.path.join(reports, time.strftime('%Y%m%d-%H%M%S') + '.json')
        json.dump(json.loads(body), open(path, 'w'), indent=1); return dict(saved=os.path.basename(path))
    return app


def drsvpt(base, device, up='y'):
    """The locate(image, K) -> T_w2c function for one tour: its map prepared once."""
    import tour_cloud
    from drs_vpt.hub import load_model
    from drs_vpt.inference import encode_map, localize
    get = lambda f: urllib.request.urlopen(base + f, timeout=120).read()
    try: box = json.loads(get('walls.json')).get('box')
    except Exception: box = None
    try: cloud, kind = tour_cloud.read_map(get('ar_map.bin')), 'mesh'                       # measured best (ar-check round 3: 28 % of pictures within 10 cm)
    except Exception: cloud, kind = tour_cloud.cloud(get('scene.compressed.ply'), grid=0.01, opacity=0.1, box=box), 'splat centres (no ar_map.bin in this tour)'
    A = ZUP if up == 'z' else np.eye(3)                                                     # y up measured right; z up 0 of 6
    model = load_model(device=device); enc = encode_map(model, np.ascontiguousarray(cloud @ A.T, np.float32), grid_size=0.01, max_points=600000)
    def locate(img, K):
        T = np.asarray(localize(model, img, enc, K)['T_w2c'], float); T[:3, :3] = T[:3, :3] @ A   # found in the turned cloud -> the tour's own axes
        return T
    return locate, dict(points=int(len(cloud)), map=kind, device=device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tour'); ap.add_argument('--port', type=int, default=8300); ap.add_argument('--device', default='cpu')
    ap.add_argument('--up', default='y', choices=['y', 'z']); ap.add_argument('--tunnel', action='store_true'); ap.add_argument('--fake', action='store_true'); a = ap.parse_args()
    key = os.environ.get('NAMMA_AR_KEY') or secrets.token_urlsafe(12)
    base = a.tour.rsplit('/', 1)[0] + '/' if a.tour else None; origin = '/'.join(a.tour.split('/')[:3]) if a.tour else '*'
    if a.fake: locate, info = (lambda img, K: np.eye(4)), dict(points=0, device='fake')
    else: assert base, 'give --tour <link>'; locate, info = drsvpt(base, a.device, a.up)
    app = make_app(locate, key, info, origin); public = f'http://127.0.0.1:{a.port}'
    if a.tunnel:
        sys.path.insert(0, os.path.join(HERE, '..')); import serve_render
        _proc, url = serve_render.start_tunnel(a.port); public = url or public
    link = f'{base}ar.html#s={public}&k={key}' if base else f'{public} key {key}'
    print('phone link (keep it private):', link, flush=True)
    if a.tunnel and base:
        try: serve_render.print_qr(link)
        except Exception: pass
    import uvicorn; uvicorn.run(app, host='0.0.0.0' if not a.tunnel else '127.0.0.1', port=a.port, log_level='warning')
