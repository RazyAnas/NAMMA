"""Look at public web pages like a visitor and keep screenshots (GitHub Actions: the Claude container cannot reach most sites).
URLS.txt: one start URL per line. For each: full-page screenshot, page text, all links; links that look like tours/demos are
opened too (up to MAX_TOURS): screenshot after load, then after dragging / pressing keys to move, so the viewer quality shows.
Everything goes to out/ (+ the private HF dataset <user>/namma-v3-out/site_look/<time>/ if HF_TOKEN is set). Public pages only."""
import json, os, re, time
from playwright.sync_api import sync_playwright
OUT = 'out'; os.makedirs(OUT, exist_ok=True); MAX_TOURS = int(os.environ.get('MAX_TOURS', 4))
urls = [u.strip() for u in open(os.path.join(os.path.dirname(__file__), 'URLS.txt')) if u.strip() and not u.startswith('#')]
log = {}

def shot(page, name):
    page.screenshot(path=f'{OUT}/{name}.jpg', type='jpeg', quality=80); print('shot', name, flush=True)

def visit_tour(ctx, url, k):
    page = ctx.new_page(); rec = dict(url=url, requests=[])
    page.on('request', lambda r: rec['requests'].append(r.url) if re.search(r'\.(sog|splat|ply|spz|ksplat|json|webp|jpg)(\?|$)', r.url) else None)
    try:
        page.goto(url, wait_until='domcontentloaded', timeout=60000); page.wait_for_timeout(35000); shot(page, f'tour{k}_a_loaded')
        W, H = page.viewport_size['width'], page.viewport_size['height']
        page.mouse.move(W / 2, H / 2); page.mouse.down(); page.mouse.move(W / 2 + 300, H / 2, steps=20); page.mouse.up(); page.wait_for_timeout(4000)
        shot(page, f'tour{k}_b_turned')
        for key in ['w'] * 12 + ['ArrowUp'] * 12: page.keyboard.press(key); page.wait_for_timeout(120)
        page.wait_for_timeout(4000); shot(page, f'tour{k}_c_moved')
        page.mouse.click(W / 2, H * 0.7); page.wait_for_timeout(6000); shot(page, f'tour{k}_d_clicked')
        rec['text'] = page.inner_text('body')[:1500]
    except Exception as e: rec['error'] = str(e)[:300]
    page.close(); return rec

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get('CHROME') or None, args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'])
    ctx = b.new_context(viewport={'width': 1440, 'height': 900}, locale='en-US')
    for i, u in enumerate(urls):
        page = ctx.new_page(); page.goto(u, wait_until='networkidle', timeout=90000); page.wait_for_timeout(5000)
        page.screenshot(path=f'{OUT}/site{i}_full.jpg', type='jpeg', quality=75, full_page=True); shot(page, f'site{i}_top')
        links = sorted({a for a in page.eval_on_selector_all('a[href]', 'els => els.map(e => e.href)')})
        text = page.inner_text('body'); log[u] = dict(links=links, text=text[:6000])
        print('LINKS', json.dumps(links, indent=0), '\nTEXT', text[:3000], flush=True)
        tours = [l for l in links if re.search(r'tour|demo|view|example|sample|showcase', l, re.I) and l.rstrip('/') != u.rstrip('/')][:MAX_TOURS]
        for k, t in enumerate(tours): log[f'tour{k}'] = visit_tour(ctx, t, k); print('TOUR', k, json.dumps(log[f'tour{k}'])[:1500], flush=True)
    b.close()
json.dump(log, open(f'{OUT}/log.json', 'w'), indent=1)
tok = os.environ.get('HF_TOKEN')
if tok:
    from huggingface_hub import HfApi
    api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"; dst = 'site_look/' + time.strftime('%Y%m%d-%H%M')
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True); api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
    print(f'uploaded: hf://datasets/{repo}/{dst}', flush=True)
