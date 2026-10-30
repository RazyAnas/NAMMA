"""Publish a built tour folder (make_tour.py build) to the public static Hugging Face Space <user>/namma-tour, like splattour's
share links: each tour in a folder with a random name (only people with the link find it), no listing page. A private static
Space cannot serve its files to a browser (the page loads, its scene/scripts do not), so the owner chose public. The link is
NOT printed (GitHub logs of this repo are public): it goes to <run>/tour_url.txt in the private results dataset.
Used by gh_run.py after a run, and on its own (.github/workflows/tour-publish.yml, v3/PUBLISH.json) to republish a run's tour:
  python tour_publish.py <run, e.g. 20261026-0309 | world/<time> | flashworld/<time>> [floorplan] [collision]   (HF_TOKEN in the environment; tour taken from the results
  dataset; 'floorplan': redraw floorplan.png from the run's model first (final/make_tour.py floorplan) and save it back there;
  'collision': rebuild collision.glb = room box + simplified mesh furniture; 'rebuild': convert the run's model.ply into the tour's
  scene file again with the current make_tour.build settings, e.g. all SH bands)"""
import os, re, secrets, sys


def publish(api, user, results_repo, run, tour_dir, say=print, url_path=None):
    sp = f'{user}/namma-tour'; api.create_repo(sp, repo_type='space', space_sdk='static', private=False, exist_ok=True)
    for f in {x.split('/')[0] for x in api.list_repo_files(sp, repo_type='space') if '/' in x}:      # old guessable folders out
        if not re.fullmatch(r'\d{8}-\d{4}(-[a-z0-9]+)?-[0-9a-f]{20}', f): api.delete_folder(f, repo_id=sp, repo_type='space')   # (digits: 'world1')
    folder = f'{run}-{secrets.token_hex(10)}'
    api.upload_folder(folder_path=tour_dir, path_in_repo=folder, repo_id=sp, repo_type='space', ignore_patterns=['plan.json'])
    page = '<!doctype html><meta charset="utf-8"><title>NAMMA</title><body style="font:16px sans-serif;padding:24px;background:#111;color:#ddd"><h1>NAMMA 3D tours</h1><p>Open a tour with its link.</p>'
    open('tours_index.html', 'w').write(page); api.upload_file(path_or_fileobj='tours_index.html', path_in_repo='index.html', repo_id=sp, repo_type='space')
    open('space_readme.md', 'w').write('---\ntitle: NAMMA tour\nemoji: 🏠\ncolorFrom: gray\ncolorTo: gray\nsdk: static\npinned: false\n---\n')
    api.upload_file(path_or_fileobj='space_readme.md', path_in_repo='README.md', repo_id=sp, repo_type='space')
    try: api.update_repo_settings(sp, repo_type='space', private=False)
    except Exception: api.update_repo_visibility(sp, private=False, repo_type='space')
    open('tour_url.txt', 'w').write(f'https://{user}-namma-tour.static.hf.space/{folder}/index.html\n')
    api.upload_file(path_or_fileobj='tour_url.txt', path_in_repo=url_path or f'{run}/tour_url.txt', repo_id=results_repo, repo_type='dataset')
    say('tour published (public Space, random folder; link in the private results dataset: tour_url.txt)')


if __name__ == '__main__':
    from huggingface_hub import HfApi, snapshot_download
    run = sys.argv[1]; tok = os.environ.get('HF_TOKEN'); assert tok, 'HF_TOKEN missing'
    api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'
    if run.startswith(('world/', 'flashworld/')):                                            # generated worlds: every sample's viewer folder
        import glob
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=[f'{run}/*/index.html', f'{run}/*/world.*'], local_dir='pub', token=tok)
        pages = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, run, '*', 'index.html')))
        assert pages, f'no viewer pages in {repo}/{run}'
        for k, pg in enumerate(pages):
            publish(api, user, repo, run.split('/')[-1] + '-world', pg, url_path=f'{run}/{"tour_url.txt" if k == 0 else f"tour{k + 1}_url.txt"}')
            print('published', os.path.basename(pg), 'as', 'tour_url.txt' if k == 0 else f'tour{k + 1}_url.txt', flush=True)
        sys.exit(0)
    fp, co, rb = 'floorplan' in sys.argv[2:], 'collision' in sys.argv[2:], 'rebuild' in sys.argv[2:]
    extra = ([f'{run}/full/{x}' for x in ('model.ply', 'level.json', 'nav.json')] if fp or rb else []) + ([f'{run}/mesh/mesh.glb'] if co or rb else [])
    if rb and os.environ.get('TOUR_MODEL'): extra.append(f"{run}/{os.environ['TOUR_MODEL']}")
    d = snapshot_download(repo, repo_type='dataset', allow_patterns=[f'{run}/tour/*', f'{run}/tour/**'] + extra, local_dir='pub', token=tok)
    tour = os.path.join(d, run, 'tour'); assert os.path.exists(f'{tour}/index.html'), f'no built tour in {repo}/{run}/tour'
    if fp or co or rb: sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'final')); import make_tour
    if fp:
        print('floor plan redrawn:', make_tour.floorplan(os.path.join(d, run), tour), flush=True)
        api.upload_file(path_or_fileobj=f'{tour}/floorplan.png', path_in_repo=f'{run}/tour/floorplan.png', repo_id=repo, repo_type='dataset')
    if co:
        make_tour.collision(os.path.join(d, run), tour)
        api.upload_file(path_or_fileobj=f'{tour}/collision.glb', path_in_repo=f'{run}/tour/collision.glb', repo_id=repo, repo_type='dataset')
    if rb:                                                                                # scene file again from the model (current settings)
        import subprocess; W = os.path.abspath('st')
        subprocess.run(f'mkdir -p {W} && cd {W} && npm init -y >/dev/null && npm i -s --no-audit --no-fund playcanvas@2.23.2 '
                       '@playcanvas/supersplat-viewer@1.37.0 @playcanvas/splat-transform@3.10.1', shell=True, check=True)
        for f in ('scene.compressed.ply', 'scene.sog'):
            if os.path.exists(f'{tour}/{f}'): os.remove(f'{tour}/{f}')
        print('rebuilt:', make_tour.build(os.path.join(d, run), tour, f'{W}/node_modules/@playcanvas/supersplat-viewer/public',
                                          f'{W}/node_modules/.bin/splat-transform'), flush=True)
        api.upload_folder(folder_path=tour, path_in_repo=f'{run}/tour', repo_id=repo, repo_type='dataset')
    publish(api, user, repo, run, tour)
