#!/usr/bin/env python3
"""Deploy the NAMMA site (the folder site/) to the static Hugging Face Space <user>/namma, with a fresh gallery.json.

Runs on a GitHub runner (.github/workflows/site.yml) with `huggingface_hub` installed and HF_TOKEN in the environment:
  1. whoami -> the user name
  2. for every tour Space named in site/gallery.src.json: the top-level folders that look like a published tour
     (<run id>-<20 hex characters>, as written by colab/real_scenes/v3/tour_publish.py), and for each one a cover picture and
     a few numbers. A tour folder is what colab/real_scenes/final/make_tour.py builds: index.html, settings.json, tour.json
     (room_box, scene_mb), overlay.json (spots with thumbs/spot<N>.jpg), walls.json (the room outline in metres),
     floorplan.png, thumbs/, and the model as scene.compressed.ply, scene.sog or lod/. Only small files are read: the two
     or three JSON files and, for scene.compressed.ply, the first 2 KB (its header holds the splat count).
  3. gallery.json = { generated, entries: [{ id, run, title, kind, created, url, cover, splats?, area_m2?, size_mb? }] },
     newest first, without the runs marked "public": false
     plus the tours other people submitted from the site: the merged files entries/*.json of the public dataset
     <user>/namma-gallery (site/js/api/jobs.js submitToGallery opens a pull request there; merging it is the owner's yes)
  4. the Space <user>/namma (static, public) is created when missing and site/ + gallery.json are uploaded in one commit;
     the Space variable SITE_OWNER tells the page whose namma-gallery dataset takes the submissions.

  python scripts/site_deploy.py                         # deploy
  python scripts/site_deploy.py --dry-run --out DIR     # only write DIR/gallery.json, upload nothing
  python scripts/site_deploy.py --make-gallery-repo     # also create the public dataset <user>/namma-gallery when missing

The token is never printed. The tour links in gallery.json are public by the owner's choice (gallery.src.json).
"""
import argparse, json, os, re, shutil, sys, tempfile, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
FOLDER_RE = re.compile(r'^\d{8}-\d{4}.*-[0-9a-f]{20}$')
IMAGES = ('.jpg', '.jpeg', '.png', '.webp')
COVER_NAMES = ('cover', 'thumb', 'preview', 'input')              # top-level pictures tried when a tour has no spot thumbnails
SCENE_FILES = ('scene.compressed.ply', 'scene.sog')
BOX_PAD = 1.24                                                    # make_tour.plan pads the room box by 12 % on each side
KEEP_OUT = ('.DS_Store', 'api.md', '__pycache__', 'gallery.json')  # not part of the published page (gallery.json is written here)
FRESH_SPACE = {'.gitattributes', 'README.md', 'index.html', 'style.css'}   # what a just-created static Space may hold
GALLERY_REPO = 'namma-gallery'                                    # GALLERY in site/js/api/jobs.js


def subdomain(repo_id):
    """'User/namma_tour' -> 'user-namma-tour': the host name part of a static Space."""
    return re.sub(r'[^a-z0-9-]', '-', repo_id.lower())


def space_url(repo_id, path=''):
    return f'https://{subdomain(repo_id)}.static.hf.space/' + urllib.parse.quote(path)


def run_id(folder):
    """The run of a published tour folder: the name minus '-' and the 20 random hex characters."""
    return folder[:-21]


def created(run):
    """'20261025-2336...' -> '2026-10-25T23:36:00' (None when the run id does not start with a real date)."""
    try: return datetime.strptime(run[:13], '%Y%m%d-%H%M').strftime('%Y-%m-%dT%H:%M:00')
    except ValueError: return None


def is_file(item):
    return getattr(item, 'size', None) is not None                # RepoFile has a size, RepoFolder has none


def natural(name):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', name)]


def hub_reader(timeout=30):
    """read(repo, path, limit=None, repo_type='space') -> bytes of a file of a PUBLIC repo, without a token (a tour the public
    cannot read does not belong in the gallery). `limit`: only the first bytes (HTTP Range)."""
    def read(repo_id, path, limit=None, repo_type='space'):
        url = f'https://huggingface.co/{repo_type}s/{repo_id}/resolve/main/' + urllib.parse.quote(path)
        head = {'User-Agent': 'namma-site-deploy'}
        if limit: head['Range'] = f'bytes=0-{int(limit) - 1}'
        with urllib.request.urlopen(urllib.request.Request(url, headers=head), timeout=timeout) as r:
            return r.read(int(limit) if limit else 4_000_000)
    return read


def tour_folders(api, space):
    """Published tour folders of a Space: [(folder, last commit date or '')]."""
    out = []
    for item in api.list_repo_tree(space, repo_type='space', recursive=False, expand=True):
        name = item.path
        if is_file(item) or '/' in name or not FOLDER_RE.match(name): continue
        lc = getattr(item, 'last_commit', None); date = getattr(lc, 'date', None) if lc is not None else None
        if date is None and isinstance(lc, dict): date = lc.get('date')
        out.append((name, date.isoformat() if hasattr(date, 'isoformat') else str(date or '')))
    return out


def tour_meta(api, read, space, folder):
    """Cover and numbers of one tour folder: dict(cover=<path in the Space or None>, splats?, area_m2?, size_mb?), or None when
    the folder holds no page. Reads only tour.json, overlay.json, walls.json and the head of scene.compressed.ply."""
    files = {}
    for item in api.list_repo_tree(space, path_in_repo=folder, repo_type='space', recursive=False):
        files[item.path[len(folder) + 1:]] = getattr(item, 'size', None)
    if 'index.html' not in files: return None

    def load(name):
        if files.get(name) is None or files[name] > 4_000_000: return {}
        try: v = json.loads(read(space, f'{folder}/{name}')); return v if isinstance(v, dict) else {}
        except Exception: return {}

    meta = dict(cover=None); tour = load('tour.json'); overlay = load('overlay.json')
    # cover: the first spot's thumbnail (a real filmed frame), then anything in thumbs/, then a picture beside the page
    spots = overlay.get('spots') if isinstance(overlay.get('spots'), list) else []
    thumb = next((s.get('thumb') for s in spots if isinstance(s, dict) and isinstance(s.get('thumb'), str)), None)
    if thumb and '..' not in thumb and not thumb.startswith('/') and thumb.lower().endswith(IMAGES): meta['cover'] = f'{folder}/{thumb}'
    if not meta['cover'] and 'thumbs' in files:
        try:
            pics = sorted((i.path for i in api.list_repo_tree(space, path_in_repo=f'{folder}/thumbs', repo_type='space', recursive=False)
                           if is_file(i) and i.path.lower().endswith(IMAGES)), key=natural)
            if pics: meta['cover'] = pics[0]
        except Exception: pass
    if not meta['cover']:
        tops = sorted((n for n, size in files.items() if size is not None and n.lower().endswith(IMAGES)), key=natural)
        pick = next((n for stem in COVER_NAMES for n in tops if n.lower().rsplit('.', 1)[0] == stem), None)
        if pick: meta['cover'] = f'{folder}/{pick}'

    # splat count: written into tour.json by newer builds, else the header of the compressed .ply
    n = next((tour[k] for k in ('splats', 'n_splats', 'gaussians') if isinstance(tour.get(k), (int, float))), None)
    if n is None and files.get('scene.compressed.ply'):
        try:
            m = re.search(rb'element vertex (\d+)', read(space, f'{folder}/scene.compressed.ply', limit=2048))
            n = int(m.group(1)) if m else None
        except Exception: n = None
    if n and n > 0: meta['splats'] = int(n)

    # floor area: the measured outline (walls.json) when there is one, else the padded room box of tour.json without its padding
    area = None; walls = load('walls.json') if 'walls.json' in files else {}
    pts = walls.get('outline') if walls.get('source') != 'box' else None
    try:
        if pts and len(pts) >= 3:
            area = abs(sum(float(a[0]) * float(b[1]) - float(b[0]) * float(a[1]) for a, b in zip(pts, pts[1:] + pts[:1]))) / 2
        elif tour.get('room_box'):
            lo, hi = tour['room_box']; area = (hi[0] - lo[0]) * (hi[2] - lo[2]) / BOX_PAD ** 2
    except Exception: area = None
    if area and 1 <= area < 100000: meta['area_m2'] = round(float(area), 1)

    size = tour.get('scene_mb') if isinstance(tour.get('scene_mb'), (int, float)) else None
    if size is None:
        raw = sum(files[n] for n in SCENE_FILES if files.get(n)); size = raw / 1e6 if raw else None
    if size: meta['size_mb'] = round(float(size), 1)
    return meta


def clean(value, n):
    """Text from a submitted file: one line, no control characters, at most n characters."""
    return re.sub(r'[\x00-\x1f\x7f]+', ' ', str(value or '')).strip()[:n]


def https(value):
    return value if isinstance(value, str) and value.startswith('https://') and not re.search(r'[\s<>"\']', value) else None


def submitted(api, read, user, say=print):
    """Tours people submitted from the site and the owner accepted: the files entries/*.json on the main branch of the public
    dataset <user>/namma-gallery, each { user, id, title, kind, url, cover, created }. [] when the dataset is not there."""
    repo = f'{user}/{GALLERY_REPO}'; out = []
    try: files = [i.path for i in api.list_repo_tree(repo, path_in_repo='entries', repo_type='dataset', recursive=False) if is_file(i) and i.path.endswith('.json')]
    except Exception as e:
        say(f'submitted tours: {GALLERY_REPO} not read ({type(e).__name__}); none listed'); return out
    for path in sorted(files)[:500]:
        try: it = json.loads(read(repo, path, repo_type='dataset'))
        except Exception: continue
        url = https(it.get('url')) if isinstance(it, dict) else None
        if not url: continue
        who = re.sub(r'[^A-Za-z0-9._-]', '', str(it.get('user') or ''))[:60]
        when = clean(it.get('created') or it.get('submitted'), 25)
        e = dict(id=f"{who}--{clean(it.get('id'), 60)}", run=None, title=clean(it.get('title'), 80) or 'Untitled', kind='splat' if it.get('kind') == 'splat' else 'tour',
                 created=when if re.match(r'^\d{4}-\d{2}-\d{2}', when) else None, url=url, cover=https(it.get('cover')))
        if who: e['by'] = who
        out.append(e)
    say(f'submitted tours: {len(out)}')
    return out


def build_gallery(api, read, user, src, say=print, now=None):
    """The gallery of every tour Space in `src` (the parsed gallery.src.json): dict(generated, entries)."""
    titles = src.get('titles') or {}; best = {}
    for name in src.get('spaces') or []:
        space = f'{user}/{name}'
        try: folders = tour_folders(api, space)
        except Exception as e:
            say(f'tour Space {name}: not read ({type(e).__name__}), skipped'); continue
        say(f'tour Space {name}: {len(folders)} tour folders')
        for folder, date in folders:
            run = run_id(folder); info = titles.get(folder) or titles.get(run) or {}
            if info.get('public', True) is False: continue
            prev = best.get(run)
            if prev and (prev[2], prev[1]) >= (date, folder): continue          # one entry per run: the folder committed last
            best[run] = (space, folder, date, info)
    entries = []
    for run, (space, folder, _, info) in best.items():
        try: meta = tour_meta(api, read, space, folder)
        except Exception as e:
            say(f'  {run}: not read ({type(e).__name__}), listed without cover'); meta = dict(cover=None)
        if meta is None:
            say(f'  {run}: no index.html, skipped'); continue
        when = created(run)
        cover = info.get('cover') or (space_url(space, meta['cover']) if meta.get('cover') else None)
        e = dict(id=folder, run=run, title=info.get('title') or f"Tour {when[:10] if when else run}", kind=info.get('kind') or 'tour',
                 created=when, url=space_url(space, f'{folder}/index.html'), cover=cover)
        if isinstance(info.get('tags'), list): e['tags'] = [str(t) for t in info['tags']]
        e.update({k: meta[k] for k in ('splats', 'area_m2', 'size_mb') if k in meta})
        entries.append(e)
    have = {e['url'] for e in entries}
    entries += [e for e in submitted(api, read, user, say) if e['url'] not in have]
    entries.sort(key=lambda e: (e['created'] or '', e['id']), reverse=True)
    stamp = (now or datetime.now(timezone.utc)).strftime('%Y-%m-%dT%H:%M:%SZ')
    return dict(generated=stamp, entries=entries)


def stage_site(site_dir, gallery, dst, viewer=None):
    """A copy of site/ with gallery.json in it: exactly what the Space will hold. `viewer`: a folder with a built splat viewer
    (its index.html), copied to viewer/ (site/js/api/jobs.js links uploaded splat files to viewer/index.html?content=...)."""
    shutil.copytree(site_dir, dst, ignore=shutil.ignore_patterns(*KEEP_OUT))
    if viewer:
        assert os.path.isfile(os.path.join(viewer, 'index.html')), f'{viewer}: no index.html'
        shutil.copytree(viewer, os.path.join(dst, 'viewer'), dirs_exist_ok=True)
    (Path(dst) / 'gallery.json').write_text(json.dumps(gallery, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return dst


VIEWER_SETTINGS = dict(version=2, tonemapping='none', highPrecisionRendering=False, background=dict(color=[0.08, 0.08, 0.09]),
                       postEffectSettings=dict(sharpness=dict(enabled=False, amount=0), bloom=dict(enabled=False, intensity=0.1, blurLevel=2),
                                               grading=dict(enabled=False, brightness=1, contrast=1, saturation=1, tint=[1, 1, 1]),
                                               vignette=dict(enabled=False, intensity=0.5, inner=0.3, outer=0.75, curvature=1),
                                               fringing=dict(enabled=False, intensity=0.5)),
                       animTracks=[], cameras=[dict(initial=dict(position=[0, 1, -3], target=[0, 0, 0], fov=60))], annotations=[], startMode='default')
VIEWER_README = ('---\ntitle: NAMMA viewer\nemoji: 🧊\ncolorFrom: gray\ncolorTo: gray\nsdk: static\npinned: false\n---\n\n'
                 'Shows a splat file people uploaded on the NAMMA site: index.html?content=<address of the file>. No sign-in, nothing stored.\n')


def stage_viewer(viewer, dst):
    """The stock SuperSplat viewer (the package's public/ folder) as its own page: the file to show comes from ?content=,
    the settings are a plain default (settings.json). Tours have their own page (make_tour.py); this is for single files."""
    assert os.path.isfile(os.path.join(viewer, 'index.html')), f'{viewer}: no index.html'
    shutil.copytree(viewer, dst, dirs_exist_ok=True)
    p = Path(dst) / 'index.html'; html = p.read_text(encoding='utf-8'); mark = 'id="sse-bootstrap">'
    if mark in html:
        a = html.index(mark) + len(mark); c = html.index('</script>', a)
        html = html[:a] + '\n' + json.dumps(dict(settingsUrl='./settings.json')) + '\n        ' + html[c:]
    p.write_text(html.replace('<title>SuperSplat Viewer</title>', '<title>NAMMA viewer</title>'), encoding='utf-8')
    (Path(dst) / 'settings.json').write_text(json.dumps(VIEWER_SETTINGS), encoding='utf-8')
    (Path(dst) / 'README.md').write_text(VIEWER_README, encoding='utf-8')
    return dst


def deploy_viewer(api, user, viewer, space='namma', say=print):
    """<user>/<space>-viewer: a Space of its own, so the viewer never shares an origin (and so the visitor's tokens) with the site."""
    repo = f'{user}/{space}-viewer'
    api.create_repo(repo, repo_type='space', space_sdk='static', private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        api.upload_folder(folder_path=stage_viewer(viewer, os.path.join(tmp, 'viewer')), repo_id=repo, repo_type='space', delete_patterns='*',
                          commit_message='Viewer for uploaded splat files')
    say('viewer deployed:', space_url(repo)); return repo


def deploy(api, user, site_dir, gallery, space='namma', force=False, make_gallery_repo=False, viewer=None, say=print):
    repo = f'{user}/{space}'
    if viewer:
        try: deploy_viewer(api, user, viewer, space, say)
        except Exception as e: say(f'viewer not deployed ({type(e).__name__}: {str(e)[:120]}): uploaded splat files cannot be opened yet')
    api.create_repo(repo, repo_type='space', space_sdk='static', private=False, exist_ok=True)
    if make_gallery_repo: api.create_repo(f'{user}/{GALLERY_REPO}', repo_type='dataset', private=False, exist_ok=True)
    have = set(api.list_repo_files(repo, repo_type='space'))
    # the upload below removes every file that is not part of the site: never do that to a Space that holds something else
    if 'gallery.src.json' not in have and have - FRESH_SPACE and not force:
        raise SystemExit(f'The Space {space} already holds files that are not this site. Nothing was changed. Use --force to replace them.')
    with tempfile.TemporaryDirectory() as tmp:
        stage = stage_site(site_dir, gallery, os.path.join(tmp, 'site'))
        api.upload_folder(folder_path=stage, repo_id=repo, repo_type='space', delete_patterns='*',
                          commit_message=f"Site deploy, {len(gallery['entries'])} tours in the gallery")
    try: api.update_repo_settings(repo, repo_type='space', private=False)
    except Exception as e: say(f'visibility not changed ({type(e).__name__}): check that the Space is public')
    try: api.add_space_variable(repo, 'SITE_OWNER', user)          # public, read by the page (jobs.js siteOwner)
    except Exception as e: say(f'Space variable SITE_OWNER not set ({type(e).__name__}): set it by hand in the Space settings')
    say('site deployed:', space_url(repo))
    return repo


def main(argv=None, api=None, read=None):
    ap = argparse.ArgumentParser(description='Build gallery.json and deploy site/ to the Space <user>/namma.')
    ap.add_argument('--dry-run', action='store_true', help='write gallery.json into --out and upload nothing')
    ap.add_argument('--out', help='folder for gallery.json (needed with --dry-run)')
    ap.add_argument('--site', default=str(SITE), help='the site folder (default: site/)')
    ap.add_argument('--space', default='namma', help='name of the site Space (default: namma)')
    ap.add_argument('--user', help='Hugging Face user name (default: whoami of HF_TOKEN)')
    ap.add_argument('--force', action='store_true', help='replace a Space that holds other files')
    ap.add_argument('--viewer', help='folder of a built splat viewer to publish as viewer/ (default: none)')
    ap.add_argument('--make-gallery-repo', action='store_true', help='create the public dataset <user>/namma-gallery when missing (people submit tours to it)')
    a = ap.parse_args(argv)
    if a.dry_run and not a.out: ap.error('--dry-run needs --out DIR')
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get('HF_TOKEN') or None)
    read = read or hub_reader()
    user = a.user or api.whoami()['name']
    src = json.loads((Path(a.site) / 'gallery.src.json').read_text(encoding='utf-8'))
    gallery = build_gallery(api, read, user, src)
    if not gallery['entries'] and not a.dry_run:
        # every listing failed or nothing is published: keep the gallery the site shows now instead of emptying it
        try:
            old = json.loads(read(f'{user}/{a.space}', 'gallery.json'))
            if old.get('entries'): gallery = old; print(f"no tours found: keeping the current gallery ({len(old['entries'])} tours)")
        except Exception: print('no tours found and no current gallery: the gallery will be empty')
    print(f"gallery: {len(gallery['entries'])} tours")
    for e in gallery['entries']: print(f"  {(e['created'] or '?')[:19]:19}  {e['title']}" + ('' if e.get('cover') else '  (no cover)'))   # no links in the log (it is public; the gallery itself is the list)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        Path(a.out, 'gallery.json').write_text(json.dumps(gallery, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print('written:', os.path.join(a.out, 'gallery.json'))
    if a.dry_run:
        print('dry run: nothing uploaded'); return gallery
    deploy(api, user, a.site, gallery, space=a.space, force=a.force, make_gallery_repo=a.make_gallery_repo, viewer=a.viewer)
    return gallery


if __name__ == '__main__':
    main()
    sys.exit(0)
