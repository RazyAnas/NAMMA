"""Publish a built tour folder (make_tour.py build) to the public static Hugging Face Space <user>/namma-tour, like splattour's
share links: each tour in a folder with a random name (only people with the link find it), no listing page. A private static
Space cannot serve its files to a browser (the page loads, its scene/scripts do not), so the owner chose public. The link is
NOT printed (GitHub logs of this repo are public): it goes to <run>/tour_url.txt in the private results dataset.
Used by gh_run.py after a run, and on its own (.github/workflows/tour-publish.yml, v3/PUBLISH.json) to republish a run's tour:
  python tour_publish.py <run, e.g. 20261026-0309>      (HF_TOKEN in the environment; tour taken from the results dataset)"""
import os, re, secrets, sys


def publish(api, user, results_repo, run, tour_dir, say=print):
    sp = f'{user}/namma-tour'; api.create_repo(sp, repo_type='space', space_sdk='static', private=False, exist_ok=True)
    for f in {x.split('/')[0] for x in api.list_repo_files(sp, repo_type='space') if '/' in x}:      # old guessable folders out
        if not re.fullmatch(r'\d{8}-\d{4}-[0-9a-f]{20}', f): api.delete_folder(f, repo_id=sp, repo_type='space')
    folder = f'{run}-{secrets.token_hex(10)}'
    api.upload_folder(folder_path=tour_dir, path_in_repo=folder, repo_id=sp, repo_type='space', ignore_patterns=['plan.json'])
    page = '<!doctype html><meta charset="utf-8"><title>NAMMA</title><body style="font:16px sans-serif;padding:24px;background:#111;color:#ddd"><h1>NAMMA 3D tours</h1><p>Open a tour with its link.</p>'
    open('tours_index.html', 'w').write(page); api.upload_file(path_or_fileobj='tours_index.html', path_in_repo='index.html', repo_id=sp, repo_type='space')
    open('space_readme.md', 'w').write('---\ntitle: NAMMA tour\nemoji: 🏠\ncolorFrom: gray\ncolorTo: gray\nsdk: static\npinned: false\n---\n')
    api.upload_file(path_or_fileobj='space_readme.md', path_in_repo='README.md', repo_id=sp, repo_type='space')
    try: api.update_repo_settings(sp, repo_type='space', private=False)
    except Exception: api.update_repo_visibility(sp, private=False, repo_type='space')
    open('tour_url.txt', 'w').write(f'https://{user}-namma-tour.static.hf.space/{folder}/index.html\n')
    api.upload_file(path_or_fileobj='tour_url.txt', path_in_repo=f'{run}/tour_url.txt', repo_id=results_repo, repo_type='dataset')
    say('tour published (public Space, random folder; link in the private results dataset: tour_url.txt)')


if __name__ == '__main__':
    from huggingface_hub import HfApi, snapshot_download
    run = sys.argv[1]; tok = os.environ.get('HF_TOKEN'); assert tok, 'HF_TOKEN missing'
    api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'
    d = snapshot_download(repo, repo_type='dataset', allow_patterns=[f'{run}/tour/*', f'{run}/tour/**'], local_dir='pub', token=tok)
    tour = os.path.join(d, run, 'tour'); assert os.path.exists(f'{tour}/index.html'), f'no built tour in {repo}/{run}/tour'
    publish(api, user, repo, run, tour)
