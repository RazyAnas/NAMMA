"""DL3DV-Benchmark (Hugging Face, gated: click "Agree" once on that account): handheld scenes, frames already extracted.
pick(csv)                      indoor scene with the fewest 'more reflection / more transparency' marks
fetch(scene, out, token)       scene = hash or 'auto' -> (hash, folder of its 960 x 540 frames)"""
REPO = 'DL3DV/DL3DV-Benchmark'

def pick(csv):
    """The meta columns are undocumented, so match on cell values: indoor (or bounded, not unbounded) first, then fewest 'more'."""
    import pandas as pd
    df = pd.read_csv(csv); print('DL3DV meta columns:', list(df.columns), flush=True)
    hc = next(c for c in df.columns if df[c].astype(str).str.fullmatch(r'[0-9a-f]{64}').mean() > 0.9)
    def score(row):
        v = ' '.join(str(x).lower() for x in row.values)
        indoor = 'indoor' in v and 'outdoor' not in v or ('bounded' in v and 'unbounded' not in v)
        return indoor * 10 - v.count('more')
    best = df.loc[df.apply(score, axis=1).idxmax()]; print('picked DL3DV scene:', best.to_dict(), flush=True)
    return str(best[hc])

def fetch(scene, out, token, res='images_4'):
    from huggingface_hub import hf_hub_download, snapshot_download
    if scene == 'auto': scene = pick(hf_hub_download(REPO, 'benchmark-meta.csv', repo_type='dataset', token=token))
    snapshot_download(REPO, repo_type='dataset', allow_patterns=[f'{scene}/nerfstudio/{res}/*'], local_dir=out, token=token)
    return scene, f'{out}/{scene}/nerfstudio/{res}'
