"""One training variant in its own process (one GPU each; run.py starts two at once in train mode)."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train_splat as T
job = json.loads(sys.argv[1]); print('variant', job['kw'], flush=True)
T.train(*job['args'], **job['kw'])
