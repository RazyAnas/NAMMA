# Compare a navmesh's standable / walkable cells with the truth navmesh in world coordinates (10 cm cells).
import sys, json, base64, zlib, numpy as np
def cells(p, key):
    d = json.load(open(p)); w, h = d['width'], d['height']; ox, oy = d['origin_floor_coordinates']; c = d['cell_size_m']
    b = d['movement']['standable_base64'] if key == 'standable' else d['walkable_grid_base64']
    g = np.frombuffer(zlib.decompress(base64.b64decode(b)), np.uint8).reshape(h, w)
    ys, xs = np.nonzero(g); return {(int(round((ox + (x + .5) * c) / c)), int(round((oy + (y + .5) * c) / c))) for x, y in zip(xs, ys)}
truth, test = sys.argv[1], sys.argv[2]; out = {}
for key in ('standable', 'walkable'):
    T, S = cells(truth, key), cells(test, key)
    out[key] = dict(truth=len(T), test=len(S), both=len(T & S), iou=len(T & S) / max(len(T | S), 1),
                    recall=len(T & S) / max(len(T), 1), false_cells=len(S - T))
print(json.dumps(out))
