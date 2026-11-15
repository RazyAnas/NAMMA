"""AI find (colab/real_scenes/final/ai_find.py) without a GPU: the geometry, the two-views-must-agree rule, the instant search
over the region index, and the agent loop driven by stand-in tools. The real models (CLIP, Grounding DINO, Qwen2.5-VL) only
run on Kaggle.   pytest -q --confcutdir=tests/v3 tests/v3/test_ai_find.py"""
import json, os, sys
import numpy as np

FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'colab', 'real_scenes', 'final')
sys.path.insert(0, os.path.abspath(FINAL))
import ai_find


def unit(v): v = np.asarray(v, np.float32); return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---- geometry
def test_lift_puts_the_box_centre_at_the_rendered_depth():
    K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]); c2w = np.eye(4); c2w[:3, 3] = [1, 2, 3]
    d = np.zeros((480, 640), np.float32); d[200:280, 380:460] = 2.0                         # a thing 2 m away, right of centre
    p = ai_find.lift(d, K, c2w, [380, 200, 460, 280])
    assert np.allclose(p, [1 + (420 - 320) * 2 / 500, 2 + (240 - 240) * 2 / 500, 3 + 2], atol=1e-6)
    R = np.array([[0, 0, 1.], [0, 1, 0], [-1, 0, 0]]); c2w2 = np.eye(4); c2w2[:3, :3] = R       # camera turned 90 deg: forward = world +x
    d2 = np.full((480, 640), 4.0, np.float32); assert np.allclose(ai_find.lift(d2, K, c2w2, [300, 220, 340, 260]), [4, 0, 0], atol=1e-6)
    assert ai_find.lift(np.zeros((480, 640), np.float32), K, c2w, [10, 10, 60, 60]) is None      # looks at nothing
    d[236:244, 416:424] = 9.0                                                                # a hole in the middle: the median ignores it
    assert abs(ai_find.lift(d, K, c2w, [380, 200, 460, 280])[2] - 5.0) < 1e-6


def test_to_tour_is_level_then_metres():
    L = np.eye(4); L[:3, 3] = [0, -1, 0]                                                     # floor at y = 1 in the model -> y = 0
    assert np.allclose(ai_find.to_tour([[2, 1, 4]], L, 0.5), [[1, 0, 2]]) and ai_find.to_tour([2, 1, 4], L, 0.5).shape == (1, 3)


# ---- two views must agree
def test_agree_needs_two_different_frames_on_the_same_place():
    c = [dict(pos=[1, 0.5, 1], frame=3, score=0.6), dict(pos=[1.2, 0.5, 1.1], frame=9, score=0.5),     # the thing, seen twice
         dict(pos=[4, 1, 0], frame=5, score=0.9),                                                      # a strong single view elsewhere
         dict(pos=[4.1, 1, 0], frame=5, score=0.8)]                                                    # the same frame again: still one view
    g = ai_find.agree(c)
    assert len(g) == 2 and g[0]['ok'] and g[0]['frames'] == [3, 9] and abs(g[0]['score'] - 0.55) < 1e-6
    assert np.allclose(g[0]['pos'], (np.array([1, 0.5, 1]) * 0.6 + np.array([1.2, 0.5, 1.1]) * 0.5) / 1.1)
    assert not g[1]['ok'] and g[1]['frames'] == [5] and g[1]['score'] == 0.9
    assert ai_find.agree([]) == [] and not ai_find.agree(c[:1])[0]['ok']
    far = [dict(pos=[0, 0, 0], frame=1, score=0.5), dict(pos=[0.6, 0, 0], frame=2, score=0.5)]          # 60 cm apart: not the same place
    assert len(ai_find.agree(far)) == 2 and not any(x['ok'] for x in ai_find.agree(far))


# ---- instant search over the region index
def room(rng, n=300, d=64):
    """A room of n regions with unrelated embeddings in a narrow band, like CLIP's (all share a common direction)."""
    base = unit(rng.normal(size=d)); E = unit(base + 0.6 * unit(rng.normal(size=(n, d))))
    items = [dict(p=[float(x) for x in rng.uniform(0, 6, 3)], f=int(rng.integers(0, 40))) for _ in range(n)]
    return base, E, items


def test_search_finds_a_thing_seen_from_two_frames_and_refuses_what_is_not_there():
    rng = np.random.default_rng(0); base, E, items = room(rng); thing = unit(base + 0.8 * unit(rng.normal(size=64)))
    for i, (f, off) in enumerate(((4, 0.0), (11, 0.15), (27, 0.25))):                         # the lamp: 3 frames, within 25 cm
        E[i] = unit(thing + 0.25 * unit(rng.normal(size=64))); items[i] = dict(p=[2.0 + off, 1.1, 3.0], f=f)
    r = ai_find.search(thing, E, items)
    assert r['found'] and set(r['found']['frames']) == {4, 11, 27} and set(r['found']['regions']) == {0, 1, 2}
    assert np.linalg.norm(np.array(r['found']['pos']) - [2.13, 1.1, 3.0]) < 0.15 and r['found']['score'] > r['threshold']
    other = unit(base + 0.8 * unit(rng.normal(size=64)))                                       # a thing that is not in the room
    r2 = ai_find.search(other, E, items)
    assert r2['found'] is None and r2['near'] is not None and len(r2['near']['pos']) == 3      # "closest match", never a pin
    # one view only (a photo of a lamp on the wall, a reflection): not enough
    E[1] = E[100]; E[2] = E[101]; r3 = ai_find.search(thing, E, items)
    assert r3['found'] is None and r3['near']['region'] == 0
    assert ai_find.search(thing, np.zeros((0, 64), np.float32), [])['found'] is None


def test_search_reference_cases_for_the_page(tmp_path):
    """The same cases as a file: tests/browser/poi_e2e.mjs feeds them to the page's aiSearch and must get the same answers."""
    rng = np.random.default_rng(1); base, E, items = room(rng, n=120, d=32); cases = []
    for k in range(6):
        q = unit(base + 0.8 * unit(rng.normal(size=32)))
        if k % 2 == 0:
            for i in range(2 + k // 2): E[10 * k + i] = unit(q + 0.2 * unit(rng.normal(size=32))); items[10 * k + i] = dict(p=[1.0 + k, 1.0, 2.0 + 0.1 * i], f=k * 5 + i)
        cases.append(q)
    out = [dict(q=[float(x) for x in q], want=ai_find.search(q, E, items)) for q in cases]
    assert [bool(c['want']['found']) for c in out] == [True, False, True, False, True, False]
    json.dump(dict(dim=32, items=items, E=[[float(x) for x in e] for e in E], cases=out), open(tmp_path / 'ai_search_cases.json', 'w'))
    if os.environ.get('WRITE_AI_CASES'): json.dump(json.load(open(tmp_path / 'ai_search_cases.json')), open(os.environ['WRITE_AI_CASES'], 'w'))


# ---- the agent loop
class Fake:
    """A room where frames 2, 5 and 8 show a fire extinguisher at (3, 1, 2), frame 6 shows a red bag the detector mistakes for
    one, and the detector only knows the thing as 'red cylinder' (the first names the planner gives find nothing)."""

    def __init__(self, real=(2, 5, 8), wrong=(6,), known='red cylinder', single=False):
        self.real, self.wrong, self.known, self.calls, self.single = set(real), set(wrong), known, [], single

    def phrases(self, q, fb): self.calls.append(('phrases', fb)); return ['fire extinguisher', 'extinguisher'] if fb is None else [self.known, 'red canister']

    def rank(self, names): return [6, 2, 5, 8, 1, 3]

    def detect(self, f, names):
        if self.known not in names: return []
        if f in self.real: return [([100, 100, 160, 220], 0.7 - 0.01 * f, self.known)]
        if f in self.wrong: return [([300, 50, 380, 150], 0.9, self.known)]
        return []

    def verify(self, f, box, q): return f in self.real

    def lift(self, f, box): return [3.0 + 0.02 * f, 1.0, 2.0] if not self.single or f == 2 else None

    def tools(self): return dict(phrases=self.phrases, rank=self.rank, detect=self.detect, verify=self.verify, lift=self.lift)


def test_agent_renames_the_thing_rejects_the_lookalike_and_needs_two_views():
    f = Fake(); r = ai_find.find_one('fire extinguisher', f.tools())
    assert r['found'] and r['confidence'] == 'high' and r['frames'] == [2, 5, 8] and r['name'] == 'red cylinder' and r['frame'] == 2
    assert np.linalg.norm(np.array(r['pos']) - [3.1, 1, 2]) < 0.1
    assert [c[0] for c in f.calls] == ['phrases', 'phrases'] and f.calls[1][1]['tried'] == ['fire extinguisher', 'extinguisher']   # told what failed
    tools_used = [t['tool'] for t in r['trace']]
    assert tools_used[:2] == ['phrases', 'rank'] and tools_used.count('agree') == 2 and tools_used.count('phrases') == 2
    ag = [t for t in r['trace'] if t['tool'] == 'agree'][-1]
    assert ag['rejected_by_verify'] == 1 and ag['candidates'] == 3                              # the red bag did not get through
    json.dumps(r)                                                                               # the trace is plain data


def test_agent_reports_one_view_as_low_confidence_and_nothing_as_not_found():
    r = ai_find.find_one('fire extinguisher', Fake(real=(2,)).tools())
    assert r['found'] and r['confidence'] == 'low' and r['frames'] == [2]                       # the admin is asked to check it
    r = ai_find.find_one('fire extinguisher', Fake(real=(2, 5), ).tools()); assert r['confidence'] == 'medium'
    r = ai_find.find_one('piano', Fake(real=(), wrong=(6,)).tools())
    assert not r['found'] and r['pos'] is None and r['confidence'] == 'none' and len([t for t in r['trace'] if t['tool'] == 'phrases']) == 2
    r = ai_find.find_one('piano', Fake(real=(2, 5), single=True).tools())                       # the second view looks at nothing (no depth)
    assert r['found'] and r['confidence'] == 'low'


def test_parse_names_and_slug():
    assert ai_find.parse_names('1. Fire extinguisher\n2) "red canister".\n- hose reel\n* a very long sentence that is not a name at all really') \
        == ['fire extinguisher', 'red canister', 'hose reel']
    assert ai_find.parse_names('sink, tap; wash basin') == ['sink', 'tap', 'wash basin'] and ai_find.parse_names('') == []
    assert ai_find.slug('Where can I wash my hands?') == 'where-can-i-wash-my-hands' and ai_find.slug('???') == 'thing'
