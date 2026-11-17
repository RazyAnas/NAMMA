"""Photo spots of a tour (make_tour.spot_frames): evenly spaced along the walk, but never facing a bare wall.
Synthetic frames only: textured pictures and plain ones. No GPU, no network."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'colab' / 'real_scenes' / 'final'))
import make_tour  # noqa: E402

Image = pytest.importorskip('PIL.Image')


def frames(tmp_path, n, bare=()):
    """n frames 320x180: random blocks (things to see), or a plain wall with a soft gradient for the numbers in `bare`."""
    rng = np.random.default_rng(1); out = []
    for i in range(n):
        if i in bare: g = np.tile(np.linspace(180, 200, 320), (180, 1))
        else: g = np.kron(rng.integers(0, 255, (18, 32)), np.ones((10, 10)))
        p = tmp_path / f'{i:04d}.jpg'; Image.fromarray(g.astype(np.uint8)).convert('RGB').save(p, quality=90); out.append(str(p))
    return out


def test_detail_tells_a_wall_from_a_room(tmp_path):
    p = frames(tmp_path, 2, bare={0})
    assert make_tour.detail(p[0]) < 2 < 20 < make_tour.detail(p[1])


def test_even_spots_stay_when_all_have_something_to_see(tmp_path):
    p = frames(tmp_path, 71)
    assert make_tour.spot_frames(p) == [10, 20, 30, 40, 50, 60]


def test_a_spot_on_a_bare_wall_moves_to_a_nearby_view(tmp_path):
    p = frames(tmp_path, 71, bare=set(range(19, 22)))                       # the wall is around spot 2 (frame 20)
    s = make_tour.spot_frames(p)
    assert s[0] == 10 and s[2:] == [30, 40, 50, 60]
    assert s[1] not in range(19, 22) and abs(s[1] - 20) <= 3                # within a third of the spacing
    assert s == sorted(set(s))                                              # walk order, no frame twice


def test_a_wall_longer_than_the_window_keeps_the_best_there_is(tmp_path):
    p = frames(tmp_path, 71, bare=set(range(14, 27)))
    s = make_tour.spot_frames(p)
    assert len(s) == 6 and abs(s[1] - 20) <= 3 and s == sorted(set(s))


def test_short_walks_are_left_alone(tmp_path):
    p = frames(tmp_path, 12, bare={3})
    assert make_tour.spot_frames(p) == [1, 3, 4, 6, 7, 9]
