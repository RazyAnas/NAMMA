"""Floor-plan outline and walls (make_tour.walls_cells) on made-up rooms: splat centres on the floor, walls, trays, cupboards.
The 16 Nov switch room is the reason: a plain rectangle with cable trays under the ceiling and tall panels along a wall came
out as a dozen loose wall pieces. Each case here is that room or one way a home differs from it."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'colab', 'real_scenes', 'final'))
import make_tour

H = 2.8


def plane(p0, e1, e2, step=0.03):
    """Points on the parallelogram p0 + s e1 + t e2."""
    p0, e1, e2 = (np.asarray(v, float) for v in (p0, e1, e2)); n1, n2 = (max(2, int(np.linalg.norm(e) / step)) for e in (e1, e2))
    s, t = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2)); return p0 + s.reshape(-1, 1) * e1 + t.reshape(-1, 1) * e2


def wall(x0, z0, x1, z1, top=H, bottom=0.0): return plane([x0, bottom, z0], [x1 - x0, 0, z1 - z0], [0, top - bottom, 0])
def floor(x0, z0, x1, z1, y=0.0): return plane([x0, y, z0], [x1 - x0, 0, 0], [0, 0, z1 - z0], 0.04)


def turn(P, deg):
    a = np.radians(deg); R = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]]); return P @ R.T


def run(parts, path, deg=0.0, pad=0.8):
    rng = np.random.default_rng(0); Q = turn(np.concatenate(parts), deg); Q = Q + rng.normal(0, 0.01, Q.shape)
    path = turn(np.asarray(path, float), deg); lo, hi = Q.min(0) - pad, Q.max(0) + pad; lo[1], hi[1] = -0.05 * H, 1.05 * H
    return make_tour.walls_cells(Q, np.full(len(Q), 0.8), lo, hi, H, path), (lambda xz: turn(np.c_[xz[0], 0, xz[1]], deg)[0, [0, 2]])


def loop(x0, z0, x1, z1, n=60, y=1.35):
    t = np.linspace(0, 2 * np.pi, n); return np.c_[(x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(t), np.full(n, y), (z0 + z1) / 2 + (z1 - z0) / 2 * np.sin(t)]


def room(x0, z0, x1, z1): return [floor(x0, z0, x1, z1), wall(x0, z0, x1, z0), wall(x1, z0, x1, z1), wall(x1, z1, x0, z1), wall(x0, z1, x0, z0)]


def corners_match(outline, want, to, tol=0.2):
    got = np.array(outline); assert len(got) == len(want), f'{len(got)} corners, wanted {len(want)}: {outline}'
    for c in want:
        d = np.linalg.norm(got - to(c), axis=1).min(); assert d < tol, f'no corner within {tol} m of {c} (nearest {d:.2f} m)'


def length(ws): return sum(float(np.hypot(w[2] - w[0], w[3] - w[1])) for w in ws)


def switch_room():
    """6 x 4 m, cable trays across it under the ceiling, a row of 2 m panels along one wall, a tall cabinet in a corner."""
    trays = [plane([0.3, 2.3, z], [5.4, 0, 0], [0, 0, 0.3]) for z in (0.8, 2.0, 3.2)] + [plane([x, 2.25, 0.3], [0.3, 0, 0], [0, 0, 3.4]) for x in (1.5, 4.5)]
    panels = [wall(5.5, 0.6, 5.5, 3.6, top=2.0), plane([5.5, 2.0, 0.6], [0.5, 0, 0], [0, 0, 3.0])]
    cabinet = [wall(0.0, 3.4, 0.6, 3.4, top=2.1), wall(0.6, 3.4, 0.6, 4.0, top=2.1)]
    return room(0, 0, 6, 4) + trays + panels + cabinet


def test_rectangle_with_trays_and_panels():
    wc, to = run(switch_room(), loop(1.2, 1.0, 4.6, 3.0))
    assert wc['source'] == 'cells' and wc['open'] == [], wc
    corners_match(wc['outline'], [(0, 0), (6, 0), (6, 4), (0, 4)], to)
    assert len(wc['walls']) == 4 and abs(length(wc['walls']) - 20) < 0.8, wc['walls']        # four sides, nothing inside the room


@pytest.mark.parametrize('deg', [20.0, -33.0, 44.0])
def test_turned_room(deg):
    wc, to = run(switch_room(), loop(1.2, 1.0, 4.6, 3.0), deg)
    corners_match(wc['outline'], [(0, 0), (6, 0), (6, 4), (0, 4)], to); assert len(wc['walls']) == 4
    assert abs(((wc['yaw'] - deg + 45) % 90) - 45) < 1.5, wc['yaw']                           # the walls' direction, mod 90 degrees


def test_l_shape():
    parts = [floor(0, 0, 6, 3), floor(0, 3, 3, 6), wall(0, 0, 6, 0), wall(6, 0, 6, 3), wall(6, 3, 3, 3), wall(3, 3, 3, 6), wall(3, 6, 0, 6), wall(0, 6, 0, 0)]
    path = np.r_[loop(1, 0.8, 5, 2.2), loop(0.8, 1.5, 2.2, 5.2)]
    wc, to = run(parts, path, 12.0)
    corners_match(wc['outline'], [(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)], to); assert wc['open'] == []


def test_two_rooms_with_a_door():
    """A wall across the middle with a 0.9 m doorway the walk goes through: one outline, the dividing wall with its door."""
    parts = room(0, 0, 8, 4) + [wall(4, 0, 4, 1.5), wall(4, 2.4, 4, 4), wall(4, 1.5, 4, 2.4, bottom=2.1)]   # lintel over the door
    t = np.linspace(0, 1, 40); through = np.c_[1 + 6 * t, np.full(40, 1.35), np.full(40, 1.95)]
    wc, to = run(parts, np.r_[loop(0.8, 0.8, 3.2, 3.2), through, loop(4.8, 0.8, 7.2, 3.2)])
    corners_match(wc['outline'], [(0, 0), (8, 0), (8, 4), (0, 4)], to)
    mid = [w for w in wc['walls'] if abs(w[0] - 4) < 0.2 and abs(w[2] - 4) < 0.2]
    assert 2.0 < length(mid) < 3.4, mid                                                    # both parts of the wall, not the doorway
    assert not any(min(w[1], w[3]) < 1.95 < max(w[1], w[3]) for w in mid), f'the doorway is walled up: {mid}'


def test_side_never_filmed_is_open_not_a_wall():
    """The camera never turned to one end: floor and side walls stop at x = 5, nothing there. That side must not be a wall."""
    parts = [floor(0, 0, 5, 4), wall(0, 0, 5, 0), wall(5, 4, 0, 4), wall(0, 4, 0, 0)]
    wc, to = run(parts, loop(0.8, 0.8, 4.0, 3.2))
    assert len(wc['outline']) == 4 and len(wc['open']) == 1, wc
    o = wc['open'][0]; assert abs(o[0] - 5) < 0.3 and abs(o[2] - 5) < 0.3 and abs(length([o]) - 4) < 0.6, o
    assert not any(abs(w[0] - 5) < 0.3 and abs(w[2] - 5) < 0.3 for w in wc['walls'])


def test_things_outside_the_window_are_not_the_room():
    """Splats far outside one wall (a building seen through the window) and loose floaters change nothing."""
    rng = np.random.default_rng(1); far = np.c_[rng.uniform(7.5, 9, 4000), rng.uniform(0, 2.6, 4000), rng.uniform(0, 4, 4000)]
    floaters = np.c_[rng.uniform(-0.6, 6.6, 1500), rng.uniform(0, 2.8, 1500), rng.uniform(-0.6, 4.6, 1500)]
    wc, to = run(switch_room() + [far, floaters], loop(1.2, 1.0, 4.6, 3.0))
    corners_match(wc['outline'], [(0, 0), (6, 0), (6, 4), (0, 4)], to); assert len(wc['walls']) == 4


def test_too_little_to_go_on():
    Q = np.random.default_rng(2).uniform(0, 3, (300, 3))
    assert make_tour.walls_cells(Q, np.full(300, 0.8), np.array([-1., -0.1, -1]), np.array([4., 3, 4]), H) is None


def test_collision_uses_the_outline(tmp_path):
    """With an outline the walk collision is a floor slab plus the plan's sides (open ones too), not the SfM box's four walls."""
    import json, trimesh
    tour = tmp_path / 'tour'; tour.mkdir()
    json.dump(dict(scale=1.0, box=[-1, -0.1, -1, 7, 3, 5]), open(tour / 'plan.json', 'w'))
    trimesh.creation.box(extents=(1, 1, 1)).export(tour / 'room_collision.glb')              # would be 12 faces if it were used
    json.dump(dict(walls=[[0, 0, 6, 0], [6, 0, 6, 4], [6, 4, 0, 4]], open=[[0, 4, 0, 0]], outline=[[0, 0], [6, 0], [6, 4], [0, 4]], height=2.8),
              open(tour / 'walls.json', 'w'))
    make_tour.collision(str(tmp_path), str(tour)); m = trimesh.load(tour / 'collision.glb', force='mesh')
    assert len(m.faces) == 12 * 5                                                           # slab + 3 walls + the open side
    assert m.bounds[0][0] < -1 and m.bounds[1][0] > 7 and abs(m.bounds[1][1] - 2.8) < 0.01
