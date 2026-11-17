"""The pose-error maths of the AR accuracy check (scripts/ar/check_drsvpt.py) and the splat reader it builds its map from
(scripts/ar/tour_cloud.py). The model itself runs only in the ar-check workflow."""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'ar'))
import check_drsvpt, tour_cloud


def rot_y(deg):
    a = np.radians(deg); return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_pose_errors():
    c2w = np.eye(4); c2w[:3, :3] = rot_y(30); c2w[:3, 3] = [1.0, 1.4, -2.0]
    assert np.allclose(check_drsvpt.errors(np.linalg.inv(c2w), c2w), (0, 0), atol=1e-4)
    off = c2w.copy(); off[:3, :3] = rot_y(37); off[:3, 3] += [0.03, 0, 0.04]                # 7 degrees and 5 cm away
    cm, deg = check_drsvpt.errors(np.linalg.inv(off), c2w); assert abs(cm - 5) < 1e-3 and abs(deg - 7) < 1e-3
    up = c2w.copy(); up[:3] = check_drsvpt.ZUP @ up[:3]                                     # the same camera in the z-up frame
    T = np.linalg.inv(c2w); T[:3, :3] = T[:3, :3] @ check_drsvpt.ZUP.T                      # a pose found in a cloud turned z up
    assert np.allclose(check_drsvpt.errors(T, up), (0, 0), atol=1e-4)
    assert abs(np.linalg.det(check_drsvpt.ZUP) - 1) < 1e-9 and np.allclose(check_drsvpt.ZUP @ [0, 1, 0], [0, 0, 1])


def test_summary_counts():
    rows = [dict(cm=3, deg=1, s=20), dict(cm=8, deg=2, s=22), dict(cm=40, deg=3, s=30), dict(cm=4, deg=9, s=21)]
    s = check_drsvpt.summary(rows)
    assert s['n'] == 4 and s['within_5cm_5deg'] == 0.25 and s['within_10cm_5deg'] == 0.5 and s['within_25cm_10deg'] == 0.75 and s['median_cm'] == 6.0


def packed(x, bits):
    out, shift = 0, sum(bits)
    for v, b in zip(x, bits): shift -= b; out |= int(round(v * ((1 << b) - 1))) << shift
    return out


def test_reads_a_compressed_splat(tmp_path):
    """Two splats in one chunk, written the way splat-transform does: position, size and opacity come back in room axes."""
    head = 'ply\nformat binary_little_endian 1.0\nelement chunk 1\n' + ''.join(f'property float {n}\n' for n in (
        'min_x min_y min_z max_x max_y max_z min_scale_x min_scale_y min_scale_z max_scale_x max_scale_y max_scale_z min_r min_g min_b max_r max_g max_b').split())
    head += 'element vertex 2\nproperty uint packed_position\nproperty uint packed_rotation\nproperty uint packed_scale\nproperty uint packed_color\nend_header\n'
    chunk = struct.pack('<18f', -2, -3, -1, 2, 1, 3, -5, -5, -5, -3, -3, -3, 0, 0, 0, 1, 1, 1)
    v = [(packed((0, 0, 0), (11, 10, 11)), 0, packed((1, 1, 1), (11, 10, 11)), packed((1, 0, 0, 1), (8, 8, 8, 8))),
         (packed((1, 1, 1), (11, 10, 11)), 0, packed((0, 0, 0), (11, 10, 11)), packed((0, 0, 0, 0.2), (8, 8, 8, 8)))]
    f = tmp_path / 'scene.compressed.ply'; f.write_bytes(head.encode() + chunk + b''.join(struct.pack('<4I', *x) for x in v))
    s = tour_cloud.read_cply(str(f))
    assert np.allclose(s['xyz'], [[2, 3, -1], [-2, -1, 3]]), s['xyz']                       # x and y negated: the viewer's flip undone
    assert np.allclose(s['size'], [np.exp(-3), np.exp(-5)]) and np.allclose(s['opacity'], [1, 0.2], atol=0.003)
    assert len(tour_cloud.cloud(str(f), grid=0.02, opacity=0.5)) == 1                       # only the solid one is a map point
