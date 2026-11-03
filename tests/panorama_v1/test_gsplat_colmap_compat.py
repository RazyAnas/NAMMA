from pathlib import Path

from gsplat_colmap_compat import patch_gsplat_colmap_parser


LEGACY_COLMAP_SOURCE = """import numpy as np
from pycolmap import SceneManager

def load(colmap_dir, camera_id, factor):
    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)

        # camera intrinsics
        cam = manager.cameras[camera_id]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= factor
        Ks_dict[camera_id] = K

        # Get distortion parameters.
        type_ = cam.camera_type
        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif type_ == 1 or type_ == "PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        if type_ == 2 or type_ == "SIMPLE_RADIAL":
            params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 3 or type_ == "RADIAL":
            params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 4 or type_ == "OPENCV":
            params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
            camtype = "fisheye"

    # 3D points and {image_name -> [point_idx]}
    points = manager.points3D.astype(np.float32)
    points_err = manager.point3D_errors.astype(np.float32)
    points_rgb = manager.point3D_colors.astype(np.uint8)
    point_indices = dict()

    image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
    for point_id, data in manager.point3D_id_to_images.items():
        for image_id, _ in data:
            image_name = image_id_to_name[image_id]
            point_idx = manager.point3D_id_to_point3D_idx[point_id]
            point_indices.setdefault(image_name, []).append(point_idx)
"""


def test_gsplat_colmap_parser_is_patched_to_official_pycolmap_and_is_idempotent(tmp_path: Path):
    gsplat_root = tmp_path / "gsplat"
    parser_path = gsplat_root / "examples" / "datasets" / "colmap.py"
    parser_path.parent.mkdir(parents=True)
    parser_path.write_text(LEGACY_COLMAP_SOURCE, encoding="utf-8")

    patch_gsplat_colmap_parser(gsplat_root)
    patched = parser_path.read_text(encoding="utf-8")

    assert "import pycolmap" in patched
    assert "pycolmap.Reconstruction(colmap_dir)" in patched
    assert "cam_from_world().rotation.matrix()" in patched
    assert "reconstruction.points3D" in patched
    assert "track_elem.image_id" in patched
    assert "SceneManager" not in patched
    assert "manager." not in patched

    patch_gsplat_colmap_parser(gsplat_root)
    assert parser_path.read_text(encoding="utf-8") == patched
