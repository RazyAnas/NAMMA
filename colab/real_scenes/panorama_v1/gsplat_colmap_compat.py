"""Compatibility patch for gsplat v1.5.3's legacy COLMAP parser.

The gsplat v1.5.3 example parser imports SceneManager from an old pycolmap fork.
Kaggle's current official pycolmap exposes Reconstruction instead. This patch
updates only the isolated gsplat checkout used by panorama_v1.
"""
from pathlib import Path


def patch_gsplat_colmap_parser(gsplat_root: Path) -> None:
    parser_path = Path(gsplat_root) / "examples" / "datasets" / "colmap.py"
    if not parser_path.is_file():
        raise RuntimeError(f"gsplat COLMAP parser is missing: {parser_path}")

    source = parser_path.read_text(encoding="utf-8")
    if "from pycolmap import SceneManager" not in source:
        if "pycolmap.Reconstruction(colmap_dir)" in source and "cam_from_world()" in source:
            return
        raise RuntimeError(
            "Unrecognized gsplat COLMAP parser; refusing an unsafe pycolmap compatibility edit."
        )

    replacements = (
        (
            "from pycolmap import SceneManager",
            "import pycolmap",
            "pycolmap import",
        ),
        (
            """        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images""",
            """        reconstruction = pycolmap.Reconstruction(colmap_dir)

        # Extract extrinsic matrices in world-to-camera format.
        imdata = reconstruction.images""",
            "Reconstruction loading",
        ),
        (
            """        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)""",
            """        for k, im in imdata.items():
            rot = im.cam_from_world().rotation.matrix()
            trans = im.cam_from_world().translation.reshape(3, 1)""",
            "camera pose API",
        ),
        (
            """            # camera intrinsics
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
                camtype = "fisheye""",
            """            # camera intrinsics
            cam = reconstruction.cameras[camera_id]
            cam_params = cam.params
            type_ = int(cam.model)
            # SIMPLE_PINHOLE=0, SIMPLE_RADIAL=2, RADIAL=3 have one focal length.
            if type_ in (0, 2, 3):
                fx = fy = cam_params[0]
                cx, cy = cam_params[1], cam_params[2]
            else:  # PINHOLE=1, OPENCV=4, OPENCV_FISHEYE=5
                fx, fy = cam_params[0], cam_params[1]
                cx, cy = cam_params[2], cam_params[3]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            if type_ in (0, 1):  # SIMPLE_PINHOLE or PINHOLE
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 2:  # SIMPLE_RADIAL
                params = np.array([cam_params[3], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3:  # RADIAL
                params = np.array([cam_params[3], cam_params[4], 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4:  # OPENCV
                params = np.array([cam_params[4], cam_params[5], cam_params[6], cam_params[7]], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5:  # OPENCV_FISHEYE
                params = np.array([cam_params[4], cam_params[5], cam_params[6], cam_params[7]], dtype=np.float32)
                camtype = "fisheye""",
            "camera model API",
        ),
        (
            """        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)""",
            """        # 3D points and {image_name -> [point_idx]}
        points3D_dict = reconstruction.points3D
        point_ids = sorted(points3D_dict.keys())
        points = np.array([points3D_dict[pid].xyz for pid in point_ids], dtype=np.float32)
        points_err = np.array([points3D_dict[pid].error for pid in point_ids], dtype=np.float32)
        points_rgb = np.array([points3D_dict[pid].color for pid in point_ids], dtype=np.uint8)
        point_indices = dict()

        point_id_to_idx = {pid: idx for idx, pid in enumerate(point_ids)}
        for point_id in point_ids:
            for track_elem in points3D_dict[point_id].track.elements:
                image_name = reconstruction.images[track_elem.image_id].name
                point_indices.setdefault(image_name, []).append(point_id_to_idx[point_id])""",
            "3D point and track API",
        ),
    )

    for old, new, label in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one legacy gsplat {label} block, found {count}; "
                "refusing to continue with a partially patched parser."
            )
        source = source.replace(old, new, 1)

    if "SceneManager" in source or "manager." in source:
        raise RuntimeError("Legacy SceneManager references remain in the patched gsplat parser.")

    compile(source, str(parser_path), "exec")
    parser_path.write_text(source, encoding="utf-8")
