"""Asset orientation and shape measurements shared by scene and video calibration."""
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def axis_rotations():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            if np.linalg.det(M) > 0:
                mats.append(M)
    return mats

def local_to_cam_cv(v_local, pose):
    """SAM 3D object-local -> PyTorch3D camera (row-vector convention) -> OpenCV camera."""
    s = np.asarray(pose["scale"])
    R = Rotation.from_quat(np.roll(pose["rotation_wxyz_p3d"], -1)).as_matrix()
    return ((v_local * s) @ R + np.asarray(pose["translation_p3d"])) * np.array([-1.0, -1.0, 1.0])

def support_area(Vw, tol=0.006):
    pts = Vw[Vw[:, 2] < Vw[:, 2].min() + tol][:, :2]
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0

def centroid_height_ratio(Vw, faces):
    tri = Vw[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    zc = (tri[:, :, 2].mean(1) * area).sum() / max(area.sum(), 1e-9)
    return (zc - Vw[:, 2].min()) / max(np.ptp(Vw[:, 2]), 1e-6)
