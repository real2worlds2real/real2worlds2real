"""Legacy fixed-articulation fit for provisional placement and diagnostics.

Deployment robot_joints calibration now refines uncertain commands with
visual_render_pipeline --joint-prior. Trimming may conceal misposed links; a
low residual here alone must not certify deployed metric scale.

apps/robot_cloud.py poses the robot's own USD at the joint readout of the capture (configs/<name>/robot_state.json)
and writes its visible surface in metres (parsing/robot_cloud.npz). The photo's and the views' robot points (SAM 3
masks lifted with the point map and the views' depth, in the photo's units, before any scale) are the same body at an
unknown scale: standing on the floor, it is fitted upright (its z along the support's normal), over yaw, position and
one scale, by trimmed ICP (the closest `keep` share of the model's points drive it: what the table hides of the robot,
its far side, finds nothing near) from a grid of yaws and scales; the scale (map units per metre) gives the scene's
metres per unit. The robot's whole body counts, not a part's width in one view. Pose errors and missing surface support
must be checked by the current image/CAD calibration route."""
import numpy as np
from scipy.spatial import cKDTree


def rot_to(a, b):
    """The rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V / (1 + c)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def icp(M, tree, S, k, yaw, t, keep, iters):
    for _ in range(iters):
        T = k * M @ rot_z(yaw).T + t
        d, j = tree.query(T)
        sel = d <= np.quantile(d, keep)
        A, B = M[sel], S[j[sel]]
        a0, b0 = A.mean(0), B.mean(0)
        Ac, Bc = A - a0, B - b0
        H = Ac[:, :2].T @ Bc[:, :2]
        yaw = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
        Rz = rot_z(yaw)
        k = float(np.sum((Ac @ Rz.T) * Bc) / np.sum(Ac * Ac))
        t = b0 - k * a0 @ Rz.T
    d, _ = tree.query(k * M @ rot_z(yaw).T + t)
    return k, yaw, t, d


def fit(model, scene, up, keep=0.6):
    """{scale (metres per map unit), residual (median of the kept, in metres), share within 2 cm, yaw, ...} of the robot
    model (metres, z up, floor at 0) fitted to its scene points (map units) standing along `up`."""
    R0 = rot_to(np.array([0.0, 0.0, 1.0]), up)  # model z -> up
    S = scene @ R0  # the scene in a frame whose z is up
    rng = np.random.default_rng(0)
    if len(S) > 60000:
        S = S[rng.permutation(len(S))[:60000]]
    vox = np.unique(np.floor(model / 0.015).astype(np.int64), axis=0, return_index=True)[1]
    M = model[vox]
    tree = cKDTree(S)
    c_s = np.median(S, 0)
    # a scale to start from: the scene's robot spans (its width across, robust) against the model's
    span_s = np.percentile(S[:, :2], 95, 0) - np.percentile(S[:, :2], 5, 0)
    span_m = np.percentile(M[:, :2], 95, 0) - np.percentile(M[:, :2], 5, 0)
    k0 = float(np.linalg.norm(span_s) / np.linalg.norm(span_m))
    best = None
    for f in (0.6, 0.8, 1.0, 1.25, 1.6):
        for yaw in np.radians(np.arange(0, 360, 30)):
            k = k0 * f
            t = c_s - k * np.median(M, 0) @ rot_z(yaw).T
            k, y, t, d = icp(M, tree, S, k, yaw, t, keep, 15)
            kept = np.sort(d)[: int(keep * len(M))].mean() / k  # in metres, comparable across scales
            if best is None or kept < best[0]:
                best = (kept, k, y, t)
    _, k, y, t = best
    k, y, t, d = icp(M, tree, S, k, y, t, keep, 60)
    dm = d / k  # metres
    kept = np.sort(dm)[: int(keep * len(M))]
    return {"scale": 1.0 / k, "residual_m": float(np.median(kept)), "within_2cm": float((dm < 0.02).mean()),
            "yaw_deg": float(np.degrees(y)), "scene_points": int(len(S)), "model_points": int(len(M)),
            "base_in_map": (R0 @ t).tolist()}
