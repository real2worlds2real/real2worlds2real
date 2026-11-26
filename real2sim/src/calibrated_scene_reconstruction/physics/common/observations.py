"""Shared physics camera projection, mask measurements and temporal association."""
import numpy as np
from calibrated_scene_reconstruction.core.media import VideoCapture
from calibrated_scene_reconstruction.core.geometry import mask_iou

def world_to_cam(T_world_cam, X):
    """World points (..., 3) into the OpenCV camera frame of camera-to-world T."""
    T = np.asarray(T_world_cam, float)
    X = np.asarray(X, float)
    return (X - T[:3, 3]) @ T[:3, :3]

def project(K, T_world_cam, X):
    """Pixels (..., 2) and depth (...,) of world points; points behind the camera get NaN pixels."""
    Xc = world_to_cam(T_world_cam, X)
    z = Xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (Xc @ np.asarray(K, float).T)[..., :2] / z[..., None]
    uv[z <= 1e-6] = np.nan
    return uv, z

def backproject(K, uv, depth):
    """Camera-frame points (..., 3) of pixels with depth along the optical axis."""
    uv = np.asarray(uv, float)
    K = np.asarray(K, float)
    x = (uv[..., 0] - K[0, 2]) / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) / K[1, 1]
    d = np.asarray(depth, float)
    return np.stack([x * d, y * d, d], -1)

def cam_to_world(T_world_cam, Xc):
    T = np.asarray(T_world_cam, float)
    return np.asarray(Xc, float) @ T[:3, :3].T + T[:3, 3]

def mask_stats(mask):
    """(centroid (2,), bbox (4,) x0 y0 x1 y1, area) of a boolean mask; NaNs and 0 when empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.full(2, np.nan), np.full(4, np.nan), 0
    return (np.array([xs.mean(), ys.mean()]), np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float),
            len(xs))

def iou(a, b):
    return float(mask_iou(a, b, empty=1.0, coerce=True))

def depth_centroid(mask, points, T_world_cam=None, erode=2):
    """Median 3D point of a mask over a camera-frame point map (H, W, 3) (MoGe-2 `points`), in the world frame when
    T is given. The mask is eroded first: depth at silhouette edges mixes object and background."""
    m = np.asarray(mask, bool)
    for _ in range(erode):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    P = np.asarray(points, float)[m]
    P = P[np.all(np.isfinite(P), 1)]
    if len(P) < 10:
        return np.full(3, np.nan)
    c = np.median(P, 0)
    return cam_to_world(T_world_cam, c) if T_world_cam is not None else c

def silhouette(K, T_world_cam, pts_world, shape, radius=1):
    """Splatted silhouette (H, W) of world surface points (dense enough samples of the object mesh) seen by the camera;
    the sim-side counterpart of a SAM 3 mask when no renderer is in the loop."""
    H, W = shape
    uv, _z = project(K, T_world_cam, pts_world)
    ok = np.all(np.isfinite(uv), 1)
    u, v = np.round(uv[ok]).astype(int).T
    m = np.zeros((H, W), bool)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            k = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            m[vv[k], uu[k]] = True
    return m

def mesh_silhouette(K, T_world_cam, vertices, faces, shape):
    """Project the actual triangles, retaining openings and disconnected parts."""
    from calibrated_scene_reconstruction.core.geometry.raster import triangle_mask

    uv, depth = project(K, T_world_cam, vertices)
    if (depth <= .05).any():
        return np.zeros(shape, bool)
    return triangle_mask(uv, faces, shape)

def convex_silhouette(K, T_world_cam, pts_world, shape):
    """Filled silhouette (H, W) of a convex body from its vertices: the convex hull of their projections, exact for a
    convex primitive (the tabletop twin's object) where splatting (silhouette) leaves holes or fattens the edge."""
    from scipy.spatial import ConvexHull, QhullError

    H, W = shape
    m = np.zeros((H, W), bool)
    uv, _z = project(K, T_world_cam, pts_world)
    uv = uv[np.all(np.isfinite(uv), 1)]
    if len(uv) < 3:
        return m
    try:
        hull = uv[ConvexHull(uv).vertices]  # counter-clockwise
    except QhullError:
        return m
    x0, y0 = np.maximum(np.floor(hull.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(hull.max(0)).astype(int) + 1, [W, H])
    if x1 <= x0 or y1 <= y0:
        return m
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))  # OpenCV: pixel centres at integers
    inside = np.ones(gx.shape, bool)
    for a, b in zip(hull, np.roll(hull, -1, 0)):
        inside &= (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0]) >= 0
    m[y0:y1, x0:x1] = inside
    return m

def associate(cands, prev_centroid, max_jump=80.0):
    """Pick one instance per frame from [{"mask", "score"}]: the highest score on the first frame, then the nearest to
    the last known centroid within `max_jump` px, else None (occluded): jumping to the best-scoring other instance
    swaps objects when the tracked one hides behind the hand.
    Returns the chosen dict or None."""
    if not cands:
        return None
    if prev_centroid is not None and np.all(np.isfinite(prev_centroid)):
        d = [np.linalg.norm(mask_stats(c["mask"])[0] - prev_centroid) for c in cands]
        i = int(np.nanargmin(d))
        return cands[i] if d[i] <= max_jump else None
    return max(cands, key=lambda c: c["score"])

def track_from_masks(masks, scores=None, t=None):
    """Track arrays (see module doc, without masks) from per-frame masks (None where the object was not found)."""
    F = len(masks)
    out = {"t": np.arange(F, dtype=float) if t is None else np.asarray(t, float), "valid": np.zeros(F, bool),
           "centroid": np.full((F, 2), np.nan), "bbox": np.full((F, 4), np.nan), "area": np.zeros(F),
           "score": np.zeros(F) if scores is None else np.asarray(scores, float)}
    for i, m in enumerate(masks):
        if m is None:
            continue
        c, b, a = mask_stats(m)
        out["valid"][i], out["centroid"][i], out["bbox"][i], out["area"][i] = a > 0, c, b, a
    return out

def read_video(path, every=1):
    import cv2

    cap = VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % every == 0:
            frames.append(f[:, :, ::-1].copy())
        k += 1
    cap.release()
    return np.stack(frames), fps / every
