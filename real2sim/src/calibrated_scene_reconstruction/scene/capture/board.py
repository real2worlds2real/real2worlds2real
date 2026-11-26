"""The scene's scale from a ChArUco board lying in it (configs/<name>/charuco_board.json, the same file calibrated_scene_reconstruction.calib
reads): an option of recon.multiview, used whenever that file exists and the board is found.

A board's corners are known to a tenth of a millimetre, so where the board is seen the point map can be scaled to
them directly, with no measuring tape and no robot model. Its ChArUco corners are found in the photo and in every view
(calibrated_scene_reconstruction.calib's detector: markers decoded on the image upscaled x2, at least `min_corners` corners); the point map's
points at those pixels (bilinear; pixels off the map's valid mask dropped) are fitted to the board's own corner
coordinates by a similarity (Umeyama: rotation, translation, one scale), corners off by more than three times the
median residual dropped and the fit repeated. The scale that takes the map onto the board is the scene's scale.

It is a size measured on the board, not a depth: a depth map drawn with a field of view other than the camera's
(DA3 keeps its own K) keeps the board's shape at a wrong depth, and the size on the table is what a grasp needs. The
board shows the scale where it lies (on the task support, ideally), not the depth error further away.
Measured object sizes are not a deployment input.

The photo's own board decides: its points are the map's own. Without it in the photo, the corners seen in two or
more views are triangulated with DA3's camera poses (the joint solution, in the photo's units) and fitted the same
way. A view's own depth does not transfer: DA3's per-view depth drifts along a walk, and a close-up of the board
shows no floor for recon.multiview's per-view correction. One view alone: its depth,
with a warning."""
import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.capture.calib import board_detector, detect_image, load_board


def umeyama(src, dst):
    """(s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def sample(points, valid, px):
    """The map's points at sub-pixel positions (OpenCV's: pixel (i, j)'s centre at x = j, y = i; bilinear), and whether
    all four neighbours were valid."""
    H, W = valid.shape
    x, y = px[:, 0], px[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < W) & (y0 + 1 < H)
    x0, y0 = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    P = (points[y0, x0] * (1 - fx) * (1 - fy) + points[y0, x0 + 1] * fx * (1 - fy) +
         points[y0 + 1, x0] * (1 - fx) * fy + points[y0 + 1, x0 + 1] * fx * fy)
    ok &= valid[y0, x0] & valid[y0, x0 + 1] & valid[y0 + 1, x0] & valid[y0 + 1, x0 + 1]
    return P, ok


def fit_board(P, B):
    """Map points P (n, 3) of corners onto the board's B (n, 3): (map -> metres scale, rms residual in metres, the
    corners kept, the board's normal and centre in the map frame)."""
    keep = np.ones(len(P), bool)
    for _ in range(2):
        s, R, t = umeyama(P[keep], B[keep])
        r = np.linalg.norm(s * P @ R.T + t - B, axis=1)
        keep = r <= max(3 * np.median(r[keep]), 0.002)
    s, R, t = umeyama(P[keep], B[keep])
    r = np.linalg.norm(s * P[keep] @ R.T + t - B[keep], axis=1)
    normal = R.T @ np.array([0.0, 0.0, 1.0])  # the board's z in the map frame
    return s, float(np.sqrt(np.mean(r ** 2))), keep, normal, P[keep].mean(0)


def triangulate(sightings):
    """Corners seen in two or more views, by linear triangulation: `sightings` {corner key: (board point, [(P (3, 4),
    pixel)])} -> (points (n, 3), board points (n, 3)); pixels in P's own convention."""
    X, B = [], []
    for b, seen in sightings.values():
        if len(seen) < 2:
            continue
        A = np.array([row for P, (u, v) in seen for row in (u * P[2] - P[0], v * P[2] - P[1])])
        x = np.linalg.svd(A)[2][-1]
        X.append(x[:3] / x[3])
        B.append(b)
    return np.array(X).reshape(-1, 3), np.array(B).reshape(-1, 3)


def find(images, board_path, min_corners=12):
    """{name: (pixels, board points)} for the images (name -> RGB array) the board is found in."""
    det, board = board_detector(load_board(board_path))
    hits = {}
    for name, rgb in images.items():
        hit = detect_image(det, board, cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR), min_corners)
        if hit is not None:
            hits[name] = hit
    return hits


def measure(name, points, valid, hit):
    """One image's board against its point map (points (H, W, 3) at the image's size): a dict, or None when too few
    corners land on valid points."""
    px, B = hit
    P, ok = sample(points, valid, px)
    if ok.sum() < 8:
        return None
    s, rms, keep, normal, centre = fit_board(P[ok], B[ok])
    return {"view": name, "corners": int(len(px)), "used": int(keep.sum()), "scale": s, "rms_m": rms,
            "normal": normal.tolist(), "centre": centre.tolist(), "px": px[ok][keep].tolist()}


def draw(img, rows):
    """The corners used, on an RGB image."""
    vis = np.asarray(img).copy()
    for r in rows:
        for u, v in r["px"]:
            cv2.circle(vis, (int(u), int(v)), 4, (255, 40, 40), -1)
    return vis
