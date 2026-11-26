"""The world frame of a monocular video: the support plane.

No camera calibration comes with a video, so the world is defined by what the video shows: z is the support plane's
normal (up), z = 0 the plane, the origin the foot of the camera on it, x the camera's right axis projected onto the
plane. The camera then sits at (0, 0, h), and h (the camera's height above the support) is the one number that carries
the metric scale of everything measured on the plane: positions, speeds and decelerations all scale with it.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np
from calibrated_scene_reconstruction.core.media import VideoCapture

G = 9.81


def fit_plane(P, tol=0.01, iters=8):
    """Plane n . X = d through camera-frame points (N, 3), by iterated trimming around a least-squares fit. Returns
    (n unit, pointing from the plane to the camera; h = distance of the camera centre to the plane; inlier mask).
    `tol` is how far a point of the plane may lie off it at a metre from the camera, and in proportion beyond (depth
    from one picture is as rough as the scene is far: a floor seen from six metres is not flat to a centimetre)."""
    P = np.asarray(P, float)
    inl = np.all(np.isfinite(P), 1) & (np.abs(P).max(1) < 1e3)  # depth flying off at an edge or into the sky
    if inl.sum() < 10:
        raise ValueError("no finite points to fit the support plane through")
    tol = tol * max(1.0, float(np.median(np.linalg.norm(P[inl], axis=1))))
    for _ in range(iters):
        c = P[inl].mean(0)
        n = np.linalg.eigh(np.cov((P[inl] - c).T))[1][:, 0]
        r = (P - c) @ n
        inl = np.abs(r - np.median(r[inl])) < tol
        if inl.sum() < 10:
            raise ValueError("support plane fit lost its inliers")
    c = P[inl].mean(0)
    n = np.linalg.eigh(np.cov((P[inl] - c).T))[1][:, 0]
    if n @ c > 0:  # the camera is at the origin: the normal points back at it
        n = -n
    return n, float(-(n @ c)), inl


CLOCK_REL = 1e-3  # what a video's own clock is good to (29.97 frames a second written as 30 is this much)
GRAZING_DEG = 10.0  # a support seen from nearer its own plane than this is seen along itself


def plane_is_one(n, h):
    """Whether the planes fitted on several frames are one plane: a mask that is a strip seen along its length (a
    plank from its end) leaves the fit free to turn about it, and it lands somewhere else on every frame."""
    n, h = np.asarray(n, float), np.asarray(h, float)
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(h)) and np.median(h) > 0):
        return False
    m = np.median(n, 0)
    m /= np.linalg.norm(m)
    return bool(np.degrees(np.arccos(np.clip(n @ m, -1, 1))).max() < 10.0 and np.std(h) < 0.25 * np.median(h))


def unusable(per, plane=True):
    """Why nothing can be read off a perception (a sentence), or None. Every reading takes one camera pose for the
    whole video: a camera that moves through the scene (perception's verdict "moving": the background does not stay
    put and no turn of the camera explains it) gives none. A reading that stands the object on its support (`plane`)
    needs that support seen as one plane: a board seen along itself, or a guess at where a support might be, is
    none. A camera whose motion could not be judged (a background without texture) is not a reason: see camera_note."""
    if "camera" in per.files and str(per["camera"]) == "moving":
        return ("the camera moves through the scene (the background does not stay put, and no turn of the camera "
                "explains how it moves): nothing is measured")
    if plane:
        ok = bool(per["plane_ok"]) if "plane_ok" in per.files else plane_is_one(per["plane_n"], per["plane_h"])
        if not ok:
            return ("the support is not seen as one plane (seen along itself, or too little of it is in view): the "
                    "object cannot be stood on it, nothing is measured")
        off = off_plane(per)
        # a strip of points always fits some plane: one the camera stands within 10 degrees of (its height above
        # it under a sixth of its distance to the object) is a board seen from its side, whatever was fitted
        if off is not None and float(np.median(per["plane_h"])) < np.sin(np.radians(GRAZING_DEG)) * off[2]:
            return (f"the camera looks along the support (it stands {float(np.median(per['plane_h'])):.2f} m off "
                    f"its plane, {off[2]:.2f} m from the object): the support is seen along itself, the object "
                    "cannot be stood on it, nothing is measured")
        # (seen from above, what shows of a thing that stands on a plane is its top, all its height off it, and
        # the size read off that top is a little under the thing's: half as much again is allowed)
        if off is not None and off[0] > max(1.5 * off[1], 0.05 * off[2]):
            return (f"the plane found for the support is not one the object is on (MoGe-2 sees the object "
                    f"{off[0]:.2f} m off it at its nearest, more than its own size): it is a wall behind it or a floor "
                    "below, and the support itself is not seen as a plane: nothing is measured")
    return None


def off_plane(per):
    """(how far the object is off the support's plane at its nearest, m; the object's size, m; its distance from
    the camera, m) in MoGe-2's points, or None where perception did not place the object (object_xyz). A thing that
    is on its support at some time is within its own size of the support's plane then."""
    if "object_xyz" not in per.files or not len(per["object_xyz"]):
        return None
    X = np.asarray(per["object_xyz"], float)
    n, h = np.atleast_2d(np.asarray(per["plane_n"], float)), np.atleast_1d(np.asarray(per["plane_h"], float))
    if len(n) != len(X):
        n, h = np.tile(np.median(n, 0), (len(X), 1)), np.full(len(X), np.median(h))
    d = np.abs((n * X).sum(1) + h)  # the plane of the same frame's points: n . X + h = 0 on it
    ok = np.isfinite(d)
    size = np.asarray(per["object_size"], float) if "object_size" in per.files else np.array([np.nan])
    if not ok.any() or not np.isfinite(size).any():
        return None
    return float(d[ok].min()), float(np.nanmedian(size)), float(np.median(np.linalg.norm(X[ok], axis=1)))


def camera_note(per):
    """A sentence for physics.json when the camera could not be seen to stand still, or None."""
    if "camera" in per.files and str(per["camera"]) == "unknown":
        return ("the background shows too little to follow: whether the camera stood still is not known, and the "
                "reading takes it that it did")
    return None


def camera_pose(n, h):
    """Camera-to-world 4x4 (OpenCV camera axes) of the plane frame described in the module doc."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, n])  # rows: the world axes in camera coordinates
    T[:3, 3] = 0, 0, h
    return T


def ray_plane(K, T_world_cam, uv, z=0.0):
    """World points (..., 3) where the pixel rays meet the horizontal plane at height z; NaN for rays that miss it."""
    uv = np.asarray(uv, float)
    K, T = np.asarray(K, float), np.asarray(T_world_cam, float)
    d = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1], np.ones(uv.shape[:-1])], -1)
    d = d @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - T[2, 3]) / d[..., 2]
    P = T[:3, 3] + s[..., None] * d
    P[~(s > 0)] = np.nan
    return P


def scale_K(K, sx, sy=None):
    """Intrinsics of the image resized by sx (and sy)."""
    K = np.array(K, float)
    K[0] *= sx
    K[1] *= sx if sy is None else sy
    return K


# --- recordings ----------------------------------------------------------------------------------------------------
def repeated_frames(frames, least=4):
    """Indices of the frames that repeat the one before them, when the video has such at a regular pace (a 25 Hz
    recording written out at 30 Hz shows every fifth frame twice: the motion then stutters on a clock that is not
    the video's), else None. A frame repeats when it differs from the one before by a small part of what its
    neighbours do; the pace may slip by a frame now and then, but a scene in which nothing moves has none."""
    import cv2
    from scipy.ndimage import median_filter

    F = len(frames)
    if F < 16:
        return None
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (96, 54), interpolation=cv2.INTER_AREA)
                      for f in frames]).astype(np.float32)
    d = np.abs(np.diff(small, axis=0)).mean((1, 2))  # d[k]: frame k + 1 against frame k
    rep = np.nonzero(d < 0.25 * median_filter(d, size=15, mode="nearest"))[0] + 1
    if len(rep) < least:
        return None
    gaps = np.diff(rep)
    P = float(np.median(gaps))
    if not (2 <= P <= 12 and np.mean(np.abs(gaps - P) <= 1) > 0.7 and len(rep) > 0.5 * F / P):
        return None
    return regular_pace(d / np.maximum(median_filter(d, size=15, mode="nearest"), 1e-9), rep, int(round(P)))


def regular_pace(rel, rep, P):
    """The repeats of a video that has one every P frames, from those found one by one (`rep`; rel[k - 1]: how
    much frame k differs from the one before it, against its neighbours). Found one by one, a few are missed and a
    few frames that only moved little are taken for repeats, most of them where little moves; the clock of what
    is left is then off by as many frames (a pendulum's length by half a percent). From the longest stretch at an
    even pace outwards there is one every P frames: the P-th after the last, or its neighbour when that one
    differs clearly less (the pace may slip by a frame)."""
    F = len(rel) + 1
    gaps = np.diff(rep)
    best, i = (0, 0), 0
    while i < len(gaps):
        j = i
        while j < len(gaps) and gaps[j] == P:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    out = [int(rep[best[0]])]
    for step in (1, -1):
        k = out[0]
        while 1 <= k + step * (P - 1) < F:
            c = [x for x in (k + step * P, k + step * (P - 1), k + step * (P + 1)) if 1 <= x < F]
            v = [rel[x - 1] for x in c]
            k = c[0] if c[0] == k + step * P and v[0] <= 1.5 * min(v) else c[int(np.argmin(v))]
            out.append(k)
    return np.array(sorted(set(out)))


def correlation_time(res):
    """How many successive residuals count as one: 1 + 2 x the sum of their autocorrelations, up to the first lag at
    which it falls under 0.05 (res (N,) or (N, k) along time; at least 1). A fit's formal errors take every sample
    for an independent one; what a model does not describe (a track that lags, a pivot a pixel off) leaves
    residuals that hold their sign for many frames, and the errors are larger by the root of this."""
    r = np.asarray(res, float)
    r = r.reshape(len(r), -1)
    r = r - r.mean(0)
    v = (r ** 2).sum()
    if len(r) < 8 or v <= 0:
        return 1.0
    tau = 1.0
    for k in range(1, len(r) // 2):
        rho = (r[k:] * r[:-k]).sum() / v
        if rho < 0.05:
            break
        tau += 2.0 * rho
    return float(tau)


def _shrunk(f, mw):
    """Frame f no wider than mw pixels (its height even)."""
    if f.shape[1] <= mw:
        return f
    import cv2

    return cv2.resize(f, (mw, int(round(f.shape[0] * mw / f.shape[1] / 2)) * 2), interpolation=cv2.INTER_AREA)


def clock_rel(path):
    """What the clock of a recording's frames is good to, as a part of any time read off them: the video's own
    (CLOCK_REL), and where frames that repeat were left out (timing.json) what their pace leaves open: each
    place the pace slips by a frame the repeat may be the frame before or after, a frame's worth either way, and
    as many slips add up as the root of their number."""
    path = Path(path)
    if not (path / "timing.json").exists():
        return CLOCK_REL
    drop = np.asarray(read_document(path / "timing.json").get("drop", []), int)
    if len(drop) < 3:
        return CLOCK_REL
    gaps = np.diff(drop)
    slips = int((gaps != int(round(float(np.median(gaps))))).sum())
    return float(np.hypot(CLOCK_REL, np.sqrt(slips) / max(drop[-1] + 1, 1)))


# kinds of motion whose reading is a rate by the video's own clock, with nothing in them that times it: of footage
# slowed by a factor nobody states (meta.json "slow_motion": "unknown") they give no number. A drop and a pendulum
# are timed by gravity and say the factor themselves; a strain and a stick-slip ratio have no time in them; an
# incline keeps what its tilt alone says (fit.fit_one)
UNTIMED = ("slide", "roll", "carried", "turntable", "beam", "sheet")


def untimed(meta):
    """Why nothing is read from footage slowed by an unknown factor, or None where the motion times itself or the
    footage is not said to be such."""
    motion = meta.get("motion", "slide")
    if meta.get("slow_motion") != "unknown" or motion not in UNTIMED:
        return None
    return (f"the footage is slowed by a factor nobody states (meta.json \"slow_motion\": \"unknown\"), and a "
            f"{motion} has nothing in it to time it by (a drop and a pendulum have gravity): nothing is measured")


def read_frames(path, every=1, stabilised=True, timing=True, raw=False):
    """(frames (F, H, W, 3) uint8 RGB, fps) of a recording directory: frames/*.png|jpg (fps from meta.json) or
    video.* (fps from the container unless meta.json gives it). meta.json may also hold
      "frames": [first, last]   the part of the video to use (inclusive, in source frames)
      "every": n                keep every n-th frame (high-speed footage)
      "max_width": px           frames wider than this are shrunk (default 960: SAM 3 and MoGe-2 gain nothing above)
      "slow_motion": k          the footage plays k times slower than it happened: the returned fps is k times higher
                                ("unknown": slowed, by a factor nobody states; read as it plays, see `untimed`)
    The returned frames are the ones every later stage indexes; when perception found the camera shaking
    (stabilised.npz) they are warped onto its reference frame, unless stabilised=False. Frames that perception found
    to repeat the one before (timing.json, repeated_frames) are left out and the fps is that of the rest, unless
    timing=False."""
    path = Path(path)
    meta = read_document(path / "meta.json")
    step = every * (1 if raw else int(meta.get("every", 1)))  # raw: every source frame (repeated_frames)
    every = 1
    a, b = meta.get("frames", [0, None])
    mw = int(meta.get("max_width", 960))
    drop = set()
    if timing and (path / "timing.json").exists():
        drop = set(read_document(path / "timing.json").get("drop", []))
    fdir = path / "frames"
    if fdir.is_dir():
        from PIL import Image

        files = sorted(p for p in fdir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        files = files[a:None if b is None else b + 1]
        fps, n = float(meta["fps"]), len(files)
        files = [f for i, f in enumerate(files) if i not in drop]
        kept = len(files)
        frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files[::step]])
    else:  # frame by frame: a long video in 4K does not fit in memory before it is thinned and shrunk
        import cv2

        video = next(p for p in sorted(path.iterdir()) if p.stem == "video")
        cap = VideoCapture(str(video))
        fps = float(meta.get("fps", cap.get(cv2.CAP_PROP_FPS) or 30.0))
        frames, k, n, kept = [], 0, 0, 0
        while True:
            ok, f = cap.read()
            if not ok or (b is not None and k > b):
                break
            if k >= a:
                if n not in drop:
                    if kept % step == 0:
                        frames.append(_shrunk(f[:, :, ::-1], mw))
                    kept += 1
                n += 1
            k += 1
        cap.release()
        frames = np.stack(frames)
    fps, every = fps * kept / max(n, 1), step
    if frames.shape[2] > mw:
        frames = np.stack([_shrunk(f, mw) for f in frames])
    if stabilised and every == int(meta.get("every", 1)) and (path / "stabilised.npz").exists():
        from .stabilise import warp

        frames = warp(frames, np.load(path / "stabilised.npz")["H"])[0]
    slow = meta.get("slow_motion", 1.0)
    return np.ascontiguousarray(frames), fps * (1.0 if slow == "unknown" else float(slow)) / every
