"""Is the camera still, and if it only shakes, undo it (CPU, OpenCV).

Everything downstream assumes one camera pose for the whole video. `camera_motion` measures how far the background
moves against a reference frame (corners followed by optical flow outside the moving masks, a RANSAC homography per
frame); `warp` brings every frame and mask back onto the reference. What moves in a scene is not only what was
segmented (a board that is tilted, a cloth, a table pushed between two takes), and it may fill most of the picture:
a camera stands still when some of what it sees does, so corners all over the picture that stay where they are
through the video settle it, whatever the rest does (`still_corners`).

A homography is exact for a camera that only rotates (a hand-held phone held in place, a tripod knocked) and for
any motion of a planar background; a camera that walks through a scene with depth is not this module's case: its
matches stop fitting one homography (`inliers` drops) and the recording should be refused rather than stabilised.
"""
import numpy as np


def still_corners(D, p0, size, still_px):
    """Which of the corners p0 (N, 2) with displacements D (F, N) px against the reference frame (NaN where one was
    not followed) stand still through the video: followed in half the frames and within still_px in nine of ten of
    those. None unless there are 25 of them, a quarter of those followed, spread over three tenths of the picture
    each way (a logo burnt into a corner of the picture stands still under any camera)."""
    seen = np.isfinite(D).mean(0) > 0.5
    still = np.zeros(len(p0), bool)
    for j in np.nonzero(seen)[0]:
        still[j] = np.nanpercentile(D[:, j], 90) < still_px
    if still.sum() < 25 or still.sum() < 0.25 * seen.sum():
        return None
    if (np.ptp(p0[still], axis=0) < 0.3 * np.asarray(size, float)).any():
        return None
    return still


def camera_motion(frames, moving=None, ref=0, max_features=600, still_px=1.5):
    """Per frame the homography H (3, 3) taking its pixels to the reference frame's, how far the background moved
    (px: the median displacement of corners of the reference frame followed into it by Lucas-Kanade flow, checked
    forwards and backwards; a person walking through a still scene does not move that median, and unlike descriptor
    matching it is not fooled by a tiled floor), and the fraction of the corners the homography explains.
    `moving` (F, H, W) bool masks out what is known to move. A homography that fewer than 25 corners or under half
    of them support is not trusted: such frames keep the identity and get inlier fraction 0. Where corners all over
    the picture stand still through the video (still_corners; still_px of a picture 848 px wide) the camera does:
    the shift is theirs and every homography the identity, whatever else in the picture moves."""
    import cv2

    F, H, W = frames.shape[:3]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    m0 = None
    if moving is not None:
        m0 = (~cv2.dilate(moving[ref].astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)).astype(np.uint8) * 255
    p0 = cv2.goodFeaturesToTrack(gray[ref], max_features, 0.01, 8, mask=m0)
    Hs, shift, inl = np.tile(np.eye(3), (F, 1, 1)), np.zeros(F), np.zeros(F)
    inl[ref] = 1.0
    if p0 is None or len(p0) < 25:  # nothing to follow: the motion is not known (verdict "unknown")
        shift[:] = np.nan
        shift[ref] = 0.0
        return Hs, shift, inl
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    D = np.full((F, len(p0)), np.nan)  # every corner's displacement against the reference frame
    D[ref] = 0.0
    found = {}
    for i in range(F):
        if i == ref:
            continue
        p1, st, _e = cv2.calcOpticalFlowPyrLK(gray[ref], gray[i], p0, None, **lk)
        pb, sb, _e = cv2.calcOpticalFlowPyrLK(gray[i], gray[ref], p1, None, **lk)
        ok = (st.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 1.0)
        if moving is not None and ok.any():  # corners that something moving covers in this frame say nothing
            u = np.clip(np.round(p1.reshape(-1, 2)).astype(int), 0, [W - 1, H - 1])
            ok &= ~cv2.dilate(moving[i].astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)[u[:, 1], u[:, 0]]
        found[i] = (p1.reshape(-1, 2)[ok], p0.reshape(-1, 2)[ok])
        D[i, ok] = np.linalg.norm(found[i][0] - found[i][1], axis=1)
    still = still_corners(D, p0.reshape(-1, 2), (W, H), still_px * W / 848.0)
    if still is not None:
        n = np.isfinite(D[:, still]).sum(1)
        shift = np.where(n >= 5, np.nanmedian(np.where(n[:, None] >= 5, D[:, still], 0.0), axis=1), np.nan)
        return Hs, shift, np.ones(F)
    for i in range(F):
        if i == ref:
            continue
        a, b = found[i]
        if len(a) < 25:
            shift[i] = np.nan
            continue
        shift[i] = float(np.median(np.linalg.norm(a - b, axis=1)))
        Hm, good = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
        if Hm is None or good.sum() < 25 or good.mean() < 0.5:
            continue
        c = np.c_[corners, np.ones(4)] @ Hm.T
        if np.linalg.norm(c[:, :2] / c[:, 2:] - corners, axis=1).max() > 0.25 * W:
            continue  # no shake moves a corner by a quarter of the picture: a degenerate fit
        Hs[i], inl[i] = Hm, float(good.mean())
    return Hs, shift, inl


def warp(frames, Hs, masks=()):
    """Frames (F, H, W, 3) and each of `masks` (F, H, W) bool warped onto the reference frame."""
    import cv2

    F, H, W = frames.shape[:3]
    out = np.stack([cv2.warpPerspective(frames[i], Hs[i], (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE) for i in range(F)])
    ms = [np.stack([cv2.warpPerspective(m[i].astype(np.uint8), Hs[i], (W, H), flags=cv2.INTER_NEAREST).astype(bool)
                    for i in range(F)]) for m in masks]
    return out, ms


def verdict(shift, inl, still_px=1.5, max_px_frac=0.15, width=848):
    """"static" | "stabilised" | "moving" | "unknown": what to do with a video whose background moved by `shift` px
    per frame (NaN where it could not be followed). Static when nine frames in ten moved less than still_px (of a
    picture 848 px wide, and in proportion: single frames with an arm across the picture do not make a camera
    move, nor does what hides the background for part of the video when it is still before and after); stabilised
    when the homographies of the frames that moved are supported; unknown when the reference frame's background
    had nothing to follow (a blank wall, a dark one: under a quarter of the frames have a shift) or was followed,
    and stood, in under half of the frames and not at both ends; else moving."""
    known = np.isfinite(shift)
    if (known.sum() <= 1 and len(shift) > 1) or known.mean() < 0.25:  # too little of it shows any background
        return "unknown"
    still_px = still_px * width / 848.0
    k = np.nonzero(known)[0]
    # something across the picture for half the video hides the background, it does not move the camera: still
    # wherever it was seen, from the video's first fifth to its last
    ends = k[0] < 0.2 * len(shift) and k[-1] >= 0.8 * len(shift) and known.mean() >= 0.25
    if np.percentile(shift[known], 90) < still_px:  # it was not seen to move: still, or not known where too
        return "static" if known.mean() >= 0.5 or ends else "unknown"  # little of the video shows the background
    if known.mean() < 0.5:
        return "moving"
    moved = known & (shift >= still_px)
    if np.percentile(shift[known], 90) > max_px_frac * width or (inl[moved] > 0).mean() < 0.8 or known.mean() < 0.8:
        return "moving"
    return "stabilised"
