"""Key frames of a walk-around video: the photo and the views of a session from one phone video.

    python -m calibrated_scene_reconstruction.keyframes data/<name>/walkaround.mp4 [--photo-at 0.5] [--max-views 16] [--sheet keys.jpg]

A video of the scene walked around has a view every frame, but Depth Anything 3 takes a dozen or so (recon.multiview),
and most frames of a walk are smeared. Every frame is read at a quarter size; its sharpness is the variance of its
Laplacian; its ORB features are kept. The photo is the sharpest frame around `photo_at` seconds (the start of the
video: the scene from the front, the robot facing the camera, as recon.multiview's scale wants it). The views are
the sharpest frame of each of `max_views` equal stretches of the video (the walk at an even pace covers the scene
evenly; a frame-change measure instead piles the views where the phone swung), none a near-duplicate of the photo
or of the previous view (ratio-tested ORB matches over the feature count above `dup`). Where two consecutive views
share fewer than `min_overlap` of their features, the sharpest frame between them that shares more with both is
added, so the views stay chained for DA3. `select` returns the photo's frame index and the views'; `extract` reads
the frames at full size (calibrated_scene_reconstruction.capture calls both).
"""
import argparse
from pathlib import Path

import cv2
from calibrated_scene_reconstruction.core.media import VideoCapture, extract_frames
import numpy as np

VIDEO = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def is_video(path):
    return Path(path).suffix.lower() in VIDEO


def scan(path, small=480, n_features=600):
    """Every frame's sharpness and ORB descriptors at a small size (frames read upright: cv2 applies the rotation tag)."""
    cap = VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orb = cv2.ORB_create(n_features)
    sharp, desc = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = small / max(f.shape[:2])
        g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        _, d = orb.detectAndCompute(g, None)
        desc.append(d if d is not None else np.zeros((0, 32), np.uint8))
    cap.release()
    return fps, np.array(sharp), desc


def match_fraction(da, db, matcher, ratio=0.75):
    """The fraction of a's features with a ratio-tested match in b."""
    if len(da) < 8 or len(db) < 8:
        return 0.0
    good = sum(1 for m in matcher.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < ratio * m[1].distance)
    return good / len(da)


def select(path, photo_at=0.5, max_views=16, min_overlap=0.1, dup=0.6, photo_window_s=0.5):
    fps, sharp, desc = scan(path)
    n = len(sharp)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    lo, hi = max(0, int((photo_at - photo_window_s) * fps)), min(n, int((photo_at + photo_window_s) * fps) + 1)
    photo = lo + int(np.argmax(sharp[lo:hi]))
    edges = np.linspace(0, n, max_views + 1).astype(int)
    keys, prev = [], photo
    for a, b in zip(edges[:-1], edges[1:]):
        for j in np.argsort(-sharp[a:b]) + a:  # the sharpest of the stretch that is not the photo / last view again
            if j != photo and match_fraction(desc[j], desc[prev], matcher) < dup and \
                    match_fraction(desc[j], desc[photo], matcher) < dup:
                keys.append(int(j))
                prev = int(j)
                break
    chain = sorted(keys + [photo])
    added = []
    for a, b in zip(chain[:-1], chain[1:]):  # a gap in the chain: the sharpest frame between that links both ends
        if b - a < 2 or match_fraction(desc[a], desc[b], matcher) >= min_overlap:
            continue
        between = [(sharp[j], j) for j in range(a + 1, b) if match_fraction(desc[a], desc[j], matcher) >= min_overlap
                   and match_fraction(desc[j], desc[b], matcher) >= min_overlap]
        if between:
            added.append(int(max(between)[1]))
    return {"fps": fps, "frames": n, "photo": photo, "views": sorted(keys + added), "linking": added,
            "sharpness": sharp.tolist(), "min_overlap": min_overlap, "dup": dup}


def board_frames(path, board_cfg, n=3, step=3, min_gap_s=1.0):
    """[(frame index, corners)]: the `n` frames showing a calibration board best (the most ChArUco corners, every
    `step`-th frame read, at least `min_gap_s` apart), for recon.board's scale."""
    from calibrated_scene_reconstruction.scene.capture.calib import board_detector, detect_image

    det, board = board_detector(board_cfg)
    cap = VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    found, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, f = cap.retrieve()
            hit = detect_image(det, board, f, min_corners=12)
            if hit is not None:
                found.append((len(hit[0]), i))
        i += 1
    cap.release()
    picked = []
    for c, j in sorted(found, reverse=True):
        if all(abs(j - k) >= min_gap_s * fps for k, _ in picked):
            picked.append((j, c))
        if len(picked) == n:
            break
    return sorted(picked)


def extract(path, indices):
    """{frame index: BGR frame} at full size, one sequential read (seeking a phone's HEVC lands off by frames)."""
    return extract_frames(path, list(indices))


def sheet(frames, order, path, tile_w=270, cols=8):
    """A contact sheet of the chosen frames in walk order, the frame index on each."""
    tiles = []
    for i in order:
        f = frames[i]
        s = tile_w / f.shape[1]
        t = cv2.resize(f, (tile_w, round(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(t, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        tiles.append(t)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--photo-at", type=float, default=0.5, help="seconds: the photo is the sharpest frame around here")
    ap.add_argument("--max-views", type=int, default=16)
    ap.add_argument("--sheet", default=None, help="write a contact sheet of the chosen frames here")
    a = ap.parse_args()
    sel = select(a.video, a.photo_at, a.max_views)
    sh = sel["sharpness"]
    print(f"{a.video}: {sel['frames']} frames at {sel['fps']:.0f} fps; photo {sel['photo']} "
          f"(sharpness {sh[sel['photo']]:.0f}), {len(sel['views'])} views: " +
          ", ".join(f"{i} ({sh[i]:.0f}{', linking' if i in sel['linking'] else ''})" for i in sel["views"]))
    if a.sheet:
        frames = extract(a.video, [sel["photo"]] + sel["views"])
        sheet(frames, sorted(frames), a.sheet)


if __name__ == "__main__":
    main()
