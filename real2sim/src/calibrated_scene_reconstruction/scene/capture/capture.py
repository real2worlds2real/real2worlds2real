"""Session input: the photo (capture/rgb.png) and, optionally, more photos of the same scene (capture/views/).

    python -m calibrated_scene_reconstruction.capture sessions/<name> data/<name>/02_front.jpg [--views data/<name>/04_top_down.jpg ...]
    python -m calibrated_scene_reconstruction.capture sessions/<name> data/<name>/walkaround.mp4 [--photo-at 0.5] [--max-views 16]

Every image is turned upright by its EXIF orientation (phone photos are often stored sideways with an orientation
tag) and scaled to at most `--max-side` pixels on its long side (the stages were tuned on 1.9-2.4 k photos; a 24 MP
phone photo only multiplies the per-pixel maps). The views keep their file stems as names; recon.multiview reads them.

A video of the scene (walked around, the robot facing the camera at `--photo-at` seconds) gives both: calibrated_scene_reconstruction.keyframes
picks the photo and the views (named f<frame index>), capture/keyframes.json records which and capture/keyframes.jpg
shows them. `--views` photos are added to a video's views.

With configs/<name>/camera.json (calibrated_scene_reconstruction.calib: the phone calibrated once on a ChArUco board) every image of the
calibrated size is undistorted first (same K: the lens's radial term removed, the field of view kept), and
recon.perceive gives MoGe-2 that field of view instead of letting it guess.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from calibrated_scene_reconstruction.scene.capture import keyframes
from calibrated_scene_reconstruction.core.paths import CONFIGS

CAMERA = None  # configs/<name>/camera.json, set by main()


def undistort(im):
    if CAMERA is None or list(im.size) != CAMERA["size"] or not any(CAMERA["dist"]):
        return im
    K, d = np.asarray(CAMERA["K"]), np.asarray(CAMERA["dist"])
    return Image.fromarray(cv2.undistort(np.asarray(im), K, d))


def load_upright(path, max_side=2400):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return fit(undistort(im), max_side)


def fit(im, max_side):
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("photo", help="a photo, or a video of the scene")
    ap.add_argument("--views", nargs="*", default=[], help="more photos of the scene (recon.multiview)")
    ap.add_argument("--max-side", type=int, default=2400)
    ap.add_argument("--photo-at", type=float, default=0.5, help="video: seconds into it where the photo is taken")
    ap.add_argument("--max-views", type=int, default=16, help="video: views taken along it")
    a = ap.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    from .supplementary import digest, in_intervals, scene_intervals, check_scene_view_camera
    session = Session(a.session)
    intervals = scene_intervals(session, a.photo, primary=True)
    for view in a.views:
        scene_intervals(session, view)
        check_scene_view_camera(session, a.photo, view)
    global CAMERA
    cam = CONFIGS / Path(a.session).name / "camera.json"
    CAMERA = json.loads(cam.read_text()) if cam.exists() else None
    if CAMERA:
        print(f"camera {cam}: fx {CAMERA['K'][0][0]:.1f} px at {CAMERA['size']}, k1 {CAMERA['dist'][0]:+.4f} (undistorting)")
    cap = Path(a.session) / "capture"
    cap.mkdir(parents=True, exist_ok=True)
    sources = [Path(name) for name in [a.photo, *a.views]]
    observations = [digest(path) for path in sources]
    vd = cap / "views"
    vd.mkdir(exist_ok=True)
    if keyframes.is_video(a.photo):
        from calibrated_scene_reconstruction.core.media import color_profile
        color = color_profile(a.photo)
        previous_path = cap/'color.json'
        if (cap/'rgb.png').exists() and (
                json.loads(previous_path.read_text()) != color if previous_path.exists()
                else color['mode'] == 'hdr_base_to_srgb'):
            raise ValueError('Video color management changed; preserve this capture and start a new session')
    provenance = cap / "observations.json"
    provenance.unlink(missing_ok=True)
    for old in vd.glob("*.png"):
        old.unlink()
    for old in (cap / "keyframes.json", cap / "keyframes.jpg"):
        old.unlink(missing_ok=True)
    views = []
    if keyframes.is_video(a.photo):
        sel = keyframes.select(a.photo, a.photo_at, a.max_views)
        board = CONFIGS / Path(a.session).name / "charuco_board.json"
        mv = CONFIGS / Path(a.session).name / "multiview.json"
        if board.exists() and (json.loads(mv.read_text()).get("board", True) if mv.exists() else True):
            from calibrated_scene_reconstruction.scene.capture.calib import load_board

            # a board in the scene sets recon.multiview's scale (recon.board): the frames that show it best join
            sel["board"] = [j for j, _ in keyframes.board_frames(a.photo, load_board(board))]
            sel["views"] = sorted(set(sel["views"]) | set(sel["board"]))
            print(f"board {board.name}: frames {sel['board']} added as views" if sel["board"] else
                  f"board {board.name}: not found in the video")
        if not in_intervals(sel['photo']/sel['fps'], intervals):
            raise ValueError('Selected photo is outside the reviewed static-scene interval; choose --photo-at inside it')
        for key in ('views', 'linking', 'board'):
            if key in sel:
                sel[key] = [i for i in sel[key] if in_intervals(i/sel['fps'], intervals)]
        frames = keyframes.extract(a.photo, [sel["photo"]] + sel["views"])
        from calibrated_scene_reconstruction.scene.capture.exposure import prepare as prepare_exposure
        prepare_exposure(session, a.photo, frames)
        im = fit(undistort(Image.fromarray(frames[sel["photo"]][:, :, ::-1])), a.max_side)
        views = [(f"f{i:04d}", fit(undistort(Image.fromarray(frames[i][:, :, ::-1])), a.max_side)) for i in sel["views"]]
        keyframes.sheet(frames, sorted(frames), cap / "keyframes.jpg")
        sh = sel.pop("sharpness")
        sel["sharpness"] = {str(i): round(sh[i]) for i in [sel["photo"]] + sel["views"]}
        (cap / "keyframes.json").write_text(json.dumps({"video": str(a.photo), **sel}, indent=1))
        previous_path.write_text(json.dumps(color, indent=2)+'\n')
        print(f"video {a.photo}: {sel['frames']} frames; photo = frame {sel['photo']} at {sel['photo'] / sel['fps']:.1f} s, "
              f"{len(sel['views'])} views (frames {sel['views']})")
    else:
        im = load_upright(a.photo, a.max_side)
    im.save(cap / "rgb.png")
    print(f"photo {a.photo} -> {cap / 'rgb.png'} {im.size[0]}x{im.size[1]}")
    views += [(Path(v).stem, load_upright(v, a.max_side)) for v in a.views]
    for name, iv in views:
        iv.save(vd / f"{name}.png")
        print(f"view {name} -> {vd / (name + '.png')} {iv.size[0]}x{iv.size[1]}")
    if observations != [digest(path) for path in sources]:
        raise ValueError("An input observation changed during capture")
    session.write_json("capture/observations.json", {"sha256": observations})


if __name__ == "__main__":
    main()
