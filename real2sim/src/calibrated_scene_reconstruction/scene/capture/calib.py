"""The phone's camera intrinsics from a video of a ChArUco board: configs/<name>/camera.json, read by calibrated_scene_reconstruction.capture
(undistortion) and recon.perceive (MoGe-2's field of view).

    python -m calibrated_scene_reconstruction.calib data/<name>/walkaround_aruco.mp4 configs/<name> [--board configs/<name>/charuco_board.json]

One calibration serves every session shot with the same phone in the same video mode: the board is needed once, not
in the scene being rebuilt (when it is, recon.board takes the scene's scale from it too). DA3's joint estimate over the views was 1554.

The board's squares are found in every frame (markers decoded on the frame upscaled x2, ChArUco corners
interpolated, at least 8); the fit keeps the principal point at the image centre and one radial term k1 (a board seen
at one distance from one side does not constrain more: the principal point wandered by 200 px when free, k2 traded
off against fx). `fx_halves` re-fits random halves of the frames: their spread is the calibration's uncertainty."""
import argparse
import json
from pathlib import Path

import cv2
from calibrated_scene_reconstruction.core.media import VideoCapture
import numpy as np

BOARD = {"squares_x": 12, "squares_y": 9, "square_length": 0.030, "marker_length": 0.0225, "dictionary": "DICT_5X5_100"}


def load_board(path=None):
    """The board's parameters: a charuco_board.json over the default (12 x 9 squares of 30 mm)."""
    return {**BOARD, **(json.loads(Path(path).read_text()) if path and Path(path).exists() else {})}


def board_detector(board_cfg):
    """(detector, board) for a board's parameters."""
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_cfg.get("dictionary", BOARD["dictionary"])))
    board = cv2.aruco.CharucoBoard((board_cfg["squares_x"], board_cfg["squares_y"]), board_cfg["square_length"],
                                   board_cfg["marker_length"], dic)
    par = cv2.aruco.DetectorParameters()
    par.minMarkerPerimeterRate = 0.005  # markers of 20-35 px in a 1920 px frame
    par.adaptiveThreshWinSizeMax = 63
    par.adaptiveThreshWinSizeStep = 4
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), par), board


def detect_image(det, board, img, min_corners=8):
    """(pixels (n, 2), board points (n, 3) in metres) of the ChArUco corners found in a BGR image, or None. The
    markers are decoded on the image upscaled x2."""
    cc, cid, _, _ = det.detectBoard(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    if cid is None or len(cid) < min_corners:
        return None
    return (cc / 2).astype(np.float32).reshape(-1, 2), board.getChessboardCorners()[cid.ravel()].astype(np.float32)


def detect(path, board_cfg, min_corners=8, step=1):
    det, board = board_detector(board_cfg)
    cap = VideoCapture(str(path))
    i, obs, size = 0, [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        size = (f.shape[1], f.shape[0])
        if i % step == 0:
            hit = detect_image(det, board, f, min_corners)
            if hit is not None:
                obs.append((i, *hit))
        i += 1
    cap.release()
    return obs, size


def fit(obs, size):
    W, H = size
    K0 = np.array([[1.2 * max(W, H), 0, W / 2], [0, 1.2 * max(W, H), H / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    return cv2.calibrateCamera([o for _, _, o in obs], [c for _, c, _ in obs], (W, H), K0, None, flags=flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("config_dir")
    ap.add_argument("--board", default=None, help="board json (default: <config_dir>/charuco_board.json, else 12x9 30 mm)")
    a = ap.parse_args()
    cdir = Path(a.config_dir)
    board_cfg = load_board(Path(a.board) if a.board else cdir / "charuco_board.json")
    obs, size = detect(a.video, board_cfg)
    if len(obs) < 10:
        raise SystemExit(f"the board is found in {len(obs)} frames only: film it closer (markers >= 20 px)")
    rms, K, dist, rv, tv = fit(obs, size)
    rng = np.random.default_rng(0)
    halves = [float(fit([obs[j] for j in sorted(rng.choice(len(obs), len(obs) // 2, replace=False))], size)[1][0, 0])
              for _ in range(8)]
    fov = float(np.degrees(2 * np.arctan(size[0] / 2 / K[0, 0])))
    dists = [float(np.linalg.norm(t)) for t in tv]
    out = {"_note": f"from {a.video} by calibrated_scene_reconstruction.calib: the board in {len(obs)} frames ({obs[0][0]}-{obs[-1][0]}) "
                    f"at {min(dists):.2f}-{max(dists):.2f} m; principal point fixed at the centre, one radial term",
           "size": list(size), "K": K.tolist(), "dist": dist.ravel()[:5].tolist(), "fov_x_deg": fov, "rms_px": rms,
           "fx_halves": halves}
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "camera.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(obs)} frames: fx {K[0, 0]:.1f} px (halves {min(halves):.0f}-{max(halves):.0f}), k1 {dist.ravel()[0]:+.4f}, "
          f"rms {rms:.2f} px, horizontal FOV {fov:.1f} deg -> {cdir / 'camera.json'}")


if __name__ == "__main__":
    main()
