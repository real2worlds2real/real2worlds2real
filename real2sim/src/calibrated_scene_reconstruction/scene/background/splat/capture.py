"""Splat background, step 1: the frames of every video of the session, their poses, the twin's world, the coverage.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.splat_capture sessions/<name>

The videos: the session's own (capture/keyframes.json) and splat.json `videos`. From each, every `step`-th frame, the sharpest `keep` of each
second, the key frames always; undistorted with configs/<name>/camera.json, named <video>_<frame>.jpg.

Cameras: each video its own.

Poses: COLMAP (pycolmap) with every video's pinhole fixed: SIFT, each video's frames matched in sequence, two
videos matched pairwise on one frame per `cross_every_s` seconds of each, each video's frames one per `loop_every_s`
seconds matched with each other beyond the sequential window (its loops: where the walk comes back), incremental
mapping; the largest model.

The twin's world: rotation and horizontal position from the key frames' cameras (the photo and the views, whose
poses recon.multiview knows); the scale from a ChArUco board in the frames (configs/<name>/charuco_board.json: its
corners triangulated over the frames of each video that show it) or else the twin's own support height; the height
from the floor (z = 0 in the twin). Checks: the support top's height in the aligned points
against the twin's, the key cameras' residual.

Coverage: the points binned in 25 cm columns over the room, per column the distinct frames that saw it and the arc
of their viewing directions (splat.grade); a top view with the cameras' paths. A property of the capture, not of
any robot camera: it says where the background will hold and what to film again.

The task zone (splat.task_zone: what the meshes take) is kept out of the training: per frame its projection
 and the zone's COLMAP points out of
the initialization (sparse/0; the full model stays in sparse/full). Without this, half the Gaussians grew on the
table, the robot and the cups and were cropped away afterwards. `--zone-only` redoes just this
on an existing model.

Writes splat/data/{images, sparse/{0, full}, database.db}, splat/frames.json, splat/align.json,
splat/coverage.{json,png}, splat/zone.json."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
from calibrated_scene_reconstruction.core.media import VideoCapture, color_profile
import numpy as np
import pycolmap

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import angular_spread, folder, grade, settings, sim3, task_zone, umeyama
from calibrated_scene_reconstruction.scene.capture.supplementary import in_intervals, scene_intervals


def frames_of(path, step, keep, always=(), intervals=None):
    """{frame index: BGR} of one video: every step-th, the sharpest `keep` of each second, `always` too."""
    cap = VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    got, sharp, i = {}, {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (i % step == 0 or i in always) and in_intervals(i/fps, intervals):
            g = cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
            sharp[i] = float(cv2.Laplacian(g, cv2.CV_64F).var())
            got[i] = f
        i += 1
    cap.release()
    kept = set(always)
    for w0 in range(0, i, int(round(fps))):
        idx = [j for j in sharp if w0 <= j < w0 + int(round(fps))]
        if idx:
            thr = np.quantile([sharp[j] for j in idx], 1 - keep)
            kept |= {j for j in idx if sharp[j] >= thr}
    return {j: got[j] for j in sorted(kept) if j in got}, fps


def by_motion(frames, motion, always=(), max_gap=15):
    """The frames kept by how far the view moved: a frame is kept when the median optical flow from the last kept
    one reaches `motion` (a share of the image's width) -- a slow pass round the task thins, a fast turn keeps its
    steps. Every step-th frame at 2600 frames took COLMAP's mapping over an hour where 900 map in 20 min, and most
    of them repeat their neighbours' view."""
    idx = sorted(frames)
    small = {j: cv2.cvtColor(cv2.resize(frames[j], None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY) for j in idx}
    w = next(iter(small.values())).shape[1]
    # the flow summed over consecutive frames (the flow straight from the last kept one fails once the view has moved
    # far: Farneback's median stayed small and a 2600-frame walk kept 31)
    kept, acc, prev = [idx[0]], 0.0, idx[0]
    for j in idx[1:]:
        f = cv2.calcOpticalFlowFarneback(small[prev], small[j], None, 0.5, 3, 21, 3, 5, 1.1, 0)
        acc += float(np.median(np.linalg.norm(f, axis=2)))
        prev = j
        if acc >= motion * w or j in always or j - kept[-1] >= max_gap:
            kept.append(j)
            acc = 0.0
    return {j: frames[j] for j in kept}


def frames_timeline(listing, fps, out_png, total=None):
    """qa: the kept frames per 2 s along each video -- the agent reads where the selection is thin or dense."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(listing), 1, figsize=(12, 2.4 * len(listing)), squeeze=False)
    for ax, (v, js) in zip(axs[:, 0], listing.items()):
        t = np.asarray(js) / fps[v]
        ax.hist(t, bins=np.arange(0, (t.max() if len(t) else 1) + 2, 2), color="#1f6f5c")
        ax.set_title(f"video {v}: {len(js)} frames kept (per 2 s)")
        ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def self_calibrate(path, work, n=150, intervals=None, *, sample_offset=0, workers=8):
    """{K, dist, fov_x_deg, reprojection_px, frames} of a video's camera from its own frames: every k-th raw frame
    (about n), SIFT, sequential matching, incremental mapping with OPENCV's focal length and distortion refined."""
    # Keep the evidence needed to investigate calibration ambiguity later. A
    # small reprojection error alone does not establish metric observability.
    # Fresh rounds also avoid erasing a previous successful or interrupted run.
    if type(n) is not int or n < 20 or type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('At least20 calibration samples and1..32 workers required')
    if type(sample_offset) is not int or sample_offset < 0:
        raise ValueError('Nonnegative calibration sampling offset required')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name[6:]) for p in work.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    work = work/f'round_{number}'
    (work / "images").mkdir(parents=True)
    shutil.copyfile(__file__, work/'capture_engine.py')
    with Path(path).open('rb') as stream:
        source_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    cap = VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    every = max(1, total // n)
    if sample_offset >= every:
        cap.release()
        raise ValueError('Calibration sampling offset must be smaller than frame stride')
    i = 0
    selected = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == sample_offset and in_intervals(i/fps, intervals):
            cv2.imwrite(str(work / "images" / f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            selected.append(i)
        i += 1
    cap.release()
    if len(selected) < 20:
        raise ValueError('Self-calibration needs at least20 sampled source frames')
    sampling = {'source': str(Path(path).resolve()), 'source_sha256': source_digest,
                'frames': selected, 'stride': every,
                'sample_offset': sample_offset, 'intervals': intervals, 'workers': workers,
                'source_color_profile': color_profile(path)}
    (work/'input.json').write_text(json.dumps(sampling, indent=2)+'\n')
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "OPENCV"
    extraction = pycolmap.FeatureExtractionOptions(); extraction.num_threads = workers
    pycolmap.extract_features(work / "db.db", work / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, extraction_options=extraction)
    seq = pycolmap.SequentialPairingOptions()
    seq.overlap = 15
    matching = pycolmap.FeatureMatchingOptions(); matching.num_threads = workers
    pycolmap.match_sequential(work / "db.db", pairing_options=seq, matching_options=matching)
    mapping = pycolmap.IncrementalPipelineOptions(); mapping.num_threads = workers
    recs = pycolmap.incremental_mapping(work / "db.db", work / "images", work / "sparse", options=mapping)
    if not recs:
        raise ValueError('Self-calibration registered no model; retained frames/database require source review')
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    if len(rec.cameras) != 1:
        raise ValueError('Single-video self-calibration unexpectedly produced multiple camera groups')
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy, k1, k2, p1, p2 = (float(x) for x in cam.params)
    if (not np.isfinite(cam.params).all() or fx <= 0 or fy <= 0 or rec.num_reg_images() < 20):
        raise ValueError('Self-calibration did not resolve a finite camera with at least20 registered frames')
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != source_digest:
            raise ValueError('Source video changed during self-calibration')
    result = {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "dist": [k1, k2, p1, p2, 0.0],
            "fov_x_deg": float(np.degrees(2 * np.arctan(cam.width / 2 / fx))),
            "reprojection_px": float(rec.compute_mean_reprojection_error()), "frames": rec.num_reg_images(),
            "size": [cam.width, cam.height], "from": "self-calibration",
            "evidence": str(work/'result.json'), "sampled_frames": len(selected),
            "registered_models": {str(k): r.num_reg_images() for k, r in recs.items()},
            "registered_source_frames": sorted(image.name for image in rec.images.values()),
            "limitation": "A single sampling fit; low reprojection error does not bound calibration or metric uncertainty"}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def video_cameras(s, videos, cfg, out):
    """{video index: {K, dist, ...}}: the session's calibration for its own video, splat.json `cameras`, else a
    self-calibration (cached in splat/cameras.json by the video's path and size)."""
    cache_path = out / "cameras.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cams = {}
    for v, path in enumerate(videos):
        intervals = scene_intervals(s, path, primary=v == 0)
        given = (cfg.get("cameras") or {}).get(path)
        if v == 0 and s.config("camera.json", {}):
            cams[v] = {**s.config("camera.json"), "from": "configs camera.json"}
        elif given:
            cams[v] = {**given, "from": "splat.json"}
        else:
            from calibrated_scene_reconstruction.core.media import color_profile
            key = f"{path}|{Path(path).stat().st_size}|color="+json.dumps(color_profile(path), sort_keys=True)
            if intervals is not None:
                key += '|scene_intervals='+json.dumps(intervals)
            if key not in cache:
                print(f"[cameras] self-calibrating {path}", flush=True)
                cache[key] = self_calibrate(path, out / "selfcal", intervals=intervals)
                cache_path.write_text(json.dumps(cache, indent=1) + "\n")
            cams[v] = cache[key]
        print(f"[cameras] video {v}: fx {cams[v]['K'][0][0]:.0f} px ({cams[v]['from']})", flush=True)
    return cams


def intrinsics(rec, img):
    """(fx, fy, cx, cy, width, height) of the frame's own camera."""
    cam = rec.cameras[img.camera_id]
    return (*(float(x) for x in cam.params[:4]), cam.width, cam.height)


def key_cameras(s):
    """{key frame name in the session's video: camera centre in the twin's world} from recon.multiview."""
    mv = json.loads((s / "parsing" / "multiview.json").read_text())
    da = dict(np.load(s / "parsing" / "multiview" / "da3.npz"))
    E = np.tile(np.eye(4), (len(da["names"]), 1, 1))
    E[:, :3, :4] = da["w2c"]
    T = np.einsum("ij,njk->nik", E[0], np.linalg.inv(E))  # view camera -> photo camera
    R_wc, o, _ = s.world_frame()
    R_wc, o = np.asarray(R_wc), np.asarray(o)
    k = float(mv["da3"]["to_point_map"]) * float(mv["scale"]["value"])
    kf = json.loads((s / "capture" / "keyframes.json").read_text())
    idx = {"photo": kf["photo"], **{f"f{v:04d}": v for v in kf["views"]}}
    out = {}
    for j, n in enumerate(str(x) for x in da["names"]):
        n = "photo" if j == 0 else n
        if n in idx:
            out[idx[n]] = (T[j][:3, 3] * k - o) @ R_wc.T
    return out


def board_scale(s, rec, img_dir):
    """Metres per COLMAP unit from the board's corners, triangulated per video over the frames that show it; per
    video its residuals -- the triangulated corners against the board's own (mm, after the fitted similarity) and the
    fitted board reprojected into the frames that show it (px) -- and qa/splat/board_fit.jpg: the detected corners
    (red) and the fitted board's (green) in four of them (S1's check of the board)."""
    path = s.config_dir / "charuco_board.json"
    if not path.exists():
        return None, {}
    from calibrated_scene_reconstruction.scene.capture.calib import board_detector, detect_image, load_board

    det, board = board_detector(load_board(path))
    obs, shots = {}, {}
    for img in rec.images.values():
        hit = detect_image(det, board, cv2.imread(str(img_dir / img.name)), 8)
        if hit is None:
            continue
        fx, fy, cx, cy, _, _ = intrinsics(rec, img)
        P = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) @ np.asarray(img.cam_from_world().matrix())[:3]
        v = img.name.split("_")[0]
        for uv, b in zip(*hit):
            obs.setdefault(v, {}).setdefault(tuple(np.round(b, 5)), (b, []))[1].append((P, uv))
        shots.setdefault(v, []).append((img.name, P, hit))
    per = {}
    for v, corners in obs.items():
        X, B = [], []
        for b, seen in corners.values():
            if len(seen) >= 3:
                A = np.array([r for P, (u, w) in seen for r in (u * P[2] - P[0], w * P[2] - P[1])])
                x = np.linalg.svd(A)[2][-1]
                X.append(x[:3] / x[3])
                B.append(b)
        if len(X) >= 12:
            X, B = np.array(X), np.array(B)
            cb, Rb, tb = umeyama(X, B)  # board ~ cb Rb X + tb
            res_mm = 1000 * np.linalg.norm(cb * X @ Rb.T + tb - B, axis=1)
            allB = board.getChessboardCorners().astype(np.float64)
            Xb = (allB - tb) @ Rb / cb  # the board's corners in COLMAP's frame
            rp = []
            for name, P, (uv, bb) in shots[v]:
                Xh = np.c_[(np.asarray(bb, float) - tb) @ Rb / cb, np.ones(len(bb))] @ P.T
                rp.extend(np.linalg.norm(Xh[:, :2] / Xh[:, 2:] - uv, axis=1).tolist())
            per[v] = (cb, len(X), float(np.median(res_mm)), float(np.percentile(res_mm, 90)), float(np.median(rp)), Xb)
    if not per:
        return None, {}
    qa = s / "qa" / "splat"
    qa.mkdir(parents=True, exist_ok=True)
    tiles = []
    for v, (_, _, _, _, _, Xb) in per.items():
        pick = shots[v][:: max(1, len(shots[v]) // 4)][:4]
        for name, P, (uv, _) in pick:
            im = cv2.imread(str(img_dir / name))
            Xh = np.c_[Xb, np.ones(len(Xb))] @ P.T
            for q in Xh[:, :2] / Xh[:, 2:]:
                cv2.circle(im, (int(q[0]), int(q[1])), 7, (0, 220, 0), 2)
            for q in uv:
                cv2.circle(im, (int(q[0]), int(q[1])), 3, (0, 0, 255), -1)
            cv2.putText(im, name, (20, 50), 0, 1.4, (0, 255, 255), 3)
            tiles.append(cv2.resize(im, (960, 540)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(qa / "board_fit.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    return float(np.median([p[0] for p in per.values()])), {
        v: {"scale": c, "corners": n, "residual_mm_median": round(r50, 2), "residual_mm_p90": round(r90, 2),
            "reprojection_px_median": round(rpx, 2)} for v, (c, n, r50, r90, rpx, _) in per.items()}


def zone_projection(s, rec, img_dir, c, R, t, crop):
    """Per registered frame, <image>_zone.png: 0 where a pixel's ray enters the task zone (the support's box with
    its margin, the robot's column, as splat.task_zone bounds them), 255 elsewhere; and the zone's points out of the
    model (the initialization). Rays are cast at half resolution and the image upsampled."""
    inside, zone = task_zone(s, crop)
    lo, hi = (np.asarray(b, float) for b in zone["support"])
    boxes = [(np.r_[lo[:2] - crop["margin"], crop["keep_floor"]], np.r_[hi[:2] + crop["margin"], zone["top"]])]
    base = zone["robot_base"]
    if base:
        r_ = crop["robot_r"]
        boxes.append((np.array([base["xy"][0] - r_, base["xy"][1] - r_, crop["keep_floor"]]),
                      np.array([base["xy"][0] + r_, base["xy"][1] + r_, crop["robot_h"]])))
    k = 2
    Rw = c * R  # COLMAP -> world (linear part)
    n_masked, n_px = 0, 0
    for img in rec.images.values():
        fx, fy, cx, cy, W, H = intrinsics(rec, img)
        u, v = np.meshgrid((np.arange(0, W, k) + 0.5), (np.arange(0, H, k) + 0.5))
        rays_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
        n_px += W * H
        T = np.asarray(img.cam_from_world().inverse().matrix())  # camera -> COLMAP world
        o = Rw @ T[:3, 3] + t
        d = rays_cam @ T[:3, :3].T @ Rw.T  # directions in the twin's world (scale does not matter for hits)
        hit = np.zeros(len(d), bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            for blo, bhi in boxes:
                t1, t2 = (blo - o) / d, (bhi - o) / d
                tn, tf = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
                hit |= (tf >= np.maximum(tn, 0)) & (tf > 0)
        m = np.where(hit, 0, 255).astype(np.uint8).reshape(u.shape)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(img_dir / (img.name.rsplit(".", 1)[0] + "_zone.png")), m)
        n_masked += int(hit.sum()) * k * k
    P = np.array([p.xyz for p in rec.points3D.values()])
    ids = list(rec.points3D)
    drop = [i for i, z in zip(ids, inside(c * P @ R.T + t)) if z]
    for i in drop:
        rec.delete_point3D(i)
    return n_masked / n_px, len(drop)


def top_extent(W, lo, hi, z_top, margin=0.4):
    """The support top's footprint in the aligned points: of the points within 2 cm of its plane and within `margin`
    of the twin's box, the xy range at the 1st / 99th percentiles along the box's axes (the twin's box came from one
    photo's mask and its views; the walks' points at the board's scale measure the slab itself)."""
    near = (np.abs(W[:, 2] - z_top) < 0.02) & np.all(W[:, :2] > lo[:2] - margin, 1) & np.all(W[:, :2] < hi[:2] + margin, 1)
    Q = W[near, :2]
    if len(Q) < 200:
        return None
    return [[float(np.percentile(Q[:, 0], 1)), float(np.percentile(Q[:, 1], 1))],
            [float(np.percentile(Q[:, 0], 99)), float(np.percentile(Q[:, 1], 99))]], int(len(Q))


MAX_LEVEL = 5.0  # deg: a placement levelled to the support's top by at most this much


def level_normal(Y, tol=0.006, iters=400):
    """A plane's up normal in points (RANSAC: the most within `tol`, refitted by least squares), or None."""
    if len(Y) < 200:
        return None
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(iters):
        q = Y[rng.choice(len(Y), 3, replace=False)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if abs(n[2]) < 0.9:  # not a level-ish plane
            continue
        k = int((np.abs((Y - q[0]) @ n) < tol).sum())
        if k > best[0]:
            best = (k, (n, q[0]))
    if best[1] is None:
        return None
    n, p0 = best[1]
    inl = Y[np.abs((Y - p0) @ n) < tol]
    _, _, vt = np.linalg.svd(inl - inl.mean(0), full_matrices=False)  # (full: an N x N U, 80 GB)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return n * np.sign(n[2])


def plane_z(z):
    """A plane's height in noisy points: the median within +-1.5 cm of their histogram's peak (1 cm bins)."""
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.01))
    peak = e[h.argmax()] + 0.005
    return float(np.median(z[np.abs(z - peak) < 0.015]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--keep-matches", action="store_true", help="reuse database.db (features and matches)")
    ap.add_argument("--reuse-model", action="store_true",
                    help="the COLMAP model as mapped before (sparse/full): only the twin's world, coverage and zone again")
    ap.add_argument("--zone-only", action="store_true", help="only the task zone's projection and initialization on an existing model")
    calibration_mode = ap.add_mutually_exclusive_group()
    calibration_mode.add_argument("--calibrate", action="store_true",
                    help="only the session video's camera, self-calibrated on its own frames, into configs/<name>/"
                         "camera.json when it has none (S0, before perceive: its field of view)")
    calibration_mode.add_argument('--calibration-audit', action='store_true',
                         help='Repeat self-calibration in QA and compare intrinsics; never adopt or overwrite the current camera')
    ap.add_argument('--calibration-samples', type=int, default=150)
    ap.add_argument('--sample-offset', type=int, default=0,
                    help='Raw-frame offset within the calibration stride; use another offset for sampling sensitivity')
    a = ap.parse_args()
    s = Session(a.session)
    if (s/'qa/independent_freeze.json').exists():
        ap.error('Frozen reconstruction: create a new capture session')
    if a.calibration_audit:
        kf = s/'capture/keyframes.json'
        baseline_path = s.config_dir/'camera.json'
        if not kf.exists() or not baseline_path.exists():
            ap.error('Calibration audit needs the current source video and an existing camera')
        video = json.loads(kf.read_text())['video']
        baseline = json.loads(baseline_path.read_text())
        baseline_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        cam = self_calibrate(video, s/'qa/camera_sampling_audit', n=a.calibration_samples,
                             intervals=scene_intervals(s, video, primary=True), sample_offset=a.sample_offset)
        report = {'diagnostic_only': True, 'adopted': False, 'task_pass': False,
                  'baseline_camera': str(baseline_path), 'baseline': baseline, 'candidate': cam,
                  'baseline_camera_sha256': baseline_digest,
                  'baseline_file_unchanged': hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_digest,
                  'relative_focal_change': [cam['K'][i][i]/baseline['K'][i][i]-1 for i in (0, 1)],
                  'scope': 'Sampling and optimizer sensitivity, not a complete uncertainty bound. Source calibration only; no task-object dimensions or contacts enter the fit.'}
        (Path(cam['evidence']).parent/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        return
    cfg = settings(s)
    out = folder(s)
    if a.calibrate:
        kf = s / "capture" / "keyframes.json"
        if (s.config_dir / "camera.json").exists() or not kf.exists():
            print("done: " + ("configs camera.json already there" if kf.exists() else "no video"))
            return
        video = json.loads(kf.read_text())["video"]
        cam = self_calibrate(video, out / "selfcal", n=a.calibration_samples,
                             intervals=scene_intervals(s, video, primary=True), sample_offset=a.sample_offset)
        cam["_note"] = (f"{video} calibrated on its own frames: COLMAP self-calibration (OPENCV, ~150 raw frames; "
                        "recon.splat_capture --calibrate), no board")
        (s.config_dir / "camera.json").write_text(json.dumps(cam, indent=1) + "\n")
        print(f"done: fx {cam['K'][0][0]:.0f} px, field of view {cam['fov_x_deg']:.1f} deg, "
              f"{cam['reprojection_px']:.2f} px over {cam['frames']} frames -> configs/{s.name}/camera.json")
        return
    data = out / "data"
    img_dir = data / "images"
    if a.zone_only:
        full = data / "sparse" / "full"
        if not full.exists():
            shutil.copytree(data / "sparse" / "0", full)
        rec = pycolmap.Reconstruction(str(full))
        c, R, t = sim3(s)
        share, dropped = zone_projection(s, rec, img_dir, c, R, t, cfg["crop"])
        shutil.rmtree(data / "sparse" / "0")
        (data / "sparse" / "0").mkdir()
        rec.write(data / "sparse" / "0")
        (out / "zone.json").write_text(json.dumps({"zone_pixel_share": share, "points_dropped": dropped}) + "\n")
        print(f"done: the zone on {100 * share:.1f} % of the pixels of {len(rec.images)} frames, {dropped} points out of the initialization")
        return
    kf_path = s / "capture" / "keyframes.json"
    if not kf_path.exists():
        sys.exit("splat_capture: the session has no video (capture/keyframes.json): nothing to build the background from")
    kf = json.loads(kf_path.read_text())
    videos = [kf["video"]] + [v for v in cfg["videos"] if v != kf["video"]]
    # Check every stream before deleting/copying any capture data.
    intervals_by_video = {v: scene_intervals(s, path, primary=v == 0) for v, path in enumerate(videos)}
    keys = key_cameras(s)

    # --- frames
    reuse = a.reuse_model and (data / "sparse" / "full").exists()
    from calibrated_scene_reconstruction.core.media import color_profile
    profiles = {str(i): color_profile(path) for i, path in enumerate(videos)}
    rectification_path = out/'rectification.json'
    previous = json.loads(rectification_path.read_text()) if rectification_path.exists() else {}
    if (reuse or a.keep_matches) and any(img_dir.glob('*.jpg')):
        old_profiles = previous.get('color_profiles')
        if old_profiles != profiles and (old_profiles is not None or
                any(p['mode'] == 'hdr_base_to_srgb' for p in profiles.values())):
            raise ValueError('Video color management changed; preserve old frames and start a new capture')
    from calibrated_scene_reconstruction.scene.capture.supplementary import assert_static_frames, load as media_plan
    if media_plan(s) is not None and (reuse or a.keep_matches):
        rates = []
        for path in videos:
            stream = VideoCapture(str(path))
            rates.append(float(stream.get(cv2.CAP_PROP_FPS)))
            stream.release()
        assert_static_frames(s, [p.name for p in img_dir.glob('*.jpg')], videos, rates)
        if reuse:
            retained = pycolmap.Reconstruction(str(data/'sparse/full'))
            assert_static_frames(s, [im.name for im in retained.images.values()], videos, rates)
    if not a.keep_matches and not reuse:
        shutil.rmtree(data, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    cams = video_cameras(s, videos, cfg, out)
    from calibrated_scene_reconstruction.scene.capture.valid_pixels import undistorted_valid
    rectification_path=out/'rectification.json'
    previous=json.loads(rectification_path.read_text()) if rectification_path.exists() else {}
    if any(img_dir.glob('*.jpg')) and previous:
        for index,camera in cams.items():
            old=previous.get('cameras',{}).get(str(index))
            if old is None or any(not np.array_equal(old.get(key),camera.get(key)) for key in ('K','dist')):
                raise ValueError('Camera rectification changed; preserve existing frames and start a new capture')
    rectification={'schema':1,'opencv_version':cv2.__version__, 'color_profiles': profiles,
                   'policy':'OpenCV undistortion; valid pixels have full linear-remap source support','cameras':{}}
    listing, fps = {}, {}
    for v, path in enumerate(videos):
        K, dist = np.asarray(cams[v]["K"], float), np.asarray(cams[v]["dist"], float)
        fr, fps[v] = frames_of(path, cfg["step"], cfg["keep"], set(keys) if v == 0 else set(),
                               intervals=intervals_by_video[v])
        if not fr:  # a wrong path reads as an empty video (a worktree's data/ is its own copy)
            sys.exit(f"splat_capture: no frames read from {path}")
        if cfg.get("motion"):  # the frames kept by how far the view moved (splat.json motion)
            n0 = len(fr)
            fr = by_motion(fr, cfg["motion"], set(keys) if v == 0 else set(), cfg.get("max_gap", 15))
            print(f"[frames] video {v}: {len(fr)} of {n0} kept by motion ({cfg['motion']} of the width)", flush=True)
        listing[v] = sorted(fr)
        height,width=next(iter(fr.values())).shape[:2]
        valid=undistorted_valid(K,dist,(width,height))
        rectification['cameras'][str(v)]={**cams[v],'size':[width,height]}
        for j, f in fr.items():
            name = img_dir / f"{v}_{j:05d}.jpg"
            if not name.exists():
                cv2.imwrite(str(name), cv2.undistort(f, K, dist), [cv2.IMWRITE_JPEG_QUALITY, 95])
            valid_path=img_dir/f'{v}_{j:05d}_valid.png'
            if not valid_path.exists():cv2.imwrite(str(valid_path),valid.astype('uint8')*255)
        print(f"[frames] video {v} {path}: {len(fr)} frames", flush=True)
    rectification_path.write_text(json.dumps(rectification,indent=2)+'\n')
    (s / "qa" / "splat").mkdir(parents=True, exist_ok=True)
    frames_timeline(listing, fps, s / "qa" / "splat" / "frames_timeline.png")

    # --- COLMAP
    db = data / "database.db"
    if reuse:  # the model as mapped (incremental mapping is not repeatable: the align would move with each run)
        rec = pycolmap.Reconstruction(str(data / "sparse" / "full"))
        recs = {0: rec}
    if not reuse:
        if not a.keep_matches:
            db.unlink(missing_ok=True)
        # A killed extraction can leave an existing but incomplete database.
        # COLMAP skips already completed feature/match rows; finish pending
        # work rather than treating mere file existence as successful matching.
        ex = pycolmap.FeatureExtractionOptions()
        workers = cfg['colmap_workers']
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError('colmap_workers must be an integer1..32')
        ex.num_threads = workers
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = workers
        ex.sift.max_num_features = 8192
        for v in listing:  # one fixed pinhole per video (frames undistorted with its camera above)
            K = np.asarray(cams[v]["K"], float)
            ro = pycolmap.ImageReaderOptions()
            ro.camera_model = "PINHOLE"
            ro.camera_params = ",".join(str(x) for x in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
            pycolmap.extract_features(db, img_dir, image_names=[f"{v}_{j:05d}.jpg" for j in listing[v]],
                                      camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=ex)
        seq = pycolmap.SequentialPairingOptions()
        seq.overlap = 20
        seq.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=seq, matching_options=matching)
        pairs = []
        if len(videos) > 1:
            pick = {v: [j for j in listing[v] if j % max(1, int(cfg["cross_every_s"] * fps[v])) < cfg["step"]]
                    for v in listing}
            pairs = [f"{v}_{i:05d}.jpg {w}_{j:05d}.jpg" for v in pick for w in pick if v < w
                     for i in pick[v] for j in pick[w]]
        if cfg.get("loop_every_s"):
            loop = {v: [j for j in listing[v] if j % max(1, int(cfg["loop_every_s"] * fps[v])) < cfg["step"]]
                    for v in listing}
            win = seq.overlap * cfg["step"] * 2  # beyond the sequential matching's reach
            pairs += [f"{v}_{i:05d}.jpg {v}_{j:05d}.jpg" for v in loop for i in loop[v] for j in loop[v] if j - i > win]
            print(f"[colmap] {sum(len(x) for x in loop.values())} frames matched as loops", flush=True)
        if pairs:
            (data / "cross_pairs.txt").write_text("\n".join(pairs) + "\n")
            po = pycolmap.ImportedPairingOptions()
            po.match_list_path = data / "cross_pairs.txt"
            pycolmap.match_image_pairs(db, pairing_options=po, matching_options=matching)
            print(f"[colmap] {len(pairs)} pairs between the videos and within each beyond its sequence", flush=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = cfg['colmap_workers']
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    # a global bundle adjustment every 30 % more frames or points (COLMAP: 10 %): 1450 frames of three walks spent
    # over an hour in global adjustments of 20-30 min each; the poses are refined once more at the end anyway
    opts.ba_global_frames_ratio = 1.3
    opts.ba_global_points_ratio = 1.3
    if not reuse:
        shutil.rmtree(data / "sparse", ignore_errors=True)
        recs = pycolmap.incremental_mapping(db, img_dir, data / "sparse", options=opts)
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        shutil.rmtree(data / "sparse", ignore_errors=True)
        (data / "sparse" / "full").mkdir(parents=True)
        rec.write(data / "sparse" / "full")
    n_all = len(list(img_dir.glob("*.jpg")))
    reg = {v: sum(1 for i in rec.images.values() if i.name.startswith(f"{v}_")) for v in listing}
    print(f"[colmap] registered {rec.num_reg_images()} / {n_all} ({reg}), {rec.num_points3D()} points, "
          f"reprojection {rec.compute_mean_reprojection_error():.2f} px; {len(recs)} model(s)", flush=True)

    # --- the twin's world
    by_name = {i.name: i for i in rec.images.values()}
    pairs = [(np.asarray(by_name[f"0_{j:05d}.jpg"].projection_center()), c) for j, c in keys.items()
             if f"0_{j:05d}.jpg" in by_name]
    if len(pairs) < 3:
        sys.exit(f"splat_capture: only {len(pairs)} key frames registered: cannot place the frames in the twin")
    src, dst = np.array([p for p, _ in pairs]), np.array([q for _, q in pairs])
    P = np.array([p.xyz for p in rec.points3D.values()])
    sg = s.scene_graph
    lo, hi = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
    photo_planes = None
    robot_anchor=s.config('twin.json',{}).get('scale') in ('robot_visual','robot_joints')
    if cfg.get('align', {}).get('photo_planes', robot_anchor):
        from calibrated_scene_reconstruction.scene.capture.placement_planes import from_photo
        photo_planes = from_photo(s, rec)
        print(f'[align] photo-labelled planes: {len(photo_planes[0])} floor, '
              f'{len(photo_planes[1])} support tracks', flush=True)

    def planes(c, R, t):
        if photo_planes is not None:
            return tuple(plane_z((c * points @ R.T + t)[:, 2]) for points in photo_planes[:2])
        W = c * P @ R.T + t
        inner = np.all(W[:, :2] > lo[:2] + 0.1, 1) & np.all(W[:, :2] < hi[:2] - 0.1, 1)
        return plane_z(W[np.abs(W[:, 2]) < 0.25, 2]), plane_z(W[inner & (np.abs(W[:, 2] - hi[2]) < 0.25), 2])

    c0, R0, t0 = umeyama(src, dst)
    if s.config('twin.json', {}).get('scale') in ('robot_visual','robot_joints'):
        bscale, per_video = None, {}  # robot-only anchors must not consume board scale
    else:
        bscale, per_video = board_scale(s, rec, img_dir)
    if bscale:
        scale, how = bscale, "board"
    else:
        f0, s0 = planes(c0, R0, t0)
        if not np.isfinite([f0, s0]).all() or s0 - f0 <= 1e-6:
            raise ValueError('Observed support must lie above the observed floor for provisional placement')
        scale, how = c0 * float(hi[2]) / (s0 - f0), "support height"
    c, R, t = umeyama(src, dst, scale)
    W = c * P @ R.T + t
    _, top0 = planes(c, R, t)
    n_top = (level_normal(c * photo_planes[0] @ R.T + t) if photo_planes is not None else
             level_normal(W[np.all(W[:, :2] > lo[:2] - 0.1, 1) & np.all(W[:, :2] < hi[:2] + 0.1, 1)
                            & (np.abs(W[:, 2] - top0) < 0.03)]))
    levelled = 0.0
    tilt0 = float(np.degrees(np.arccos(np.clip(n_top[2], -1, 1)))) if n_top is not None else 0.0
    # the floor's own plane (the points well under the top): a top and a floor parallel to each other are level
    # together whatever the photo's gravity said -- then levelled however far
    n_floor = level_normal(W[W[:, 2] < top0 - 0.3]) if n_top is not None and tilt0 > MAX_LEVEL and photo_planes is None else None
    agree = photo_planes is not None or (n_floor is not None and
            float(np.degrees(np.arccos(np.clip(n_floor @ n_top, -1, 1)))) < 2.0)
    if agree and photo_planes is None:
        print(f"[align] the support's top and the floor both {tilt0:.1f} deg off level, parallel: the photo's "
              "gravity was off -- levelled to them", flush=True)
    if tilt0 > MAX_LEVEL and not agree:  # not the photo's gravity a little off: the plane found is not the top's (the scene
        # graph's support box from another world frame took a wall's slice of points: 20.9 deg) -- left as placed
        print(f"[align] WARNING the support's top plane {tilt0:.1f} deg off level: not levelled (over {MAX_LEVEL} "
              "deg: the support's box is not where the frames put the top -- the scene graph's world frame?)",
              flush=True)
    elif n_top is not None and n_top[2] < np.cos(np.radians(0.2)):
        v = np.cross(n_top, [0.0, 0.0, 1.0])
        sn, cs_ = float(np.linalg.norm(v)), float(n_top[2])
        Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rl = np.eye(3) + Vx + Vx @ Vx * ((1 - cs_) / max(sn * sn, 1e-12))
        piv = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, top0])  # about the support's middle
        R, t = Rl @ R, Rl @ (t - piv) + piv
        levelled = float(np.degrees(np.arccos(np.clip(cs_, -1, 1))))
        label = 'photo-labelled floor' if photo_planes is not None else "support's top plane"
        print(f"[align] levelled by {levelled:.2f} deg to the {label}", flush=True)
    floor_before_zeroing = planes(c, R, t)[0]
    t[2] -= floor_before_zeroing
    floor, top = planes(c, R, t)
    if photo_planes is not None and (not np.isfinite([floor, top]).all() or top <= floor):
        rejected = s/'qa/placement_alignment_rejected.json'
        rejected.parent.mkdir(parents=True, exist_ok=True)
        rejected.write_text(json.dumps({'pass':False,'reason':'Observed support is not above the observed floor',
                 'floor_z':float(floor),'support_z':float(top),'scale':float(c),
                 'action':'Review current-capture floor/support semantics; reuse its own COLMAP model after correction'},
                indent=2)+'\n')
        raise ValueError('Observed floor/support placement is inconsistent; refused to generate posed frames')
    res = np.linalg.norm(c * src @ R.T + t - dst, axis=1)
    ext = top_extent(c * P @ R.T + t, lo, hi, top)
    align = {"sim3": {"scale": c, "R": R.tolist(), "t": t.tolist()}, "scale_from": how, "board": per_video,
             "levelled_deg": round(levelled, 2),
             "key_frames_scale": c0, "checks": {"floor_z": floor, "support_top_z": top,
                                                "twin_support_top_z": float(hi[2]),
                                                "key_residual_m": [float(np.median(res)), float(res.max())],
                                                "support_top_xy": ext[0] if ext else None,
                                                "support_top_points": ext[1] if ext else 0,
                                                "twin_support_xy": [lo[:2].tolist(), hi[:2].tolist()]}}
    if photo_planes is not None:
        align['photo_planes'] = {**photo_planes[2], 'floor_tracks': len(photo_planes[0]),
                                 'support_tracks': len(photo_planes[1]),
                                 'floor_z_before_zeroing': float(floor_before_zeroing),
                                 'note': 'Observed tracked surfaces select height layers. Support draft height is provisional; robot/board calibration still requires S1 review.'}
    (out / "align.json").write_text(json.dumps(align, indent=1) + "\n")
    print(f"[align] {c:.4f} m per unit from the {how} (the key frames alone {c0:.4f}: {100 * (c0 / c - 1):+.1f} %); "
          f"support top {top:.3f} m (twin {hi[2]:.3f}); key cameras off by {100 * np.median(res):.1f} cm median",
          flush=True)
    for v, b in per_video.items():
        print(f"[board] video {v}: {b['corners']} corners, against the board {b['residual_mm_median']:.1f} mm median "
              f"({b['residual_mm_p90']:.1f} p90), reprojected {b['reprojection_px_median']:.1f} px median "
              f"(qa/splat/board_fit.jpg)", flush=True)
    if ext:
        (a, b), n = ext
        print(f"[align] support top in the points: x {a[0]:.3f}..{b[0]:.3f} ({b[0] - a[0]:.3f} m), y {a[1]:.3f}..{b[1]:.3f} "
              f"({b[1] - a[1]:.3f} m) from {n} points; the twin's box {hi[0] - lo[0]:.3f} x {hi[1] - lo[1]:.3f} m",
              flush=True)

    # --- coverage
    if (s / "scene" / "manifest.json").exists():
        man = s.manifest["surfaces"]
        (x0, x1), (y0, y1) = man["room_x"], man["room_y"]
    else:  # a twin's poses stage runs before the layout: the room as the frames' points spread
        man = None
        x0, x1, y0, y1 = np.inf, -np.inf, np.inf, -np.inf
    centres = {i: c * np.asarray(img.projection_center()) @ R.T + t for i, img in rec.images.items()}
    # the room as the frames see it: the twin's room and every point the frames placed within 10 m of the support
    W = c * P @ R.T + t
    near = W[np.linalg.norm(W[:, :2] - (lo[:2] + hi[:2]) / 2, axis=1) < 10]
    x0, y0 = min(x0, float(np.percentile(near[:, 0], 1))), min(y0, float(np.percentile(near[:, 1], 1)))
    x1, y1 = max(x1, float(np.percentile(near[:, 0], 99))), max(y1, float(np.percentile(near[:, 1], 99)))
    cell = 0.25
    nx, ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    seen = [[set() for _ in range(ny)] for _ in range(nx)]
    dirs = [[[] for _ in range(ny)] for _ in range(nx)]
    for p in rec.points3D.values():
        X = c * np.asarray(p.xyz) @ R.T + t
        i, j = int((X[0] - x0) / cell), int((X[1] - y0) / cell)
        if not (0 <= i < nx and 0 <= j < ny) or not -0.1 < X[2] < (man["ceiling_z"] if man else 3.0) + 1.0:
            continue
        for el in p.track.elements:
            seen[i][j].add(el.image_id)
            d = centres[el.image_id][:2] - X[:2]
            dirs[i][j].append(np.arctan2(d[1], d[0]))
    n = np.array([[len(seen[i][j]) for j in range(ny)] for i in range(nx)])
    sp = np.array([[angular_spread(dirs[i][j]) for j in range(ny)] for i in range(nx)])
    g = [[grade(int(n[i, j]), float(sp[i, j])) for j in range(ny)] for i in range(nx)]
    colours = {"unseen": (40, 40, 40), "thin": (200, 60, 60), "one-sided": (230, 160, 40), "good": (60, 180, 60)}
    px = 24
    img = np.zeros((ny * px, nx * px, 3), np.uint8)
    for i in range(nx):
        for j in range(ny):
            img[(ny - 1 - j) * px:(ny - j) * px, i * px:(i + 1) * px] = colours[g[i][j]]

    def to_px(x, y):
        return int((x - x0) / cell * px), int((y1 - y) / cell * px)

    if man:  # the mesh twin's room
        cv2.rectangle(img, to_px(man["room_x"][0], man["room_y"][1]), to_px(man["room_x"][1], man["room_y"][0]),
                      (120, 120, 120), 1)
    cv2.rectangle(img, to_px(lo[0], hi[1]), to_px(hi[0], lo[1]), (255, 255, 255), 2)
    for v, col in zip(listing, [(0, 220, 255), (255, 80, 255), (255, 255, 0), (0, 255, 160)]):
        path = np.array([to_px(*centres[i][:2]) for i in sorted(centres) if rec.images[i].name.startswith(f"{v}_")])
        if len(path):
            cv2.polylines(img, [path.reshape(-1, 1, 2)], False, col, 2)
    for k, lab in enumerate(colours):
        cv2.rectangle(img, (8, 8 + 22 * k), (24, 24 + 22 * k), colours[lab], -1)
        cv2.putText(img, lab, (30, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out / "coverage.png"), img[:, :, ::-1])
    share = {lab: round(float(sum(row.count(lab) for row in g)) / (nx * ny), 3) for lab in colours}
    (out / "coverage.json").write_text(json.dumps({"origin": [x0, y0], "cell_m": cell, "shape": [nx, ny],
                                                   "share": share, "frames": n.tolist(),
                                                   "spread_deg": sp.round(0).tolist()}) + "\n")
    (out / "frames.json").write_text(json.dumps({"videos": videos, "frames": {v: listing[v] for v in listing},
                                                 "registered": reg}, indent=1) + "\n")
    # the task zone out of the training: masks per frame, its points out of the initialization (sparse/0)
    z_share, dropped = zone_projection(s, rec, img_dir, c, R, t, cfg["crop"])
    (data / "sparse" / "0").mkdir(parents=True, exist_ok=True)  # (--reuse-model: written before)
    rec.write(data / "sparse" / "0")  # 3DGRUT reads <path>/sparse/0
    (out / "zone.json").write_text(json.dumps({"zone_pixel_share": z_share, "points_dropped": dropped}) + "\n")
    print(f"[zone] on {100 * z_share:.1f} % of the pixels, {dropped} of its points out of the initialization", flush=True)
    print(f"[coverage] {nx} x {ny} columns of {cell} m: {share}", flush=True)
    print(f"done: {rec.num_reg_images()} frames posed, scale from the {how}")


if __name__ == "__main__":
    main()
