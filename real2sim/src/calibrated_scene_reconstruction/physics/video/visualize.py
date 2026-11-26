"""A recording's result as one video, for a person (or the agent) to check what was measured from what (CPU).

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.visualize real/<name> [...]

Writes real/<name>/summary.mp4. Left: the video with the object's mask outlined in green, whatever touches it tinted
red, the tracked point as a white dot and, during a free phase, where the fitted constant-deceleration model puts it
as a yellow cross (the two agree when the fit explains the motion). Right: the path on the support plane seen from
above (metres; track in grey, each free phase's fit in colour), the speed along time with each phase's fitted line,
and the parameters with where each comes from. Of a carried object the plot is its acceleration instead (what its
bound is read from), with four times the share of it the track's jitter makes; a refused one has empty panels.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.resolution import scaled_even_size
from calibrated_scene_reconstruction.physics.common.observations import project
from .fit import slide_positions, speeds
from .scene import read_frames

COLOURS = [(255, 200, 0), (0, 200, 255), (255, 90, 200), (120, 255, 120), (255, 140, 60), (160, 160, 255)]
PANEL_W = 520


def length_as_filmed(q):
    """The pendulum's length that draws the swing in the video's own time: the fitted one, also where the length
    that is given is another (footage that is slowed) or none (a clock in doubt)."""
    return q["pendulum.length"].get("as_filmed") or q["pendulum.length"]["value"]


def _axes(img, box, xlim, ylim, title, cv2):
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 70), 1)
    cv2.putText(img, title, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def to_px(x, y):
        u = x0 + (np.asarray(x, float) - xlim[0]) / max(xlim[1] - xlim[0], 1e-9) * (x1 - x0)
        v = y1 - (np.asarray(y, float) - ylim[0]) / max(ylim[1] - ylim[0], 1e-9) * (y1 - y0)
        return np.stack([u, v], -1)

    for val, horiz in [(ylim[0], True), (ylim[1], True), (xlim[0], False), (xlim[1], False)]:
        p = to_px(xlim[0], val) if horiz else to_px(val, ylim[0])
        pos = (x0 - 44, int(p[1]) + 4) if horiz else (int(p[0]) - 14, y1 + 14)
        cv2.putText(img, f"{val:.2f}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
    return to_px


def _poly(img, pts, col, cv2, thick=1):
    pts = pts[np.all(np.isfinite(pts), 1)]
    if len(pts) > 1:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, thick, cv2.LINE_AA)


def render(path):
    import cv2

    path = Path(path)
    frames, fps = read_frames(path)
    per, tr = np.load(path / "perception.npz"), np.load(path / "track.npz")
    twin, phys = read_document(path / "twin.json"), read_document(path / "physics.json")
    meta = read_document(path / "meta.json")
    F, H, W = frames.shape[:3]
    K, T = np.asarray(twin["K"], float), np.asarray(twin["T_world_cam"], float)
    o = twin["object"]
    hc = o.get("centroid_height", o["dims"]["height"] / 2)
    t, xy, valid = tr["t"], tr["xy"], tr["valid"]
    motion = phys.get("motion", "slide")
    phases = phys.get("free_phases", []) if motion in ("slide", "roll", "incline") else []
    model = np.full((F, 2), np.nan)   # the fitted position per frame
    which = np.full(F, -1)
    path_uv, along = np.full((F, 2), np.nan), np.full(F, np.nan)
    for j, p in enumerate(phases):
        k = np.arange(p["first_frame"], p["last_frame"] + 1)
        if motion == "incline" and "path_start" in p:
            # a support seen along itself (incline.run_along_path): the object's place along its own line, which
            # is a line in the camera's frame, and where the mask's centroid says it is along it
            from calibrated_scene_reconstruction.physics.common.observations import mask_stats

            tt = t[k] - t[k[0]]
            c_, u_ = np.array(p["path_camera"]["point"]), np.array(p["path_camera"]["direction"])
            s_ = p["path_start"] + p["release_speed"] * tt + 0.5 * p["acceleration"] * tt ** 2
            X = c_ + np.outer(s_, u_)
            path_uv[k] = (X @ K.T)[:, :2] / X[:, 2:]
            uv_o = np.array([mask_stats(m)[0] for m in per["object"][k]])
            r_ = np.c_[(uv_o[:, 0] - K[0, 2]) / K[0, 0], (uv_o[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(k))]
            r_ /= np.linalg.norm(r_, axis=1, keepdims=True)
            ru = r_ @ u_
            along[k] = ((r_ @ c_) * ru - c_ @ u_) / np.maximum(1 - ru ** 2, 1e-6)
            model[k], which[k] = np.c_[s_, np.zeros(len(k))], j
            continue
        if motion == "incline":  # a constant acceleration along the heading (incline.fit_ramp)
            tt = t[k] - t[k[0]]
            u = np.array([np.cos(np.radians(p["heading_deg"])), np.sin(np.radians(p["heading_deg"]))])
            model[k] = np.array(p["start_xy"]) + np.outer(p["release_speed"] * tt + 0.5 * p["acceleration"] * tt ** 2, u)
            which[k] = j
            continue
        prm = [p["start_xy"][0], p["start_xy"][1], np.radians(p["heading_deg"]), p["release_speed"],
               p["coefficient"] * 9.81]
        model[k], which[k] = slide_positions(prm, t[k] - t[k[0]]), j
    uv_track = project(K, T, np.c_[xy, np.full(F, hc)])[0]
    uv_model = project(K, T, np.c_[model, np.full(F, hc)])[0]
    if np.isfinite(path_uv).any():  # no track on a plane: the place along the line, level in the panel
        uv_model, uv_track = path_uv, np.full((F, 2), np.nan)
        xy, valid = np.c_[along, np.zeros(F)], np.isfinite(along)
    v = speeds(t, np.where(valid[:, None], xy, np.nan))
    drop = phys.get("motion") == "drop" and phys.get("free_phase")
    plot_title = "speed (m/s) over time (s)"
    if drop:  # a flight: the fitted path of the centre, and the height the mask's centroid has above that path
        fl = phys["free_phase"]["flight"]
        kf, Pf = np.array(fl["frames"]), np.array(fl["position"], float)
        uv_model[kf] = project(K, T, Pf)[0]
        uv_track[:] = np.nan
        model3 = np.full((F, 3), np.nan)
        model3[kf] = Pf
        cam = T[:3, 3]
        from calibrated_scene_reconstruction.physics.common.observations import mask_stats
        from .scene import ray_plane
        zobs = np.full(F, np.nan)
        for i in kf:
            d = ray_plane(K, T, mask_stats(per["object"][i])[0], z=-1e3) - cam
            if np.all(np.isfinite(d)):
                s_ = ((model3[i, :2] - cam[:2]) @ d[:2]) / max(d[:2] @ d[:2], 1e-12)
                zobs[i] = cam[2] + s_ * d[2]
        xy = model3[:, :2].copy()
        v = zobs
        plot_title = "height of the centre (m) over time (s): mask (grey), fitted flight"
    swing = None
    if motion == "pendulum" and phys.get("free_phase"):  # the fitted swing, and the bob along it in the image
        from calibrated_scene_reconstruction.physics.common.observations import mask_stats
        from .pendulum import bob, swing as swing_of

        fp, q = phys["free_phase"], phys["parameters"]
        k = np.arange(fp["first_frame"], min(fp.get("last_frame", F - 1) + 1, F))
        th = swing_of(np.radians(fp["theta0_deg"]), np.radians(fp["omega0_deg"]), length_as_filmed(q),
                      q["pendulum.damping_rate"]["value"], q["pendulum.dry_friction"]["value"], t[k] - t[k[0]])
        uv_model[:] = np.nan
        uv_model[k] = bob(K, np.array(fp["gravity_camera"]), fp["pivot_uv"], fp["pivot_distance"],
                          np.radians(fp["azimuth_deg"]), length_as_filmed(q), th)[0]
        uv_obs = np.array([mask_stats(m)[0] for m in per["object"]])
        uv_obs[per["score"] <= 0.05] = np.nan
        c0 = np.nanmean(uv_model[k], 0)
        ax = np.linalg.eigh(np.cov((uv_model[k] - c0).T))[1][:, -1]
        swing = {"obs": (uv_obs - c0) @ ax, "model": (uv_model - c0) @ ax, "k": k, "uv": uv_obs,
                 "pivot": np.array(fp["pivot_uv"], float),
                 "arc": bob(K, np.array(fp["gravity_camera"]), fp["pivot_uv"], fp["pivot_distance"],
                            np.radians(fp["azimuth_deg"]), length_as_filmed(q),
                            np.linspace(-np.abs(th).max(), np.abs(th).max(), 80))[0]}
        uv_track[:] = np.nan
        xy = np.full((F, 2), np.nan)
        v = np.full(F, np.nan)
        plot_title = "the bob along its swing in the image (px) over time (s): seen (grey), fitted"
    turn = None
    if motion == "turntable" and phys.get("path"):  # the path in the plane the fit used, and its distance from the axis
        xy = np.array([q if q else [np.nan, np.nan] for q in phys["path"]], float)
        axis = np.array(phys["axis"], float)
        v = np.linalg.norm(xy - axis, axis=1)
        turn = {"axis": axis, "rides": phys.get("rides", []), "slides": phys.get("slides", [])}
        plot_title = "distance from the axis (m) over time (s): seen (grey), rides and slides"
    carry = None
    if motion == "carried":  # what is read is the object's acceleration, not its speed: that, the jitter's share
        from .carried import series  # of it, and the part above that where a friction is read

        carry = series(path)
        if "note" in carry or "carried" not in phys:  # refused: no acceleration of a track it would not stand by
            carry = None
            xy, v = np.full((F, 2), np.nan), np.full(F, np.nan)
        else:
            v = np.full(F, np.nan)
            plot_title = "acceleration (g) over time (s): seen (grey), 4 x jitter's (dim), read (colour)"
    xy = np.where(np.abs(xy) < 50.0, xy, np.nan)  # a track off its plane (an object in the air) has no place on it
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & (valid | bool(drop))
    pad = 0.05
    xlim = (np.nanmin(xy[ok, 0]) - pad, np.nanmax(xy[ok, 0]) + pad) if ok.any() else (-1, 1)
    ylim = (np.nanmin(xy[ok, 1]) - pad, np.nanmax(xy[ok, 1]) + pad) if ok.any() else (-1, 1)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])  # equal aspect
    xlim = (np.mean(xlim) - span / 2, np.mean(xlim) + span / 2)
    ylim = (np.mean(ylim) - span / 2, np.mean(ylim) + span / 2)
    v = np.where(np.abs(v) < 50.0, v, np.nan)
    vmax = max(float(np.nanmax(v)) if np.isfinite(v).any() else 1.0, 0.1) * 1.1
    vmin = 0.0
    if carry is not None:
        inner = np.isfinite(carry["sure"])  # the ends a spline rings at are not read, and not what sets the scale
        g_all = np.linalg.norm(carry["acc"], axis=1)[inner if inner.any() else slice(None)]
        vmax = max(float(g_all.max()), float(4 * carry["noise"].max())) / 9.81 * 1.3
    if swing is not None:
        vmax = float(np.nanmax(np.abs(swing["model"]))) * 1.15
        vmin = -vmax
    s = 720 / H if H > 720 else 1.0
    VW, VH = scaled_even_size(W, H, s)
    OH = max(VH, 720)
    tmp = path / "summary_raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 60.0), (VW + PANEL_W, OH))
    lines = [f"{path.name}", f"object: {meta['object']}   motion: {phys.get('motion', 'slide')}",
             f"scale: {twin['scale']['source']} (+-{100 * twin['scale']['rel_std']:.0f} %)", ""]
    for name, prm in phys["parameters"].items():
        err = f" +- {prm['stderr']:.3f}" if "stderr" in prm else ""
        if prm.get("value") is None and ("lower_bound" in prm or "upper_bound" in prm):  # a bound is all it gives
            lines.append(f"{name} " + (f">= {prm['lower_bound']:.3f}" if "lower_bound" in prm
                                       else f"< {prm['upper_bound']:.3f}"))
        elif prm.get("value") is None:  # not taken, and its source says why
            lines.append(f"{name}: not given")
        else:
            lines.append(f"{name} = {prm['value']:.3f}{err}" + (f" {prm['unit']}" if "unit" in prm else ""))
        lines.append(f"    {prm['source']}" + (f", {prm['phases']} phase(s)" if "phases" in prm else ""))
    if "sim" in phys:
        for key in ("support.kinetic_friction", "support.rolling_resistance", "object.restitution"):
            if key in phys["sim"]:
                lines.append(f"PhysX fit: {key.split('.')[1]} {phys['sim'][key]:.3f}")
    if phys.get("note"):
        lines += textwrap.wrap(phys["note"], 62)[:6]
    obj_m, man_m = per["object"], per["manipulator"]  # once: an .npz member is unpacked whole at every access
    for i in range(F):
        img = frames[i].astype(np.float32)
        m = man_m[i]
        img[m] = 0.55 * img[m] + 0.45 * np.array([255, 40, 40])
        img = img.astype(np.uint8)
        om = obj_m[i].astype(np.uint8)
        cs, _ = cv2.findContours(om, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lost = swing is not None and not np.isfinite(swing["uv"][i, 0])  # a mask the segmenter did not stand by
        if not lost:
            cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
        if np.all(np.isfinite(uv_track[i])) and valid[i] and not drop:
            cv2.circle(img, tuple(np.round(uv_track[i]).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if np.all(np.isfinite(uv_model[i])):
            cv2.drawMarker(img, tuple(np.round(uv_model[i]).astype(int)), (255, 230, 0), cv2.MARKER_CROSS, 18, 2)
        state = "contact" if tr["contact"][i] else (f"free phase {which[i] + 1}" if which[i] >= 0 else
                                                    ("not tracked" if not valid[i] else ""))
        if lost and not tr["contact"][i]:
            state = "bob not segmented (a streak): frame not used"
        img = cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"t = {t[i]:.2f} s  {state}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"t = {t[i]:.2f} s  {state}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 80, 80) if tr["contact"][i] else (255, 255, 255), 2, cv2.LINE_AA)
        canvas = np.zeros((OH, VW + PANEL_W, 3), np.uint8)
        canvas[:VH, :VW] = img
        pan = canvas[:, VW:]
        pan[:] = 24
        if swing is not None:  # a pendulum has no path on a support: the bob in the image, and its fitted arc
            if i == 0:
                pts = np.concatenate([swing["uv"][np.isfinite(swing["uv"][:, 0])], swing["arc"], swing["pivot"][None]])
                mid, half = 0.5 * (pts.min(0) + pts.max(0)), 0.5 * float(np.ptp(pts, 0).max()) + 15
                to1 = _axes(pan, (56, 30, 56 + 210, 30 + 210), (mid[0] - half, mid[0] + half), (-mid[1] - half, -mid[1] + half),
                            "the bob in the image (px, y up): seen (grey), fitted, pivot (+)", cv2)
                for a in np.nonzero(np.isfinite(swing["uv"][:, 0]))[0]:
                    cv2.circle(pan, tuple(np.round(to1(swing["uv"][a, 0], -swing["uv"][a, 1])).astype(int)), 1,
                               (150, 150, 150), -1, cv2.LINE_AA)
                _poly(pan, to1(swing["arc"][:, 0], -swing["arc"][:, 1]), COLOURS[0], cv2, 2)
                cv2.drawMarker(pan, tuple(np.round(to1(swing["pivot"][0], -swing["pivot"][1])).astype(int)),
                               (235, 235, 235), cv2.MARKER_CROSS, 10, 1)
                base1, to_bob = pan[:262].copy(), to1
            pan[:262] = base1
            if np.all(np.isfinite(uv_model[i])):
                cv2.circle(pan, tuple(np.round(to_bob(uv_model[i, 0], -uv_model[i, 1])).astype(int)), 4, (255, 255, 255),
                           -1, cv2.LINE_AA)
        else:
            to1 = _axes(pan, (56, 30, 56 + 210, 30 + 210), xlim, ylim, "path on the support, top view (m)", cv2)
            _poly(pan, to1(xy[ok, 0], xy[ok, 1]), (120, 120, 120), cv2)
        for j, p in enumerate(phases):
            k = which == j
            _poly(pan, to1(model[k, 0], model[k, 1]), COLOURS[j % len(COLOURS)], cv2, 2)
        if turn is not None:
            ang = np.linspace(0, 2 * np.pi, 73)
            cv2.drawMarker(pan, tuple(np.round(to1(*turn["axis"])).astype(int)), (235, 235, 235), cv2.MARKER_CROSS, 8, 1)
            for j, r_ in enumerate(turn["rides"]):
                _poly(pan, to1(turn["axis"][0] + r_["radius"] * np.cos(ang), turn["axis"][1] + r_["radius"] * np.sin(ang)),
                      COLOURS[j % len(COLOURS)], cv2, 1)
            for j, s_ in enumerate(turn["slides"]):
                fit_ = np.array(s_["fitted"])
                _poly(pan, to1(fit_[:, 0], fit_[:, 1]), COLOURS[(len(turn["rides"]) + j) % len(COLOURS)], cv2, 2)
        if ok[i]:
            cv2.circle(pan, tuple(np.round(to1(xy[i, 0], xy[i, 1])).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        to2 = _axes(pan, (56, 290, PANEL_W - 16, 290 + 150), (t[0], t[-1]), (vmin, vmax), plot_title, cv2)
        if swing is not None:
            if i == 0:
                base2 = pan[280:446].copy()
                for a in np.nonzero(np.isfinite(swing["obs"]))[0]:
                    cv2.circle(base2, tuple(np.round(to2(t[a], swing["obs"][a])).astype(int) - [0, 280]), 1,
                               (170, 170, 170), -1, cv2.LINE_AA)
                _poly(base2, to2(t[swing["k"]], swing["model"][swing["k"]]) - [0, 280], COLOURS[0], cv2, 1)
            pan[280:446] = base2
        elif turn is not None:
            for a in np.nonzero(np.isfinite(v))[0]:
                cv2.circle(pan, tuple(np.round(to2(t[a], v[a])).astype(int)), 1, (170, 170, 170), -1, cv2.LINE_AA)
            for j, r_ in enumerate(turn["rides"]):
                _poly(pan, to2(np.array([t[r_["first_frame"]], t[r_["last_frame"]]]), np.full(2, r_["radius"])),
                      COLOURS[j % len(COLOURS)], cv2, 2)
            for j, s_ in enumerate(turn["slides"]):
                _poly(pan, to2(t[np.array(s_["frames_used"])], np.linalg.norm(np.array(s_["fitted"]) - turn["axis"], axis=1)),
                      COLOURS[(len(turn["rides"]) + j) % len(COLOURS)], cv2, 2)
        elif carry is not None:
            if i == 0:
                base2 = pan[280:446].copy()
                tk_, g_ = carry["tk"], np.minimum(np.linalg.norm(carry["acc"], axis=1) / 9.81, vmax)
                for a in range(len(tk_)):
                    cv2.circle(base2, tuple(np.round(to2(tk_[a], g_[a])).astype(int) - [0, 280]), 1, (170, 170, 170),
                               -1, cv2.LINE_AA)
                _poly(base2, to2(tk_, np.minimum(4 * carry["noise"] / 9.81, vmax)) - [0, 280], (90, 90, 90), cv2, 1)
                read = np.isfinite(carry["sure"]) & (carry["sure"] > 0)
                for a in np.nonzero(read)[0]:
                    cv2.circle(base2, tuple(np.round(to2(tk_[a], g_[a])).astype(int) - [0, 280]), 2, COLOURS[0],
                               -1, cv2.LINE_AA)
                if "carried" in phys:
                    c_ = phys["carried"]
                    cv2.drawMarker(base2, tuple(np.round(to2(c_["peak_time"], c_["peak_acceleration"] / 9.81)
                                                         ).astype(int) - [0, 280]), (235, 235, 235),
                                   cv2.MARKER_TRIANGLE_DOWN, 10, 1)
            pan[280:446] = base2
        elif drop:
            for a in np.nonzero(np.isfinite(v))[0]:
                cv2.circle(pan, tuple(np.round(to2(t[a], v[a])).astype(int)), 2, (170, 170, 170), -1, cv2.LINE_AA)
            _poly(pan, to2(t[kf], Pf[:, 2]), COLOURS[0], cv2, 2)
        else:
            _poly(pan, to2(t, np.nan_to_num(v, nan=0.0)), (150, 150, 150), cv2)
        for j, p in enumerate(phases):
            if motion == "incline":
                ts = np.array([p["t0"], t[p["last_frame"]]])
                _poly(pan, to2(ts, p["release_speed"] + p["acceleration"] * (ts - p["t0"])), COLOURS[j % len(COLOURS)], cv2, 2)
                continue
            ts = np.array([p["t0"], min(p["t0"] + p["stop_time"], t[p["last_frame"]])])
            _poly(pan, to2(ts, np.maximum(p["release_speed"] - p["coefficient"] * 9.81 * (ts - p["t0"]), 0)),
                  COLOURS[j % len(COLOURS)], cv2, 2)
        cx = int(to2(t[i], 0)[0])
        cv2.line(pan, (cx, 290), (cx, 440), (255, 255, 255), 1)
        y = 480
        for j, b in enumerate(phys.get("bounces", []) if drop else []):
            txt = f"bounce {j + 1} at {b['t']:.2f} s: in at {b['speed_in']:.2f} m/s -> restitution {b['restitution']:.2f}"
            cv2.putText(pan, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOURS[(j + 1) % len(COLOURS)], 1, cv2.LINE_AA)
            y += 17
        for j, r_ in enumerate(turn["rides"] if turn else []):
            txt = (f"ride {j + 1}: {r_['radius']:.3f} m from the axis at {abs(r_['omega']):.2f} rad/s needs "
                   f"{r_['needed']:.3f}")
            cv2.putText(pan, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOURS[j % len(COLOURS)], 1, cv2.LINE_AA)
            y += 17
        for j, s_ in enumerate(turn["slides"] if turn else []):
            txt = f"slide {j + 1}: |a| = {s_['acceleration']:.2f} m/s^2 over {s_['seconds']:.2f} s -> {s_['friction']:.3f}"
            cv2.putText(pan, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        COLOURS[(len(turn["rides"]) + j) % len(COLOURS)], 1, cv2.LINE_AA)
            y += 17
        if carry is not None and "carried" in phys:
            c_ = phys["carried"]
            for txt in (f"carried {100 * c_['travel']:.0f} cm, at up to {100 * c_['top_speed']:.1f} cm/s",
                        f"largest read at {c_['peak_time']:.2f} s: {c_['peak_acceleration'] / 9.81:.3f} g, of which "
                        f"jitter {c_['acceleration_noise'] / 9.81:.3f} g",
                        f"less 4 x jitter, less 2 x the scale's uncertainty: friction >= "
                        f"{phys['parameters']['support.static_friction']['lower_bound']:.3f}"):
                cv2.putText(pan, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOURS[0], 1, cv2.LINE_AA)
                y += 17
        for j, p in enumerate(phases):
            if motion == "incline":
                txt = (f"slide {j + 1}: tilt {p['tilt_deg']:.1f} deg, {p['acceleration']:.2f} m/s^2 over "
                       f"{p['distance']:.2f} m -> {p['coefficient']:.3f}{'' if p['ok'] else '  (rejected)'}")
            else:
                txt = (f"phase {j + 1}: {p['release_speed']:.2f} m/s, {p['distance']:.2f} m -> {p['coefficient']:.3f}"
                       f"{'' if p['ok'] else '  (rejected)'}{'' if p['stopped'] else '  (unfinished)'}")
            cv2.putText(pan, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOURS[j % len(COLOURS)], 1, cv2.LINE_AA)
            y += 17
        y = max(y + 8, OH - 17 * len(lines) - 8)
        for ln in lines:
            cv2.putText(pan, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
            y += 17
        vw.write(np.ascontiguousarray(canvas[:, :, ::-1]))
    vw.release()
    out = path / "summary.mp4"
    try:  # H.264 plays everywhere; OpenCV's mp4v does not
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
    except (OSError, subprocess.CalledProcessError):
        tmp.replace(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    args = ap.parse_args()
    for r in args.recording:
        if not (Path(r) / "physics.json").exists():  # an earlier stage failed on it: the others go on
            print(f"{r}: no physics.json, skipped", flush=True)
            continue
        print(render(r))


if __name__ == "__main__":
    main()
