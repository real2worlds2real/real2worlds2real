"""A long body stood on a support and drawn along it by its other end: its foot follows in jumps, and what the
jumps measure (meta.json "motion": "stickslip"; perception as for a slide: the body, the support's plane).

What holds the foot back is friction; what draws it on is whatever gives between the foot and the hand that
holds the body (the body's own bending, the grip), which is wound up while the foot sticks and let go when it
slips. The video shows neither force. It shows how far the upper end leads the foot, the lean, which is that
spring's stretch: it climbs while the foot sticks, to the stretch static friction holds (x_s), and falls in a
slip to x_e. A slip under a steady kinetic friction is half a swing about the stretch that friction holds, so that
one is midway, (x_s + x_e) / 2, and

    kinetic friction / static friction = (x_s + x_e) / (2 x_s)

with no stiffness, no load and no scale in it: the lean is read as an angle, and only its zero matters, which
is the lean before the drawing begins. The foot's load is taken to stay what it is and the link to be linear. A
foot that springs back past where it was drawn from (x_e < 0) is not sliding against a steady friction: the body
rings, and nothing is reported.

The lean is the direction of a line through the body's mask (many pixels: a tenth of a degree), set in the
upright plane through the foot's path over the support.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
from pathlib import Path

import numpy as np

from .scene import camera_pose, ray_plane, read_frames, unusable

LEAST_SLIPS = 3
NEAR = 0.4  # the part of the way, from where the body stood at rest, whose slips give the ratio
LEAST_NEAR = 6  # slips there, or the whole way's are taken


def axis_of(mask, image=None, towards=None):
    """The line of a long body: (foot end (2,), unit direction from the foot along the body, length), px; None
    for a mask that is not long. The foot is the end towards `towards` (a direction in the image, e.g.
    gravity's; down the image if None). The direction is the mask's long axis, and with the image (grey, float)
    that of the image's own edges along the body, within its mask: the edges weigh by their contrast squared, so
    the stretch of the body that stands out from what is behind it says where it points, and a mask that is
    ragged where the body is dark on dark, or holds what grips its other end, does not."""
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    if w[1] < 16 * max(w[0], 0.25):
        return None
    e = V[:, 1]
    towards = np.array([0.0, 1.0]) if towards is None else np.asarray(towards, float)
    if e @ towards > 0:
        e = -e  # from the foot on
    a = (P - c) @ e
    lo, hi = np.percentile(a, [0.5, 99.5])
    if image is not None:
        x0, x1, y0, y1 = max(xs.min() - 8, 0), xs.max() + 9, max(ys.min() - 8, 0), ys.max() + 9
        g = cv2.GaussianBlur(image[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.0)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        near = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        gx, gy = gx[near].astype(float), gy[near].astype(float)
        m2 = gx ** 2 + gy ** 2
        for gate in (0.42, 0.17):  # edges within 25 deg of the body's line, then within 10
            use = np.abs(gx * e[0] + gy * e[1]) < gate * np.sqrt(m2)
            if m2[use].sum() <= 0:
                break
            # the edges' gradients as directions without a sign: the mean of their doubled angles
            phi = 0.5 * np.arctan2(np.sum(2 * gx[use] * gy[use]), np.sum(gx[use] ** 2 - gy[use] ** 2))
            d = np.array([-np.sin(phi), np.cos(phi)])
            e = d if d @ e > 0 else -d
    foot = c + lo * V[:, 1] * np.sign(V[:, 1] @ e) + ((P[a <= lo + 2] - c) @ np.array([-V[1, 1], V[0, 1]])).mean() \
        * np.array([-V[1, 1], V[0, 1]])
    return foot, e, float(hi - lo)


def sawtooth(t, x, least=None):
    """Slips of a lead x(t) that climbs and falls: [(i0, i1, x_s, x_e)] with the lead a line through the climb
    before gives at the slip's first frame and the line through the climb after at its last, and the lead's
    scatter about those lines. A fall counts from five times that scatter (or `least`), and is over within a tenth
    of a second (three frames at the least: a slip is the foot's own half swing, tens of milliseconds); the climb
    before the first slip and after the last is read over a second."""
    x = np.asarray(x, float)
    dt = float(np.median(np.diff(t)))
    quick, long_ = max(3, int(round(0.1 / dt))), max(8, int(round(1.0 / dt)))
    ok = np.isfinite(x)
    xi = np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok])
    noise = 1.4826 * np.median(np.abs(np.diff(xi, 2))) / np.sqrt(6) + 1e-9
    thr = max(5 * noise, least or 0.0)
    fall = np.zeros(len(x), bool)  # frames in a fall: the lead two frames on is lower by the threshold
    for w in range(1, quick + 1):
        d = xi[w:] - xi[:-w]
        for i in np.nonzero(d < -thr)[0]:
            fall[i:i + w + 1] = True
    slips, i = [], 0
    while i < len(x):
        if fall[i]:
            j = i
            while j + 1 < len(x) and fall[j + 1]:
                j += 1
            a = i + int(np.argmax(xi[i:j + 1]))
            b = a + int(np.argmin(xi[a:j + 1]))
            if b > a:
                slips.append([a, b])
            i = j
        i += 1
    out, res = [], []
    for k, (a, b) in enumerate(slips):
        before = np.arange(slips[k - 1][1] if k else max(a - long_, 0), a + 1)
        after = np.arange(b, (slips[k + 1][0] if k + 1 < len(slips) else min(b + long_, len(x) - 1)) + 1)
        ends = []
        for seg, at in ((before, a), (after, b)):
            seg = seg[ok[seg]]
            if len(seg) >= 4 and np.ptp(t[seg]) > 0:
                kf = np.polyfit(t[seg] - t[at], x[seg], 1)
                res.extend(x[seg] - np.polyval(kf, t[seg] - t[at]))
                ends.append(float(kf[1]))
            else:
                ends.append(float(xi[at]))
        out.append((int(a), int(b), ends[0], ends[1]))
    return out, float(np.std(res)) if len(res) > 4 else float(noise)


def ratio_of(t, lead, zero):
    """The slips of a lead (any unit) whose value before the drawing begins is `zero`: the dict that goes into
    physics.json, with `ratio` None and a `reason` when the slips are not a slide against a steady friction."""
    slips, scatter = sawtooth(t, lead)
    xs = np.array([s[2] for s in slips]) - zero
    xe = np.array([s[3] for s in slips]) - zero
    out = {"slips": len(slips), "scatter": scatter, "ratio": None,
           "slip_frames": [[s[0], s[1]] for s in slips]}
    if len(slips) < LEAST_SLIPS:
        return dict(out, reason=f"{len(slips)} slips seen: the foot is drawn along without sticking, or stays")
    steady = xs > 0.6 * np.median(xs)  # not the first, before the lead has built up
    xs, xe = xs[steady], xe[steady]
    out.update(lead_at_slip=float(np.median(xs)), lead_after=float(np.median(xe)), fall=float(np.median(xs - xe)),
               period=float(np.median(np.diff([t[s[0]] for s in slips]))) if len(slips) > 1 else None)
    if np.median(xs) < 8 * scatter:
        return dict(out, reason="the lead at a slip is within the scatter of its own reading")
    if np.median(xe) <= 0 or np.mean(xe < 0) > 0.25:
        return dict(out, reason="the foot springs back past where it is drawn from: the body rings, and no steady "
                                "friction is sliding")
    r = (xs + xe) / (2 * xs)
    return dict(out, ratio=float(np.median(r)), ratio_std=float(np.std(r)), ratio_n=int(len(r)))


def measure(per, frames=None):
    """physics.json of a recording whose motion is "stickslip"; `frames` grey (F, H, W) for the body's line."""
    t, obj, K = per["t"], per["object"], per["K"]
    out = {"motion": "stickslip", "measured": False, "fps": float(1.0 / np.median(np.diff(t))), "parameters": {}}
    n, h = np.atleast_2d(per["plane_n"]), np.atleast_1d(per["plane_h"])
    n, h = np.median(n, 0), float(np.median(h))
    T = camera_pose(n, h)
    # gravity in the image, at the body: the way its foot is
    c = np.array(np.nonzero(obj.any(0)))[::-1].mean(1)
    down = (K @ (np.array([(c[0] - K[0, 2]) / K[0, 0], (c[1] - K[1, 2]) / K[1, 1], 1.0]) - 0.05 * n))
    down = down[:2] / down[2] - c
    ax = [axis_of(m, None if frames is None else frames[i], down) for i, m in enumerate(obj)]
    ok = np.array([a is not None for a in ax])
    if ok.sum() < 30:
        return dict(out, reason="no long body is seen for long enough")
    foot = np.full((len(t), 2), np.nan)
    top = np.full((len(t), 2), np.nan)
    foot[ok] = [a[0] for a in ax if a is not None]
    top[ok] = [a[0] + 0.8 * a[2] * a[1] for a in ax if a is not None]  # a place up the body's line
    P = ray_plane(K, T, foot)[:, :2]  # the foot on the support
    ok &= np.all(np.isfinite(P), 1)
    c0 = np.median(P[ok], 0)
    V = np.linalg.eigh(np.cov((P[ok] - c0).T))[1]
    travel = float(np.ptp((P[ok] - c0) @ V[:, 1]))
    d = V[:, 1] * np.sign((P[ok][-10:].mean(0) - P[ok][:10].mean(0)) @ V[:, 1])
    m = np.array([-d[1], d[0], 0.0])  # the normal of the upright plane the body is drawn in
    o = T[:3, 3]

    def in_plane(uv):  # where a pixel's ray meets that plane: (along the path, up), m
        r = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], -1)
        r = r @ T[:3, :3].T
        s = ((np.append(c0, 0.0) - o) @ m) / (r @ m)
        X = o + s[:, None] * r
        return np.stack([(X[:, :2] - c0) @ d, X[:, 2]], 1), np.abs(r @ m) / np.linalg.norm(r, axis=1)

    F, facing = in_plane(foot)
    U, _ = in_plane(top)
    if np.nanmedian(facing) < 0.25:
        return dict(out, reason="the camera looks along the way the body is drawn: its lean is not seen")
    lean = (U[:, 0] - F[:, 0]) / (U[:, 1] - F[:, 1])  # the upper end's lead over the foot, per height
    height = float(np.nanmedian(U[:, 1] - F[:, 1]))
    if not travel > 0.2 * height:  # 3 cm of a body 13 cm up
        return dict(out, reason="the foot goes nowhere")
    # before the drawing begins: the first frames, as long as the foot and the lean stay where they were
    first = np.nonzero(ok)[0][:5]
    l0, s0 = np.median(lean[first]), np.median(F[first, 0])
    _, scatter = sawtooth(t, lean)
    moved = ok & ((np.abs(lean - l0) > 6 * scatter) | (np.abs(F[:, 0] - s0) > 0.03 * height))  # 4 mm of 13 cm
    start = int(np.argmax(moved)) if moved.any() else len(t)
    rest = ok & (np.arange(len(t)) < start)
    stick = {"travel": travel, "height": height, "rest_frames": int(rest.sum()),
             "lean_at_rest_deg": float(np.degrees(np.arctan(np.median(lean[rest])))) if rest.any() else None,
             "speed": travel / float(t[ok][-1] - t[max(start, np.nonzero(ok)[0][0])] + 1e-9),
             "foot_off_support": float(np.nanmedian(np.abs(F[:, 1])))}
    out["stickslip"] = stick
    if rest.sum() < 3:
        return dict(out, reason="the body is drawn from the first frame on: its lean with nothing drawing is not seen")
    lead, zero = np.where(ok, lean, np.nan), float(np.median(lean[rest]))
    whole = ratio_of(t, lead, zero)
    # the lean's zero was read where the body stood at rest, and what the camera's pose is off by grows from
    # there on: the ratio is that of the slips over the first of the way, if they are enough
    gone = np.nonzero(ok & (F[:, 0] - s0 > NEAR * travel))[0]
    cut = int(gone[0]) if len(gone) else len(t)
    r = ratio_of(t[:cut], lead[:cut], zero)
    stick["ratio_over"] = NEAR if r["ratio"] is not None and r["ratio_n"] >= LEAST_NEAR else 1.0
    if stick["ratio_over"] == 1.0:
        r = whole
    stick.update({k: whole[k] for k in ("slips", "slip_frames", "period") if k in whole})
    if whole["ratio"] is not None:
        stick["ratio_whole_way"] = whole["ratio"]
    for k in ("lead_at_slip", "lead_after", "fall", "scatter"):  # as lengths at the upper end, m
        if k in r:
            stick[k] = r[k] * height
    stick["lean"] = [float(v) if g else None for v, g in zip(lean, ok)]
    stick["foot_along"] = [float(v) if g else None for v, g in zip(F[:, 0], ok)]
    if r["ratio"] is None:
        return dict(out, reason=r["reason"])
    stick.update(ratio=r["ratio"], ratio_std=r["ratio_std"], ratio_n=r["ratio_n"])
    # the slips of one trial share what they are read against, so their mean is no better than the ratio holds
    # along the way: beside their own scatter, what it differs by over the whole way
    drift = abs(r["ratio"] - whole["ratio"]) if whole["ratio"] is not None else 0.0
    out["parameters"]["support.kinetic_over_static_friction"] = {
        "value": r["ratio"], "stderr": float(np.hypot(r["ratio_std"] / np.sqrt(r["ratio_n"]), drift)),
        "source": "measured",
        "how": "the lean of the body when its foot lets go and when it holds again, over the slips of the first "
               f"{stick['ratio_over']:.0%} of the way: (x_s + x_e) / (2 x_s)"}
    out["measured"] = True
    return out


def fit(path):
    path = Path(path)
    per = np.load(path / "perception.npz")
    why = unusable(per)
    if why is not None:
        return {"motion": "stickslip", "measured": False, "parameters": {}, "reason": why}
    frames, _fps = read_frames(path)  # as perception left them: on the reference frame's camera, like its masks
    grey = [f.astype(np.float32).mean(2) for f in frames]
    return measure(per, grey)


def describe(path, o):
    s = o.get("stickslip", {})
    if not o.get("measured"):
        return f"{path}: not measured: {o['reason']}" + (f" ({s['slips']} slips)" if "slips" in s else "")
    return (f"{path}: drawn {s['travel'] * 100:.0f} cm at {s['speed'] * 1000:.0f} mm/s, {s['slips']} slips every "
            f"{s['period']:.2f} s: the upper end leads by {s['lead_at_slip'] * 1000:.1f} mm when the foot lets go and "
            f"{s['lead_after'] * 1000:.1f} mm when it holds again: kinetic / static friction {s['ratio']:.2f} "
            f"(sd {s['ratio_std']:.2f} over {s['ratio_n']})")


def summary(path):
    """summary.mp4: the video with the body's line as it was read (yellow) and its foot (red); beside it the lean
    along time with every slip marked, and what the slips say."""
    import subprocess

    import cv2

    path = Path(path)
    phys = read_document(path / "physics.json")
    if "lean" not in phys.get("stickslip", {}):
        return None
    st = phys["stickslip"]
    per = np.load(path / "perception.npz")
    obj, t = per["object"], per["t"]
    frames, fps = read_frames(path)
    H, W = frames.shape[1:3]
    scale = 720.0 / H if H > 720 else 1.0
    h, w, PW = int(H * scale), int(W * scale), 460
    lean = np.degrees(np.arctan(np.array(st["lean"], float)))
    lean -= st["lean_at_rest_deg"] or 0.0
    box = (w + 20, 40, w + PW - 20, 260)
    lo, hi = min(np.nanmin(lean), 0.0), max(np.nanmax(lean), 0.0)
    lo, hi = lo - 0.05 * (hi - lo + 1e-6), hi + 0.05 * (hi - lo + 1e-6)
    px = lambda tt, y: (int(box[0] + (tt - t[0]) / max(t[-1] - t[0], 1e-6) * (box[2] - box[0])),
                        int(box[3] - (y - lo) / (hi - lo) * (box[3] - box[1])))
    if phys.get("measured"):
        text = [f"{st['slips']} slips, one every {st['period']:.2f} s",
                f"the lean when the foot lets go: {np.degrees(np.arctan(st['lead_at_slip'] / st['height'])):.1f} deg",
                f"   when it holds again: {np.degrees(np.arctan(st['lead_after'] / st['height'])):.1f} deg",
                f"kinetic / static friction {st['ratio']:.2f}",
                f"   (sd {st['ratio_std']:.2f} over {st['ratio_n']} slips)",
                "the friction itself: not measured", "   (no force is in the picture)"]
    else:
        text = [f"{st.get('slips', 0)} slips", "not measured:"] + [
            "   " + phys["reason"][i:i + 46] for i in range(0, len(phys["reason"]), 46)]
    grey = [f.astype(np.float32).mean(2) for f in frames]
    tmp = path / "summary_raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + PW, h))
    pts = np.array([px(t[i], lean[i]) for i in range(len(t)) if np.isfinite(lean[i])], np.int32)
    for k in range(len(frames)):
        im = frames[k][:, :, ::-1].copy()
        a = axis_of(obj[k], grey[k])
        if a is not None:
            f, e, length = a
            cv2.line(im, tuple(int(round(v)) for v in f), tuple(int(round(v)) for v in f + length * e), (0, 230, 255),
                     1, cv2.LINE_AA)
            cv2.circle(im, tuple(int(round(v)) for v in f), 3, (0, 0, 255), -1)
        img = np.zeros((h, w + PW, 3), np.uint8)
        img[:, :w] = cv2.resize(im, (w, h)) if scale != 1.0 else im
        cv2.rectangle(img, box[:2], box[2:], (70, 70, 70), 1)
        cv2.line(img, px(t[0], 0.0), px(t[-1], 0.0), (70, 70, 70), 1)
        for a0, _b in st.get("slip_frames", []):
            cv2.line(img, (px(t[a0], 0)[0], box[3] - 8), (px(t[a0], 0)[0], box[3]), (0, 0, 255), 1)
        cv2.polylines(img, [pts], False, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "lean beyond its lean at rest, deg (red: slips)", (box[0] + 6, box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(img, (px(t[k], 0)[0], box[1]), (px(t[k], 0)[0], box[3]), (255, 255, 255), 1)
        for j, line in enumerate(text):
            cv2.putText(img, line, (w + 16, box[3] + 34 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1,
                        cv2.LINE_AA)
        cv2.putText(img, "yellow: the body's line, read from its edges   red: its foot", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out = path / "summary.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
    except (OSError, subprocess.CalledProcessError):
        tmp.rename(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--summary", action="store_true", help="summary.mp4 of recordings already measured")
    args = ap.parse_args()
    for p in args.recording:
        if args.summary:
            print(summary(p), flush=True)
            continue
        if args.skip_done and (Path(p) / "physics.json").exists():
            continue
        if not (Path(p) / "perception.npz").exists():
            continue
        try:
            o = fit(p)
        except Exception as e:  # one recording's failure is its own
            o = {"motion": "stickslip", "measured": False, "reason": f"{type(e).__name__}: {e}"}
        write_document(Path(p) / "physics.json", o, indent=1)
        print(describe(p, o), flush=True)


if __name__ == "__main__":
    main()
