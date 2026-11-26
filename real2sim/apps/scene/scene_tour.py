"""A video of the whole scene: the twin (or a cousin: --tag, --variant) with the robot (--robot) standing where the
views stand it (the real robot's base when accepted, standpoints.view_bases), first from above and outside the room,
then around the task support from inside it.

    tools/run_isaac.sh apps/scene/scene_tour.py sessions/<name> [--seconds 60] [--outside 0.5] [--frames-only 12]

1. Outside: one turn round the content of the room (the support, the furniture, the objects, the robot; not the
   empty floor a long room keeps) from above, looking down at its centre, the ceiling hidden and the walls cut down
   to `--cut` m: a model of the room, every piece in view at once.
2. Inside: a sweep round the task support at head height, looking at it, as far out as the room lets: along each
   direction the camera keeps `--margin` from the walls and from anything taller than itself (a low desk passes
   under it) and clears the support, what stands on it and the robot; the radius is eroded over the neighbouring
   directions and smoothed, and the sweep takes the longest run of directions where the camera fits (a full turn
   when it fits all round), so it never noses into a corner and never cuts through a piece.
3. Standing still, turning round (`--pans` of the time): at the photo camera's place and beside the robot, a full
   turn each at eye height, so the room is seen in every direction, not only past the support's edges.

`--path capture_free` (the splat twin's video, the pipeline's deliver stage): nothing from outside and no mesh-only
frame -- the capture's own camera path (55 % of the time), then a free view: an arc in front of the support over
the directions the capture faced it from, centred on the photo camera's side, then in to the task's objects for a
close-up, a +-30 deg orbit there, and back out.

A fly-through's camera path (apps/scene/flythrough.py, views.json key frames) is drawn by hand per scene and showed the
background only in passing; this one is derived from the scene. Writes renders/tour/tour.mp4 (a cousin:
renders/tour/<tag>/tour.mp4); `--frames-only N` renders N stills along the path (renders/tour/stills/) instead.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default=None, help="robot spec (default: the one the session's placements are for)")
ap.add_argument('--no-robot', action='store_true', help='Render the task workspace without the optional robot CAD')
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--outside", type=float, default=0.3, help="the share of the video outside the room")
ap.add_argument("--pans", type=float, default=0.4, help="the share of the video standing still and turning round")
ap.add_argument("--fov", type=float, default=70.0, help="horizontal field of view, degrees")
ap.add_argument("--path", default="auto", choices=["auto", "capture", "sweep", "capture_free", "free"],
                help="inside: the capture's own camera path (auto: with the splat background) or the sweep round the support")
ap.add_argument("--eye", type=float, default=1.75, help="the inside camera's height, m")
ap.add_argument("--margin", type=float, default=0.8, help="the inside camera's distance from walls and tall pieces, m")
ap.add_argument("--cut", type=float, default=1.1, help="outside: the walls' height, m")
ap.add_argument("--frames-only", type=int, default=0)
ap.add_argument("--background", default="mesh", choices=["mesh", "splat", "code", "zone", "blender"],
                help="splat: the twin's Gaussian-splat background (recon.splat_export, sim.splat)")
ap.add_argument("--splat-cousin", default="", help="with --background splat: a splat cousin's background "
                "(recon.splat_cousin); renders/tour/<it>/splat/")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant", default=None, help="a cousin's json (default: the twin)")
ap.add_argument("--environment-variant", default="twin", help="Background-only Blender variant from environment.json")
args = ap.parse_args()
if args.background == "blender" and args.variant:
    ap.error("Use --environment-variant for background cousins; task-changing --variant is incompatible")

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 1280, 720
app = SimulationApp({"headless": True, "renderer": "PathTracing" if args.background == 'blender' else "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.render_robot import capture_spec  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import encode_video, grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import PATH as SPLAT_PATH, add_background  # noqa: E402
from calibrated_scene_reconstruction.sim.sweep import RobotSweep, clear_robot, summarize, support_collisions  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, look_at_quat, set_pose  # noqa: E402
from calibrated_scene_reconstruction.robots.viewpoints import coverage_score, is_photo_base, view_bases  # noqa: E402

configure_render()
s = Session(args.session)
tag = args.splat_cousin or ("" if args.tag == "twin" else args.tag)
out = s / "renders" / "tour" / tag / (args.background if args.background in ("splat", "code", "zone", "blender") else "")
frames_dir = out / ("stills" if args.frames_only else "frames")
frames_dir.mkdir(parents=True, exist_ok=True)
for f in frames_dir.glob("*"):
    f.unlink()
m = s.manifest
sg = s.scene_graph

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = json.loads(Path(args.variant).read_text()) if args.variant else {}
ts = build_scene(stage, prepare_scene(s.dir), photo_goal(s.dir, prepare_variant(s.dir, variant)))
name = args.robot or next((p.stem for p in sorted((s / "placements").glob("*.json")) if p.stem != "placements"),
                          "vega_1p_sharpa")
spec = capture_spec(s, load_spec(name))
pl = s.read_json("placements/placements.json" if name == "vega_1p_sharpa" else f"placements/{name}.json")
photo_base = (sg.get("photo_robot") or {}).get("base")
cands, warn = view_bases(pl["candidates"], photo_base)
if warn:
    print(f"[warn] {warn}")
bx, by, byaw = (cands[0] if is_photo_base(cands[0], photo_base) else max(cands, key=coverage_score))["base"]
sup = sg["furniture"][sg["task"]["support"]]["box"]
zt = float(sup[1][2])
if not args.no_robot:
    add_robot(stage, spec, (bx, by), np.radians(byaw), support_top=zt)
    support_collisions(stage, False)
# splat: the background inside; outside, above the room, the mesh model (no frame looked down on the room)
if args.background == "code":  # the room written as code (sim.code_scene)
    from calibrated_scene_reconstruction.sim.code_stage import add_code_background  # noqa: E402

    add_code_background(stage, prepare_background(s.dir))
    hidden = []
else:
    hidden = [stage.GetPrimAtPath(p) for p in add_background(stage, prepare_background(s.dir), args.background,
            args.environment_variant if args.background == "blender" else args.splat_cousin)]
hidden = [p for p in hidden if p.GetName() != "ceiling"]  # the ceiling has its own rule below (shown inside only)
splat_bg = stage.GetPrimAtPath(SPLAT_PATH) if args.background == "splat" else None

# --- the room: its walls (inward normals), the ceiling, the content's extent, the tall pieces
walls, ceiling = {}, []
for pname, part in m["parts"].items():
    if part.get("kind") != "structure" or "normal" not in part:
        continue
    n = np.asarray(part["normal"], float)
    prim = stage.GetPrimAtPath(f"/World/Room/{pname}")
    if not prim.IsValid():
        continue
    if n[2] < -0.7:
        ceiling.append(prim)
    elif abs(n[2]) < 0.3:
        V = np.load(s / "scene" / part["npz"])["V"]
        t = np.array([-n[1], n[0], 0.0])
        a, b = (V @ t).min(), (V @ t).max()
        p = V.mean(0)
        walls[pname] = (prim, n, p, p + (a - p @ t) * t, p + (b - p @ t) * t)  # inward normal, centre, 2D ends
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
skip = set(walls) | {c.GetName() for c in ceiling} | {"floor"}


def boxes(root):
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return []
    out_ = []
    for c in prim.GetChildren():
        if c.GetName() in skip or c.GetName().startswith("collider_") or not c.IsA(UsdGeom.Imageable):
            continue
        r = cache.ComputeWorldBound(c).ComputeAlignedRange()
        if not r.IsEmpty():
            out_.append((c.GetName(), np.asarray(r.GetMin()), np.asarray(r.GetMax())))
    return out_


pieces = [b for root in ("/World/Room", "/World/Static", "/World/Support", "/World/Synthetic", "/World/Objects")
          for b in boxes(root)]
lo = np.min([b[1] for b in pieces] + [[bx - 0.4, by - 0.4, 0]], 0)
hi = np.max([b[2] for b in pieces] + [[bx + 0.4, by + 0.4, 1.6]], 0)
centre = (lo + hi) / 2
sf = m["surfaces"]
zc = float(sf["ceiling_z"])
print(f"content {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}; walls {sorted(walls)}; ceiling "
      f"{[c.GetName() for c in ceiling]}", flush=True)

# --- 1. outside: a turn round the content from above
half = float(np.linalg.norm(hi[:2] - lo[:2])) / 2
fx = W / 2 / np.tan(np.radians(args.fov) / 2)
r_out = half / np.tan(np.radians(args.fov) / 2 * 0.8) * 0.75 + 1.0  # the content's diagonal about fills the view
z_out = zc + max(1.5, 0.55 * r_out)
look_out = np.array([centre[0], centre[1], 0.5])
a0 = np.arctan2(*(np.array([bx, by]) - centre[:2])[::-1]) + np.pi  # start opposite the robot: its front first

# --- 2. inside: a turn round the support, as far out as the walls and the tall pieces let
c_in = np.array([(sup[0][0] + sup[1][0]) / 2, (sup[0][1] + sup[1][1]) / 2])
look_in = np.array([c_in[0], c_in[1], zt + 0.15])
s_lo, s_hi = np.asarray(sup[0][:2]) - 0.05, np.asarray(sup[1][:2]) + 0.05
# the support with what stands on it or round it (the frame, the objects) and the robot stay inside the turn; the
# walls and every other piece taller than the camera outside it
on_sup = [np.all((lo_[:2] + hi_[:2]) / 2 >= s_lo) and np.all((lo_[:2] + hi_[:2]) / 2 <= s_hi) for _, lo_, hi_ in pieces]
inner = [(lo_[:2], hi_[:2]) for (_, lo_, hi_), o in zip(pieces, on_sup) if o]
inner.append((np.array([bx - 0.45, by - 0.45]), np.array([bx + 0.45, by + 0.45])))
outer = [(lo_[:2], hi_[:2]) for (_, lo_, hi_), o in zip(pieces, on_sup) if not o and hi_[2] > args.eye - 0.25]


def slab(d, blo, bhi):
    """(entry, exit) of the ray from the support's centre along d through a box's footprint, or None."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (blo - c_in) / d, (bhi - c_in) / d
    tn, tf = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
    return (max(tn, 0.0), tf) if tf >= max(tn, 0) else None


def wall_hit(o, d):
    """The distance along d from o to the nearest wall segment (6 m when none)."""
    best = 6.0
    for _, n, p, e0, e1 in walls.values():
        u = e1[:2] - e0[:2]
        A = np.array([d, -u]).T
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        t, k = np.linalg.solve(A, e0[:2] - o)
        if t > 0 and -0.05 <= k <= 1.05:
            best = min(best, t)
    return best


def reach(theta):
    """(the least radius clear of the inner pieces, the most radius short of the walls and the outer pieces)."""
    d = np.array([np.cos(theta), np.sin(theta)])
    near = max([h[1] for b in inner if (h := slab(d, *b))] + [0.0])
    far = wall_hit(c_in, d)
    for b in outer:
        if (h := slab(d, *b)) and h[0] > near:
            far = min(far, h[0])
    return near, far


def circ(v, k, how):
    return np.array([how(v[[(i + j) % len(v) for j in range(-k, k + 1)]]) for i in range(len(v))])


thetas = np.radians(np.arange(0, 360, 2))
near, far = np.array([reach(t) for t in thetas]).T
# +-12 degrees: the camera's sides keep clear too, then a +-20 degree mean smooths the path
r_lo = circ(circ(near + 0.5, 6, np.max), 10, np.mean)
r_hi = circ(circ(far - args.margin, 6, np.min), 10, np.mean)
ok = (r_hi >= np.maximum(r_lo, 1.2))
# the camera sweeps the longest run of directions where it fits between the inner pieces and the walls, not a forced full turn
if ok.all():
    run = (0, len(thetas))
else:
    start = int(np.argmin(ok))  # a blocked direction: runs counted from just after it
    best, cur = (0, 0), None
    for j in range(1, len(thetas) + 1):
        i = (start + j) % len(thetas)
        if ok[i]:
            cur = (cur[0], j) if cur else (j, j)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    run = (start + best[0], start + best[1] + 1)
r_in = np.clip(r_hi, np.maximum(r_lo, 1.2), 3.2)
arc = (thetas[run[0] % len(thetas)], 2 * np.pi * (run[1] - run[0]) / len(thetas))
if run[1] - run[0] < len(thetas):
    arc = (arc[0] + np.radians(15), max(arc[1] - np.radians(30), np.radians(60)))
print(f"inside: a sweep of {np.degrees(arc[1]):.0f} degrees from {np.degrees(arc[0]):.0f} "
      f"({int((~ok).sum())} of {len(thetas)} directions blocked)", flush=True)
print(f"inside: radius {r_in.min():.2f}-{r_in.max():.2f} m round {np.round(c_in, 2).tolist()}, eye {args.eye} m; "
      f"outside: radius {r_out:.2f} m at {z_out:.2f} m round {np.round(centre[:2], 2).tolist()}", flush=True)


def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)


# 3. standing still, turning round: the room from where it was seen (the sweep looks at the support, the background
# only passes its edges: the user saw almost nothing of what the robot faces). Two standpoints at eye height, a
# full turn each, 5 degrees down: the photo camera's place (the walk's start, the best-covered spot) and beside the
# robot, the turn beginning where it faces
T_photo = np.asarray(m["photo_camera"]["T_world_cam_cv"])
fwd_r = np.array([np.cos(np.radians(byaw)), np.sin(np.radians(byaw)), 0.0])
left_r = np.array([-fwd_r[1], fwd_r[0], 0.0])
stands = [(np.array([T_photo[0, 3], T_photo[1, 3], args.eye]), np.arctan2(T_photo[1, 2], T_photo[0, 2])),
          (np.array([bx, by, args.eye]) - 0.3 * fwd_r + 0.9 * left_r, np.radians(byaw))]
n = int(args.seconds * 30)
n_out = int(round(n * args.outside))
n_pan = int(round(n * args.pans))
n_in = n - n_out - n_pan
path = []
for f in range(n_out):  # a full turn, eased at both ends
    a = a0 + 2 * np.pi * ease(f / max(n_out - 1, 1))
    path.append(("out", np.array([centre[0] + r_out * np.cos(a), centre[1] + r_out * np.sin(a), z_out]), look_out))
def capture_cams():
    """The capture's cameras: the splat's (scene/splat/cameras.npz), else the frames posed in the twin's world
    (parsing/posed_frames/cameras.npz: background zone has no splat) -- names, centres, viewing axes."""
    f = s.dir / "scene" / "splat" / "cameras.npz"
    if f.exists() and "names" in np.load(f):
        z = np.load(f)
        return [str(x) for x in z["names"]], z["centre"], z["axis"]
    z = np.load(s.dir / "parsing" / "posed_frames" / "cameras.npz")
    return [str(x) for x in z["names"]], z["centre"], z["R_world_cam"][:, :, 2]  # OpenCV: +z looks forward


def capture_path(n_frames, sigma=10):
    """Smooth each captured camera trajectory and resample it at regular travel intervals."""
    from scipy.ndimage import gaussian_filter1d

    names, C_all, A_all = capture_cams()
    if args.background == "zone":  # the task zone alone: only the stretches of the walk that look at it
        v_ = np.r_[c_in, zt][None] - C_all
        look_ = np.sum(v_ * A_all, 1) / np.maximum(np.linalg.norm(v_, axis=1), 1e-6) > np.cos(np.radians(30))
        if look_.sum() >= 20:
            # each unbroken stretch its own segment (a cut, not a flight across what was left out)
            run, out_, prev = 0, [], None
            for nm, k in zip(names, look_):
                if not k:
                    prev = None
                    continue
                if prev is None:
                    run += 1
                prev = nm
                out_.append(f"{nm.split('_')[0]}-{run:03d}_{nm.split('_', 1)[1]}")
            names = out_
            C_all, A_all = C_all[look_], A_all[look_]
    vids = sorted({nm.split("_")[0] for nm in names})
    segs = []
    for v in vids:
        idx = [i for i, nm in enumerate(names) if nm.split("_")[0] == v]
        if len(idx) < 10:
            continue
        P = gaussian_filter1d(C_all[idx], sigma, axis=0, mode="nearest")
        A = gaussian_filter1d(A_all[idx], sigma, axis=0, mode="nearest")
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        step = np.linalg.norm(np.diff(P, axis=0), axis=1) + np.arccos(np.clip(np.sum(A[1:] * A[:-1], 1), -1, 1))
        segs.append((P, A, np.r_[0.0, np.cumsum(step)]))
    total = sum(sg[2][-1] for sg in segs)
    out = []
    for P, A, s_ in segs:
        k = max(2, int(round(n_frames * s_[-1] / max(total, 1e-9))))
        for u in np.linspace(0, s_[-1], k):
            j = int(np.clip(np.searchsorted(s_, u), 1, len(s_) - 1))
            w = (u - s_[j - 1]) / max(s_[j] - s_[j - 1], 1e-9)
            p, a = (1 - w) * P[j - 1] + w * P[j], (1 - w) * A[j - 1] + w * A[j]
            out.append(("in", p, p + 2.0 * a / np.linalg.norm(a)))
    return out[:n_frames]


def free_path(n_frames):
    """A free view that stays where the capture looked: an arc in front of the support over the directions the
    capture's cameras faced it from (their centres' azimuths round it, the largest gap left out, 150 deg at most), at
    their median distance and height, looking at the support; then in to the task's objects (its target and goal, or
    every dynamic object) for a close-up -- 0.45 m off, 0.28 m above them, from the arc's middle --, a +-30 deg orbit
    there, and back out."""
    _, C, A = capture_cams()
    v = np.r_[c_in, zt][None] - C
    facing = np.sum(v * A, 1) / np.maximum(np.linalg.norm(v, axis=1), 1e-6) > np.cos(np.radians(40))
    C = C[facing] if facing.sum() >= 20 else C
    d = C[:, :2] - c_in
    az = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    gaps = np.diff(np.r_[az, az[0] + 2 * np.pi])
    g = int(np.argmax(gaps))  # the arc the cameras covered: from after the largest gap round to before it
    a_lo, cover = az[(g + 1) % len(az)], 2 * np.pi - gaps[g]
    # centred on the photo camera's side (the robot faces it: nothing of the robot between the camera and the
    # objects), kept inside the covered arc
    a_ph = np.arctan2(T_photo[1, 3] - c_in[1], T_photo[0, 3] - c_in[0])
    rel = float(np.clip((a_ph - a_lo) % (2 * np.pi), 0.0, cover))
    span = min(cover * 0.8, np.radians(150))
    rel = float(np.clip(rel, span / 2, cover - span / 2)) if cover > span else cover / 2
    mid = a_lo + rel
    r_c = float(np.clip(np.median(np.linalg.norm(d, axis=1)), 1.3, 2.5))
    h_c = float(np.clip(np.median(C[:, 2]), zt + 0.3, zt + 0.9))
    names_ = [sg["task"].get("target"), sg["task"].get("goal")]
    prims = [ts.objects[nm_] for nm_ in names_ if nm_ in ts.objects] or list(ts.objects.values())
    bb = [cache.ComputeWorldBound(p_).ComputeAlignedRange() for p_ in prims]
    obj = np.mean([(np.asarray(r_.GetMin()) + np.asarray(r_.GetMax())) / 2 for r_ in bb], 0)
    look_arc = np.array([c_in[0], c_in[1], zt + 0.1])
    arc_centre = np.asarray(c_in)
    if args.no_robot and args.background == 'zone':
        # A standalone workspace tour must show the complete support/feet,
        # then approach the task. Fit the wide segment to actual modeled
        # bounds rather than the capture's usually close tabletop distance.
        support = stage.GetPrimAtPath('/World/Room/' + sg['task']['support'])
        complete = [*bb]
        if support:
            complete.append(cache.ComputeWorldBound(support).ComputeAlignedRange())
        low = np.min([np.asarray(b.GetMin()) for b in complete], axis=0)
        high = np.max([np.asarray(b.GetMax()) for b in complete], axis=0)
        look_arc = (low + high) / 2
        arc_centre = look_arc[:2]
        radius = float(np.linalg.norm(high - low) / 2)
        half_vertical_fov = np.arctan(H / (2 * fx))
        distance = 1.08 * radius / np.sin(half_vertical_fov)
        h_c = max(h_c, float(look_arc[2]) + .35)
        r_c = max(r_c, float(np.sqrt(max(distance ** 2 - (h_c - look_arc[2]) ** 2, 0.))))

    def on_arc(a):
        return np.array([arc_centre[0] + r_c * np.cos(a), arc_centre[1] + r_c * np.sin(a), h_c])

    toward = on_arc(mid)[:2] - obj[:2]
    toward /= np.linalg.norm(toward)
    a_obj = np.arctan2(toward[1], toward[0])

    def close(a):
        return np.array([obj[0] + 0.45 * np.cos(a), obj[1] + 0.45 * np.sin(a), obj[2] + 0.28])

    n_a, n_in_, n_orb = int(0.45 * n_frames), int(0.18 * n_frames), int(0.22 * n_frames)
    n_out_ = n_frames - n_a - n_in_ - n_orb
    out = []
    for f in range(n_a):  # across the covered arc and back to its middle
        u = f / max(n_a - 1, 1)
        a = mid - span / 2 + span * ease(min(1.0, 1.5 * u)) if u < 2 / 3 else mid + span / 2 - span / 2 * ease(3 * u - 2)
        out.append(("in", on_arc(a), look_arc))
    for f in range(n_in_):  # in to the objects
        w = ease(f / max(n_in_ - 1, 1))
        out.append(("in", (1 - w) * on_arc(mid) + w * close(a_obj), (1 - w) * look_arc + w * obj))
    for f in range(n_orb):  # round them, +-30 deg
        u = f / max(n_orb - 1, 1)
        out.append(("in", close(a_obj + np.radians(30) * np.sin(2 * np.pi * u)), obj))
    for f in range(n_out_):  # back out
        w = ease(f / max(n_out_ - 1, 1))
        out.append(("in", (1 - w) * close(a_obj) + w * on_arc(mid), (1 - w) * obj + w * look_arc))
    print(f"free view: an arc of {np.degrees(span):.0f} deg at {r_c:.2f} m, {h_c:.2f} m high round the support, a "
          f"close-up of {[nm_ for nm_ in names_ if nm_ in ts.objects] or 'the objects'} at {np.round(obj, 3).tolist()}",
          flush=True)
    return out


if args.path == 'free' or (args.background == 'zone' and args.path == 'auto'):
    # A task-only scene has no room to pan around. Keep a continuous free
    # camera aimed at the modeled support/objects rather than an empty dome.
    path = free_path(n)
    n_in, n_pan = 0, 0
if args.path == "capture_free":  # with the splat: the capture's path, then the free view; nothing from outside
    path = capture_path(int(n * 0.55)) + free_path(n - int(n * 0.55))
    n_in, n_pan = 0, 0
    print(f"capture_free: {len(path)} frames, the capture's path then the free view", flush=True)
use_capture = args.background == "splat" and args.path in ("auto", "capture") and \
    (s.dir / "scene" / "splat" / "cameras.npz").exists() and "names" in np.load(s.dir / "scene" / "splat" / "cameras.npz")
if use_capture:  # inside: the capture's own path, the pans' time too (standing turns leave its views)
    path += capture_path(n_in + n_pan)
    n_pan = 0
    print(f"inside: the capture's camera path, {len(path) - n_out} frames", flush=True)
for f in range(0 if use_capture else n_in):  # inside: the sweep, eased at both ends
    a = arc[0] + arc[1] * ease(f / max(n_in - 1, 1))
    r = float(np.interp(a % (2 * np.pi), np.r_[thetas, 2 * np.pi], np.r_[r_in, r_in[0]]))
    path.append(("in", np.array([c_in[0] + r * np.cos(a), c_in[1] + r * np.sin(a), args.eye]), look_in))
per = n_pan // len(stands)


def seen_arc(eye, n_dir=72, dists=(2.0, 3.5, 5.0), min_frames=15):
    """Find the widest arc seen in enough captured frames for a camera tour."""
    f = s.dir / "scene" / "splat" / "cameras.npz"
    if args.background != "splat" or not f.exists():
        return None
    z = np.load(f)
    C, A, HF = z["centre"], z["axis"], z["half_fov"]
    a = np.arange(n_dir) * 2 * np.pi / n_dir
    ok = np.zeros(n_dir, bool)
    for k in range(n_dir):
        n_seen = 0
        for d in dists:
            p = eye + d * np.array([np.cos(a[k]), np.sin(a[k]), -0.05])
            v = p - C
            dist = np.linalg.norm(v, axis=1)
            cosang = np.sum(v * A, 1) / np.maximum(dist, 1e-6)
            n_seen = max(n_seen, int(np.sum((cosang > np.cos(HF.min(1))) & (dist > 0.3))))
        ok[k] = n_seen >= min_frames
    if ok.all():
        return None
    if not ok.any():
        return (0.0, 0.0)
    run, best, start = 0, (0, 0), 0
    for k in range(2 * n_dir):  # the longest circular run of seen directions
        if ok[k % n_dir]:
            run += 1
            if run > best[0]:
                best = (run, start)
        else:
            run, start = 0, k + 1
    span = min(best[0], n_dir) * 2 * np.pi / n_dir
    return (a[best[1] % n_dir], span)


for eye, a_start in stands:
    arc_seen = seen_arc(eye)
    if arc_seen is not None:
        print(f"pan at ({eye[0]:.2f}, {eye[1]:.2f}): the capture saw {np.degrees(arc_seen[1]):.0f} deg of it from "
              f"{np.degrees(arc_seen[0]):.0f}", flush=True)
    for f in range(per):
        if arc_seen is None:
            a = a_start + 2 * np.pi * ease(f / max(per - 1, 1))
        else:  # across the seen arc and back
            u = f / max(per - 1, 1)
            a = arc_seen[0] + arc_seen[1] * ease(2 * u if u < 0.5 else 2 - 2 * u)
        tgt = eye + np.array([np.cos(a), np.sin(a), -np.tan(np.radians(5))])
        path.append(("in", eye, tgt))
if args.frames_only:
    path = [path[i] for i in np.linspace(0, len(path) - 1, args.frames_only).astype(int)]

K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
cam = add_camera(stage, "/World/Cams/tour", K, W, H, eye=path[0][1], target=path[0][2])
world.reset()
vega = None if args.no_robot else Robot(world, spec)
if vega is not None:
    vega.teleport(vega.home())
for _ in range(round(.75/world.get_physics_dt())):
    world.step(render=False)
if vega is not None and pl.get('scope') != 'presentation_only':
    sweep = RobotSweep(stage, vega)
    sweep.sample()
    print(summarize(clear_robot(ts, sweep)), flush=True)
support_collisions(stage, True)
ann = rgb_annotator(cam.GetPath(), W, H)
shown = {}


def show(prim, on):
    if shown.get(prim.GetPath()) != on:
        (UsdGeom.Imageable(prim).MakeVisible if on else UsdGeom.Imageable(prim).MakeInvisible)()
        shown[prim.GetPath()] = on


# outside, the walls are cut down to `--cut` m (scaled about the floor), as a model of the room: every piece in view,
# the walls' places and colours kept (hiding the near ones left a long white side wall lit across the foreground)
cut_ops = {pname: UsdGeom.Xformable(w[0]).AddScaleOp() for pname, w in walls.items()}
for f, (mode, eye, tgt) in enumerate(path):
    if f == 0 or path[f - 1][0] != mode:
        for c in ceiling:
            show(c, mode == "in")
        for op in cut_ops.values():
            op.Set(Gf.Vec3f(1.0, 1.0, 1.0 if mode == "in" else args.cut / zc))
        if splat_bg is not None:
            show(splat_bg, mode == "in")
            for q in hidden:
                show(q, mode != "in")
    set_pose(cam.GetPrim(), eye, look_at_quat(eye, tgt))
    if vega is not None:
        vega.command(vega.home())
    world.step(render=True)
    if f == 0 or (args.frames_only and f > 0) or (f > 0 and path[f - 1][0] != mode) or \
            (f > 0 and np.linalg.norm(eye - path[f - 1][1]) > 0.5):  # a cut: to a standpoint too
        for _ in range(8):  # a cut (the first frame, the move inside, each still): let the renderer settle
            world.step(render=True)
    Image.fromarray(grab(ann)).save(frames_dir / f"{f:05d}.jpg", quality=92)
if args.frames_only:
    print("stills in", frames_dir)
else:
    encode_video(frames_dir, out / "tour.mp4")
    print("saved", out / "tour.mp4")
app.close()
