"""The free phase of an RGB video (real/<name>/, calibrated_scene_reconstruction.vidphys) replayed in its twin: the object starts where and as
fast as the video's fit says it was let go, PhysX carries it from there, and its silhouette in the video's camera is
compared with the video's masks.

    tools/run_isaac.sh apps/video_physics.py real/<name> [--calibrate [--batch N] [--iters G] [--active p1,p2,...]]
                                                         [--scan p1,p2,... [--batch N]] [--mass KG] [--render]
                                                         [--seconds S]

Needs perception.npz, twin.json, track.npz and physics.json (python -m calibrated_scene_reconstruction.vidphys.{perceive,track,fit}; with
vidphys.mesh in between the object is its SAM 3D mesh instead of a primitive).
No hand or robot is simulated: the free phase starts after the last contact, so the pusher only enters through the
release state (position, heading, speed, yaw, yaw rate), which is fitted beside the physical parameters as a
nuisance. Of a video with several free phases the longest measured one is replayed.

What is replayed, by physics.json's motion and what the video showed of it:

    slide, roll, drop   on a level support that stands still
    incline             a slide down a support that is not level: the twin's frame is the support's, gravity leans
    stay                an object that stays on such a support: the least friction that holds the twin there
    tilting             a support that tilts until the object slips: the frame is the support's, gravity turns in it
                        by the tilt of every frame, the camera moves against it (incline.board_frames)
    ride, pulled        a support that turns under the object (a turntable): a disc that turns as the object's own
                        ride says (turntable.support_turn), or, where it slides from rest and never rode, spins up
                        at a rate fitted as a nuisance
    pendulum            a ball on a revolute joint at the fitted pivot (an articulation of that one joint)

default      one rollout with physics.json's closed-form friction and release state -> sim/replay.json, overlay.mp4
--calibrate  CMA-ES over --active in --batch scene copies per generation (calibrated_scene_reconstruction.physics.replay.optim) -> sim/fit.json, and
             physics.json gains "sim": the fitted values, the loss before and after, the closed-form value beside it.
             Two estimates of the same friction by different routes (a parabola through plane positions; a physics
             engine scored on silhouettes): their agreement is the check, their difference an error bar.
             Of a stay or a ride, which bound the friction from below: the batch is a ladder of frictions, and the
             least one that holds the twin where the video has the object is PhysX's bound beside the closed form's.
--scan       the loss over a grid of each named parameter, the others at the fit (else the closed form) ->
             sim/scan.json. A flat curve is the measured statement that this video does not show that parameter:
             object.mass is flat for every free slide (friction force and inertia both scale with it).
--render     the replay (sim/fit.json's parameters when it exists) rendered from the video's camera -> sim/render.mp4
             and sim/side_by_side.mp4: the video beside its twin. The twin's picture is the video's own background
             (plate.png of vidphys.mesh, else the frames' median) with the rendered object on it and the rendered
             shadow as a darkening: only the object and its motion are simulated, so only they are drawn. Where the
             background itself moves (a tilting board, a turntable, a pendulum's string) it is every frame of the
             video with the object taken out.
--seconds    replay no more than this much of the phase (a pendulum swings for minutes)
Writes real/<name>/sim/{replay.json | fit.json, sim_track.npz, overlay.mp4, twin.usd}.
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--active", help="comma-separated parameters to fit (default: ACTIVE of the recording's motion)")
ap.add_argument("--scan", help="comma-separated parameters: loss over a grid of each (identifiability)")
ap.add_argument("--mass", type=float, help="object mass in kg (default meta.json prior.mass, else 0.3)")
ap.add_argument("--render", action="store_true")
ap.add_argument("--seconds", type=float, help="replay at most this much of the phase")
ap.add_argument("--set", help="name=value,...: parameters of the default replay other than the closed form's")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument('--object-route', choices=('agent_blender', 'template', 'sam3d'), default='agent_blender')
args = ap.parse_args()

if args.object_route == 'agent_blender':
    from calibrated_scene_reconstruction.physics.video.model_inputs import require_model
    require_model(args.recording, applied=True)

import numpy as np  # noqa: E402

from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.physics.video.scene import G, read_frames  # noqa: E402

rec = Path(args.recording)
frames, _ = read_frames(rec)
IMG_H, IMG_W = frames.shape[1:3]

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, **({"renderer": "RaytracedLighting", "width": IMG_W, "height": IMG_H}
                                           if args.render else {})})

import cv2  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation, RigidPrim  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from calibrated_scene_reconstruction.physics.replay import loss, observe, optim, params  # noqa: E402
from calibrated_scene_reconstruction.physics.replay.params import Param  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import apply_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import quat_between  # noqa: E402
from calibrated_scene_reconstruction.physics.video import raster  # noqa: E402
from calibrated_scene_reconstruction.sim import video_physics_stage as vstage  # noqa: E402

STRIDE = 2
out = rec / "sim"
out.mkdir(exist_ok=True)
meta, tw, phys = vstage.load(rec)
per, tr = np.load(rec / "perception.npz"), np.load(rec / "track.npz")
MOTION = phys.get("motion", "slide")
fp = phys.get("free_phase")  # the longest measured one
KIND = MOTION
if MOTION == "incline":
    KIND = ("tilting" if "tilt_deg_t" in tw else "incline") if fp else "stay" if phys.get("stays") and "tilt_deg_t" not in tw else None
elif MOTION == "turntable":
    KIND = "pulled" if phys.get("slides") else "ride" if phys.get("rides") else None
elif fp is None:
    KIND = None
if KIND is None:
    sys.exit(f"{rec}: no free phase in physics.json: nothing to replay")
if KIND not in ("slide", "roll", "drop", "incline", "stay", "tilting", "ride", "pulled", "pendulum"):
    sys.exit(f"{rec}: motion {MOTION}: not replayed")
SLIDES = KIND not in ("roll", "drop", "pendulum")  # the body stands on its base and its mesh is its collider
RATE = 240.0 if KIND in ("drop", "pendulum") else 120.0  # Hz: a bounce is over in a few steps
K, T = np.asarray(tw["K"], float), np.asarray(tw["T_world_cam"], float)
if MOTION == "turntable":  # in the plane the fit read the path in (the object's ride refines the outline's)
    T = np.asarray(phys.get("support_plane", {}).get("T_world_cam", T), float)
if KIND == "stay":
    _s = max(phys["stays"], key=lambda s: s["seconds"])
    i0, i1 = _s["first_frame"], _s["last_frame"]
elif KIND == "tilting":  # the end of its ride and the slide
    i0 = max([s["first_frame"] for s in phys.get("stays", []) if s.get("rode") and s["last_frame"] <= fp["first_frame"] + 1]
             or [fp["first_frame"]])
    i0, i1 = max(i0, fp["first_frame"] - 30), fp["last_frame"]
elif KIND == "ride":
    i0, i1 = phys["rides"][0]["first_frame"], phys["rides"][-1]["last_frame"]
else:
    i0, i1 = fp["first_frame"], fp["last_frame"]
if args.seconds:
    i1 = min(i1, int(np.searchsorted(per["t"], per["t"][i0] + args.seconds)))
fps = 1.0 / float(np.median(np.diff(per["t"])))
Ks = K * np.array([[1 / STRIDE], [1 / STRIDE], [1]])
frames_idx = np.arange(i0, i1 + 1)
masks = per["object"][frames_idx][:, ::STRIDE, ::STRIDE]
centroid = np.array([observe.mask_stats(m)[0] * STRIDE for m in masks])
valid = (tr["valid"] & ~tr["border"])[frames_idx]
TK = np.repeat(T[None], len(frames_idx), 0)  # the camera in the twin's frame at every frame of the phase
mass0 = args.mass or meta.get("prior", {}).get("mass", 0.3)
obj = tw["object"]
DISC = GRAVITY_T = None
if KIND == "tilting":
    from calibrated_scene_reconstruction.physics.video.incline import board_frames, place_on_board

    _T, _g = board_frames(per, tw)
    BOARD_T = _T
    TK = _T[frames_idx]
    valid &= np.isfinite(TK[:, 0, 0])
    _k = np.nonzero(np.isfinite(_g[frames_idx, 0]))[0]
    if len(_k) < 5 or not valid[0]:
        sys.exit(f"{rec}: the support is not seen well enough to carry a frame: nothing to replay")
    # gravity in the support's frame at the time of every step
    GRAVITY_T = lambda s: np.array([np.interp(s, per["t"][frames_idx[_k]] - per["t"][i0], _g[frames_idx[_k], j])
                                    for j in range(3)])
    _uv = np.full((len(per["t"]), 2), np.nan)
    _uv[frames_idx] = centroid
    _hc = float(obj.get("centroid_height", obj["dims"]["height"] / 2))
    _xy = place_on_board(K, _T, _uv, [i for i in frames_idx if i < fp["first_frame"] - 2] or frames_idx[:3], _hc)
    if _xy is None:
        sys.exit(f"{rec}: the object is not seen riding its support: nothing to replay")
    for _i in np.nonzero(~np.isfinite(TK[:, 0, 0]))[0]:  # a frame without the support: the last one's (not scored)
        TK[_i] = TK[_i - 1]
elif MOTION == "turntable":
    from calibrated_scene_reconstruction.physics.video.turntable import support_turn

    AXIS = np.array(phys["axis"], float)
    PATH = np.array([p if p else [np.nan, np.nan] for p in phys["path"]], float)
    valid = np.all(np.isfinite(PATH[frames_idx]), 1)
    _reach = float(np.nanmax(np.linalg.norm(PATH - AXIS, axis=1)))
    DISC = {"centre": AXIS.tolist(), "radius": max(tw.get("support_circle", {}).get("radius", 0.0), 1.15 * _reach)}
elif KIND == "pendulum":
    valid = (np.all(np.isfinite(centroid), 1) & ~tr["border"][frames_idx] & (per["score"][frames_idx] > 0.05))
obs = {"t": per["t"][frames_idx] - per["t"][i0], "valid": valid, "masks": masks, "centroid": centroid}
if KIND == "pendulum":
    # the pendulum of pendulum.py in the twin's frame: the pivot, gravity, the level direction it swings along, and
    # the axis it turns about (a turn by theta about it takes "hanging" to pivot + L (sin(theta) h + cos(theta) g))
    _d = np.asarray(fp["gravity_camera"], float)
    _e1 = np.cross(_d, [0.0, 0.0, 1.0])
    _e1 /= np.linalg.norm(_e1)
    _phi = np.radians(fp["azimuth_deg"])
    _h = np.cos(_phi) * _e1 + np.sin(_phi) * np.cross(_d, _e1)
    PIVOT = T[:3, :3] @ np.asarray(fp["pivot_camera"], float) + T[:3, 3]
    DOWN, SWING = T[:3, :3] @ _d, T[:3, :3] @ _h
    HINGE = np.cross(DOWN, SWING)
    # (as filmed: what replays the video in its own time, where the length given is another or none)
    L0 = phys["parameters"]["pendulum.length"].get("as_filmed") or phys["parameters"]["pendulum.length"]["value"]
    # the ball's size: its mask's, at the distance the fitted swing has it at in that frame, where it is sharp (a
    # streak is larger than the ball: the lower fifth of the frames)
    from calibrated_scene_reconstruction.physics.video.pendulum import bob as _bob, swing as _swing

    _th = _swing(np.radians(fp["theta0_deg"]), np.radians(fp["omega0_deg"]), L0,
                 phys["parameters"]["pendulum.damping_rate"]["value"], phys["parameters"]["pendulum.dry_friction"]["value"],
                 np.asarray(per["t"][frames_idx] - per["t"][i0], float))
    _z = _bob(K, _d, fp["pivot_uv"], fp["pivot_distance"], _phi, L0, _th)[1][:, 2]
    _r = float(np.percentile((np.sqrt(per["object"][frames_idx].sum((1, 2)) / np.pi) * _z / K[0, 0])[valid], 20))
    Vb, Fb = vstage.sphere_mesh(_r)
    _px = frames[frames_idx[np.argmax(valid)]][per["object"][frames_idx[np.argmax(valid)]]]
    _m = {"V": Vb, "F": Fb, "UV": None, "albedo": None, "Vs": Vb, "Fs": Fb,
          "color": (np.median(_px, 0) / 255.0).tolist() if len(_px) else (0.2, 0.2, 0.2)}
    ROUND = {"type": "sphere", "radius": _r, "centre": [0.0, 0.0, _r]}
else:
    # what rolls or bounces is a sphere or a lying cylinder to PhysX, with the mesh as its look (stage.round_collider)
    _m = vstage.object_mesh(rec, obj, flat=SLIDES,
                            radius=None if SLIDES else fp.get("radius", obj["dims"]["height"] / 2))
    ROUND = None if SLIDES else vstage.round_collider(_m["V"])
    if ROUND and ROUND["type"] == "sphere" and obj.get("mesh_iou", 1.0) < 0.7:
        # a mesh that does not look like the ball it is (a small blurred ball gives SAM 3D little): a plain sphere in
        # the ball's colour instead
        Vb, Fb = vstage.sphere_mesh(ROUND["radius"])
        _px = frames[tw["ref_frame"]][per["object"][tw["ref_frame"]]]
        _m = {"V": Vb, "F": Fb, "UV": None, "albedo": None, "Vs": Vb, "Fs": Fb,
              "color": (np.median(_px, 0) / 255.0).tolist() if len(_px) else (0.8, 0.3, 0.2)}
        ROUND = dict(ROUND, centre=[0.0, 0.0, ROUND["radius"]])
if _m["albedo"] is None and "color" not in _m:  # a primitive: in the colour the video shows the object in
    _px = frames[tw["ref_frame"]][per["object"][tw["ref_frame"]]]
    if len(_px):
        _m["color"] = (np.median(_px, 0) / 255.0).tolist()
V, F_TRI = _m["Vs"], _m["Fs"]  # the outline mesh: silhouettes


def yaw_rate0():
    k = np.arange(i0, min(i0 + 6, i1 + 1))
    k = k[np.isfinite(tr["yaw"][k])]
    if len(k) < 3:
        return 0.0
    y = np.degrees(np.unwrap(np.radians(tr["yaw"][k])))
    return float(np.clip(np.polyfit(per["t"][k], y, 1)[0], -700, 700))


YAW0 = float(tr["yaw"][i0]) if np.isfinite(tr["yaw"][i0]) else float(obj["pose"][3])
# the parameter vector per motion: what the motion measures, the release state around the closed-form fit as a
# nuisance, and what the motion should not show (--scan them).
COMMON = [
    Param("object.mass", 0.002, 3.0, float(np.clip(mass0, 0.002, 3.0)), log=True),
    Param("object.com_offset.z", -0.3, 0.05, 0.0),
]
PLACE = [Param("release.dx", -0.03, 0.03, 0.0), Param("release.dy", -0.03, 0.03, 0.0),
         Param("release.yaw_deg", YAW0 - 30.0, YAW0 + 30.0, YAW0), Param("object.restitution", 0.0, 0.6, 0.05)]
LADDER = None  # of a stay or a ride: the closed form's lower bound on the friction, which --calibrate climbs past
if KIND == "drop":
    # the flight's start (ballistic.py): the centre's position and velocity at the first frame of the phase
    P0, V0 = np.array(fp["start"]["position"], float), np.array(fp["start"]["velocity"], float)
    e0 = phys["parameters"]["object.restitution"]["value"]
    MAIN, LABEL = "object.restitution", "restitution"
    PARAMS = COMMON + [
        Param("object.restitution", 0.0, 0.98, float(np.clip(e0, 0.0, 0.98))),
        Param("support.friction", 0.02, 1.5, 0.4, log=True),
        Param("release.dx", -0.03, 0.03, 0.0), Param("release.dy", -0.03, 0.03, 0.0),
        Param("release.dz", -0.03, 0.03, 0.0),
        Param("release.dvx", -0.3, 0.3, 0.0), Param("release.dvy", -0.3, 0.3, 0.0),
        Param("release.dvz", -0.4, 0.4, 0.0),
        Param("release.yaw_deg", YAW0 - 30.0, YAW0 + 30.0, YAW0),
    ]
    ACTIVE = "object.restitution,release.dx,release.dy,release.dz,release.dvx,release.dvy,release.dvz"
    mu0 = e0
elif KIND == "incline":
    # a slide down a support that is not level (incline.py): gravity is the twin's "gravity", the body starts at
    # the fitted place with the fitted speed along the fitted heading, and friction sets how fast it gains speed
    XY0 = np.array(fp["start_xy"], float)
    mu0 = fp["coefficient"]
    MAIN, LABEL = "support.friction", "friction"
    v_hi = max(0.3, 2.5 * abs(fp["release_speed"]))
    PARAMS = COMMON + [
        Param("support.friction", 0.02, 1.5, float(np.clip(mu0, 0.02, 1.5)), log=True),
        Param("release.speed", 0.0, v_hi, float(np.clip(fp["release_speed"], 0.0, v_hi))),
        Param("release.heading_deg", fp["heading_deg"] - 15, fp["heading_deg"] + 15, fp["heading_deg"]),
        Param("release.dx", -0.03, 0.03, 0.0),
        Param("release.dy", -0.03, 0.03, 0.0),
        Param("release.yaw_deg", YAW0 - 30.0, YAW0 + 30.0, YAW0),
        Param("release.yaw_rate_deg", -720.0, 720.0, 0.0),
        Param("object.restitution", 0.0, 0.6, 0.05),
    ]
    ACTIVE = "support.friction,release.speed,release.dx,release.dy"
elif KIND == "stay":
    # it stands where it was put on a support that is not level: any friction above tan(tilt) holds it there
    XY0 = np.nanmedian(tr["xy"][frames_idx][valid], 0)
    _st = phys["parameters"]["support.static_friction"]
    mu0 = LADDER = _st.get("needed", _st["lower_bound"])  # what holds it at the twin's own scale
    MAIN, LABEL = "support.friction", "static friction >="
    PARAMS = COMMON + [Param("support.friction", 0.01, 3.0, float(np.clip(mu0, 0.01, 3.0)), log=True)] + PLACE
    ACTIVE = "support.friction"
elif KIND == "tilting":
    # it rides a support that tilts and slips where gravity along it passes what static friction holds; from
    # there kinetic friction sets how fast it gains speed (incline.py). It starts at rest where it rode. When it
    # lets go is the rollout's doing, not PhysX's (the body is held in place until tan(tilt) passes its static
    # friction): PhysX's own threshold moved with the run, the same twin slipping at 0.7 to 1.0 of its static
    # friction as one copy, in a batch, or with the renderer on
    XY0 = np.asarray(_xy, float)
    mu0 = fp["coefficient"]
    mus0 = phys["parameters"]["support.static_friction"]["value"]
    MAIN, LABEL = "support.friction", "friction"
    PARAMS = COMMON + [
        Param("support.friction", 0.02, 1.5, float(np.clip(mu0, 0.02, 1.5)), log=True),
        Param("support.static_friction", 0.02, 1.5, float(np.clip(max(mus0, mu0), 0.02, 1.5)), log=True),
    ] + PLACE
    ACTIVE = "support.friction,support.static_friction,release.dx"
elif KIND == "ride":
    # it rides a support that turns as its own path says (turntable.support_turn): the disc's angular velocity at
    # every step is that of the object's angle about the axis, and the object starts on its circle, going round
    TURN = support_turn(per["t"] - per["t"][i0], np.where(np.arange(len(PATH))[:, None] >= i0, PATH, np.nan)[: i1 + 1],
                        AXIS)
    _a0 = float(TURN(0.0))  # on its first circle, at the angle the turn starts from
    XY0 = AXIS + phys["rides"][0]["radius"] * np.array([np.cos(_a0), np.sin(_a0)])
    _st = phys["parameters"]["support.static_friction"]
    mu0 = LADDER = _st.get("needed") or _st.get("lower_bound") or _st.get("value") or _st.get("upper_bound")
    MAIN, LABEL = "support.friction", "static friction >="
    PARAMS = COMMON + [Param("support.friction", 0.01, 3.0, float(np.clip(mu0, 0.01, 3.0)), log=True)] + PLACE
    ACTIVE = "support.friction"
elif KIND == "pulled":
    # it slides on a support that spins up under it (turntable.fit_pulled): friction pulls it with mu g towards
    # where the support's surface goes. How the support turns is not seen in an object that never rode it: its
    # angular acceleration, when it starts, its top speed and where its axis is (the outline's middle is a few
    # cm off along a grazing view) are fitted as the release state is, with the friction held at the closed
    # form's. The friction is not fitted here: the closed form reads the path as a slide from its first motion
    # on, and a twin free to choose both draws the same path with a slower support the object rides and twice
    # the friction (--scan support.friction says so). The support is held to the closed form's reading: it spins
    # up at least half as fast again as friction could take the object along (r alpha >= 1.5 mu g).
    _fit = np.array(max(phys["slides"], key=lambda s: s["frames"])["fitted"], float)
    XY0 = _fit[0]
    _far = _fit[np.argmax(np.linalg.norm(_fit - _fit[0], axis=1) > 0.01)] - _fit[0]
    SPIN = float(np.sign(np.cross(XY0 - AXIS, _far)) or 1.0)  # which way it is dragged round
    mu0 = fp["friction"]
    MAIN, LABEL = "support.friction", "friction"
    _moves = float(obs["t"][np.argmax(np.linalg.norm(_fit - _fit[0], axis=1) > 0.003)])
    _least = 1.5 * mu0 * G / float(np.linalg.norm(XY0 - AXIS))
    PARAMS = COMMON + [
        Param("support.friction", 0.02, 1.5, float(np.clip(mu0, 0.02, 1.5)), log=True),
        Param("support.spin_up", _least, 12.0 * _least, 2.0 * _least, log=True),   # rad/s^2
        Param("support.spin_start", _moves - 0.25, _moves + 0.1, _moves - 0.04),   # s into the phase
        Param("support.spin_top", 2.0, 80.0, 10.0, log=True),                      # rad/s
        Param("support.axis_dx", -0.04, 0.04, 0.0), Param("support.axis_dy", -0.04, 0.04, 0.0),
    ] + PLACE
    ACTIVE = ("support.spin_up,support.spin_start,support.spin_top,support.axis_dx,support.axis_dy,"
              "release.dx,release.dy")
elif KIND == "pendulum":
    # pendulum.py's swing: the equivalent simple pendulum's length (the ball's own inertia is left out of the twin
    # as it is out of the fit), a damping in proportion to the speed (the joint's damping, c m L^2) and one at a
    # constant rate (PhysX has none: the rollout takes it out of the velocity, as rolling resistance)
    q = phys["parameters"]
    mu0 = L0
    MAIN, LABEL = "pendulum.length", "length (m)"
    PARAMS = COMMON + [
        Param("pendulum.length", 0.7 * L0, 1.4 * L0, L0, log=True),
        Param("pendulum.damping_rate", 0.0, 5.0, q["pendulum.damping_rate"]["value"]),
        Param("pendulum.dry_friction", 0.0, 5.0, q["pendulum.dry_friction"]["value"]),
        Param("release.theta_deg", fp["theta0_deg"] - 15, fp["theta0_deg"] + 15, fp["theta0_deg"]),
        Param("release.omega_deg", fp["omega0_deg"] - 120, fp["omega0_deg"] + 120, fp["omega0_deg"]),
    ]
    ACTIVE = "pendulum.length,pendulum.damping_rate,release.theta_deg,release.omega_deg"
else:
    # where the twin's object stands when it is let go: the fit's start, with the yaw it was tracked at. The release
    # speed is not a parameter but sqrt(2 c g distance), c the friction or the rolling resistance: the distance
    # covered is what the images pin down, so c then moves the timing alone, where c and the speed as a pair trade
    # off along a narrow valley CMA-ES crawls through.
    XY0 = np.array(fp["start_xy"], float)
    mu0 = fp["coefficient"]
    MAIN, LABEL = (("support.friction", "friction") if KIND == "slide"
                   else ("support.rolling_resistance", "rolling resistance"))
    PARAMS = COMMON + [
        Param("support.friction", 0.02, 1.5, mu0 if KIND == "slide" else 0.6, log=True),
        Param("release.distance", 0.8 * fp["distance"], 1.25 * fp["distance"], fp["distance"], log=True),
        Param("release.heading_deg", fp["heading_deg"] - 15, fp["heading_deg"] + 15, fp["heading_deg"]),
        Param("release.dx", -0.04, 0.04, 0.0),
        Param("release.dy", -0.04, 0.04, 0.0),
        Param("release.yaw_deg", YAW0 - 30.0, YAW0 + 30.0, YAW0),
        Param("release.yaw_rate_deg", -720.0, 720.0, yaw_rate0()),
        Param("object.restitution", 0.0, 0.6, 0.05),
    ]
    ACTIVE = f"{MAIN},release.distance,release.heading_deg,release.dx,release.dy"
    if KIND == "slide":
        ACTIVE += ",release.yaw_rate_deg"
    else:  # the deceleration of a free roll over g: PhysX has none, the rollout takes it out of the velocity
        PARAMS.append(Param("support.rolling_resistance", 0.001, 0.6, float(np.clip(mu0, 0.001, 0.6)), log=True))


def release_speed(flat):
    if KIND == "drop":
        return float(np.linalg.norm(V0 + [flat["release.dvx"], flat["release.dvy"], flat["release.dvz"]]))
    if KIND == "incline":
        return float(flat["release.speed"])
    if KIND in ("stay", "tilting", "pulled"):
        return 0.0
    if KIND in ("ride", "pendulum"):
        return float(np.linalg.norm(start(flat)[2]))  # of a pendulum: at release.theta_deg
    return float(np.sqrt(2 * flat[MAIN] * G * flat["release.distance"]))


def turn_rate(flat, s):
    """The support's angular velocity about its axis (rad/s) at s seconds into the phase."""
    if KIND == "ride":
        return float(TURN.derivative()(min(max(s, 0.0), float(obs["t"][-1]))))
    return SPIN * float(np.clip(flat["support.spin_up"] * (s - flat["support.spin_start"]), 0.0, flat["support.spin_top"]))


def quat_z(yaw_deg):
    h = np.radians(yaw_deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def swung(flat, theta):
    """The ball of a pendulum at the angles theta (N,) rad from hanging: (its origin (N, 3), its orientation
    (N, 4) w x y z)."""
    R = Rotation.from_rotvec(np.outer(np.atleast_1d(theta), HINGE))
    return (PIVOT + R.apply(flat["pendulum.length"] * DOWN - np.asarray(ROUND["centre"], float)),
            R.as_quat(scalar_first=True))


def start(flat):
    """The body at the first frame of the phase: (its origin (3,), its orientation (w, x, y, z), velocity (3,),
    angular velocity (3,)). A pendulum is authored hanging; its joint is then set to release.theta_deg, going at
    release.omega_deg."""
    if KIND == "pendulum":
        p, q = swung(flat, 0.0)
        om = np.radians(flat["release.omega_deg"])
        return p[0], q[0], om * flat["pendulum.length"] * SWING, om * HINGE
    if KIND == "drop":
        yaw = flat["release.yaw_deg"]
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cx, cy, cz = ROUND["centre"]  # the collider's centre in the body: the fitted flight is its path
        p = P0 + [flat["release.dx"], flat["release.dy"], flat["release.dz"]] - [c * cx - s_ * cy, s_ * cx + c * cy, cz]
        return (np.r_[p[:2], max(float(p[2]), 0.0005)], quat_z(yaw),
                V0 + [flat["release.dvx"], flat["release.dvy"], flat["release.dvz"]], np.zeros(3))
    xy = XY0 + [flat["release.dx"], flat["release.dy"]]
    if KIND in ("stay", "tilting", "pulled"):  # at rest on its support
        return np.r_[xy, 0.001], quat_z(flat["release.yaw_deg"]), np.zeros(3), np.zeros(3)
    if KIND == "ride":  # going round with its support
        w = np.array([0.0, 0.0, turn_rate(flat, 0.0)])
        return np.r_[xy, 0.001], quat_z(flat["release.yaw_deg"]), np.cross(w, np.r_[xy - AXIS, 0.0]), w
    h = np.radians(flat["release.heading_deg"])
    v = release_speed(flat) * np.array([np.cos(h), np.sin(h), 0.0])
    if SLIDES:
        return (np.r_[xy, 0.001], quat_z(flat["release.yaw_deg"]), v,
                np.array([0.0, 0.0, np.radians(flat["release.yaw_rate_deg"])]))
    yaw = flat["release.yaw_deg"]
    if ROUND["type"] == "cylinder":  # it rolls across its axis: of the two such yaws, the one nearer the tracked
        yaw = flat["release.heading_deg"] + 90.0 - ROUND["axis_deg"]
        yaw += 180.0 * round((flat["release.yaw_deg"] - yaw) / 180.0)
    return np.r_[xy, 0.0005], quat_z(yaw), v, np.cross([0.0, 0.0, 1.0], v) / ROUND["radius"]  # rolling without slipping


def physics_of(flat):
    """sim.physics.apply_scene's dict from the flat parameters. The centre of mass stays PhysX's (the collider's,
    at uniform density) unless a height offset is asked for."""
    o = {"mass": flat["object.mass"], "restitution": flat.get("object.restitution", 0.0)}
    if flat["object.com_offset.z"] != 0.0:
        o["com_offset"] = [0.0, 0.0, flat["object.com_offset.z"]]
    if KIND == "pendulum":
        return {"object": o}
    mu = flat["support.friction"]
    if KIND == "tilting":  # the rollout lets it go at its static friction; PhysX slides it from there
        return {"object": o, "support": {"static_friction": mu, "dynamic_friction": mu}}
    return {"object": o, "support": {"static_friction": max(flat.get("support.static_friction", mu), mu),
                                     "dynamic_friction": mu}}


world = World(stage_units_in_meters=1.0, physics_dt=1 / RATE, rendering_dt=1 / RATE)
stage = omni.usd.get_context().get_stage()
n_env = args.batch if args.calibrate or args.scan else 1
ENVS = vstage.build(stage, rec, tw, mass0, n_env, mesh=_m, collider=ROUND, disc=DISC, slab=KIND != "pendulum")
SCENES = [UsdPhysics.Scene(_p) for _p in stage.Traverse() if _p.IsA(UsdPhysics.Scene)]


def set_gravity(g, running=False):
    """Gravity's direction (3,) in the twin's frame; running: while the simulation runs."""
    g = np.asarray(g, float) / np.linalg.norm(g)
    if running:
        import carb

        world.physics_sim_view.set_gravity(carb.Float3(*[float(x) for x in 9.81 * g]))
    else:
        for sc in SCENES:
            sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(*[float(x) for x in g]))


if KIND == "pendulum":
    set_gravity(DOWN)
    for _p in stage.Traverse():  # nothing stands under it: no shadow to lay on the picture
        if _p.GetName() == "look" and _p.GetParent().GetName() == "Support":
            UsdGeom.Imageable(_p).MakeInvisible()
    # an articulation of one joint: an anchor fixed in the world at the pivot, and the ball on a revolute joint to
    # it. (As two bodies and a joint between them the swing gained energy from the joint's position correction:
    # 72 deg grew to 82 in 20 s at 240 Hz. Joint coordinates have no such error to correct.)
    _q = quat_between((1.0, 0.0, 0.0), HINGE)
    _q = Gf.Quatf(float(_q[0]), *[float(x) for x in _q[1:]])
    for _off, _root in ENVS:
        _a = UsdGeom.Xform.Define(stage, f"{_root}/Objects/anchor")
        vstage.set_pose(_a.GetPrim(), PIVOT)
        UsdPhysics.RigidBodyAPI.Apply(_a.GetPrim())
        _ma = UsdPhysics.MassAPI.Apply(_a.GetPrim())
        _ma.CreateMassAttr(1.0)
        _ma.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))
        _fx = UsdPhysics.FixedJoint.Define(stage, f"{_root}/Objects/fix")
        _fx.CreateBody1Rel().SetTargets([f"{_root}/Objects/anchor"])
        _fx.CreateLocalPos0Attr(Gf.Vec3f(*[float(x) for x in PIVOT + _off]))
        _j = UsdPhysics.RevoluteJoint.Define(stage, f"{_root}/Objects/pivot")
        _j.CreateBody0Rel().SetTargets([f"{_root}/Objects/anchor"])
        _j.CreateBody1Rel().SetTargets([f"{_root}/Objects/object"])
        _j.CreateAxisAttr("X")
        _j.CreateLocalRot0Attr(_q)
        _j.CreateLocalRot1Attr(_q)
        _dr = UsdPhysics.DriveAPI.Apply(_j.GetPrim(), "angular")  # the joint's damping: a torque against its rate
        _dr.CreateTypeAttr("force")
        _dr.CreateStiffnessAttr(0.0)
        _dr.CreateTargetVelocityAttr(0.0)
        _b = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(f"{_root}/Objects/object"))
        _b.CreateLinearDampingAttr(0.0)
        _b.CreateAngularDampingAttr(0.0)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"{_root}/Objects"))
        _ar = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(f"{_root}/Objects"))
        _ar.CreateSleepThresholdAttr(0.0)
        _ar.CreateSolverPositionIterationCountAttr(16)
elif KIND == "tilting":
    set_gravity(GRAVITY_T(0.0))
    for _off, _root in ENVS:  # at rest for seconds while gravity turns: it must not fall asleep
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(f"{_root}/Objects/object")).CreateSleepThresholdAttr(0.0)
elif "gravity" in tw:  # a support that is not level: the twin's frame is the support's, gravity leans in it
    set_gravity(tw["gravity"])
N_STEPS = int(np.ceil(float(obs["t"][-1]) * RATE)) + 2  # to the last frame's own timestamp (phones drift)
T_SIM = np.arange(N_STEPS) / RATE


def silhouette(Kc, Tc, R, p, shape):
    """Filled silhouette of the object at rotation R (3, 3), position p, from the camera at Tc."""
    return raster.fill(observe.project(Kc, Tc, V @ R.T + p)[0][F_TRI], shape)


def author(i, f):
    """What a copy's parameters are beyond apply_scene's, on the stopped stage."""
    root = ENVS[i][1]
    body = stage.GetPrimAtPath(f"{root}/Objects/object")
    if KIND == "pendulum":
        # the pivot in the ball's frame; no inertia of its own (the fitted length is the simple pendulum's)
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{root}/Objects/pivot")).CreateLocalPos1Attr().Set(
            Gf.Vec3f(*[float(x) for x in np.asarray(ROUND["centre"]) - f["pendulum.length"] * DOWN]))
        UsdPhysics.MassAPI(body).CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(f["object.mass"]) * 1e-6] * 3))
        # the damping rate c (1/s) as the joint's damping, c m L^2 (N m s / rad; USD's angular drives are per degree)
        UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"{root}/Objects/pivot"), "angular").CreateDampingAttr().Set(
            float(f["pendulum.damping_rate"] * f["object.mass"] * f["pendulum.length"] ** 2 * np.pi / 180.0))
    if DISC is not None:
        vstage.set_pose(stage.GetPrimAtPath(f"{root}/Support/disc"),
                        [AXIS[0] + f.get("support.axis_dx", 0.0), AXIS[1] + f.get("support.axis_dy", 0.0), 0.0])


def spin(arts, rates):
    """The pendulums' joint velocities (rad/s). Isaac's setter also makes them the joint drives' targets, and a
    drive that aims at the rate the joint has damps nothing: the targets go back to rest."""
    arts.set_joint_velocities(np.asarray(rates, float)[:, None])
    arts.set_joint_velocity_targets(np.zeros((len(rates), 1)))


def rollout(flats, on_frame=None):
    """The free phase in len(flats) copies at once; per copy the sim dict of calibrated_scene_reconstruction.physics.replay.loss (+ poses).
    on_frame(k) is called when the simulation stands at video frame k of the phase (rendering)."""
    n = len(flats)
    world.stop()  # author on a stopped timeline (see apps/real2sim_replay.py)
    starts = [start(f) for f in flats]
    for i, f in enumerate(flats):
        vstage.pose(stage, ENVS[i][1], starts[i][0], starts[i][1])
        apply_scene(stage, physics_of(f), f"{ENVS[i][1]}/Objects/object", supports=(f"{ENVS[i][1]}/Support",),
                    tag=f"vidphys_env{i}")
        author(i, f)
    if GRAVITY_T is not None:
        set_gravity(GRAVITY_T(0.0))
    world.reset()
    offs = np.stack([ENVS[i][0] for i in range(n)])
    bodies = RigidPrim([f"{ENVS[i][1]}/Objects/object" for i in range(n)], name="objects")  # all copies in one call
    bodies.initialize()
    discs = arts = None
    if DISC is not None:
        discs = RigidPrim([f"{ENVS[i][1]}/Support/disc" for i in range(n)], name="discs")
        discs.initialize()
    if KIND == "pendulum":
        arts = Articulation([f"{ENVS[i][1]}/Objects" for i in range(n)], name="pendulums")
        arts.initialize()
        arts.set_joint_positions(np.array([[np.radians(f["release.theta_deg"])] for f in flats]))
        spin(arts, [np.radians(f["release.omega_deg"]) for f in flats])
        dry = np.array([f["pendulum.dry_friction"] for f in flats])
    else:
        bodies.set_linear_velocities(np.stack([s[2] for s in starts]).astype(float))
        bodies.set_angular_velocities(np.stack([s[3] for s in starts]).astype(float))
    P, Q = np.zeros((n, N_STEPS, 3)), np.zeros((n, N_STEPS, 4))
    TH = np.zeros((n, N_STEPS))  # a pendulum's joint angle
    # on a support that tilts: held where it rode until gravity along the support passes what static friction holds
    held = np.full(n, GRAVITY_T is not None)
    rest_p, rest_q = np.stack([s[0] for s in starts]) + offs, np.stack([s[1] for s in starts])
    holds = np.array([max(f.get("support.static_friction", 0.0), f.get("support.friction", 0.0)) for f in flats])
    due = 0
    for k in range(N_STEPS):
        if arts is not None:
            TH[:, k] = np.asarray(arts.get_joint_positions(), float)[:, 0]
        else:
            p, q = bodies.get_world_poses()
            P[:, k], Q[:, k] = np.asarray(p, float) - offs, np.asarray(q, float)
        while on_frame is not None and due < len(obs["t"]) and T_SIM[k] >= obs["t"][due] - 1e-9:
            on_frame(due)
            due += 1
        if GRAVITY_T is not None:
            g = GRAVITY_T(T_SIM[k])
            set_gravity(g, running=True)
            held &= ~(np.hypot(g[0], g[1]) > -g[2] * holds)
            idx = np.nonzero(held)[0]
            if len(idx):
                bodies.set_world_poses(rest_p[idx], rest_q[idx], indices=idx)
                bodies.set_linear_velocities(np.zeros((len(idx), 3)), indices=idx)
                bodies.set_angular_velocities(np.zeros((len(idx), 3)), indices=idx)
        if discs is not None:  # the support turns at its rate about its axis, whatever stands on it
            discs.set_linear_velocities(np.zeros((n, 3)))
            discs.set_angular_velocities(np.array([[0.0, 0.0, turn_rate(f, T_SIM[k])] for f in flats]))
        world.step(render=False)
        if KIND == "roll":  # rolling resistance: both velocities shrink by what a deceleration c g takes in a step
            v, w = np.asarray(bodies.get_linear_velocities(), float), np.asarray(bodies.get_angular_velocities(), float)
            c = np.array([f["support.rolling_resistance"] for f in flats])
            keep = np.maximum(0.0, 1.0 - c * G / RATE / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9))
            bodies.set_linear_velocities(np.c_[v[:, :2] * keep[:, None], v[:, 2]])
            bodies.set_angular_velocities(w * keep[:, None])
        elif arts is not None and dry.any():  # the dry part of its damping: b tanh(theta' / 0.02) off the joint's rate
            om = np.asarray(arts.get_joint_velocities(), float)[:, 0]
            spin(arts, om - np.sign(om) * np.minimum(np.abs(om), dry * np.tanh(np.abs(om) / 0.02) / RATE))
    mass = np.asarray(bodies.get_masses(), float)
    sims = []
    shape = masks.shape[1:]
    for i in range(n):
        if arts is not None:
            Pv, Qv = swung(flats[i], loss.sample_at(T_SIM, TH[i], obs["t"]))
        else:
            Pv, Qv = loss.sample_at(T_SIM, P[i], obs["t"]), loss.sample_at(T_SIM, Q[i], obs["t"])
        R = Rotation.from_quat(Qv / np.linalg.norm(Qv, axis=1, keepdims=True), scalar_first=True).as_matrix()
        sil = np.stack([silhouette(Ks, TK[f], R[f], Pv[f], shape) for f in range(len(Pv))])
        sims.append({"t": obs["t"], "mask": sil, "uv": np.array([observe.mask_stats(m)[0] * STRIDE for m in sil]),
                     "pos": Pv, "quat_wxyz": Qv, "mass": float(mass[i]),
                     "tilt": float(np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1))).max())})
    return sims


TILT = 12.0  # deg: a twin leaning further than this in a slide has toppled, and the video's object did not


def score(sim):
    total, parts = loss.discrepancy(sim, obs, fps)
    if SLIDES and sim["tilt"] > TILT:
        total, parts = total + 50.0, dict(parts, toppled=sim["tilt"])
    return total, parts


def upright_com(flat):
    """The centre-of-mass offset (m, <= 0) at which the twin slides without toppling, as the video's object did: the
    smallest lowering out of a ladder down to the support, None when none stands. A slide the object survives
    upright bounds its centre of mass from above (height < base half-width / friction); the mesh's own, at uniform
    density, may lie above that (a bottle is heavier at its base, and a mesh from one view has a rough base)."""
    hz = float(V[:, 2].max()) / 2
    ladder = [0.0] + [-hz * x for x in np.linspace(0.15, 0.9, max(n_env - 1, 1))]
    # with a third more friction than measured: a twin that only just stands falls over in the next rollout
    hard = dict(flat, **{"support.friction": 1.3 * flat["support.friction"]})
    tilt = [s["tilt"] for s in rollout([dict(hard, **{"object.com_offset.z": z}) for z in ladder[:n_env]])]
    print("[calibrate] tilt (deg) by centre-of-mass offset (m): " + " ".join(f"{z:.3f}:{a:.0f}" for z, a in zip(ladder, tilt)),
          flush=True)
    for z, a in zip(ladder, tilt):
        if a <= TILT:
            return z, tilt[0]
    return None, tilt[0]


def strayed(sim):
    """How far (m) the twin's object is from where the video has it on its support, at the worst: of a stay, how
    far it went from where it stood; of a ride, the most its distance from the axis differs from the path's."""
    if KIND == "stay":
        return float(np.linalg.norm(sim["pos"][:, :2] - sim["pos"][0, :2], axis=1).max())
    r_sim = np.linalg.norm(sim["pos"][:, :2] - AXIS, axis=1)
    r_obs = np.linalg.norm(PATH[frames_idx] - AXIS, axis=1)
    return float(np.nanmax(np.abs(r_sim - r_obs)[obs["valid"]]))


def left(sim):
    """Whether the twin's object left its support where the video's did not: a stay that went a centimetre; a ride
    that ends off the disc, or half as far again from the axis as the video's object ever is."""
    if KIND == "stay":
        return strayed(sim) > 0.01
    r_sim = np.linalg.norm(sim["pos"][:, :2] - AXIS, axis=1)
    return bool(r_sim.max() > min(DISC["radius"], 1.5 * float(np.nanmax(np.linalg.norm(PATH[frames_idx] - AXIS, axis=1)))))


def least_friction(flat, rounds=2):
    """A ladder of frictions about the closed form's bound in the copies, and a finer one where it matters.
    Returns {"holds_from": the least friction from which the twin's object stays on its support (left),
    "best": the friction of the least loss among those, "loss_best", "loss_held": the loss at the ladder's upper
    end, where nothing slips, "ladder": [(friction, strayed, loss, left)]}. Of a ride whose support reverses, the
    object slips outwards a little at every reversal: a twin with more friction keeps its first circle, one with
    less flies off, and the loss has a minimum where the twin slips as far as the video's object. That "best" is
    the friction to replay the video with, and not a measurement of it: support motion is estimated from the smoothed image trajectory."""
    lo, hi, table = 0.3 * LADDER, 3.0 * LADDER, []
    for _ in range(rounds):
        grid = np.geomspace(lo, hi, n_env)
        sims = rollout([dict(flat, **{MAIN: float(m)}) for m in grid])
        table += [(float(m), strayed(s), score(s)[0], left(s)) for m, s in zip(grid, sims)]
        table.sort()
        first = next((j for j in range(len(table)) if not any(r[3] for r in table[j:])), None)
        if first is None:
            return {"holds_from": None, "best": None, "ladder": table}
        best = first + int(np.argmin([r[2] for r in table[first:]]))
        lo, hi = table[max(min(first, best) - 1, 0)][0], table[min(best + 1, len(table) - 1)][0]
        if hi <= lo * 1.02:
            break
    top = [r[2] for r in table if r[0] >= 2.0 * LADDER] or [table[-1][2]]
    return {"holds_from": table[first][0], "best": table[best][0], "loss_best": table[best][2],
            "loss_held": float(np.median(top)), "strayed_best": table[best][1], "ladder": table}


def overlay(sim, path):
    """The video's free phase with the mask's outline in green and the sim silhouette's in red."""
    from scipy.ndimage import binary_erosion

    raw = Path(path).with_suffix(".raw.mp4")  # OpenCV's mp4v, which no browser plays
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
    for k, f in enumerate(frames_idx):
        img = frames[f].copy()
        for m, col in ((masks[k], (0, 255, 0)), (sim["mask"][k], (255, 0, 0))):
            m = np.kron(m, np.ones((STRIDE, STRIDE), bool))[:IMG_H, :IMG_W]
            img[m & ~binary_erosion(m, iterations=2)] = col
        vw.write(img[:, :, ::-1])
    vw.release()
    h264(raw, path)


def save(sim, flat, name, extra=None):
    total, parts = score(sim)
    np.savez_compressed(out / "sim_track.npz", **{k: v for k, v in sim.items() if k not in ("mask", "tilt")})
    res = {"loss": total, "parts": parts, "parameters": flat, "applied_mass": sim["mass"], "tilt_deg": sim["tilt"],
           "kind": KIND, **(extra or {})}
    (out / name).write_text(json.dumps(res, indent=1))
    overlay(sim, out / "overlay.mp4")
    return res


def h264(src, dst):
    import subprocess

    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "19",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
        Path(src).unlink()
    except (OSError, subprocess.CalledProcessError):
        Path(src).replace(dst)


def swing_of(sim, flat):
    """Of a pendulum's rollout: its angle from hanging at every frame (rad) and its period (s; None under three
    swings), by pendulum.period_of as for the video."""
    from calibrated_scene_reconstruction.physics.video.pendulum import period_of

    R = Rotation.from_quat(sim["quat_wxyz"] / np.linalg.norm(sim["quat_wxyz"], axis=1, keepdims=True), scalar_first=True)
    c = sim["pos"] + R.apply(np.asarray(ROUND["centre"], float)) - PIVOT
    th = np.arctan2(c @ SWING, c @ DOWN)
    return th, period_of(np.asarray(obs["t"], float), th)


def backgrounds(sim=None):
    """What the rendered twin is laid over at every frame of the phase (F, H, W, 3) float32: the video's background
    plate, or, where the background itself moves, the video's own frame with what the twin draws taken out of it:
    the object (filled from the plate), a board that tilts together with the object (the twin draws the board
    too), a pendulum with what it hangs by (sim: its rollout, for where it is)."""
    from PIL import Image

    from calibrated_scene_reconstruction.physics.video.mesh import plate as make_plate

    plate_file = rec / "plate.png"
    if KIND == "tilting":  # the room without the board
        plate = make_plate(frames, per["support_t"] | per["object"] | per["manipulator"], tw["ref_frame"]).astype(np.float32)
    elif KIND == "pendulum":  # the room without the pendulum: it is nowhere for long while it swings wide
        plate = np.median(frames[frames_idx[np.unique(np.linspace(0, len(frames_idx) - 1, 60).round().astype(int))]], 0).astype(np.float32)
    else:
        plate = (np.asarray(Image.open(plate_file).convert("RGB")) if plate_file.exists()
                 else make_plate(frames, per["object"] | per["manipulator"], tw["ref_frame"])).astype(np.float32)
    if KIND not in ("tilting", "ride", "pulled", "pendulum"):
        return [plate] * len(frames_idx), plate
    back = []
    for k, f in enumerate(frames_idx):
        m = per["object"][f].astype(np.uint8)
        if KIND == "tilting":
            m |= per["support_t"][f].astype(np.uint8)
        elif KIND == "pendulum" and sim is not None:  # along the twin's own pendulum, widely (the video's is there)
            c = observe.project(K, T, (sim["pos"][k] + Rotation.from_quat(sim["quat_wxyz"][k], scalar_first=True).apply(
                np.asarray(ROUND["centre"], float)))[None])[0][0]
            pv = observe.project(K, T, PIVOT[None])[0][0]
            cv2.line(m, tuple(int(round(v)) for v in pv), tuple(int(round(v)) for v in c), 1, 30)
            cv2.circle(m, tuple(int(round(v)) for v in c), int(1.8 * ROUND["radius"] * K[0, 0] / max(
                observe.world_to_cam(T, (sim["pos"][k])[None])[0][2], 0.05)) + 14, 1, -1)
        m = cv2.dilate(m, np.ones((11, 11), np.uint8))
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)[..., None]
        back.append(plate * soft + frames[f].astype(np.float32) * (1 - soft))
    return back, plate


def box_mesh(lo, hi, top=True):
    """A box from corner lo to corner hi (3,): (V (8, 3), F); top False leaves its upper face out."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = [[0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return V, np.array(F + ([[4, 5, 6], [4, 6, 7]] if top else []))


def add_board():
    """The support that tilts, drawn in its twin: a board with the picture the video shows of its top face
    (incline.board_look), in the frame the twin's object stands in. Returns its outline mesh (V, F) or None."""
    from calibrated_scene_reconstruction.sim.usd import add_mesh, color_material, textured_material
    from calibrated_scene_reconstruction.physics.video.incline import board_look

    look = board_look(frames, per["support_t"], per["object"] | per["manipulator"], K, BOARD_T, frames_idx)
    if look is None:
        return None
    (x0, y0), (x1, y1), thick = look["lo"], look["hi"], look["thick"]
    cv2.imwrite(str(out / "support.png"), look["texture"][:, :, ::-1])
    UsdGeom.Scope.Define(stage, "/World/Carrier")
    add_mesh(stage, "/World/Carrier/top", [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], [[0, 1, 2], [0, 2, 3]],
             UV=[[0, 0], [1, 0], [1, 1], [0, 1]],
             material=textured_material(stage, "/World/Looks/carrier", str((out / "support.png").resolve()), roughness=0.8))
    Vs, Fs = box_mesh([x0, y0, -thick], [x1, y1, -0.0002], top=False)
    add_mesh(stage, "/World/Carrier/sides", Vs, Fs, material=color_material(
        stage, "/World/Looks/carrier_side", tuple(0.6 * look["texture"].reshape(-1, 3).mean(0) / 255.0), roughness=0.8))
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Support/look")).MakeInvisible()
    return box_mesh([x0, y0, -thick], [x1, y1, 0.0])


def add_link(plate):
    """What the pendulum's bob hangs by, drawn in its twin: a rod from the pivot to the ball, as wide and of the
    colour the video shows between them (pendulum.link_look, on the frame the ball is seen best in), a part of the
    ball's body. Returns (its mesh in the ball's frame (V, F), its colour) or None when the video shows nothing
    there."""
    from calibrated_scene_reconstruction.sim.usd import add_mesh, color_material
    from calibrated_scene_reconstruction.physics.video.pendulum import link_look

    k = int(np.argmax(np.where(obs["valid"], per["score"][frames_idx], -1.0)))
    f = frames_idx[k]
    uvb, dist = observe.mask_stats(per["object"][f])[0], float(_z[k])
    got = link_look(frames[f], plate, fp["pivot_uv"], uvb, ROUND["radius"] * K[0, 0] / dist)
    if got is None:
        return None
    width, colour = got
    r = max(0.5 * width * dist / K[0, 0], 0.001)
    c = np.asarray(ROUND["centre"], float)
    up = -DOWN  # from the ball's centre to the pivot, hanging
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    ring = np.array([np.cos(a) * e1 + np.sin(a) * e2 for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]) * r
    V = np.concatenate([c + 0.8 * ROUND["radius"] * up + ring, c + L0 * up + ring])
    F = np.array([[i, (i + 1) % 8, 8 + i] for i in range(8)] + [[(i + 1) % 8, 8 + (i + 1) % 8, 8 + i] for i in range(8)])
    add_mesh(stage, "/World/Objects/object/link", V, F,
             material=color_material(stage, "/World/Looks/link", tuple(float(x) for x in colour), roughness=0.7))
    return (V, F), np.asarray(colour, float) * 255.0


def fill_of(Kc, Tc, R, p, mesh):
    """Where a mesh (V, F) at rotation R, position p shows in the image (H, W) bool."""
    return raster.fill(observe.project(Kc, Tc, mesh[0] @ R.T + p)[0][mesh[1]], (IMG_H, IMG_W))


def tint(render, where, want):
    """The gain (3,) that brings a rendered part to the colour `want` (3,) the video shows it in: the renderer's
    lights are not the room's."""
    if where.sum() < 20:
        return np.ones(3)
    return np.clip(np.asarray(want, float) / np.maximum(np.median(render[where], 0), 1.0), 0.15, 4.0)


def render(flat):
    """sim/render.mp4 and side_by_side.mp4 (module doc): the rollout rendered at every video frame of the phase, the
    empty support rendered once (at every frame, from a camera that moves), and the object and its shadow laid over
    the video's background. What carries the object and moves with it in the picture is drawn too: the board of a
    support that tilts, what a pendulum hangs by."""
    from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator

    cam = vstage.add_view(stage, dict(tw, T_world_cam=TK[0].tolist()), IMG_W, IMG_H)
    annot = rgb_annotator(cam.GetPath(), IMG_W, IMG_H)
    moving = KIND == "tilting"
    board = add_board() if KIND == "tilting" else None
    link = None
    if KIND == "pendulum":
        link = add_link(np.median(frames[frames_idx[np.unique(np.linspace(0, len(frames_idx) - 1, 60).round().astype(int))]], 0))
    shots = []

    def shoot(k=None):
        if moving and k is not None:
            vstage.view_pose(cam, TK[k])
        for _ in range(6):  # the path tracer's accumulation
            world.render()
        shots.append(grab(annot).astype(np.float32))

    sim = rollout([flat], on_frame=shoot)[0]
    body = shots[: len(obs["t"])]
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Objects/object")).MakeInvisible()
    for _ in range(4):
        world.render()
    shots.clear()
    for k in (range(len(body)) if moving else [None]):
        shoot(k)
    empty = [np.maximum(s, 1.0) for s in shots] * (1 if moving else len(body))
    back, plate = backgrounds(sim)
    R = Rotation.from_quat(sim["quat_wxyz"] / np.linalg.norm(sim["quat_wxyz"], axis=1, keepdims=True),
                           scalar_first=True).as_matrix()
    look = (_m["V"], _m["F"])
    # the renderer's lights are not the room's: the object's colours are scaled to the video's where both show it,
    # on the frame its mask and the twin overlap best (on the first one it may be a streak, or half in a gripper)
    k0 = int(np.argmax([observe.iou(sim["mask"][k], masks[k]) if obs["valid"][k] and masks[k].any() else 0.0
                        for k in range(len(body))]))
    both = cv2.erode((fill_of(K, TK[k0], R[k0], sim["pos"][k0], look) & per["object"][frames_idx[k0]]).astype(np.uint8),
                     np.ones((5, 5), np.uint8)).astype(bool)
    if both.sum() <= 50:  # a small object: without the margin
        both = fill_of(K, TK[k0], R[k0], sim["pos"][k0], look) & per["object"][frames_idx[k0]]
    gain = (np.clip(np.median(frames[frames_idx[k0]][both], 0) / np.maximum(np.median(body[k0][both], 0), 1.0), 0.15, 2.0)
            if both.sum() > 12 else np.ones(3))
    gain_board = gain_link = np.ones(3)
    if board is not None:  # the board's colours, where the video and the twin both show it bare
        kb = int(np.argmax(TK[:, 2, 3]))
        bare = (fill_of(K, TK[kb], np.eye(3), np.zeros(3), board) & per["support_t"][frames_idx[kb]]
                & ~cv2.dilate(per["object"][frames_idx[kb]].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
                & ~fill_of(K, TK[kb], R[kb], sim["pos"][kb], look))
        gain_board = tint(empty[kb], bare, np.median(frames[frames_idx[kb]][bare], 0) if bare.sum() >= 20 else np.ones(3))
    if link is not None:
        on = fill_of(K, TK[k0], R[k0], sim["pos"][k0], link[0]) & ~fill_of(K, TK[k0], R[k0], sim["pos"][k0], look)
        gain_link = tint(body[k0], on, link[1])
    slips = fit_path.exists() and json.loads(fit_path.read_text()).get("slips_as_the_video")
    label = (f"twin: PhysX, friction {flat[MAIN]:.3f} (static {max(flat['support.static_friction'], flat[MAIN]):.3f})"
             if KIND == "tilting" else f"twin: PhysX at friction {flat[MAIN]:.3f} (measured: >= {LADDER:.3f})" if slips
             else f"twin: PhysX, {LABEL} {flat[MAIN]:.3f}")
    raw = out / "render_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
    raw2 = out / "side_raw.mp4"
    vw2 = cv2.VideoWriter(str(raw2), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * IMG_W, IMG_H))

    def soft(mask, sigma=0.8):
        return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)[..., None]

    for k, f in enumerate(frames_idx):
        comp = back[k]
        if board is not None:  # the board as the twin has it, in place of the video's
            comp = comp * (1 - soft(fill_of(K, TK[k], np.eye(3), np.zeros(3), board))) + empty[k] * gain_board * soft(
                fill_of(K, TK[k], np.eye(3), np.zeros(3), board))
        shade = np.clip(body[k].mean(-1) / empty[k].mean(-1), 0.25, 1.0)  # the object's shadow on the support
        shade = cv2.GaussianBlur(shade, (0, 0), 1.5)[..., None]
        if KIND == "pendulum":
            shade = 1.0  # nothing in the twin to throw a shadow on
        if link is not None:
            a = soft(fill_of(K, TK[k], R[k], sim["pos"][k], link[0]))
            comp = comp * (1 - a) + body[k] * gain_link * a
            shade = 1.0
        sil = cv2.erode(fill_of(K, TK[k], R[k], sim["pos"][k], look).astype(np.uint8), np.ones((3, 3), np.uint8))
        alpha = soft(sil)                                             # inside the rendered object's edge
        comp = comp * shade * (1 - alpha) + body[k] * gain * alpha
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        vw.write(comp[:, :, ::-1])
        pair = np.hstack([frames[f], comp])
        for txt, x in (("video", 12), (label, IMG_W + 12)):
            cv2.putText(pair, txt, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(pair, txt, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw2.write(np.ascontiguousarray(pair[:, :, ::-1]))
    vw.release(), vw2.release()
    h264(raw, out / "render.mp4"), h264(raw2, out / "side_by_side.mp4")
    return sim


sp0 = params.Space([], params=PARAMS)
flat0 = sp0.flat([])
if args.set:
    flat0.update({k: float(v) for k, v in (kv.split("=") for kv in args.set.split(","))})
fit_path = out / "fit.json"
if fit_path.exists() and json.loads(fit_path.read_text()).get("kind", KIND) != KIND:
    fit_path.unlink()  # of another reading of this video
if args.scan:
    centre = json.loads(fit_path.read_text())["parameters"] if fit_path.exists() else flat0
    scan = {}
    for name in args.scan.split(","):
        sp = params.Space([name], fixed=centre, params=PARAMS)
        U = np.linspace(0, 1, n_env)[:, None]
        res = [score(s) for s in rollout([sp.flat(u) for u in U])]
        scan[name] = {"values": [float(sp.to_values(u)[0]) for u in U], "loss": [r[0] for r in res]}
        print(f"[scan] {name}: " + " ".join(f"{v:.3g}:{f:.2f}" for v, f in zip(scan[name]["values"],
                                                                           scan[name]["loss"])), flush=True)
    (out / "scan.json").write_text(json.dumps(scan, indent=1))
elif args.calibrate and LADDER is not None:
    lad = least_friction(flat0)
    print("[calibrate] friction: strayed mm, loss  " + " ".join(f"{m:.3f}:{1000 * d:.0f},{f:.1f}{'x' if o else ''}"
                                                                for m, d, f, o in lad["ladder"]), flush=True)
    if lad["holds_from"] is None:
        print(f"[calibrate] the twin leaves its support at every friction up to {lad['ladder'][-1][0]:.2f}", flush=True)
        flat = flat0
    else:
        # a friction, where the loss says the twin slips as the video's object does; else the bound, a little over
        slips = KIND == "ride" and lad["loss_best"] < lad["loss_held"] - max(1.0, 0.15 * lad["loss_held"])
        flat = dict(flat0, **{MAIN: lad["best"] if slips else 1.05 * lad["holds_from"]})
        lad["slips_as_the_video"] = bool(slips)
    r = save(rollout([flat])[0], flat, "fit.json", dict(lad, closed_form_lower_bound=LADDER))
    phys["sim"] = {"support.static_friction_lower_bound": lad["holds_from"], "closed_form_lower_bound": LADDER,
                   "loss": r["loss"], "parts": r["parts"], "mass_used": flat["object.mass"],
                   **{k: lad[k] for k in ("loss_best", "loss_held", "strayed_best", "slips_as_the_video", "ladder")
                      if k in lad}}
    if lad.get("slips_as_the_video"):  # the friction of the replay that looks most like the video: not a measurement
        phys["sim"]["replay_friction"] = lad["best"]
    (rec / "physics.json").write_text(json.dumps(phys, indent=1))
    if lad["holds_from"] is not None:
        print(f"[calibrate] the twin keeps its place from friction {lad['holds_from']:.3f} (closed form: >= {LADDER:.3f})"
              + (f"; it slips as the video's object does at {lad['best']:.3f} (loss {lad['loss_best']:.2f}, "
                 f"{lad['loss_held']:.2f} where nothing slips)" if lad["slips_as_the_video"] else "")
              + f", loss {r['loss']:.2f}", flush=True)
elif args.calibrate:
    com_z, tilt0 = upright_com(flat0) if KIND in ("slide", "incline") else (0.0, 0.0)
    if com_z is None:
        print(f"[calibrate] the twin topples ({tilt0:.0f} deg) wherever its centre of mass is put", flush=True)
        com_z = 0.0
    elif com_z < 0:
        print(f"[calibrate] the twin topples ({tilt0:.0f} deg) with the mesh's centre of mass: lowered by "
              f"{-com_z:.3f} m, where it slides upright as in the video", flush=True)
    flat0 = dict(flat0, **{"object.com_offset.z": com_z})
    sp = params.Space([n for n in (args.active or ACTIVE).split(",") if n], fixed=flat0, params=PARAMS)
    f_default = score(rollout([flat0])[0])[0]
    opt = optim.CMAES(sp.default_unit(), sigma=0.06, popsize=n_env, seed=args.seed)
    u, fu, best = optim.minimize(lambda U: np.array([score(s)[0] for s in rollout([sp.flat(x) for x in U])]), opt,
                                 iters=args.iters,
                                 log=lambda g, b: print(f"[calibrate] generation {g}: best {b:.2f}", flush=True))
    flat = sp.flat(u) if fu < f_default else flat0  # the search starts beside the closed form, not on it
    spread = [score(s)[0] for s in rollout([flat] * n_env)]  # the same parameters in every copy: the loss's noise
    if flat is not flat0 and score(rollout([flat])[0])[0] > f_default:  # a best that does not repeat is no best
        flat = flat0
    sim = rollout([flat])[0]
    r = save(sim, flat, "fit.json", {
        "active": sp.names(), "loss_closed_form": f_default, "generations": len(best),
        "loss_copies": [float(np.mean(spread)), float(np.std(spread))]})
    phys["sim"] = {{"support.friction": "support.kinetic_friction"}.get(MAIN, MAIN): flat[MAIN], "loss": r["loss"],
                   "parts": r["parts"], "collider": ROUND,
                   "loss_closed_form": f_default, "release_speed": release_speed(flat), "mass_used": flat["object.mass"],
                   "object.com_offset.z": flat["object.com_offset.z"], "tilt_deg": r["tilt_deg"]}
    more = ""
    if KIND == "tilting":
        phys["sim"]["support.static_friction"] = max(flat["support.static_friction"], flat[MAIN])
        more = f", static {phys['sim']['support.static_friction']:.3f} (closed form {mus0:.3f})"
    elif KIND == "pulled":
        phys["sim"]["support"] = {k: flat[k] for k in ("support.spin_up", "support.spin_start", "support.spin_top",
                                                       "support.axis_dx", "support.axis_dy")}
        phys["sim"]["friction_fitted"] = MAIN in sp.names()
        phys["sim"]["support"]["spin"] = SPIN
    elif KIND == "pendulum":
        phys["sim"]["period"] = swing_of(sim, flat)[1]
        phys["sim"].update({k: flat[k] for k in ("pendulum.damping_rate", "pendulum.dry_friction")})
        more = f", period {phys['sim']['period']:.4f} s (video {fp['period_mean']:.4f})"
    (rec / "physics.json").write_text(json.dumps(phys, indent=1))
    print(f"[calibrate] loss {f_default:.2f} (closed form, {LABEL} {mu0:.3f}) -> {r['loss']:.2f} ({LABEL} "
          f"{flat[MAIN]:.3f}, release {release_speed(flat):.2f} m/s{more})", flush=True)
elif not args.render:
    sim = rollout([flat0])[0]
    extra = {}
    if KIND == "pendulum":  # the twin's swing beside the video's: its period, and how far its bob is from the mask's
        th, period = swing_of(sim, flat0)
        e = np.linalg.norm(sim["uv"] - obs["centroid"], axis=1)[obs["valid"]]
        from calibrated_scene_reconstruction.physics.video.pendulum import period_of, swing

        th_fit = swing(np.radians(fp["theta0_deg"]), np.radians(fp["omega0_deg"]), L0, q["pendulum.damping_rate"]["value"],
                       q["pendulum.dry_friction"]["value"], np.asarray(obs["t"], float))
        tail = slice(-max(len(th) // 8, 2), None)
        extra = {"period": period, "period_closed_form": period_of(np.asarray(obs["t"], float), th_fit),
                 "swing_end_deg": float(np.degrees(np.abs(th[tail]).max())),
                 "swing_end_closed_form_deg": float(np.degrees(np.abs(th_fit[tail]).max())),
                 "angle_rms_deg": float(np.degrees(np.sqrt(np.mean((th - th_fit) ** 2)))),
                 "bob_rms_px": float(np.sqrt(np.mean(e ** 2))), "bob_typical_px": float(1.2 * np.sqrt(np.median(e ** 2))),
                 "bob_typical_px_closed_form": fp["typical_px"], "seconds": float(obs["t"][-1])}
    elif LADDER is not None:
        extra = {"strayed": strayed(sim)}
    r = save(sim, flat0, "replay.json", extra)
    print(f"[replay] loss {r['loss']:.2f} {r['parts']} (PhysX mass {r['applied_mass']:.3f} kg) {extra}", flush=True)
if args.render:
    flat = json.loads(fit_path.read_text())["parameters"] if fit_path.exists() else flat0
    sim = render(flat)
    print(f"[render] {out / 'side_by_side.mp4'} loss {score(sim)[0]:.2f}", flush=True)
stage.GetRootLayer().Export(str(out / "twin.usd"))
app.close()
