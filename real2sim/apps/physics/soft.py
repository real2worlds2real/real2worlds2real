"""A body that deforms (real/<name>/, calibrated_scene_reconstruction.physics.video.soft and .beam) replayed in PhysX: the beam's chain as an
articulation, let go where the video's was, and its silhouette in the video's camera compared with the video's masks.

    tools/run_isaac.sh apps/video_soft.py real/<name> [--calibrate [--batch N]] [--render]

Needs perception.npz and physics.json (python -m calibrated_scene_reconstruction.vidphys.{soft,beam}).

The twin is the chain of vidphys.beam, in PhysX's own joints: every lump (a stiff part, or a slice of a soft one) is a
link, every element four joints in a row at its middle (half a hinge, a slide along the body, a slide across it, the
other half of the hinge), whose drives are the element's springs: stretch E A / L, shear 1 / (L / (k G A) +
L^3 / (12 E I)), bending E I / L (twice that in each half). That is a Timoshenko element exactly, for small strains
of the element itself; the two half hinges put the slides in the frame turned half-way between the element's ends,
as the chain measures its strains against the element's chord, so the ends may turn far against each other (with
the whole hinge on one side a beam hanging at 70 degrees was 7 mm from the chain's own integration, 34 at worst).
The material lets go as the fit says it does (a standard linear solid): each drive is as stiff as the fast modulus
and pulls towards a target that creeps after the joint with the relaxation time, which is that solid written as
one spring with a moving rest.

Not PhysX's deformable bodies. Those were tried on the cantilever of foam and rubber cubes first (FEM, attached to
rigid cubes): a cube of foam of 2.5 g that carries 270 g of rubber stretched to two and four times its length and
never came to rest, and the sag changed with the step rate and the solver's iterations (240 Hz to 2 kHz, 40 to 255).

default      one rollout with physics.json's beam (sim/fit.json's moduli when it exists) -> sim/replay.json (PhysX
             beside the video's masks and beside the chain's own integration), sim/sim_track.npz
--calibrate  the two moduli fitted in PhysX itself: a grid of --batch articulations about the chain's values, let
             go together, and a finer one about the best, scored by how far the silhouette's middle is from the
             video's over the whole record -> sim/fit.json, and physics.json gains "sim". The two integrations of
             one chain differ where its ends turn far (a few millimetres; PhysX swings some percent slower), so
             PhysX's own moduli are what make its twin swing as the video does.
--render     the rollout rendered from the video's camera -> sim/render.mp4, sim/side_by_side.mp4: the video beside
             its twin. The twin's picture is the room without the body (every pixel from the frames in which the
             body is not on it) with the rendered body on it.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--batch", type=int, default=25, help="articulations let go together (a square number)")
ap.add_argument("--render", action="store_true")
ap.add_argument("--rate", type=float, default=480.0, help="physics steps a second")
args = ap.parse_args()

import numpy as np  # noqa: E402

from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.physics.video import beam  # noqa: E402
from calibrated_scene_reconstruction.physics.video.scene import read_frames  # noqa: E402

rec = Path(args.recording)
phys = json.loads((rec / "physics.json").read_text())
if phys.get("motion") != "beam" or not phys.get("measured") or not phys["beam"].get("modulus_fast"):
    raise SystemExit(f"{rec}: no fitted beam to replay")
B = phys["beam"]
frames, FPS = read_frames(rec)
IMG_H, IMG_W = frames.shape[1:3]
per = np.load(rec / "perception.npz")
T, OBJ = per["t"], per["object"]
chain = beam.chain_of(B)
view = beam.View(phys["K"], phys["T_world_cam"], B["plane"]["origin"], B["plane"]["along"], B["plane"]["across"],
                 B.get("tilt", 0.0))
RATE = args.rate
out = rec / "sim"
out.mkdir(exist_ok=True)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, **({"renderer": "RaytracedLighting", "width": IMG_W, "height": IMG_H}
                                           if args.render else {})})

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation, RigidPrim  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

# the chain's frame in the world: x along the body as it is held, z across it in its plane (upwards), y the rest
E_S = view.world(1.0, 0.0) - view.world(0.0, 0.0)
E_Z = view.world(0.0, 1.0) - view.world(0.0, 0.0)
E_Y = np.cross(E_Z, E_S)
R_WC = np.stack([E_S, E_Y, E_Z], 1)
Q_WC = Rotation.from_matrix(R_WC).as_quat()  # xyzw


def stiffness(E, nu):
    """Each element's springs for a modulus E, in its joints' order: (half hinge N m/rad, stretch N/m, shear N/m,
    half hinge), each (elements,)."""
    L, A, I = chain.L0, chain.area, chain.inertia_area
    GA = beam.SHEAR_K * E / (2 * (1 + nu)) * A
    return 2 * E * I / L, E * A / L, 1.0 / (L / GA + L ** 3 / (12 * E * I)), 2 * E * I / L


def xform(path, pos):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    x.AddOrientOp().Set(Gf.Quatf(float(Q_WC[3]), float(Q_WC[0]), float(Q_WC[1]), float(Q_WC[2])))
    return x.GetPrim()


def link(path, pos, mass, inertia):
    p = xform(path, pos)
    UsdPhysics.RigidBodyAPI.Apply(p)
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr(float(mass))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(inertia)] * 3))
    b = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
    b.CreateLinearDampingAttr(0.0)
    b.CreateAngularDampingAttr(0.0)
    b.CreateSleepThresholdAttr(0.0)
    return p


def joint(kind, path, body0, body1, pos0, pos1, axis):
    j = (UsdPhysics.PrismaticJoint if kind == "slide" else UsdPhysics.RevoluteJoint).Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in pos1]))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if kind == "slide" else "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(0.0)
    return j


world = World(stage_units_in_meters=1.0, physics_dt=1 / RATE, rendering_dt=1 / RATE)
stage = omni.usd.get_context().get_stage()
N_ENV = max(int(round(np.sqrt(args.batch))) ** 2, 2) if args.calibrate else 2  # the replay: the fit's and the chain's
centre = 0.5 * (chain.lumps[:, 0] + chain.lumps[:, 1])
light = 1e-3 * float(chain.mass.min())  # the three links inside an element carry nothing
DOFS = []  # joint names in stiffness()'s order, element by element
ROOTS = [f"/World/Beam{i}" for i in range(N_ENV)]  # copies in the one place: nothing in them collides
for root in ROOTS:
    UsdGeom.Xform.Define(stage, root)
    base = link(f"{root}/clamp", view.world(0.0, 0.0), 1.0, 1e-3)
    fx = UsdPhysics.FixedJoint.Define(stage, f"{root}/fix")
    fx.CreateBody1Rel().SetTargets([base.GetPath()])
    fx.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in view.world(0.0, 0.0)]))
    fx.CreateLocalRot0Attr(Gf.Quatf(float(Q_WC[3]), float(Q_WC[0]), float(Q_WC[1]), float(Q_WC[2])))
    for i in range(chain.n):
        link(f"{root}/lump{i}", view.world(centre[i], 0.0), chain.mass[i], chain.J[i])
    for e in range(len(chain.L0)):
        a, b = int(chain.a[e]), int(chain.b[e])
        mid = chain.x0[e] + 0.5 * chain.L0[e]
        body_a = base.GetPath() if a < 0 else f"{root}/lump{a}"
        off_a = mid - (0.0 if a < 0 else centre[a])
        for k in ("h", "x", "z"):
            link(f"{root}/e{e}{k}", view.world(mid, 0.0), light, light * 1e-4)
        joint("hinge", f"{root}/e{e}_bend0", body_a, f"{root}/e{e}h", (off_a, 0, 0), (0, 0, 0), "Y")
        joint("slide", f"{root}/e{e}_stretch", f"{root}/e{e}h", f"{root}/e{e}x", (0, 0, 0), (0, 0, 0), "X")
        joint("slide", f"{root}/e{e}_shear", f"{root}/e{e}x", f"{root}/e{e}z", (0, 0, 0), (0, 0, 0), "Z")
        joint("hinge", f"{root}/e{e}_bend1", f"{root}/e{e}z", f"{root}/lump{b}", (0, 0, 0),
              (mid - centre[b], 0, 0), "Y")
        if root == ROOTS[0]:
            DOFS += [f"e{e}_bend0", f"e{e}_stretch", f"e{e}_shear", f"e{e}_bend1"]
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
    art_api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root))
    art_api.CreateSleepThresholdAttr(0.0)
    art_api.CreateSolverPositionIterationCountAttr(32)
    art_api.CreateEnabledSelfCollisionsAttr(False)
SECONDS = float(T[-1] - B["release"])
AT = np.clip(np.round((T - B["release"]) * RATE).astype(int), 0, int(round(SECONDS * RATE)))  # the step of every frame
NU = B["poisson_assumed"]


def rollout(pars):
    """The chains' states q (len(pars), F, 3 n) at the video's frames: each let go straight and still at the
    release, with its own (modulus, fast modulus, relaxation time)."""
    n = len(pars)
    world.reset()
    art = Articulation(ROOTS[:n], name="beams")
    art.initialize()
    lumps = RigidPrim([f"{r}/lump{i}" for r in ROOTS[:n] for i in range(chain.n)], name="lumps")
    lumps.initialize()
    order = [list(art.dof_names).index(d) for d in DOFS]
    k_slow = np.stack([np.stack(stiffness(E, NU), 1).ravel() for E, _Ef, _r in pars])
    k_fast = np.stack([np.stack(stiffness(Ef, NU), 1).ravel() for _E, Ef, _r in pars])
    relax = np.array([r for _E, _Ef, r in pars])[:, None]
    kp = np.zeros((n, art.num_dof))
    kp[:, order] = k_fast
    art.set_gains(kps=kp, kds=np.zeros_like(kp))
    share = 1.0 - k_slow / k_fast  # of the fast spring, the part that lets go
    xv = np.zeros((n, len(DOFS)))  # where the dashpots are
    target = np.zeros((n, art.num_dof))
    steps = int(AT.max())
    keep = {int(k): np.nonzero(AT == k)[0] for k in np.unique(AT)}
    q = np.zeros((n, len(T), 3 * chain.n))
    q[:] = chain.rest[:3 * chain.n]
    o = view.world(0.0, 0.0)
    for k in range(1, steps + 1):
        x = np.asarray(art.get_joint_positions(), float)[:, order]
        xv += (x - xv) / relax / RATE
        target[:, order] = share * xv
        art.set_joint_position_targets(target)
        world.step(render=False)
        if k in keep:
            p, quat = lumps.get_world_poses()
            p, quat = np.asarray(p, float) - o, np.asarray(quat, float)
            ax = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).apply([1.0, 0.0, 0.0])
            qk = np.stack([p @ E_S, p @ E_Z, np.arctan2(ax @ E_Z, ax @ E_S)], 1).reshape(n, 3 * chain.n)
            q[:, keep[k]] = qk[:, None]
    return q


def middle_of(mask):
    import cv2

    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] > 0 else np.array([np.nan, np.nan])


CEN = np.array([middle_of(o) for o in OBJ])
REL = int(B["release_frame"])
USE = np.isfinite(CEN[:, 0]) & (T > B["release"] - 0.3)


def score(q):
    """Of one rollout (F, 3 n): {"loss": how far its silhouette's middle is from the video's, rms px over the
    record from the release on, "iou": (F,), "mid": (F, 2)}."""
    sil = [beam.silhouette(chain, x, view, OBJ[0].shape) for x in q]
    mid = np.array([middle_of(m) for m in sil])
    d = np.nan_to_num(np.linalg.norm(mid - CEN, axis=1)[USE], nan=50.0)
    return {"loss": float(np.sqrt(np.mean(d ** 2))), "mid": mid,
            "iou": np.array([beam.iou(m, o) for m, o in zip(sil, OBJ)])}


closed = (B["modulus"], B["modulus_fast"], B["relax_time"])
fit_path = out / "fit.json"
if args.calibrate:
    side = int(round(np.sqrt(N_ENV)))
    best, span, ladder = closed, 0.3, []
    for rnd in range(2):  # a grid about the chain's values, then a finer one about the best
        f = np.exp(np.linspace(-span, span, side))
        pars = [(best[0] * a, best[1] * b, best[2]) for a in f for b in f]
        res = [score(q) for q in rollout(pars)]
        ladder += [(p[0], p[1], r["loss"]) for p, r in zip(pars, res)]
        if rnd == 0:
            loss_closed = res[(side // 2) * side + side // 2]["loss"]  # the grid's middle is the chain's own
        best = pars[int(np.argmin([r["loss"] for r in res]))]
        span /= side - 1
    loss_fit = min(l for _a, _b, l in ladder)
    (fit_path).write_text(json.dumps({"kind": "beam", "parameters": {"modulus": best[0], "modulus_fast": best[1],
                                                                     "relax_time": best[2]},
                                      "loss": loss_fit, "loss_closed_form": loss_closed, "grid": ladder}, indent=1))
    phys["sim"] = {"modulus": best[0], "modulus_fast": best[1], "relax_time": best[2], "loss": loss_fit,
                   "loss_closed_form": loss_closed, "closed_form": {"modulus": closed[0], "modulus_fast": closed[1]}}
    (rec / "physics.json").write_text(json.dumps(phys, indent=1))
    print(f"[calibrate] loss {loss_closed:.2f} -> {loss_fit:.2f} px; modulus {closed[0] / 1e3:.1f} -> "
          f"{best[0] / 1e3:.1f} kPa, swinging {closed[1] / 1e3:.1f} -> {best[1] / 1e3:.1f} kPa", flush=True)
PAR = closed
if fit_path.exists():
    fp = json.loads(fit_path.read_text())["parameters"]
    PAR = (fp["modulus"], fp["modulus_fast"], fp["relax_time"])
runs = rollout([PAR] if PAR == closed else [PAR, closed])
Qf = runs[0]
Qc = chain.swing(T, closed[0], NU, closed[1] - closed[0], closed[2], 0.0, B["release"])[0]
sc = score(Qf)
first = np.arange(REL, min(REL + int(round(1.5 / float(np.median(np.diff(T))))), len(T)))
after = T > B["release"] + 0.1
tip = lambda q: chain.axis(q, [chain.length])[0]
xz = [k for k in range(Qf.shape[1]) if k % 3 != 2]
rep = {"kind": "beam", "rate": RATE, "seconds": SECONDS,
       "modulus": PAR[0], "modulus_fast": PAR[1], "relax_time": PAR[2], "calibrated": PAR != closed,
       "loss": sc["loss"], "sag": float(-tip(Qf[-1])[1]), "sag_chain": float(-tip(Qc[-1])[1]),
       "period": beam.period_of(T[after], sc["mid"][after, 1]), "period_video": B["period"],
       "period_chain": B["period_model"],
       "iou_hanging": float(np.mean(sc["iou"][-beam.last_frames(T):])), "iou_swing": float(np.mean(sc["iou"][first])),
       "iou_hanging_chain": B["iou_hanging"], "iou_swing_chain": B["iou_swing"],
       # the two integrations of the one chain with the same moduli: PhysX's joints and beam.Chain.swing's
       # elements, lump by lump
       "against_chain_mm": float(1e3 * np.sqrt(np.mean((runs[-1] - Qc)[:, xz] ** 2))),
       "against_chain_worst_mm": float(1e3 * np.abs((runs[-1] - Qc)[:, xz]).max())}
(out / "replay.json").write_text(json.dumps(rep, indent=1))
np.savez_compressed(out / "sim_track.npz", t=T, q=Qf, q_chain=Qc, iou=sc["iou"])
print(f"[replay] loss {rep['loss']:.2f} px; PhysX sag {rep['sag'] * 1e3:.0f} mm, period {rep['period']:.3f} s (video "
      f"{rep['period_video']:.3f}), IoU hanging {rep['iou_hanging']:.2f} swinging {rep['iou_swing']:.2f} (chain "
      f"{rep['iou_hanging_chain']:.2f} {rep['iou_swing_chain']:.2f}); with the chain's moduli PhysX is "
      f"{rep['against_chain_mm']:.1f} mm rms from the chain, {rep['against_chain_worst_mm']:.1f} at worst", flush=True)


def h264(src, dst):
    import subprocess

    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "21",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
        Path(src).unlink()
    except (OSError, subprocess.CalledProcessError):
        Path(src).replace(dst)


def skin(q, s):
    """World points (len(s) * 4, 3) of the body's cross-sections at the stations s (m from the clamp, at rest)."""
    s = np.asarray(s, float)
    if s[-1] <= 0:  # the clamp's own part of the body, before the chain: where it was held
        c, d = np.stack([s, np.zeros(len(s))], 1), np.tile([1.0, 0.0], (len(s), 1))
    else:
        s = np.clip(s, 1e-9, chain.length - 1e-9)
        c = chain.axis(q, s)
        d = chain.axis(q, np.clip(s + 1e-3, s[0], s[-1])) - chain.axis(q, np.clip(s - 1e-3, s[0], s[-1]))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    h, w = 0.5 * chain.side, 0.5 * chain.area / chain.side
    corners = [view.world(c[:, 0] - h * sz * d[:, 1], c[:, 1] + h * sz * d[:, 0], np.full(len(s), w * sw))
               for sz, sw in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    return np.stack(corners, 1).reshape(-1, 3)


def skin_faces(n):
    """Triangles of n sections of four corners joined along the body, and its two ends."""
    F = []
    for j in range(n - 1):
        for c in range(4):
            a, b, a2, b2 = 4 * j + c, 4 * j + (c + 1) % 4, 4 * (j + 1) + c, 4 * (j + 1) + (c + 1) % 4
            F += [[a, b, b2], [a, b2, a2]]
    e = 4 * (n - 1)
    return np.array(F + [[0, 2, 1], [0, 3, 2], [e, e + 1, e + 2], [e, e + 2, e + 3]])


def room():
    """The picture without the body: every pixel's median over the frames in which the body is not on it, and
    where it always is (at the clamp), what surrounds it."""
    import cv2

    pick = np.unique(np.linspace(0, len(frames) - 1, 80).round().astype(int))
    on = np.stack([cv2.dilate(OBJ[k].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) for k in pick])
    stack = frames[pick].astype(np.float32)
    stack[on] = np.nan
    with np.errstate(all="ignore"):
        plate = np.nanmedian(stack, 0)
    never = ~np.isfinite(plate[..., 0])
    plate = np.nan_to_num(plate).astype(np.uint8)
    if never.any():
        plate = cv2.inpaint(plate, never.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return plate.astype(np.float32)


def render():
    import cv2

    from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator
    from calibrated_scene_reconstruction.sim.usd import add_mesh, color_material
    from calibrated_scene_reconstruction.physics.video import raster
    from calibrated_scene_reconstruction.sim import video_physics_stage as vstage

    cam = vstage.add_view(stage, {"K": phys["K"], "T_world_cam": phys["T_world_cam"]}, IMG_W, IMG_H)
    annot = rgb_annotator(cam.GetPath(), IMG_W, IMG_H)
    stations, x = [], 0.0
    if B.get("clamp_lead", 0.0) > 0.2 * chain.side:  # a stiff part before the first soft one: it stays with the clamp
        stations.append(np.array([-B["clamp_lead"], 0.0]))
    for p in B["parts"]:
        stations.append(np.linspace(x, x + p["length"], 9 if p["soft"] else 2))
        x += p["length"]
    faces = [skin_faces(len(s)) for s in stations]

    def masks_of(q):
        return [raster.fill(view.project(skin(q, s))[f], OBJ[0].shape) for s, f in zip(stations, faces)]

    # every part in the colour the video shows it in while it is held (sharp, the parts side by side), read along
    # the middle of the long face the camera sees: a part's whole picture will not do, its cut towards the camera
    # lies over its neighbour's face
    start = max(int(B["release_frame"]) - 20, 0)
    ref = max(int(B["release_frame"]) - 4, start)
    h, w = 0.5 * chain.side, 0.5 * chain.area / chain.side
    mid0 = view.world(0.5 * chain.length, 0.0)
    fz, fw = max([(h, 0.0), (-h, 0.0), (0.0, w), (0.0, -w)],
                 key=lambda f: (view.world(0.5 * chain.length, f[0], f[1]) - mid0) @ (view.T[:3, 3] - mid0))

    def along(st, image):
        """The pixels of `image` along the middle of that face of the part with stations st, held straight."""
        ss = np.linspace(st[0] + 0.2 * (st[-1] - st[0]), st[-1] - 0.2 * (st[-1] - st[0]), 15)
        uv = np.round(view.pixels(ss, np.full(15, 0.8 * fz), np.full(15, 0.8 * fw))).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
        return image[uv[ok, 1], uv[ok, 0]].astype(np.float32)

    want = [np.median(along(st, frames[ref]), 0) for st in stations]
    prims = []
    for k, (s, f) in enumerate(zip(stations, faces)):
        # the video's colours are display values, a material's are linear: as they are, black and white come out
        # two greys
        mat = color_material(stage, f"/World/Looks/part{k}", tuple(float(c / 255.0) ** 2.2 for c in want[k]),
                             roughness=0.8)
        prims.append(add_mesh(stage, f"/World/Body/part{k}", skin(Qf[0], s), f, material=mat))
    shots = []
    for k in range(start, len(T)):
        for prim, s in zip(prims, stations):
            UsdGeom.Mesh(prim).GetPointsAttr().Set([Gf.Vec3f(*map(float, v)) for v in skin(Qf[k], s)])
        for _ in range(8 if k == start else 5):  # the path tracer's accumulation
            world.render()
        shots.append(grab(annot).astype(np.float32))
    # the renderer's lights are not the room's: one gain for the picture, that brings the parts' rendered colours
    # on that frame to the video's
    seen = np.concatenate([along(st, shots[ref - start]) for st in stations])
    real = np.concatenate([along(st, frames[ref]) for st in stations])
    gain = np.clip((seen * real).sum(0) / np.maximum((seen * seen).sum(0), 1.0), 0.15, 4.0)
    plate = room()
    label = (f"twin: PhysX, soft parts {PAR[0] / 1e3:.1f} kPa at rest, {PAR[1] / 1e3:.0f} kPa swinging "
             f"({PAR[2] * 1e3:.0f} ms)")
    raw, raw2 = out / "render_raw.mp4", out / "side_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))
    vw2 = cv2.VideoWriter(str(raw2), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2 * IMG_W, IMG_H))
    for j, k in enumerate(range(start, len(T))):
        comp = plate.copy()
        body = np.any(masks_of(Qf[k]), 0)
        a = cv2.GaussianBlur(cv2.erode(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32),
                             (0, 0), 0.8)[..., None]
        comp = comp * (1 - a) + shots[j] * gain * a
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        vw.write(comp[:, :, ::-1])
        pair = np.hstack([frames[k], comp])
        for txt, x0 in (("video", 12), (label, IMG_W + 12)):
            cv2.putText(pair, txt, (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(pair, txt, (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw2.write(np.ascontiguousarray(pair[:, :, ::-1]))
    vw.release(), vw2.release()
    h264(raw, out / "render.mp4"), h264(raw2, out / "side_by_side.mp4")
    print(f"[render] {out / 'side_by_side.mp4'}", flush=True)


if args.render:
    render()
app.close()
