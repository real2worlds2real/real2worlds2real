"""Replay a real robot recording (real/<name>/, calibrated_scene_reconstruction.physics.replay.data) in the reconstructed twin and compare the object's
motion with the video; with --calibrate, fit the physical parameters (calibrated_scene_reconstruction.physics.replay.params) so that they match.

    tools/run_isaac.sh apps/real2sim_replay.py real/<name> [--params configs/<scene>/physics.json]
                       [--calibrate [--batch N] [--iters G] [--optimizer cmaes|cem] [--active p1,p2,...]
                                    [--repeats R [--jitter M]]]
                       [--scan p1,p2,... [--batch N]] [--synthetic truth.json]

Needs real/<name>/track.npz (python -m calibrated_scene_reconstruction.physics.replay.observe real/<name>, GPU), "base" [x, y, yaw_deg] (the real
robot's base in the twin frame; optional "base_z") and a twin: meta.json "session" (the twin reconstructed from a frame of
the same scene), else real/<name>/twin.json (python -m calibrated_scene_reconstruction.physics.replay.tabletop real/<name>: support box + primitive
object fitted to the first frame, sim.tabletop), whose calibration goes to real/<name>/replay/physics.json.
The recorded joint positions (resampled to the 120 Hz physics rate) are position targets of the robot's drives, their
finite-difference velocities the velocity targets (without them the PD drives lag by kd / kp = 0.1 s: 15 mm of flange
error at DROID's push speeds); the
object's centre (and TCP) is projected into the real camera at every video frame and scored by
calibrated_scene_reconstruction.physics.replay.loss.discrepancy. --calibrate rolls out one generation of parameter sets at once, set i in scene copy i
(the episode.py --batch layout: every collider, rigid body and material copied beside the room, a robot per copy),
then writes configs/<scene>/physics.json, which sim.scene.build_scene applies from then on.
--scan is the identifiability check: per named parameter, the loss at --batch values spread over its bounds (the
others at --params, e.g. a fit, else their defaults), one generation each -> replay/scan.json. A flat curve means the
recording says nothing about that parameter, whatever the optimizer returns.
--repeats R scores every candidate in R copies (mean loss; batch / R candidates per generation), the object of
repeat r > 0 shifted by --jitter metres on a circle (tabletop twin): an expected loss for knife-edge contacts, where
one copy's outcome flips with round-off and CMA-ES would chase lucky copies.
--synthetic truth.json replaces the video's track by the replay's own under those parameters (centroids and
silhouettes, every frame valid) and writes to replay_synthetic/: a sim-to-sim check of what this robot motion could
identify with perfect observations, before blaming the real data.
Writes real/<name>/replay/{sim_track.npz, loss.json[, history.json, physics.json]}.

Parameters are authored into USD and take effect at world.reset() (stop / play re-parses the physics scene); the replay
reads PhysX's mass of each copy back after the reset as a check (replay/loss.json "applied"; a warning when it
differs). The timeline is stopped before authoring: edits made while playing are undone by the reset.
gravity is one per physics scene: it cannot vary across copies, so it is only fitted in a sequential (--batch 1) run.
"""
import argparse
import json
from pathlib import Path

from calibrated_scene_reconstruction.physics.replay import params

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("--params", help="physics.json to replay with, the scans' centre, the calibration's start (default: the "
                "session's, else the defaults)")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--batch", type=int, default=16, help="scene copies per generation (--calibrate)")
ap.add_argument("--iters", type=int, default=30, help="generations")
ap.add_argument("--optimizer", default="cmaes", choices=("cmaes", "cem"))
ap.add_argument("--active", default=",".join(params.DEFAULT_ACTIVE), help="comma-separated parameter names")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scan", help="comma-separated parameters: loss over a grid of each (identifiability)")
ap.add_argument("--synthetic", help="physics.json of the ground truth: fit the sim's own track (sim-to-sim)")
ap.add_argument("--repeats", type=int, default=1, help="copies per candidate (--calibrate): mean loss")
ap.add_argument("--jitter", type=float, default=0.0, help="object shift (m) of repeats r > 0 (tabletop twin)")
ap.add_argument("--out", help="output directory under the recording (default replay, replay_synthetic)")
ap.add_argument('--object-route', choices=('agent_blender', 'template'), default='agent_blender')
args = ap.parse_args()

if args.object_route == 'agent_blender':
    from calibrated_scene_reconstruction.physics.replay.data import load as load_recording
    recording = load_recording(args.recording)
    if recording.meta.get('session'):
        from calibrated_scene_reconstruction.scene.session import Session
        from calibrated_scene_reconstruction.scene.objects.routes import ensure
        from calibrated_scene_reconstruction.core.paths import resolve
        scene_session = Session(resolve(recording.meta['session']))
        ensure(scene_session, require_reviewed=True)
        default_calibration = scene_session.config('physics.json', {})
    else:
        from calibrated_scene_reconstruction.physics.video.model_inputs import require_model
        require_model(recording.path, mode='replay', applied=True)
        default_calibration = {}
    active = args.active.split(',') if args.calibrate else []
    active += args.scan.split(',') if args.scan else []
    calibrations = [json.loads(Path(args.params).read_text()) if args.params else default_calibration]
    if args.synthetic:
        calibrations.append(json.loads(Path(args.synthetic).read_text()))
    params.require_fixed_geometry(active, calibrations)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import Sdf, Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.physics.replay import data, loss, observe, optim, tabletop  # noqa: E402
from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot, hand_links  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import SUPPORT_ROOTS, apply_robot, apply_scene, hand_prims  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.tabletop import author_twin, build_tabletop  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose  # noqa: E402

RATE = 120.0

rec = data.load(args.recording)
out = rec.path / (args.out or ("replay_synthetic" if args.synthetic else "replay"))
out.mkdir(parents=True, exist_ok=True)
obs = dict(np.load(out.parent / "track.npz"))
fps = 1.0 / float(np.median(np.diff(obs["t"])))
s = Session(ROOT / rec.meta["session"]) if rec.meta.get("session") else None
twin = None if s else json.loads((rec.path / "twin.json").read_text())
spec = load_spec(rec.meta["robot"])
side = next(iter(spec["arms"]))
target = rec.meta.get("object_name", s.scene_graph["task"]["target"] if s else twin["object"]["name"])
if "video_offset_s" not in rec.meta:  # cross-correlate video motion with joint speed
    frames, vfps = observe.read_video(rec.video)
    lag, corr = data.estimate_offset(np.arange(len(frames)) / vfps, data.video_motion(frames), rec.t,
                                     data.speed(rec.t, rec.q))
    rec.meta["video_offset_s"] = lag
    print(f"video offset {lag:.3f} s (correlation {corr:.2f}); set meta.json video_offset_s to override", flush=True)
    del frames

world = World(stage_units_in_meters=1.0, physics_dt=1 / RATE, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
if s:
    ts = build_scene(stage, prepare_scene(s.dir), prepare_variant(s.dir, None))
    support_top = float(s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"][1][2])
    rx, ry = ts.manifest["surfaces"]["room_x"], ts.manifest["surfaces"]["room_y"]
else:  # the robot base is the world frame of a dataset recording: base_z (default 0) replaces the support mount
    build_tabletop(stage, twin)
    support_top = float(rec.meta.get("base_z", 0.0))
    (x0, y0, _), (x1, y1, _) = twin["support"]["box"]
    rx, ry = (min(x0, -1.0), max(x1, 1.0)), (min(y0, -1.0), max(y1, 1.0))  # the robot's reach around its base
bx, by, byaw = rec.meta["base"]
n_env = args.batch if args.calibrate or args.scan else 1
ENVS = []  # [(offset, root, robot path, object path)]; env 0 is the room itself
add_robot(stage, spec, (bx, by), np.radians(byaw), support_top=support_top)
ENVS.append((np.zeros(3), "/World", "/World/Robot", f"/World/Objects/{target}"))
layer = stage.GetRootLayer()
cols = int(np.ceil(np.sqrt(max(n_env - 1, 1))))
for i in range(1, n_env):  # same layout as apps/episode.py --batch (lights stay behind: nothing is rendered)
    off = np.array([(1 + (i - 1) % cols) * (rx[1] - rx[0] + 2.0), ((i - 1) // cols) * (ry[1] - ry[0] + 2.0), 0.0])
    root = f"/World/Batch/env{i}"
    set_pose(UsdGeom.Xform.Define(stage, root).GetPrim(), off)
    for part in ("Room", "Static", "Objects", "Support"):
        if layer.GetPrimAtPath(f"/World/{part}"):
            Sdf.CopySpec(layer, f"/World/{part}", layer, f"{root}/{part}")
    rpath = f"/World/Batch/robot{i}"
    add_robot(stage, spec, (bx + off[0], by + off[1]), np.radians(byaw), path=rpath, support_top=support_top)
    ENVS.append((off, root, rpath, f"{root}/Objects/{target}"))
links = hand_links(stage, spec, side)
obj_prim = stage.GetPrimAtPath(f"/World/Objects/{target}")
object_meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(obj_prim) if p.IsA(UsdGeom.Mesh)]
parts = [np.asarray(mesh.GetPointsAttr().Get()) for mesh in object_meshes]
pts = np.concatenate(parts)
c_local = (pts.max(0) + pts.min(0)) / 2
if args.object_route == 'agent_blender':
    faces, offset = [], 0
    for mesh, vertices in zip(object_meshes, parts):
        if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
            raise ValueError('Reviewed Blender geometry must contain triangles')
        faces.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3) + offset)
        offset += len(vertices)
    mesh_faces = np.concatenate(faces)
    surf = pts
else:
    surf = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 4000), replace=False)]  # legacy silhouette splats


def sil(K, T, X, shape):
    """Reviewed meshes retain triangle coverage; explicit fallback keeps legacy scoring."""
    if args.object_route == 'agent_blender':
        return observe.mesh_silhouette(K, T, X, mesh_faces, shape)
    return observe.convex_silhouette(K, T, X, shape) if twin else observe.silhouette(K, T, X, shape)


def author(i, cal):
    """Write one parameter set into scene copy i (USD; effective at the next world.reset())."""
    _off, root, rpath, opath = ENVS[i]
    tag = f"real2sim_env{i}"
    sup = SUPPORT_ROOTS if i == 0 else tuple(f"{root}/{p}" for p in ("Room", "Support", "Static"))
    if twin:  # every copy is re-authored (it may hold another scale from the previous generation), before the CoM
        tw = tabletop.scaled(twin, rec.T_world_cam, depth_scale(cal), float(cal.get("twin", {}).get("support_dz", 0)))
        tw["object"]["pose"][:2] = (np.asarray(tw["object"]["pose"][:2]) + JITTER[i]).tolist()
        author_twin(stage, tw, root)
    apply_scene(stage, cal, opath, supports=sup, tag=tag)
    apply_robot(stage, cal, hand_prims(stage, links, rpath), robot_path=rpath, tag=tag)


def depth_scale(cal):
    return float(cal.get("twin", {}).get("depth_scale", 1.0)) if twin else 1.0


APPLIED = []
REPEATS = max(1, args.repeats) if args.calibrate else 1
_a = 2 * np.pi * (np.arange(n_env) % REPEATS) / max(REPEATS - 1, 1)  # copy i is repeat i % R; repeat 0 unshifted
JITTER = np.where((np.arange(n_env) % REPEATS > 0)[:, None], args.jitter * np.c_[np.cos(_a), np.sin(_a)], 0.0)
EE_LINK = rec.meta.get("ee_link") if rec.ee_pose is not None else None  # the link trajectory ee_pose records


def rollout(cals):
    """Play the recording in len(cals) copies at once; returns per copy the sim dict of calibrated_scene_reconstruction.physics.replay.loss."""
    world.stop()  # author on a stopped timeline: stop restores the state saved at play, which would undo edits
    for i, cal in enumerate(cals):
        author(i, cal)
    world.reset()
    robs = [Robot(world, spec, path=ENVS[i][2]) for i in range(len(cals))]
    objs = [SingleRigidPrim(ENVS[i][3]) for i in range(len(cals))]
    for o_ in objs:
        o_.initialize()
    APPLIED[:] = [float(o_.get_mass()) for o_ in objs]  # PhysX's mass after the reset: the authored value took effect
    al = [data.align(rec, RATE, time_scale=c.get("time_scale", 1.0)) for c in cals]
    cmd, vel = [], []  # position targets and their velocities (feed-forward: no drive lag behind the recording)
    for r_, a in zip(robs, al):
        q0 = r_.home()
        Q = np.repeat(q0[None], len(a["ts"]), 0)
        for j, n in enumerate(rec.joints):
            if n in r_.idx:
                Q[:, r_.idx[n]] = a["q"][:, j]
        if a["gripper"] is not None:
            Q = np.stack([r_.with_ee(q, float(g), side) for q, g in zip(Q, a["gripper"])])
        cmd.append(Q)
        vel.append(np.gradient(Q, 1.0 / RATE, axis=0) if len(Q) > 1 else np.zeros_like(Q))
        r_.teleport(Q[0])
    for _ in range(60):  # settle in the first recorded pose
        for r_, Q in zip(robs, cmd):
            r_.command(Q[0], np.zeros_like(Q[0]))
        world.step(render=False)
    n = max(len(Q) for Q in cmd)
    P = np.zeros((len(cals), n, 3))
    Rq = np.zeros((len(cals), n, 4))
    tcp = np.zeros((len(cals), n, 3))
    eep = np.zeros((len(cals), n, 3))
    links0 = np.zeros((n, len(robs[0].body_names), 3))  # every link of copy 0: the robot drawn over the video
    for k in range(n):
        for r_, Q, V in zip(robs, cmd, vel):
            r_.command(Q[min(k, len(Q) - 1)], V[min(k, len(V) - 1)])
        world.step(render=False)
        for i, (r_, o_) in enumerate(zip(robs, objs)):
            p, q = o_.get_world_pose()
            P[i, k], Rq[i, k] = np.asarray(p, float) - ENVS[i][0], np.asarray(q, float)
            tcp[i, k] = r_.tcp_pose(side)[0] - ENVS[i][0]
            if EE_LINK:
                eep[i, k] = r_.link_pose(EE_LINK)[0] - ENVS[i][0]
        tf = robs[0].view._physics_view.get_link_transforms()
        links0[k] = (tf.cpu().numpy() if hasattr(tf, "cpu") else np.asarray(tf))[0, :, :3]
    sims = []
    K, T = rec.K, rec.T_world_cam
    stride = int(obs.get("mask_stride", 2))
    shape = (int(rec.camera["height"]) // stride, int(rec.camera["width"]) // stride)
    Ks = K * np.array([[1 / stride], [1 / stride], [1]])
    for i, a in enumerate(al):
        sc = depth_scale(cals[i])  # the copy's object is the twin's scaled about its bottom centre
        ts_ = np.arange(n) / RATE + a["video_t"][0]
        Pv, Qv = loss.sample_at(ts_, P[i], obs["t"]), loss.sample_at(ts_, Rq[i], obs["t"])
        R = Rotation.from_quat(Qv / np.linalg.norm(Qv, axis=1, keepdims=True), scalar_first=True)
        centre = Pv + R.apply(sc * c_local)
        sim = {"t": obs["t"], "uv": observe.project(K, T, centre)[0],
               "uv_tcp": observe.project(K, T, loss.sample_at(ts_, tcp[i], obs["t"]))[0]}
        if "masks" in obs:
            sim["mask"] = np.stack([sil(Ks, T, Pv[f] + Rotation.from_quat(Qv[f], scalar_first=True)
                                        .as_matrix().dot(sc * surf.T).T, shape) for f in range(len(Pv))])
        sim.update({"p_world": centre, "pos": Pv, "quat_wxyz": Qv, "obj_pts_local": sc * surf})
        if EE_LINK:  # kinematic check: the replayed arm's link vs the recorded pose of the same link
            sim["ee_err"] = np.linalg.norm(loss.sample_at(np.arange(n) / RATE, eep[i], rec.t * cals[i].get(
                "time_scale", 1.0)) - rec.ee_pose[:, :3], axis=1)
        if i == 0:
            sim["link_pos"] = loss.sample_at(ts_, links0, obs["t"])
        sims.append(sim)
    return sims


GRASP_FRAME = None  # slip is scored only when the real TCP is known (trajectory ee_pose) and the gripper closes
if rec.ee_pose is not None and rec.gripper is not None and (rec.gripper > 0.5).any():
    vt = rec.t + rec.meta["video_offset_s"]
    obs["uv_tcp"] = observe.project(rec.K, rec.T_world_cam, loss.sample_at(vt, rec.ee_pose[:, :3], obs["t"]))[0]
    t_close = vt[int(np.argmax(rec.gripper > 0.5))] + 0.3  # the fingers need a moment to close
    GRASP_FRAME = min(int(np.searchsorted(obs["t"], t_close)), len(obs["t"]) - 1)


def score(sim):
    return loss.discrepancy(sim if GRASP_FRAME is not None else {k: v for k, v in sim.items() if k != "uv_tcp"}, obs,
                            fps, grasp_frame=GRASP_FRAME)


def replay(cal, tag=""):
    """One rollout with `cal`; writes replay/sim_track<tag>.npz and loss<tag>.json."""
    sim = rollout([cal])[0]
    total, parts = score(sim)
    np.savez_compressed(out / f"sim_track{tag}.npz", **{k: v for k, v in sim.items() if k != "mask"})
    ee = {"ee_err_mean_m": float(sim["ee_err"].mean()), "ee_err_max_m": float(sim["ee_err"].max())} \
        if "ee_err" in sim else {}
    (out / f"loss{tag}.json").write_text(json.dumps({"loss": total, "parts": parts, "physics": cal, **ee,
                                                     "applied": {"object.mass": APPLIED[0]}}, indent=1))
    print(f"[replay{tag}] loss {total:.2f} {parts} (PhysX mass {APPLIED[0]:.3f} kg) {ee}", flush=True)
    return total


DEFAULTS = params.nested(params.Space([]).flat([]))  # every parameter at its default (params.PARAMS)
if args.synthetic:  # the observation becomes the replay under the true parameters
    truth = json.loads(Path(args.synthetic).read_text())
    syn = rollout([truth])[0]
    obs.update({"valid": np.ones(len(obs["t"]), bool), "centroid": syn["uv"], "masks": syn["mask"]})
    (out / "truth.json").write_text(json.dumps(params.flatten(truth), indent=1))
    print(f"[synthetic] observation = replay under {args.synthetic}", flush=True)
PHYS_OUT = s.config_dir / "physics.json" if s else out / "physics.json"
cal0 = (json.loads(Path(args.params).read_text()) if args.params
        else s.config("physics.json", {}) if s else DEFAULTS)
if args.scan:
    scan = {}
    for name in args.scan.split(","):
        sp = params.Space([name], fixed=params.flatten(cal0))  # the others at --params (a fit), else defaults
        U = np.linspace(0, 1, n_env)[:, None]
        cals = [params.nested(sp.flat(u)) for u in U]
        sims = rollout(cals)
        res = [score(sim) for sim in sims]
        for k, sim in enumerate(sims):
            np.savez_compressed(out / f"scan_{name}_{k}.npz", **{key: v for key, v in sim.items() if key != "mask"})
        scan[name] = {"values": [float(sp.to_values(u)[0]) for u in U], "loss": [r[0] for r in res],
                      "parts": [r[1] for r in res], "applied_mass": list(APPLIED)}
        print(f"[scan] {name}: " + " ".join(f"{v:.3g}:{f:.2f}" for v, f in zip(scan[name]["values"],
                                                                           scan[name]["loss"])), flush=True)
    (out / "scan.json").write_text(json.dumps(scan, indent=1))
elif not args.calibrate:
    replay(cal0)
else:
    # --params: the start of the search and the values of the inactive parameters (e.g. a previous fit)
    sp = params.Space([n for n in args.active.split(",") if n], fixed=params.flatten(cal0) if args.params else None)
    pop = n_env // REPEATS
    opt = (optim.CMAES(sp.default_unit(), sigma=0.25, popsize=pop, seed=args.seed) if args.optimizer == "cmaes"
           else optim.CEM(sp.default_unit(), popsize=pop, seed=args.seed))
    hist = []

    def f_batch(U):
        f = np.zeros(len(U))
        for b in range(0, len(U), pop):
            chunk = U[b:b + pop]
            cals = [params.nested(sp.flat(u)) for u in chunk for _ in range(REPEATS)]
            sims = rollout(cals)
            f[b:b + len(chunk)] = np.reshape([score(sim)[0] for sim in sims], (len(chunk), REPEATS)).mean(1)
            bad = [i for i, c in enumerate(cals) if abs(APPLIED[i] - c["object"]["mass"]) > 1e-3 * c["object"]["mass"]]
            if bad:
                print(f"[calibrate] WARNING: copies {bad} simulate another mass than authored", flush=True)
        hist.append({"f": f.tolist(), "x": [sp.flat(u) for u in U], "applied_mass": list(APPLIED)})
        return f

    f_default = replay(params.nested(sp.flat(sp.default_unit())), "_default")

    u, fu, best = optim.minimize(f_batch, opt, iters=args.iters,
                                 log=lambda g, b: print(f"[calibrate] generation {g}: best {b:.2f}", flush=True))
    # the fit in every copy: copies differ only by their place in the scene (float round-off), so a spread here is
    # the contact's sensitivity to tiny perturbations, the noise floor of the loss
    spread = [score(sim)[0] for sim in rollout([params.nested(sp.flat(u))] * n_env)]
    phys = params.write_physics(PHYS_OUT, sp.flat(u), source=f"real/{rec.name}", loss=fu, loss_default=f_default,
                                loss_copies=[float(np.mean(spread)), float(np.std(spread))],
                                object_name=target, active=sp.names(), generations=len(best))
    (out / "history.json").write_text(json.dumps(hist))
    replay(params.nested(sp.flat(u)), "_calibrated")
    print(f"[calibrate] loss {f_default:.2f} (defaults) -> {fu:.2f} -> {PHYS_OUT}: {json.dumps(phys)}", flush=True)
app.close()
