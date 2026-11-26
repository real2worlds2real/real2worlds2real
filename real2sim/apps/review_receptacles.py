"""Agent review of the place targets (assets.receptacles) a scene samples: per kind, a few goals drawn as
apps/episode.py draws them (grasping.place.sample_goal around the task target, clear of the scene environment, the
objects on the support and the keep-out areas; no robot here, so no reach check), each built in the twin, checked
automatically (grasping.place_review) and rendered for the agent.

    tools/run_isaac.sh apps/review_receptacles.py sessions/<name> [--kinds coaster,tray,...] [--samples 3] [--seed 0]
                       [--robot NAME]

--robot: the target as that robot's episodes see it (a parallel gripper narrower than the target gets the largest
YCB cup that fits, as apps/episode.py does; a smaller object fits more kinds, a coaster for instance).

Kinds: every kind that belongs in the scene's room (scene graph task.place.room; also the ones the agent disabled, so
a decision can be revisited), or --kinds. Checks per sample:
  support   rays down through the footprint (receptacle absent) all hit the support's collider at its top (within
            3 mm): no floating over a lower surface or off the edge, no sinking into a higher one, nothing else under it
  inside    the footprint outline inside the support top by edge_margin
  clear     the footprint clear of the environment boxes, other objects and keep-out areas
  settle    the target object dropped onto the place surface and settled for 1 s rests on it (within 5 mm of its
            resting height on the bare support), upright (10 deg from its resting tilt), where it was put (2 cm) and
            inside the accept region
  holds     the place surface holds the object's base (a container leaves room for the fingers)
  size      the footprint within 20 % of the support top's area and 60 % of its shorter side
  texture   every textured part shows its texture at its real-world size (within 25 %)
  visible   the receptacle changes at least 0.1 % of the robot view (rendered with and without it)
  text      the task text names the kind's label
Writes sessions/<name>/qa/receptacles/<kind>.jpg (rows: samples; columns: photo view, robot view, close view, with
the object settled on the receptacle), qa/receptacles/sheet.jpg (one row per kind: robot and close view of each
sample) and qa/receptacles.json; prints one line per sample. The agent looks at the sheet and judges each kind on
that table in that room (plausible there? a real object's material, thickness, colour and scale?) and records the
decision in the scene graph's task.place (weights, keep_out, a `_review` note): docs/pick_place_policy.md.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--kinds", default=None, help="comma-separated kinds (default: every kind of the scene's room)")
ap.add_argument("--samples", type=int, default=3, help="goals per kind")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--robot", default=None, help="the target as this robot's episodes see it (gripper substitute)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 640, 400
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from calibrated_scene_reconstruction.assets import receptacles as rcp  # noqa: E402
from calibrated_scene_reconstruction.grasping import place_review as pr  # noqa: E402
from calibrated_scene_reconstruction.grasping.place import PlaceCfg, keep_out, receptacle_weights, sample_goal, support_obstacles  # noqa: E402
from calibrated_scene_reconstruction.grasping.targets import scene_env_world  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.receptacle import add_receptacle, remove_receptacle  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, look_at_quat, pinhole, set_pose  # noqa: E402

configure_render()
s = Session(args.session)
sg = s.scene_graph
place = sg["task"].get("place") or {}
target, support = sg["task"]["target"], sg["task"]["support"]
sbox = sg["furniture"][support]["box"]
top = float(sbox[1][2])
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = {}
if args.robot:
    from calibrated_scene_reconstruction.robots.spec import load_spec
    arm = next(iter(load_spec(args.robot)["arms"].values()))
    if not arm["ee"].get("bodex") and arm["ee"]["type"] == "parallel":
        from calibrated_scene_reconstruction.scene.cousins.variants import fit_target
        variant, sub = fit_target(s, variant, 0.8 * float(arm["ee"].get("max_width", 0.08)))
        print(f"[receptacles] target {target} as the {args.robot} episodes see it: {sub or 'unchanged'}", flush=True)
build_scene(stage, prepare_scene(s.dir), prepare_variant(s.dir, variant))
world.reset()
obj = SingleRigidPrim(f"/World/Objects/{target}")
p_obj, q_obj = (np.asarray(v, float) for v in obj.get_world_pose())   # the prim's origin is the object's bottom
for _ in range(120):  # how the object rests on the bare support: the settle check's reference
    world.step(render=False)
p_rest, q_rest = (np.asarray(v, float) for v in obj.get_world_pose())
_tp = stage.GetPrimAtPath(f"/World/Objects/{target}")
_xf = {op: _tp.GetAttribute(f"xformOp:{op}") for op in ("translate", "orient")}


def author_target(p, q):
    """The target's authored pose (its prim sits directly under /World/Objects), for the next world.reset(); authored
    with the simulation stopped (stopping restores what was authored while it ran). Moving the body while the
    simulation runs instead leaves the renderer showing it at its old pose too."""
    world.stop()
    _xf["translate"].Set(type(_xf["translate"].Get())(*[float(v) for v in p]))
    _xf["orient"].Set(type(_xf["orient"].Get())(*[float(v) for v in q]))


pts = np.concatenate([np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in
                      Usd.PrimRange(stage.GetPrimAtPath(f"/World/Objects/{target}")) if p.IsA(UsdGeom.Mesh)])
ext = pts.max(0) - pts.min(0)
r_obj = 0.5 * float(np.hypot(ext[0], ext[1]))
tf = TaskFrame(sg, p_obj)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
boxes = []
for root in ("/World/Objects", "/World/Static"):
    for prim in stage.GetPrimAtPath(root).GetChildren() if stage.GetPrimAtPath(root) else []:
        if prim.GetName() != target:
            b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            boxes.append((np.array(b.GetMin()), np.array(b.GetMax())))
env = [b for b in scene_env_world(s, "twin") if b[3] != f"room/{support}"]  # not the support itself
obst = support_obstacles(top, env, boxes) + keep_out(sg)
weights = receptacle_weights(sg)
kinds = args.kinds.split(",") if args.kinds else rcp.kinds_for(place.get("room"))
if place.get("room") is None:
    print("[receptacles] no task.place.room in the scene graph: every kind is sampled (name the room)", flush=True)
cfg = PlaceCfg(tries=2000)  # the kind's feasibility on this support: more draws than an episode's
sqi = get_physx_scene_query_interface()

# goals per kind, as the episodes sample them (one rng per kind: adding a kind leaves the others' draws alone)
plan = {}
for kind in kinds:
    rng = np.random.default_rng([args.seed, sum(map(ord, kind))])
    goals = [g for g in (sample_goal(tf, p_obj, ext, sbox, rng, cfg, obstacles=obst, support=support,
                                     weights={kind: 1.0}) for _ in range(args.samples)) if g is not None]
    plan[kind] = goals


def ray_down(xy):
    """(z, collider path) of the first collider under xy from 15 cm above the support top, (None, None) if none."""
    h = sqi.raycast_closest((float(xy[0]), float(xy[1]), top + 0.15), (0.0, 0.0, -1.0), 1.0)
    return (float(h["position"][2]), str(h["collision"])) if h["hit"] else (None, None)


eye, at = (np.asarray(v, float) for v in s.config("views.json")["third_person"])
K = pinhole(W, H)
cams = {"photo": add_camera(stage, "/World/Cams/review_photo", K, W, H, eye, at),
        "robot": add_camera(stage, "/World/Cams/review_robot", K, W, H, eye, at),
        "close": add_camera(stage, "/World/Cams/review_close", K, W, H, eye, at)}
annot = {k: rgb_annotator(c.GetPath(), W, H) for k, c in cams.items()}


def shoot(names, frames=12):
    """Frames with physics running (everything at rest by then)."""
    for _ in range(frames):
        world.step(render=True)
    return {k: grab(annot[k]) for k in names}


try:
    FONT = ImageFont.load_default(16)
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()


def label(img, lines, ok=True):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((8, 6 + 19 * i), t, fill=(255, 255, 255) if ok or i else (255, 90, 90), font=FONT,
               stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


out_dir = s / "qa" / "receptacles"
out_dir.mkdir(parents=True, exist_ok=True)
report = {"scene": s.name, "room": place.get("room"), "target": target, "robot": args.robot, "support": support,
          "r_obj": r_obj,
          "weights": weights, "keep_out": place.get("keep_out", {}),
          "obstacles": [[np.round(b[0], 3).tolist(), np.round(b[1], 3).tolist()] for b in obst], "kinds": {}}
sheet_rows = []
for kind, goals in plan.items():
    entry = {"enabled": weights.get(kind, 0) > 0, "weight": weights.get(kind, 0), "samples": []}
    report["kinds"][kind] = entry
    if not goals:
        entry["note"] = "no goal: the kind does not fit the object or the free support"
        print(f"[receptacles] {kind}: no goal (does not fit the target / the support)", flush=True)
        continue
    rows, tiles = [], []
    for i, g in enumerate(goals):
        rd = g.receptacle
        rec = rcp.rebuild(rd)
        remove_receptacle(stage, "/World/Goal/receptacle")
        author_target(p_obj, q_obj)
        world.reset()
        checks = pr.geometry_checks(rec, rd, sbox, r_obj, target, obst, cfg)
        checks["support"] = pr.support_check([ray_down(p) for p in pr.footprint_points(rec, rd)], top, support)
        UsdGeom.Xform.Define(stage, "/World/Goal")
        prim = add_receptacle(stage, "/World/Goal/receptacle", rec, rd["origin"], rd["yaw"])
        # the object dropped onto the place surface (its resting height on the bare support kept), 1 s of physics
        p_goal = np.asarray(g.position, float) + [0, 0, p_rest[2] - p_obj[2]]
        author_target(p_goal + [0, 0, 0.003], q_rest)
        world.reset()
        for _ in range(120):
            world.step(render=False)
        p_end, q_end = (np.asarray(v, float) for v in obj.get_world_pose())
        checks["settle"] = pr.settle_check(p_goal, p_end, q_end, rd, q_rest)
        c = np.array([*rcp.to_world(rd, [0, 0]), top + 0.5 * rd["rim_z"]])
        view = tf.origin - 0.45 * tf.forward + np.array([0, 0, 0.7])      # a head camera behind the approach edge
        set_pose(cams["robot"].GetPrim(), view, look_at_quat(view, c))
        close = c - 0.30 * tf.forward + 0.08 * tf.right + np.array([0, 0, 0.22])
        set_pose(cams["close"].GetPrim(), close, look_at_quat(close, c))
        UsdGeom.Imageable(prim).MakeInvisible()
        without = shoot(["robot"])["robot"]
        UsdGeom.Imageable(prim).MakeVisible()
        checks["visible"] = pr.visible_check(without, shoot(["robot"])["robot"])
        checks["text"]["ok"] &= prim.GetCustomDataByKey("calibrated_scene_reconstruction:receptacle") == kind
        imgs = shoot(["photo", "robot", "close"], frames=30)
        failed = [k for k, v in checks.items() if not v["ok"]]
        look = rec.info["look"]
        head = [f"{kind} #{i} {look}" + (f"  FAIL: {', '.join(failed)}" if failed else "  ok")]
        rows.append(np.concatenate([label(imgs["photo"], head + [checks["text"]["task"]], not failed),
                                    imgs["robot"], imgs["close"]], 1))
        tiles += [label(imgs["robot"], head, not failed), imgs["close"]]
        entry["samples"].append({"seed": rec.info["seed"], "look": look, "origin": rd["origin"], "yaw": rd["yaw"],
                                 "half": rd["half"], "place_z": rd["place_z"], "rim_z": rd["rim_z"],
                                 "ok": not failed, "failed": failed, "checks": checks})
        print(f"[receptacles] {kind} #{i} {look}: " + ("ok" if not failed else "FAIL " + ", ".join(failed)) +
              f" | settle dz {checks['settle']['dz_m'] * 1000:+.1f} mm tilt {checks['settle']['tilt_deg']:.1f} deg"
              f" | support gap {checks['support']['max_gap_m'] or 0:.4f} m | visible "
              f"{100 * checks['visible']['changed_frac']:.1f} % | size {checks['size']['footprint_m'][0]:.2f} x "
              f"{checks['size']['footprint_m'][1]:.2f} m", flush=True)
    Image.fromarray(np.concatenate(rows, 0)).save(out_dir / f"{kind}.jpg", quality=88)
    small = [np.asarray(Image.fromarray(t).resize((W // 2, H // 2))) for t in tiles]
    small += [np.zeros_like(small[0])] * (2 * args.samples - len(small))
    sheet_rows.append(np.concatenate(small, 1))
    entry["ok"] = all(x["ok"] for x in entry["samples"])
remove_receptacle(stage, "/World/Goal/receptacle")
if sheet_rows:
    Image.fromarray(np.concatenate(sheet_rows, 0)).save(out_dir / "sheet.jpg", quality=88)
s.write_json("qa/receptacles.json", report)
n = sum(len(e["samples"]) for e in report["kinds"].values())
bad = sum(not x["ok"] for e in report["kinds"].values() for x in e["samples"])
print(f"[receptacles] {n} samples of {len(plan)} kinds, {bad} failing -> {out_dir / 'sheet.jpg'}, "
      f"{s / 'qa' / 'receptacles.json'}", flush=True)
app.close()
