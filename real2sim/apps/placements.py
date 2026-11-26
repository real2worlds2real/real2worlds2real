"""Sample base placements of a robot (configs/robots/<robot>.json) around the target and score three constraints:
(1) collision-free: the robot swept over home -> pre-grasp -> grasp -> lift (sim.sweep) must not penetrate the room or
a background object; movable furniture (stools) in its way slides aside, (2) IK reach to pre-grasp and grasp,
(3) coverage of content that was actually observed in the photo by the robot's head camera (robot pixels excluded;
robots without a head camera, e.g. arms with a wrist camera, are scored on (1) and (2) only).

    tools/run_isaac.sh apps/placements.py sessions/<name> [--robot NAME] [--n 16] [--min-coverage 0.55]

Bases are sampled in the task frame (sim.taskframe: the support edge with free floor, facing the support) within the
spec's standoff / lateral / yaw ranges; the reference grasp is the first candidate of the robot's grasp family. When
the photo shows the real robot (scene graph `photo_robot.base`, from recon.multiview), the first candidate is its base.
Writes placements/<robot>.json (Vega: placements/placements.json, as before) + placements/<robot>/candXX_*.jpg.
"""
import argparse

DEFAULT_ROBOT = "vega_1p_sharpa"
ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default=DEFAULT_ROBOT)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-coverage", type=float, default=0.55)
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 768, 480
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.sim.robot_articulation import (Robot, add_cameras, add_robot, clear_support, hand_links,  # noqa: E402
                                       hand_points, place_robot)
from calibrated_scene_reconstruction.grasping.targets import hand_name  # noqa: E402
from calibrated_scene_reconstruction.robots.grasps import candidates  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.render_robot import capture_spec  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import world_points  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render, robot_semantics  # noqa: E402
from calibrated_scene_reconstruction.sim.sweep import RobotSweep, children, clear_robot, summarize, support_collisions  # noqa: E402
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, pinhole  # noqa: E402

configure_render()
s = Session(args.session)
spec = capture_spec(s, load_spec(args.robot))
side = next(iter(spec["arms"]))
arm = spec["arms"][side]
default = args.robot == DEFAULT_ROBOT
out = s / "placements" / ("" if default else args.robot)
out.mkdir(parents=True, exist_ok=True)
sg = s.scene_graph
target = sg["task"]["target"]
support_top = float(sg["furniture"][sg["task"]["support"]]["box"][1][2])

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = prepare_variant(s.dir, {})
ts = build_scene(stage, prepare_scene(s.dir), variant)
pose0 = SingleRigidPrim(f"/World/Objects/{target}").get_world_pose() \
    if stage.GetPrimAtPath(f"/World/Objects/{target}").IsValid() else None
obj0 = np.asarray(pose0[0], float) if pose0 is not None else None
tf0 = TaskFrame(sg, obj0 if obj0 is not None else np.mean(sg["furniture"][sg["task"]["support"]]["box"], 0))
x0, y0, yaw0 = tf0.sample_base(spec, np.random.default_rng(12345))  # a first pose, replaced per sample
add_robot(stage, spec, (x0, y0), yaw0, support_top=support_top)
cams = add_cameras(stage, spec, size=(W, H))
head_name = next((c["name"] for c in spec["cameras"] if c.get("role") == "head"), None)
head = cams[head_name][0] if head_name else None  # coverage is a head-camera criterion (a wrist camera sees the arm)
over = add_camera(stage, "/World/Cams/over", pinhole(W, H, 0.55), W, H, *s.config("views.json")["overview"])
robot_semantics(stage)
support_collisions(stage, False)
world.reset()
rob = Robot(world, spec)
rob.teleport(rob.home())
for _ in range(60):
    world.step(render=False)
obj = SingleRigidPrim(f"/World/Objects/{target}")
p_obj = np.asarray(obj.get_world_pose()[0], float)
if obj0 is not None and np.linalg.norm(p_obj - obj0) > 0.05:
    # the robot's first pose stands at the edge and its home posture can knock the target off: put it back where the scene built it
    print(f"target moved {np.linalg.norm(p_obj - obj0):.2f} m while the robot settled at its first pose: put back",
          flush=True)
    obj.set_world_pose(*pose0)
    obj.set_linear_velocity(np.zeros(3))
    obj.set_angular_velocity(np.zeros(3))
    p_obj = obj0
pts = world_points(stage.GetPrimAtPath(f"/World/Objects/{target}"), 20000)  # a twin mesh's geom or a kit's parts
ext = pts.max(0) - pts.min(0)
tf = TaskFrame(sg, p_obj)
body = RobotSweep(stage, rob)
bodex = hand_name(spec, side) is not None
hl = hand_links(stage, spec, side)
c0 = clear_support(candidates(tf, p_obj, ext, arm)[:1], hand_points(rob, side, body), tf.support_top)[0]
if bodex:  # synthesized grasps come from any side: the reference is the hand above the target (position only), and
    # the hand's own contacts are left to the trajectory planner and physics (apps/episode.py)
    p_above = p_obj + np.array([0.0, 0.0, ext[2] + 0.10])
    c0 = {"family": "above_target", "params": [round(float(ext[2] + 0.10), 3)], "pre": p_above, "grasp": p_above,
          "lift": p_above, "R": np.eye(3)}
head_rgb, head_seg = rgb_annotator(head.GetPath(), W, H, semantic=True) if head else (None, None)
over_rgb = rgb_annotator(over.GetPath(), W, H)
print(f"robot {spec['name']}: task frame forward {np.round(tf.forward, 2)}, edge free {tf.edge_info['free']:.2f}, "
      f"reference grasp {c0['family']} {c0['params']}", flush=True)


def render(n):
    for _ in range(n):
        world.step(render=True)


furniture = {p: UsdGeom.Xformable(p).GetOrderedXformOps()[0].Get() for p in children(stage, "/World/Support")}
rng = np.random.default_rng(args.seed)
results = []
real_base = sg.get("photo_robot", {}).get("base")  # where the real robot stood in the photo (recon.multiview)
for i in range(args.n):
    bx, by, yaw = tf.sample_base(spec, rng)
    if i == 0 and real_base:  # tried first: the twin's robot where the real one is
        (bx, by), yaw = real_base["xy"], float(np.radians(real_base["yaw_deg"]))
    res = {"id": i, "base": [bx, by, float(np.degrees(yaw))]}
    for prim, t in furniture.items():  # furniture back where the scene put it
        prim.SetActive(True)
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(t)
    place_robot(stage, spec, (bx, by), yaw, support_top=support_top)
    world.reset()
    rob = Robot(world, spec)
    q_home = rob.home()
    rw = 0.0 if bodex else 0.6  # position-only reach for synthesized grasps
    q_pre, e1 = rob.solve_ik(side, c0["pre"], c0["R"], q_home, iters=200, rot_weight=rw)
    q_g, e2 = rob.solve_ik(side, c0["grasp"], c0["R"], q_pre, iters=200, rot_weight=rw)
    q_l, e3 = rob.solve_ik(side, c0["lift"], c0["R"], q_g, iters=200, rot_weight=rw)
    if bodex:
        e1, e2, e3 = ((e[0], 0.0) for e in (e1, e2, e3))
    res["ik_err"] = [list(e1), list(e2), list(e3)]
    res["reachable"] = bool(max(e1[0], e2[0]) < 0.01 and max(e1[1], e2[1]) < 0.05)
    sweep = body.restart(rob)
    for qa, qb, n in ((q_home, q_pre, 8), (q_pre, q_g, 4), (q_g, q_l, 4)):  # the motion, joint-space waypoints
        for a in np.linspace(0, 1, n + 1)[(qa is not q_home):]:
            rob.teleport(qa + a * (qb - qa))
            world.step(render=False)
            sweep.sample()
    rc = clear_robot(ts, sweep, target)
    if bodex:  # hand contacts near the target are the grasp planner's business
        rc["hand"] = [x for x in rc["penetration"] if x["link"] in hl]
        rc["penetration"] = [x for x in rc["penetration"] if x["link"] not in hl]
        rc["ok"] = not rc["penetration"]
    res["robot_clearance"] = rc
    res["stools_moved"] = [m["item"] for m in rc["moved"]]
    res["collision_free"] = rc["ok"]
    rob.teleport(q_home)
    render(8)
    imgs = {"over": grab(over_rgb)}
    if head:
        imgs["head"] = grab(head_rgb)
        ts.coverage_mode(True)
        render(4)
        cov_img = grab(head_rgb)
        ts.coverage_mode(False)
        sd = head_seg.get_data()
        robot_ids = [int(k) for k, v in sd["info"]["idToLabels"].items() if "robot" in str(v)]
        robot_px = np.isin(np.asarray(sd["data"]), robot_ids)
        valid = ~robot_px
        res["coverage"] = float(((cov_img.mean(-1) > 127) & valid).sum() / max(valid.sum(), 1))
        cov_img[robot_px] = [90, 90, 160]
        imgs["cov"] = cov_img
    else:
        res["coverage"] = 1.0
    res["coverage_ok"] = res["coverage"] >= args.min_coverage
    # the real robot's own base needs no coverage: coverage keeps a sampled base from looking at what the photo never
    # saw, and the real robot's head saw this view
    real = i == 0 and bool(real_base)
    res["accepted"] = bool(res["collision_free"] and res["reachable"] and (res["coverage_ok"] or real))
    if real and res["accepted"] and not res["coverage_ok"]:
        res["coverage_waived"] = "the real robot's base"
    for k, im in imgs.items():
        Image.fromarray(im).save(out / f"cand{i:02d}_{k}.jpg", quality=92)
    results.append(res)
    print(f"cand {i}: base=({bx:.2f},{by:.2f},{np.degrees(yaw):.0f}) free={res['collision_free']} "
          f"reach={res['reachable']} cov={res['coverage']:.2f} -> {'ACCEPT' if res['accepted'] else 'reject'} | "
          f"{summarize(rc)}", flush=True)
if not any(r["accepted"] for r in results):  # coverage is relative to the scene: keep the best-covered feasible bases
    ok = [r for r in results if r["collision_free"] and r["reachable"]]
    if ok:
        bar = 0.8 * max(r["coverage"] for r in ok)
        for r in ok:
            if r["coverage"] >= bar:
                r.update(accepted=True, coverage_relaxed=True)
                print(f"cand {r['id']}: ACCEPT (coverage {r['coverage']:.2f} >= 0.8 x the best feasible)", flush=True)
s.write_json(f"placements/{'placements' if default else args.robot}.json",
             {"robot": spec["name"], "object": p_obj.tolist(), "task_frame": tf.to_dict(), "candidates": results})
app.close()
