"""The robot's visible surface at the joint angles it held while the scene was filmed, as points in its base frame.

    tools/run_isaac.sh apps/robot_cloud.py sessions/<name> [--robot NAME] [--state configs/<name>/robot_state.json]

The state is the robot's joint readout during the capture (configs/<name>/robot_state.json: a {joint: rad} map, or a
hardware snapshot whose "joints" list holds {joint, position_rad}). The spec's USD is posed at it (joints the readout
lacks keep the spec's home; hand joints matched by name, else by their suffix on the same side) and every visual
mesh's vertices go to the base frame through its link's PhysX pose. recon.multiview fits them to the photo and its views for the scale, recon.robot_fit to the splat. Writes
parsing/robot_cloud.npz (points, link index per point, link names, the joint vector) and robot_cloud.json (matched,
unmatched joints).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default=None)
ap.add_argument("--state", default=None, help="the readout's json, or 'home' (none: the spec's home pose)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from calibrated_scene_reconstruction.sim.robot_articulation import ROBOT_PATH, Robot, add_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402

s = Session(args.session)
sg = s.config("scene_graph.json", {})  # also before the scene graph: recon.multiview takes its scale from the cloud
spec = load_spec(args.robot or (sg.get("photo_robot") or {}).get("spec") or "vega_1p_sharpa_arm_lean")
if args.state == "home":  # no readout of the capture: the spec's home (recon.robot_fit --links fits the parts the
    state_path, state = Path("home"), {}  # joints do not move: the base, the wheels, the lowest torso link)
else:
    state_path = Path(args.state) if args.state else s.config_dir / "robot_state.json"
    raw = json.loads(state_path.read_text())
    state = {j["joint"]: float(j["position_rad"]) for j in raw["joints"]} if "joints" in raw else {k: float(v) for k, v in raw.items() if not k.startswith("_")}

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
add_robot(stage, spec, (0.0, 0.0), 0.0)

# each visual mesh's vertices in its link's frame, from the authored (zero) pose: the mesh-to-link transform is fixed
link_names = None
meshes = []
xc = UsdGeom.XformCache()
world.reset()
rob = Robot(world, spec)
link_names = rob.body_names
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH), Usd.TraverseInstanceProxies()):  # visuals are instanced
    if not p.IsA(UsdGeom.Mesh) or "collision" in str(p.GetPath()).lower():
        continue
    if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
        continue
    pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
    if not pts:
        continue
    q = p.GetParent()
    while q and q.GetName() not in link_names:
        q = q.GetParent()
    if not q:
        continue
    M = np.asarray(xc.GetLocalToWorldTransform(p), float).T  # USD row vectors -> column convention
    L = np.asarray(xc.GetLocalToWorldTransform(q), float).T
    P = np.asarray(pts, float)
    P = (np.linalg.inv(L) @ (M @ np.c_[P, np.ones(len(P))].T)).T[:, :3]
    meshes.append((link_names.index(q.GetName()), P))

# the joint vector: the readout where it names the joint, else the same-side suffix, else the spec's home
q = rob.home()
matched, unmatched = {}, []
for n, i in rob.idx.items():
    side = "right" if n.lower().startswith(("r_", "right")) else "left" if n.lower().startswith(("l_", "left")) else ""
    key = n if n in state else next((k for k in state if side and k.lower().startswith(side) and
                                     k.split("_", 1)[-1] == n.split("_", 1)[-1]), None)
    if key is None:
        unmatched.append(n)
        continue
    q[i] = float(np.clip(state[key], rob.lower[i], rob.upper[i]))
    matched[n] = key
rob.teleport(q)
for _ in range(3):
    world.step(render=False)
rob.sync()
tf = rob.link_transforms()
base = tf[0] if ROBOT_PATH else None
out_pts, out_link = [], []
for li, P in meshes:
    t = tf[li]
    R = Rotation.from_quat(t[3:7]).as_matrix()
    out_pts.append(P @ R.T + t[:3])
    out_link.append(np.full(len(P), li, np.int32))
pts = np.concatenate(out_pts)
dst = s / "parsing"
dst.mkdir(exist_ok=True)
stem = "robot_cloud_home" if args.state == "home" else "robot_cloud"  # (home: not a readout -- recon.multiview
# takes robot_cloud.npz as the readout's robot for the robot_joints scale)
np.savez_compressed(dst / f"{stem}.npz", points=pts.astype(np.float32), link=np.concatenate(out_link),
                    links=np.array(link_names), q=q)
(dst / f"{stem}.json").write_text(json.dumps({
    "spec": spec["name"], "state": state_path.name, "matched": matched, "unmatched_kept_home": unmatched,
    "readout_unused": sorted(set(state) - set(matched.values())), "points": int(len(pts)),
    "extent": [pts.min(0).round(3).tolist(), pts.max(0).round(3).tolist()]}, indent=1) + "\n")
print(f"done: {len(pts)} points from {len(meshes)} meshes, {len(matched)} joints from the readout, "
      f"{len(unmatched)} at home, extent {pts.min(0).round(2)} .. {pts.max(0).round(2)}", flush=True)
app.close()
