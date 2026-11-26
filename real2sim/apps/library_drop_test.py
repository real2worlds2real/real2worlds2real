"""Library QA: drop every converted model as a rigid body onto a plane; its visible mesh must end up resting on it.

    tools/run_isaac.sh apps/library_drop_test.py

Write the signed height of the lowest visible point after settling to each model's
rest_offset_m entry. The cousin builder uses this offset to select supported props.
"""
import argparse
import json

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from calibrated_scene_reconstruction.core.paths import ABO_DIR, POLYHAVEN_DIR  # noqa: E402
from calibrated_scene_reconstruction.sim.assets import add_library_model, library_index  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import world_points  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
UsdGeom.Xformable(ground).AddScaleOp().Set(Gf.Vec3f(80, 80, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

names = sorted(library_index()["usd"])
YAWS = (0.0, 1.3, 2.8, 4.4)  # collision / visual mismatches can depend on the orientation
prims, cells = {}, [(name, yaw) for name in names for yaw in YAWS]
side = int(np.ceil(np.sqrt(len(cells))))
for i, (name, yaw) in enumerate(cells):  # one 2.5 m cell per (model, yaw), natural size
    xy = (2.5 * (i % side) - 1.25 * side, 2.5 * (i // side) - 1.25 * side)
    prims.setdefault(name, []).append(add_library_model(stage, f"/World/m_{i}", name, xy, 0.0, yaw=yaw, dynamic=True,
                                                        mass=0.3)[0])
world.reset()
bodies, start = {}, {}
for name, ps in prims.items():
    for p in ps:
        b = SingleRigidPrim(str(p.GetPath()))
        b.initialize()
        bodies[p.GetPath()] = b
        pos, q = b.get_world_pose()
        start[p.GetPath()] = (world_points(p), np.asarray(pos), Rotation.from_quat(np.roll(q, -1)))
for _ in range(180):
    world.step(render=False)
offset = {}
for name, ps in prims.items():  # physics poses (Fabric) applied to the initial visual points
    lows = []
    for p in ps:
        P0, p0, R0 = start[p.GetPath()]
        pos, q = bodies[p.GetPath()].get_world_pose()
        if len(P0):
            R1 = Rotation.from_quat(np.roll(q, -1))
            lows.append(float(((P0 - p0) @ (R1 * R0.inv()).as_matrix().T + np.asarray(pos))[:, 2].min()))
    offset[name] = round(max(lows, key=abs), 4) if lows else None
for path in (POLYHAVEN_DIR / "index.json", ABO_DIR / "usd_index.json"):
    if not path.exists():
        continue
    idx = json.loads(path.read_text())
    for name in idx.get("meta", {}):
        idx["meta"][name].pop("drop_sink_m", None)
        if name in offset:
            idx["meta"][name]["rest_offset_m"] = offset[name]
    path.write_text(json.dumps(idx, indent=1))
bad = {n: v for n, v in offset.items() if v is None or abs(v) > 0.01}
print(f"drop test: {len(offset) - len(bad)}/{len(offset)} models rest on their visuals; off: "
      + (", ".join(f"{n} {v * 100:+.1f}cm" if v is not None else f"{n} (no mesh)" for n, v in bad.items()) or "none"))
app.close()
