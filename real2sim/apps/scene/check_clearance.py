"""QA: no placed item intersects the room or another item, and nothing gets ejected once physics runs.

    tools/run_isaac.sh apps/scene/check_clearance.py sessions/<name> --tag twin [--variant cousins/cousin_00.json]

1. geometry: build_scene already resolves penetrations (sim.clearance); its report says what intersected what, how
   deep, how far the item was pushed and whether it is clear now;
2. semantics (sim.semantics): stools seat at counter height with any backrest turned away from the counter, picture
   frames face the camera, elongated swaps keep the photo's long axis;
3. physics: 2 s of PhysX with no robot. A manipulable object that pops up (> 1 cm: it was still penetrating),
   drops (> 2 cm: it rested on nothing), slides away (> 5 cm) or topples (> 45 deg) fails. Round produce is held to
   the same limits: the scene lays it on its steadiest flat spot with rolling resistance (assets.physprops).
Writes qa/clearance_<tag>.json and prints one PASS / FAIL line.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant")
ap.add_argument("--background", default="mesh", choices=["mesh", "code", "zone"],
                help="code: the room written as code (sim.code_scene), its pieces settled by PhysX too")
ap.add_argument("--static-big", action="store_true", help="code: the big pieces (desks, cabinets) held still")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import summarize  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import load_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.semantics import check as semantic_check  # noqa: E402
from calibrated_scene_reconstruction.sim.semantics import summarize as semantic_summary  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import add_background  # noqa: E402

s = Session(args.session)
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = prepare_variant(s.dir, load_variant(args.variant))
ts = build_scene(stage, prepare_scene(s.dir), variant)
if args.background == 'zone':
    add_background(stage, prepare_background(s.dir), 'zone')
semantics = semantic_check(ts, s.read_json("scene/twin.json").get("static_swaps"))
code = {}
if args.background == "code":  # every piece of the code room a rigid body for the test: one that floats drops, one
    from pxr import UsdPhysics  # noqa: E402  (that penetrates another is pushed out; static otherwise)

    from calibrated_scene_reconstruction.scene.background.code_room import BIG  # noqa: E402
    from calibrated_scene_reconstruction.sim.code_stage import add_code_background  # noqa: E402

    add_code_background(stage, prepare_background(s.dir))
    for p in stage.GetPrimAtPath("/World/Code").GetChildren():
        if p.GetCustomDataByKey("calibrated_scene_reconstruction:kit") in (None, "panel"):
            continue
        rb = UsdPhysics.RigidBodyAPI.Apply(p)
        UsdPhysics.MassAPI.Apply(p).CreateMassAttr(10.0)
        if args.static_big and p.GetCustomDataByKey("calibrated_scene_reconstruction:kit") in BIG:  # a support held: what topples does so
            rb.CreateKinematicEnabledAttr(True)  # on its own
            continue
        code[p.GetName()] = p
world.reset()
# A linked assembly's container is not a rigid body. Wrapping that container
# would author an extra parent body above its real rigid/cloth parts.
bodies = {}
for name, root in {**ts.objects, **{f'code/{n}':p for n,p in code.items()}}.items():
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        suffix = str(prim.GetPath())[len(str(root.GetPath())):]
        bodies[name+suffix] = SingleRigidPrim(str(prim.GetPath()),reset_xform_properties=False)
for b in bodies.values():
    b.initialize()
p0 = {n: (np.asarray(b.get_world_pose()[0]), np.asarray(b.get_world_pose()[1])) for n, b in bodies.items()}
for _ in range(round(2/world.get_physics_dt())):
    world.step(render=False)
physics = []
for n, b in bodies.items():
    p, q = (np.asarray(v) for v in b.get_world_pose())
    if not all(np.isfinite(v).all() for v in [p,q,*p0[n]]):
        physics.append({'item':n,'ok':False,'why':'Nonfinite native rigid state',
                        'moved_m':None,'dz_m':None,'tilt_deg':None})
        continue
    tilt = np.degrees(2 * np.arccos(np.clip(abs(np.dot(q, p0[n][1])), 0, 1)))
    d = p - p0[n][0]
    moved = float(np.linalg.norm(d))
    physics.append({"item": n, "moved_m": round(moved, 4), "dz_m": round(float(d[2]), 4), "tilt_deg": round(float(tilt), 1),
                    "ok": bool(d[2] < 0.01 and d[2] > -0.02 and np.linalg.norm(d[:2]) < 0.05 and tilt < 45)})
bad_geo = [r for r in ts.clearance if not r["clear"]]
bad_phys = [r for r in physics if not r["ok"]]
bad_sem = [r for r in semantics if not r["ok"]]
res = {"tag": args.tag, "geometry": ts.clearance, "semantics": semantics, "physics": physics,
       "pass": not bad_geo and not bad_phys and not bad_sem}
s.write_json(f"qa/clearance_{args.tag}.json", res)
print(f"[{args.tag}] {'PASS' if res['pass'] else 'FAIL'} | {summarize(ts.clearance)} | {semantic_summary(semantics)} | physics: "
      + (", ".join(r['item']+': '+r['why'] if r.get('why') else f"{r['item']} moved {r['moved_m'] * 100:.1f}cm dz {r['dz_m'] * 100:+.1f}cm tilt {r['tilt_deg']:.0f}"
                   for r in bad_phys)
         if bad_phys else f"{len(physics)} objects at rest"), flush=True)
app.close()
