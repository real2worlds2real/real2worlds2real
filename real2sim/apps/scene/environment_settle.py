"""Release exported background assemblies under gravity in an isolated scene.

Only reconstructed background and read-only support geometry are copied in.
Task objects and robot are absent; source USD/configuration is never modified.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job')
parser.add_argument('--output', required=True)
args = parser.parse_args()
if Path(args.output).exists():
    raise FileExistsError('Passive checks require a new immutable result path: '+args.output)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from calibrated_scene_reconstruction.sim.usd import add_mesh  # noqa: E402

job = json.loads(Path(args.job).read_text())
z = np.load(job['geometry'])
cfg = job['contract']
by_entity = {}
for i, row in enumerate(job['parts']):
    by_entity.setdefault(row['entity'], []).append(i)
world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/30)
stage = omni.usd.get_context().get_stage()
material = UsdShade.Material.Define(stage, '/World/ValidationMaterial')
physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
physical.CreateStaticFrictionAttr(.6)
physical.CreateDynamicFrictionAttr(.5)
physical.CreateRestitutionAttr(0.)
dynamic, start = {}, {}
masses = {}
for number, (entity, indices) in enumerate(by_entity.items()):
    spec = cfg.get('entities', {}).get(entity, {})
    active = spec.get('mode') == 'resting' and entity not in job['external_entities']
    path = '/World/Validation/body_'+str(number)
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    vertices = np.concatenate([z[f'v{i}'] for i in indices])
    # The geometry remains at exactly its exported world pose. A shared assembly
    # root is released; internal members remain mechanically connected.
    centre = (vertices.min(0)+vertices.max(0))/2
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    for i in indices:
        mesh = add_mesh(stage, path+'/part_'+str(i), z[f'v{i}']-centre, z[f'f{i}']).GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)
        if active:
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexHull')
        PhysxSchema.PhysxCollisionAPI.Apply(mesh).CreateContactOffsetAttr(.001)
        PhysxSchema.PhysxCollisionAPI.Apply(mesh).CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    if active:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # Share one density hypothesis rather than assigning a cardboard box and
        # a large cabinet the same mass. This agrees with the static audit's
        # volume-weighted gravity-centre approximation; actual mass is unknown.
        volume=0.
        for i in indices:
            tri=(z[f'v{i}']-centre)[z[f'f{i}']]
            volume+=abs(float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6))
        mass=float(spec.get('validation_mass_kg',max(.01,volume*300.)))
        if not np.isfinite(mass) or not .001<=mass<=10000:raise ValueError('Finite plausible validation mass required')
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        masses[entity]=mass
        properties = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        properties.CreateSolverPositionIterationCountAttr(32)
        properties.CreateSolverVelocityIterationCountAttr(8)
        dynamic[entity] = path
    start[entity] = centre.tolist()
world.reset()
bodies = {}
for entity, path in dynamic.items():
    body = SingleRigidPrim(path)
    body.initialize()
    bodies[entity] = body
initial = {key: (np.asarray(body.get_world_pose()[0]), np.asarray(body.get_world_pose()[1])) for key, body in bodies.items()}
middle = {}
for step in range(240):
    world.step(render=False)
    if step == 119:
        middle = {key: np.asarray(body.get_world_pose()[0]) for key, body in bodies.items()}
rows = []
for entity, body in bodies.items():
    position, quaternion = (np.asarray(v) for v in body.get_world_pose())
    p0, q0 = initial[entity]
    displacement = float(np.linalg.norm(position-p0))
    rotation = float(np.degrees((Rotation.from_quat(np.roll(quaternion, -1))*Rotation.from_quat(np.roll(q0, -1)).inv()).magnitude()))
    drift = float(np.linalg.norm(position-middle[entity]))
    finite = bool(np.isfinite([displacement, rotation, drift]).all())
    passed = finite and displacement <= .003 and rotation <= 1. and drift <= .001
    rows.append({'entity': entity, 'pass': passed, 'displacement_m': displacement if finite else None,
                 'rotation_deg': rotation if finite else None, 'late_drift_m': drift if finite else None,
                 'validation_mass_kg':masses[entity],
                 'initial_position': p0.tolist(), 'final_position': position.tolist() if finite else None})
result = {'pass': all(row['pass'] for row in rows), 'entities': rows, 'duration_s': 2., 'physics_hz': 120,
          'method': 'Release all resting background assemblies together under gravity. Fixed/foundation/read-only supports stay static. Conservative per-member convex colliders, shared300kg/m3 density hypothesis (10g minimum mass) unless declared, friction0.6/0.5. No robot/task simulation or source-pose correction.',
          'assumptions': 'Checks modeled rigid-body stability, not material strength, actual mass/friction or unobserved attachment truth.'}
with Path(args.output).open('x') as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
print('Environment passive stability', result['pass'], len(rows), 'released bodies', flush=True)
app.close()
