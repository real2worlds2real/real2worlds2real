"""Numerical fluid-volume and passive-motion control in analytic open boxes.

Independent of any reconstruction. Compare resolutions without fitting source
objects, masses or force inputs. This diagnostic never grants task acceptance.
"""
import argparse
import json
from pathlib import Path
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--hz', default=480, type=int)
parser.add_argument('--mass-scale', default=1., type=float)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
import hashlib
input_paths=[Path(__file__),Path('src/calibrated_scene_reconstruction/sim/native_fluid.py'),Path('src/calibrated_scene_reconstruction/sim/simulation.py')]
inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in input_paths}
(args.output/'inputs.json').write_text(json.dumps(dict(files=inputs,arguments=vars(args)),default=str,indent=2)+'\n')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True})
try:
    import carb
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from omni.physx.bindings import _physx as bindings
    from pxr import UsdGeom, UsdPhysics, Gf
    from scipy.spatial import ConvexHull
    from calibrated_scene_reconstruction.sim.native_fluid import NativeFluid
    from calibrated_scene_reconstruction.sim.simulation import apply
    from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors

    hz=args.hz
    world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/30)
    context=world.get_physics_context();context.enable_gpu_dynamics(True)
    context.set_broadphase_type('GPU');context.enable_fabric(False)
    settings=carb.settings.get_settings();settings.set('/physics/suppressReadback',False)
    for key in (bindings.SETTING_UPDATE_TO_USD,bindings.SETTING_UPDATE_VELOCITIES_TO_USD,
                bindings.SETTING_UPDATE_PARTICLES_TO_USD):settings.set(key,True)
    stage=omni.usd.get_context().get_stage()
    apply(stage,dict(physics_hz=hz,solver='TGS',friction_type='patch',gpu_dynamics=True,
                    rigid_iterations=[64,16],basis='Analytic container numerical control'))
    scene=next(p for p in stage.Traverse()if p.IsA(UsdPhysics.Scene))
    errors=PhysicsErrors(args.output/'native_errors.json');errors.guard_steps(world)
    cases=[];width=.04;fill=.03
    for index,spacing in enumerate([.0016,.0012,.0008]):
        origin=np.array([index*.1,0.,0.]);root=f'/World/Case{index}'
        UsdGeom.Xform.Define(stage,root)
        solids=[('floor',[0,0,-.002],[width+.008,width+.008,.004]),
                ('left',[-width/2-.002,0,.03],[.004,width+.008,.06]),
                ('right',[width/2+.002,0,.03],[.004,width+.008,.06]),
                ('back',[0,-width/2-.002,.03],[width,.004,.06]),
                ('front',[0,width/2+.002,.03],[width,.004,.06])]
        for name,position,size in solids:
            cube=UsdGeom.Cube.Define(stage,root+'/'+name);cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.array(position)+origin)))
            cube.AddScaleOp().Set(Gf.Vec3d(*size));UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        axis=np.arange(-width/2+spacing,width/2-spacing/2,spacing)
        height=np.arange(spacing,fill-spacing/2,spacing)
        xyz=np.stack(np.meshgrid(axis,axis,height,indexing='ij'),-1).reshape(-1,3)+origin
        spec=dict(basis='Analytic box, water prior and native preset; no task measurements',
                  spacing_m=spacing,density_kg_m3=1000.*args.mass_scale,iterations=64,isosurface=True,
                  pbd=dict(friction=.12,cohesion=.01,surface_tension=.0074,viscosity=.0000017,adhesion=0.,damping=0.))
        fluid=NativeFluid(stage,root+'/Fluid',scene.GetPath(),xyz,spec)
        cases.append(dict(fluid=fluid,origin=origin,spacing=spacing,initial=xyz,history=[]))
    world.reset();settings.set('/physics/suppressReadback',False)
    for step in range(3*hz):
        world.step(render=False)
        if step%max(1,hz//4)==0 or step==3*hz-1:
            for case in cases:
                p,v=case['fluid'].state();speed=np.linalg.norm(v,axis=1)
                case['history'].append(dict(t=(step+1)/hz,centre=p.mean(0).tolist(),
                    z_percentiles=np.percentile(p[:,2],[0,50,95,99,100]).tolist(),
                    speed_percentiles=np.percentile(speed,[50,95,99,100]).tolist(),
                    rms_speed=float(np.sqrt(np.mean(speed**2))),mean_velocity=v.mean(0).tolist()))
    for _ in range(5):world.render()
    stage.GetRootLayer().Export(str(args.output/'native_final.usda'))
    result=[]
    for index,case in enumerate(cases):
        p,v=case['fluid'].state();local=p-case['origin'];d=case['spacing']
        interior=(np.abs(local[:,:2])<width/2-3*d).all(1)
        top=float(np.quantile(local[interior,2],.99)+d/2)
        hull=ConvexHull(local).volume
        nominal=case['fluid'].initial['initial_volume_m3']
        bulk_density=case['fluid'].initial['initial_mass_kg']/(width**2*top)
        np.savez_compressed(args.output/f'state_{index}.npz',points=p,velocity=v,initial=case['initial'])
        result.append(dict(spacing_m=d,initial=case['fluid'].initial,ledger=case['fluid'].ledger(),
                           box_cross_section_m2=width**2,interior_top_m=top,
                           estimated_bulk_volume_m3=width**2*top,
                           bulk_to_nominal_ratio=width**2*top/nominal,estimated_bulk_density_kg_m3=bulk_density,
                           particle_centres_convex_hull_m3=float(hull),history=case['history']))
    errors.check()
    (args.output/'result.json').write_text(json.dumps(dict(accepted=False,inputs=inputs,inputs_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items()),physics_hz=hz,mass_scale=args.mass_scale,target_bulk_density_kg_m3=1000.,cases=result,
         native_physics=errors.report(),scope=__doc__),indent=2)+'\n')
    print('FLUID_RESOLUTION',[(r['spacing_m'],r['bulk_to_nominal_ratio'],r['history'][-1]['rms_speed'])for r in result],flush=True)
    errors.close()
except Exception:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    app.close()
