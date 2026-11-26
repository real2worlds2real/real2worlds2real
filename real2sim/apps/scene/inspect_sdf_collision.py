"""Measure a native cooked SDF on its actual source surface; no task approval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage');p.add_argument('--prim',required=True);p.add_argument('--output',required=True)
p.add_argument('--spacing-m',type=float,required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'source.py').write_bytes(Path(__file__).read_bytes())
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
_original_hook=sys.excepthook

def _failure(kind,error,traceback):
    _original_hook(kind,error,traceback);app.app.post_quit(2)
sys.excepthook=_failure
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SdfShapePrim  # noqa: E402
from pxr import UsdGeom,UsdPhysics,PhysxSchema  # noqa: E402
from calibrated_scene_reconstruction.assets.rigid_initial_state import surface_samples  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402

errors=PhysicsErrors(out/'native_errors.json')
try:
    if not omni.usd.get_context().open_stage(str(Path(a.stage).resolve())):raise ValueError('Could not read source stage')
    stage=omni.usd.get_context().get_stage();prim=stage.GetPrimAtPath(a.prim)
    if not prim or not prim.IsA(UsdGeom.Mesh) or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='sdf':
        raise ValueError('An existing native SDF collider is required')
    mesh=UsdGeom.Mesh(prim);V=np.asarray(mesh.GetPointsAttr().Get(),float)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());F=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    if not np.all(counts==3):raise ValueError('Triangle source mesh required')
    T=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0));scales=np.linalg.norm(T[:3,:3],axis=1)
    if not np.allclose(T[:3,:3]@T[:3,:3].T,np.eye(3)*scales[0]**2,rtol=1e-5,atol=1e-10):
        raise ValueError('Metric SDF inspection requires rigid or uniform source scale')
    unit=UsdGeom.GetStageMetersPerUnit(stage)*scales[0]
    points=surface_samples(V,F,a.spacing_m/unit);batch=min(65536,len(points))
    world=World(stage_units_in_meters=1.,device='cuda:0',backend='torch')
    errors.guard_steps(world)
    # SDF queries require the GPU tensor pipeline. No simulation motion is run.
    world.get_physics_context().enable_gpu_dynamics(True)
    world.get_physics_context().set_broadphase_type('GPU')
    view=SdfShapePrim(a.prim,num_query_points=batch,prepare_sdf_schemas=False,
                      reset_xform_properties=False,disable_stablization=False)
    world.reset();view.initialize()
    values=[]
    for start in range(0,len(points),batch):
        selected=points[start:start+batch];n=len(selected)
        padded=np.concatenate([selected,np.repeat(selected[-1:],batch-n,axis=0)])
        raw=view.get_sdf_and_gradients(torch.as_tensor(padded[None],dtype=torch.float32,device='cuda:0'))
        if raw is None:raise ValueError('Native SDF query unavailable')
        raw=raw.detach().cpu().numpy()[0,:n]
        # Native tensors (and NVIDIA's test_sdf_shape_view.py) use xyz gradient
        # followed by signed distance, despite the high-level docstring order.
        values.append(raw[:,3]*unit)
    distances=np.concatenate(values)
    if not np.isfinite(distances).all():raise ValueError('Nonfinite native SDF samples')
    np.savez_compressed(out/'source_surface_samples.npz',points=points,native_sdf_m=distances)
    report={'task_approved':False,'diagnostic_only':True,'prim':a.prim,'spacing_m':a.spacing_m,
            'samples':len(points),'maximum_absolute_surface_error_m':float(np.abs(distances).max()),
            'minimum_signed_distance_m':float(distances.min()),'maximum_signed_distance_m':float(distances.max()),
            'absolute_distance_quantiles_m':np.quantile(np.abs(distances),[.5,.95,.99,1]).tolist(),
            'native_errors':errors.report(),'precision':errors.precision_audit(),
            'stage_sha256':hashlib.sha256(Path(a.stage).read_bytes()).hexdigest(),
            'scope':'Finite source-surface samples queried against actual native SDF, not continuous shape or task proof.'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['native_errors']['pass']:raise ValueError('Native SDF query reported a physics error')
finally:
    errors.close();app.close()
