"""Read actual Isaac convex cooking output; no scene edits or task approval."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage');p.add_argument('--prim',required=True);p.add_argument('--output',required=True)
p.add_argument('--cooking-plan',help='Diagnostic explicit cooking settings, authored only in the session layer')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import sys  # noqa: E402
_exception_hook=sys.excepthook
def _failed_inspection(kind,error,traceback):
    _exception_hook(kind,error,traceback)
    app.app.post_quit(2)
sys.excepthook=_failed_inspection
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_cooking_interface  # noqa: E402
from pxr import PhysicsSchemaTools,UsdGeom,UsdPhysics,UsdUtils  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402

errors=PhysicsErrors(out/'native_errors.json')
try:
    if not omni.usd.get_context().open_stage(str(Path(a.stage).resolve())):raise ValueError('Could not open source stage')
    stage=omni.usd.get_context().get_stage();prim=stage.GetPrimAtPath(a.prim)
    if not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('An existing native mesh collider is required')
    approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
    if approximation not in {'convexDecomposition','convexHull'}:
        raise ValueError('Inspector cannot substitute a different collision approximation')
    cooking=None
    if a.cooking_plan:
        if approximation!='convexDecomposition':raise ValueError('Decomposition settings cannot configure a single hull')
        from calibrated_scene_reconstruction.sim.convex_cooking import apply
        stage.SetEditTarget(stage.GetSessionLayer())
        cooking=apply(prim,json.loads(Path(a.cooking_plan).read_text()))
        shutil.copy2(a.cooking_plan,out/'cooking_plan.json')
    result={}
    def ready(status,convexes):
        hulls=[]
        for hull in convexes:
            indices=np.asarray(hull.indices,int);faces=[]
            for polygon in hull.polygons:
                ids=indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
                faces.extend([[int(ids[0]),int(ids[j]),int(ids[j+1])] for j in range(1,len(ids)-1)])
            hulls.append({'V':np.asarray(hull.vertices,float).tolist(),'F':faces})
        result.update(status=str(status),hulls=hulls)
    cache=UsdUtils.StageCache.Get();sid=cache.GetId(stage)
    if not sid.IsValid():sid=cache.Insert(stage)
    get_physx_cooking_interface().request_convex_collision_representation(stage_id=sid.ToLongInt(),
        collision_prim_id=PhysicsSchemaTools.sdfPathToInt(a.prim),run_asynchronously=False,on_result=ready)
    if not result.get('hulls'):raise ValueError('No actual cooked convex hulls returned: '+str(result))
    mesh=UsdGeom.Mesh(prim);counts=np.array(mesh.GetFaceVertexCountsAttr().Get());indices=np.array(mesh.GetFaceVertexIndicesAttr().Get())
    if not np.all(counts==3):raise ValueError('Triangle source collider required')
    np.savez_compressed(out/'source_mesh.npz',V=np.array(mesh.GetPointsAttr().Get()),F=indices.reshape(-1,3),
                        local_to_world=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)))
    native=errors.report();precision=errors.precision_audit()
    (out/'native_errors.json').write_text(json.dumps(native,indent=2))
    result.update(native_physics_pass=native['pass'],precision_audit=precision,approximation=approximation,
        stage=str(Path(a.stage).resolve()),stage_sha256=hashlib.sha256(Path(a.stage).read_bytes()).hexdigest(),
        prim=a.prim,cooking=cooking,source_stage_changed=False,task_approved=False,
        scope='Actual native convex cooking geometry in collider local coordinates; fidelity and task checks remain separate.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('Cooked hulls:',len(result['hulls']),flush=True)
    if not native['pass'] or not precision['pass']:raise ValueError('Native cooking rejected the requested profile; see result.json')
finally:
    errors.close();app.close()
