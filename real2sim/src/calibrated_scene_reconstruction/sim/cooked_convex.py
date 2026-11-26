"""Audit SDK-cooked convex representations against each authored convex cell.

Queries the source stage through PhysX's collision-representation interface.
No rigid or flexible simulation state is changed. This is a geometry audit,
not contact or task validation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh


def source_reference(vertices,faces,representation='authored_cell'):
    """Respect the declared collider interpretation, not an implicit repair.

    A CAD triangle soup declared as convexHull is a point-cloud hull in PhysX.
    The explicit alternate mode audits that hull, not the visual mesh surface.
    Authored source cells retain their original closed-surface requirement.
    """
    if representation=='declared_convex_hull':
        from calibrated_scene_reconstruction.assets.convex_source_solids import _hull
        return _hull(vertices)
    if representation!='authored_cell':raise ValueError('Explicit source convex representation required')
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    if not mesh.is_watertight:raise ValueError('Closed authored source cell required')
    return mesh


def audit(stage,root,output,*,tolerance_m,source_representation='authored_cell'):
    import omni.physx
    import omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics,PhysicsSchemaTools,UsdUtils
    from omni.physx.bindings import _physx
    if not np.isfinite(tolerance_m) or tolerance_m<=0:raise ValueError('Positive metric cooking audit tolerance required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cooking=omni.physx.get_physx_cooking_interface();stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    transforms=UsdGeom.XformCache();rows=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if (not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
                or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull'):continue
        response=[]
        cooking.request_convex_collision_representation(stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,
            on_result=lambda result,convexes:response.append((result,convexes)))
        if len(response)!=1 or response[0][0]!=_physx.PhysxCollisionRepresentationResult.RESULT_VALID or len(response[0][1])!=1:
            raise ValueError('Expected one valid cooked hull per authored source cell: '+str(prim.GetPath()))
        native=response[0][1][0];vertices=np.array([[v.x,v.y,v.z] for v in native.vertices],float);indices=np.asarray(native.indices,int);faces=[]
        for polygon in native.polygons:
            ids=indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
            faces.extend([[int(ids[0]),int(ids[k]),int(ids[k+1])] for k in range(1,len(ids)-1)])
        body=prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body:raise ValueError('Cooked collider has no rigid owner')
        frame=np.linalg.inv(np.asarray(transforms.GetLocalToWorldTransform(body)).T)@np.asarray(transforms.GetLocalToWorldTransform(prim)).T
        raw=UsdGeom.Mesh(prim);source_v=np.asarray(raw.GetPointsAttr().Get(),float)
        source_f=np.asarray(raw.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        source_v=source_v@frame[:3,:3].T+frame[:3,3];vertices=vertices@frame[:3,:3].T+frame[:3,3]
        source=source_reference(source_v,source_f,source_representation)
        cooked=source_reference(vertices,faces,source_representation)
        if not cooked.is_watertight:raise ValueError('Closed cooked hull required')
        def samples(mesh):return np.r_[mesh.vertices,mesh.triangles_center,mesh.triangles[:,:2].mean(1)]
        forward=trimesh.proximity.closest_point(cooked,samples(source))[1]
        backward=trimesh.proximity.closest_point(source,samples(cooked))[1]
        error=float(max(forward.max(),backward.max()));file=output/('cell_'+str(len(rows))+'.npz')
        np.savez_compressed(file,source_vertices=source_v,source_faces=source_f,cooked_vertices=vertices,cooked_faces=faces,mesh_to_body=frame,
            reference_vertices=source.vertices,reference_faces=source.faces)
        rows.append({'collider':str(prim.GetPath()),'pass':error<=tolerance_m,'sampled_surface_error_m':error,
            'source_volume_m3':float(source.volume),'cooked_volume_m3':float(cooked.volume),
            'source_vertices':len(source_v),'cooked_vertices':len(vertices),'source_representation':source_representation,'data':file.name,
            'data_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    result={'pass':bool(rows) and all(row['pass'] for row in rows),'cells':rows,'tolerance_m':tolerance_m,
        'scope':'SDK collision-representation query in the source stage, transformed into physical rigid coordinates. Finite surface samples; not simulation/contact/task approval.'}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
