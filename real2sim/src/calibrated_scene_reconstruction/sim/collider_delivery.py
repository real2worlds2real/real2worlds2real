"""Compare delivered collision recipes and geometry with current source authoring."""
import hashlib
import numpy as np
from pxr import Sdf,Usd,UsdGeom,UsdPhysics


def colliders(stage,body):
    cache=UsdGeom.XformCache();inverse=cache.GetLocalToWorldTransform(body).GetInverse();result={}
    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        mesh=UsdGeom.Mesh(prim)
        if not mesh:raise ValueError('Task collider is not an authored source mesh')
        name=str(prim.GetPath()).removeprefix(str(body.GetPath()))
        transform=np.asarray(cache.GetLocalToWorldTransform(prim)*inverse,float)
        properties={a.GetName():a.Get() for a in prim.GetAttributes()
            if a.GetName().startswith(('physxSDFMeshCollision:','physxConvexHullCollision:',
                'physxConvexDecompositionCollision:','physxCollision:'))
                or a.GetName()=='physics:approximation'}
        result[name]={'vertices':np.asarray(mesh.GetPointsAttr().Get(),float),
            'faces':np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int),
            'counts':np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int),
            'transform':transform,'properties':properties}
    return result


def compare(actual,expected):
    if not actual or actual.keys()!=expected.keys():raise ValueError('Delivered collider paths differ from source recipe')
    result={}
    for name,a in actual.items():
        e=expected[name]
        for key in ['vertices','faces','counts','transform']:
            if a[key].shape!=e[key].shape or not np.allclose(a[key],e[key],rtol=0,atol=1e-7):
                raise ValueError('Delivered collider geometry/transform differs: '+name+'/'+key)
        if a['properties']!=e['properties']:
            raise ValueError('Delivered collision precision/cooking differs: '+name)
        digest=hashlib.sha256()
        for key in ['vertices','faces','counts','transform']:digest.update(a[key].tobytes())
        result[name]={'properties':a['properties'],'geometry_sha256':digest.hexdigest(),
                      'matches_current_source_recipe':True}
    return result


def source_stage(stage):
    """Mixed source authoring needs the already verified native scene context."""
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('One explicit delivery physics scene required')
    expected = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(expected, 1.)
    UsdGeom.SetStageUpAxis(expected, UsdGeom.Tokens.z)
    path = scenes[0].GetPath()
    Sdf.CreatePrimInLayer(expected.GetRootLayer(), path.GetParentPath())
    if not Sdf.CopySpec(stage.Flatten(), path, expected.GetRootLayer(), path):
        raise ValueError('Cannot copy the verified native physics context')
    return expected


def audit(stage,session,links):
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    from calibrated_scene_reconstruction.sim.mechanism_stage import add_links
    from calibrated_scene_reconstruction.sim.physics import physics_material
    expected=source_stage(stage)
    material=physics_material(expected,'/World/SourceMaterial',.4,.3,.03)
    for obj,spec in session.scene_graph['dynamic_objects'].items():
        fix=dict(spec['fix']);fix.pop('look',None)
        add_links(expected,'/World/Objects/'+obj,asset(fix),fix,spec,material,look_base=session.dir)
    report={}
    for ref,link in links.items():
        if link['kind']!='rigid':continue
        path=link['path'];a=stage.GetPrimAtPath(path);e=expected.GetPrimAtPath(path)
        if not a or not e:raise ValueError('Delivered body differs from source: '+ref)
        report.update({path+key:value for key,value in compare(colliders(stage,a),colliders(expected,e)).items()})
    return report
