"""Native Isaac5.1 particle-cloth bonds, with explicit physical ownership."""
import json
import numpy as np
from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,Vt


def audit_bond(stage,prim):
    """Read the actual legacy native endpoints and declared material nodes."""
    from .rigid_partition import OWNER,verify
    a=prim.GetRelationship('actor0').GetTargets();b=prim.GetRelationship('actor1').GetTargets()
    if len(a)!=1 or len(b)!=1 or prim.GetAttribute('attachmentEnabled').Get() is not True:
        raise ValueError('Particle root requires enabled real physical endpoints')
    if prim.GetCustomDataByKey('calibrated_scene_reconstruction:root_filter_only'):return None
    source=stage.GetPrimAtPath(a[0]);target=stage.GetPrimAtPath(b[0])
    if not source.HasAPI(PhysxSchema.PhysxParticleClothAPI):raise ValueError('Particle bond source has no native cloth')
    owner=target.GetCustomDataByKey(OWNER)
    if owner:
        info=verify(stage,owner)
        if not info or str(b[0]) not in info['carriers'] or prim.GetCustomDataByKey(OWNER)!=owner:
            raise ValueError('Particle bond has an unverified numerical rigid owner')
    elif not target.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Particle bond cannot target an unsupported body or world pin')
    declared=prim.GetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices')
    ids=json.loads(declared) if declared else list(prim.GetAttribute('collisionFilterIndices0').Get() or [])
    points=np.asarray(UsdGeom.Mesh(source).GetPointsAttr().Get());indices=np.asarray(ids,int)
    p0=np.asarray(prim.GetAttribute('points0').Get(),float);p1=np.asarray(prim.GetAttribute('points1').Get(),float)
    if (indices.ndim!=1 or not len(indices) or len(set(ids))!=len(ids) or indices.min()<0 or indices.max()>=len(points)
            or p0.shape!=(len(ids),3) or p1.shape!=p0.shape or not np.isfinite([p0,p1]).all()
            or not np.allclose(p0,points[indices],rtol=0.,atol=2e-7)):
        raise ValueError('Particle attachment points differ from actual source root vertices')
    cache=UsdGeom.XformCache();m0=np.asarray(cache.GetLocalToWorldTransform(source));m1=np.asarray(cache.GetLocalToWorldTransform(target))
    gap=np.linalg.norm(p0@m0[:3,:3]+m0[3,:3]-(p1@m1[:3,:3]+m1[3,:3]),axis=1).max()
    if gap*UsdGeom.GetStageMetersPerUnit(stage)>1e-6:raise ValueError('Particle root endpoints do not initially coincide')
    return (str(a[0]),owner or str(b[0])),ids


def audit_filters(stage,surface,rigid,indices):
    """Only declared real root vertices may lose their owner's contact."""
    from pxr import Usd
    filtered=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rigid).GetParent()):
        if prim.GetTypeName()!='PhysxPhysicsAttachment':continue
        if list(map(str,prim.GetRelationship('actor0').GetTargets()))!=[surface]:continue
        if list(map(str,prim.GetRelationship('actor1').GetTargets()))!=[rigid]:continue
        ids=list(prim.GetAttribute('collisionFilterIndices0').Get() or [])
        if not ids:continue
        if prim.GetAttribute('filterType0').Get()!='Vertices':raise ValueError('Particle root filter must address actual vertices')
        filtered.extend(map(int,ids))
    if len(filtered)!=len(set(filtered)) or sorted(filtered)!=sorted(indices):
        raise ValueError('Particle owner contact filter differs from the actual bonded vertices')
    count=len(UsdGeom.Mesh(stage.GetPrimAtPath(surface)).GetPointsAttr().Get())
    if not 0<len(filtered)<count:raise ValueError('Particle root filter must retain free contacting nodes')
    return {'surface':surface,'rigid':rigid,'filtered_root_vertices':sorted(filtered),
            'unfiltered_free_vertices':count-len(filtered),'backend':'particle'}


def author_many(stage,records):
    from collections import defaultdict
    from .rigid_partition import create,verify,OWNER
    groups=defaultdict(list)
    for record in records:groups[record['rigid']].append(record)
    result=[]
    for owner,items in groups.items():
        modes={r.get('backend','direct') for r in items}
        if 'legacy_direct' in modes:
            if modes!={'legacy_direct'}:raise ValueError('A physical particle-cloth owner must use one root backend')
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(owner)))
            inverse=np.linalg.inv(matrix)
            for record in items:
                surface=stage.GetPrimAtPath(record['surface']);indices=np.asarray(record['indices'],int)
                world=np.asarray(record['points'])[indices]
                mesh_inverse=np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(surface)))
                local0=world@mesh_inverse[:3,:3]+mesh_inverse[3,:3]
                local1=world@inverse[:3,:3]+inverse[3,:3]
                bond=PhysxSchema.PhysxPhysicsAttachment.Define(stage,record['path'])
                bond.CreateActor0Rel().SetTargets([record['surface']]);bond.CreateActor1Rel().SetTargets([owner])
                bond.CreatePoints0Attr(Vt.Vec3fArray.FromNumpy(np.asarray(local0,np.float32)))
                bond.CreatePoints1Attr(Vt.Vec3fArray.FromNumpy(np.asarray(local1,np.float32)))
                bond.CreateFilterType0Attr('Vertices');bond.CreateCollisionFilterIndices0Attr(indices.tolist())
                surface.SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps(indices.tolist()))
                bond.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',record['basis'])
                result.append({'indices':indices.tolist(),'numerical_attachments':[record['path']],
                               'physical_rigid_owner':owner,'backend':'legacy_direct'})
            continue
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(owner)))
        inverse=np.linalg.inv(matrix);world=np.concatenate([np.asarray(r['points'])[r['indices']] for r in items])
        anchors=world@inverse[:3,:3]+inverse[3,:3]
        frames=create(stage,owner,items[0]['rigid_geometry'],len(anchors),anchors=anchors);cursor=0
        for record in items:
            path=record['path'];indices=list(map(int,record['indices']));surface=stage.GetPrimAtPath(record['surface'])
            surface.SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps(indices))
            # A filter-only attachment preserves actual owner collisions away
            # from bonded vertices; numerical frames carry no collision shape.
            filter_=PhysxSchema.PhysxPhysicsAttachment.Define(stage,path+'_root_filter')
            filter_.CreateActor0Rel().SetTargets([record['surface']]);filter_.CreateActor1Rel().SetTargets([owner])
            filter_.CreatePoints0Attr([]);filter_.CreatePoints1Attr([])
            filter_.CreateFilterType0Attr('Vertices');filter_.CreateCollisionFilterIndices0Attr(indices)
            filter_.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:root_filter_only',True)
            world_to_surface=np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(surface)))
            points=np.asarray(record['points'])[indices]@world_to_surface[:3,:3]+world_to_surface[3,:3]
            paths=[]
            for j,(vertex,point) in enumerate(zip(indices,points)):
                frame=frames[cursor];cursor+=1
                bond=PhysxSchema.PhysxPhysicsAttachment.Define(stage,path+'_point_'+str(j))
                bond.CreateActor0Rel().SetTargets([record['surface']]);bond.CreateActor1Rel().SetTargets([frame])
                bond.CreatePoints0Attr(Vt.Vec3fArray([Gf.Vec3f(*map(float,point))]));bond.CreatePoints1Attr(Vt.Vec3fArray([Gf.Vec3f(0.)]))
                bond.GetPrim().SetCustomDataByKey(OWNER,owner);bond.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps([vertex]))
                bond.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',record['basis']);paths.append(str(bond.GetPath()))
            result.append({'indices':indices,'numerical_attachments':paths,'physical_rigid_owner':owner})
        verify(stage,owner)
    return result
