"""Explicit physical root bonds for native surface/rigid assemblies.

Use only for an evidenced real connection, never to hold a loose object still.
The installed backend's attachment stiffness/damping are not modeled here.
Dynamic rigid-drive controls qualify this path; coupled nodal forcing does not.
"""
import numpy as np


from calibrated_scene_reconstruction.assets.surface_bindings import binding


def audit_filters(stage, surface, rigid, indices):
    """Verify that only the declared bonded faces lose owner contact."""
    from pxr import Usd, UsdGeom, UsdPhysics
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(surface))
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not len(counts) or not (counts==3).all():
        raise ValueError('Root contact audit requires actual triangle material geometry')
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    expected=np.flatnonzero(np.isin(faces,indices).any(axis=1)).tolist()
    if not expected or len(expected)==len(faces):
        raise ValueError('Root contact filter must retain a free material surface')
    owner=stage.GetPrimAtPath(rigid);colliders=set()
    for prim in Usd.PrimRange(owner):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if body==owner:colliders.add(str(prim.GetPath()))
    if not colliders:raise ValueError('Root contact owner has no actual collision surfaces')
    found=set()
    for prim in Usd.PrimRange(owner.GetParent()):
        if prim.GetTypeName()!='OmniPhysicsElementCollisionFilter':continue
        a=list(map(str,prim.GetRelationship('omniphysics:src0').GetTargets()))
        b=list(map(str,prim.GetRelationship('omniphysics:src1').GetTargets()))
        if a!=[surface] or not set(b)&colliders:continue
        def values(name):return list(prim.GetAttribute('omniphysics:'+name).Get() or [])
        if (len(b)!=1 or b[0] in found or values('groupElemCounts0')!=[len(expected)]
                or sorted(values('groupElemIndices0'))!=expected
                or values('groupElemCounts1')!=[0] or values('groupElemIndices1')):
            raise ValueError('Root contact filter differs from the declared bonded material faces')
        found.add(b[0])
    if found!=colliders:
        raise ValueError('Root contact filters do not cover every owned collider')
    return {'surface':surface,'rigid':rigid,'filtered_root_faces':expected,
            'unfiltered_free_faces':len(faces)-len(expected),'colliders':sorted(found)}


def author_many(stage,records):
    """Author real bonds, optionally through an equivalent rigid partition.

    Partitioned frames are ordinary fixed rigid bodies. They preserve source
    mass/inertia and introduce no world constraint or articulation freedom.
    The direct beta multipoint path still needs coupled-force qualification.
    """
    from collections import defaultdict
    from pxr import Gf,UsdGeom,Vt
    from .rigid_partition import create,verify,OWNER
    groups=defaultdict(list)
    for record in records:
        mode=record.get('backend','direct')
        if mode not in {'direct','partitioned','centered','patches'}:raise ValueError('Unknown physical root numerical backend')
        groups[record['rigid']].append(record)
    frames={};cursors={}
    for owner,items in groups.items():
        modes={r.get('backend','direct') for r in items}
        if len(modes)!=1:raise ValueError('A rigid physical owner must use one consistent root backend')
        if modes=={'partitioned'}:
            world=np.concatenate([np.asarray(r['points'])[r['indices']] for r in items])
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(owner)))
            inverse=np.linalg.inv(matrix);anchors=world@inverse[:3,:3]+inverse[3,:3]
            frames[owner]=create(stage,owner,items[0]['rigid_geometry'],len(anchors),anchors=anchors);cursors[owner]=0
        elif modes=={'centered'}:
            # One well-conditioned equivalent frame at the source COM. Unlike
            # per-node frames this has half the source mass/full inertia, not
            # hundreds of tiny masses. The ordinary fixed weld preserves the
            # aggregate source moments and introduces no world constraint.
            import trimesh
            part=items[0]['rigid_geometry']
            solid=trimesh.Trimesh(part['V'],part['F'],process=True)
            frames[owner]=create(stage,owner,part,1,anchors=np.asarray(solid.center_mass)[None])
        elif modes=={'patches'}:
            # One frame per actual finite bond footprint avoids both a long
            # COM-to-root lever arm and the tiny masses of per-node frames.
            # The partition still preserves source mass/COM/full inertia.
            world=np.array([np.asarray(r['points'])[r['indices']].mean(0) for r in items])
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(owner)))
            inverse=np.linalg.inv(matrix);anchors=world@inverse[:3,:3]+inverse[3,:3]
            frames[owner]=create(stage,owner,items[0]['rigid_geometry'],len(items),anchors=anchors)
            cursors[owner]=0
    reports=[]
    for record in records:
        import json
        if record.get('clamp') is not None:
            from calibrated_scene_reconstruction.assets.bond_footprint import measure
            footprint=measure(record['points'],record['faces'],record['indices'],**record['clamp'])
            if not footprint['pass']:raise ValueError('Actual native root has no declared finite clamp footprint')
            stage.GetPrimAtPath(record['surface']).SetCustomDataByKey('calibrated_scene_reconstruction:clamp_footprint',json.dumps(footprint))
        stage.GetPrimAtPath(record['surface']).SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps(list(map(int,record['indices']))))
        data=author(stage,record['path'],record['surface'],record['rigid'],record['points'],record['faces'],record['indices'],
                    connection_basis=record['basis'])
        if record['rigid'] in frames:
            stage.RemovePrim(record['path']);paths=[]
            if record.get('backend') in {'centered','patches'}:
                owner=record['rigid'];index=cursors[owner] if record.get('backend')=='patches' else 0
                frame=frames[owner][index]
                if record.get('backend')=='patches':cursors[owner]+=1
                matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(frame)))
                inverse=np.linalg.inv(matrix)
                points=np.asarray(record['points'])[record['indices']]@inverse[:3,:3]+inverse[3,:3]
                prim=stage.DefinePrim(record['path'],'OmniPhysicsVtxXformAttachment')
                prim.GetRelationship('omniphysics:src0').SetTargets([record['surface']])
                prim.GetRelationship('omniphysics:src1').SetTargets([frame])
                prim.GetAttribute('omniphysics:vtxIndicesSrc0').Set(list(map(int,record['indices'])))
                prim.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(np.asarray(points,np.float32)))
                prim.SetCustomDataByKey(OWNER,record['rigid'])
                prim.SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',record['basis'])
                data.update(numerical_attachments=[record['path']],physical_rigid_owner=record['rigid'])
                reports.append(data)
                continue
            for i,vertex in enumerate(record['indices']):
                owner=record['rigid'];frame=frames[owner][cursors[owner]];cursors[owner]+=1
                matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(frame)))
                inverse=np.linalg.inv(matrix);point=np.asarray(record['points'])[vertex]@inverse[:3,:3]+inverse[3,:3]
                path=record['path']+'_point_'+str(i);prim=stage.DefinePrim(path,'OmniPhysicsVtxXformAttachment')
                prim.GetRelationship('omniphysics:src0').SetTargets([record['surface']]);prim.GetRelationship('omniphysics:src1').SetTargets([frame])
                prim.GetAttribute('omniphysics:vtxIndicesSrc0').Set([int(vertex)])
                prim.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray([Gf.Vec3f(*map(float,point))]))
                prim.SetCustomDataByKey(OWNER,owner);prim.SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',record['basis'])
                paths.append(path)
            data['numerical_attachments']=paths;data['physical_rigid_owner']=record['rigid']
        reports.append(data)
    for owner in frames:verify(stage,owner)
    return reports


def author(stage, path, surface, rigid, points, faces, indices, *, connection_basis):
    from pxr import Usd,UsdGeom,UsdPhysics,Vt
    if not isinstance(connection_basis,str) or not connection_basis.strip():
        raise ValueError('A real physical root bond needs its observed/inferred construction basis')
    body=stage.GetPrimAtPath(rigid)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Physical root must reference an actual rigid body')
    if UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
        raise ValueError('This qualified root path requires a dynamic rigid body')
    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(body))
    data=binding(points,faces,indices,matrix)
    attachment=stage.DefinePrim(path,'OmniPhysicsVtxXformAttachment')
    attachment.GetRelationship('omniphysics:src0').SetTargets([surface])
    attachment.GetRelationship('omniphysics:src1').SetTargets([rigid])
    attachment.GetAttribute('omniphysics:vtxIndicesSrc0').Set(data['indices'])
    attachment.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(np.asarray(data['local_positions'],np.float32)))
    colliders=[]
    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        owner=prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
        if owner==body:colliders.append(prim)
    if not colliders:raise ValueError('Physical root body has no owned collision surface')
    for index,collider in enumerate(colliders):
        filtering=stage.DefinePrim(path+f'_root_filter_{index}','OmniPhysicsElementCollisionFilter')
        filtering.GetRelationship('omniphysics:src0').SetTargets([surface])
        filtering.GetRelationship('omniphysics:src1').SetTargets([collider.GetPath()])
        filtering.GetAttribute('omniphysics:groupElemCounts0').Set([len(data['filtered_root_faces'])])
        filtering.GetAttribute('omniphysics:groupElemIndices0').Set(data['filtered_root_faces'])
        filtering.GetAttribute('omniphysics:groupElemCounts1').Set([0])
        filtering.GetAttribute('omniphysics:groupElemIndices1').Set([])
    data['filtered_root_colliders']=[str(p.GetPath()) for p in colliders]
    attachment.SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',connection_basis)
    return data
