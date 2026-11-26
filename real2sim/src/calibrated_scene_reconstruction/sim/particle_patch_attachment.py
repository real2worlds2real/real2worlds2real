"""Experimental one-frame-per-root-patch native particle connections.

Each real node retains its own native bond, while nodes on the same evidenced
bond footprint share an equivalent rigid frame. Source mass, COM and full
inertia are retained. This needs native coupled-force and task qualification;
authoring equivalence alone never approves the backend.
"""
import json
from collections import defaultdict
import numpy as np
from pxr import Gf,PhysxSchema,UsdGeom,Vt


def author_many(stage,records):
    from .rigid_partition import create,verify,OWNER
    grouped=defaultdict(list)
    for record in records:grouped[record['rigid']].append(record)
    result=[]
    for owner,items in grouped.items():
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(owner)))
        inverse=np.linalg.inv(matrix)
        anchors=np.array([np.asarray(r['points'])[r['indices']].mean(0) for r in items])
        frames=create(stage,owner,items[0]['rigid_geometry'],len(items),
                      anchors=anchors@inverse[:3,:3]+inverse[3,:3])
        for record,frame in zip(items,frames):
            surface=stage.GetPrimAtPath(record['surface']);indices=np.asarray(record['indices'],int)
            surface.SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps(indices.tolist()))
            filter_=PhysxSchema.PhysxPhysicsAttachment.Define(stage,record['path']+'_root_filter')
            filter_.CreateActor0Rel().SetTargets([record['surface']]);filter_.CreateActor1Rel().SetTargets([owner])
            filter_.CreatePoints0Attr([]);filter_.CreatePoints1Attr([])
            filter_.CreateFilterType0Attr('Vertices');filter_.CreateCollisionFilterIndices0Attr(indices.tolist())
            filter_.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:root_filter_only',True)
            mesh_inverse=np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(surface)))
            frame_inverse=np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(frame))))
            world=np.asarray(record['points'])[indices]
            local0=world@mesh_inverse[:3,:3]+mesh_inverse[3,:3]
            local1=world@frame_inverse[:3,:3]+frame_inverse[3,:3]
            paths=[]
            for vertex,p0,p1 in zip(indices,local0,local1):
                path=record['path']+'_point_'+str(int(vertex))
                bond=PhysxSchema.PhysxPhysicsAttachment.Define(stage,path)
                bond.CreateActor0Rel().SetTargets([record['surface']]);bond.CreateActor1Rel().SetTargets([frame])
                bond.CreatePoints0Attr(Vt.Vec3fArray([Gf.Vec3f(*map(float,p0))]))
                bond.CreatePoints1Attr(Vt.Vec3fArray([Gf.Vec3f(*map(float,p1))]))
                prim=bond.GetPrim();prim.SetCustomDataByKey(OWNER,owner)
                prim.SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps([int(vertex)]))
                prim.SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',record['basis'])
                paths.append(path)
            result.append({'indices':indices.tolist(),'numerical_attachments':paths,
                           'physical_rigid_owner':owner,'backend':'patches'})
        verify(stage,owner)
    return result
