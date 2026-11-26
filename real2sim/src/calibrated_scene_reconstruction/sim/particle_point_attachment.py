"""Experimental independent particle/rigid point bonds without mass frames.

Unlike a multipoint legacy attachment, each declared node has an independent
native constraint. No numerical rigid body or mass is added. Qualification
still requires native passive, coupled-load, contact and complete task tests.
"""
import json
import numpy as np
from pxr import Gf,PhysxSchema,UsdGeom,Vt


def author_many(stage,records):
    result=[]
    for record in records:
        surface=stage.GetPrimAtPath(record['surface']);owner=record['rigid']
        indices=np.asarray(record['indices'],int);world=np.asarray(record['points'])[indices]
        surface.SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps(indices.tolist()))
        inverse0=np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(surface)))
        inverse1=np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(owner))))
        local0=world@inverse0[:3,:3]+inverse0[3,:3]
        local1=world@inverse1[:3,:3]+inverse1[3,:3]
        paths=[]
        for vertex,p0,p1 in zip(indices,local0,local1):
            path=record['path']+'_point_'+str(int(vertex))
            bond=PhysxSchema.PhysxPhysicsAttachment.Define(stage,path)
            bond.CreateActor0Rel().SetTargets([record['surface']]);bond.CreateActor1Rel().SetTargets([owner])
            bond.CreatePoints0Attr(Vt.Vec3fArray([Gf.Vec3f(*map(float,p0))]))
            bond.CreatePoints1Attr(Vt.Vec3fArray([Gf.Vec3f(*map(float,p1))]))
            bond.CreateFilterType0Attr('Vertices');bond.CreateCollisionFilterIndices0Attr([int(vertex)])
            prim=bond.GetPrim();prim.SetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices',json.dumps([int(vertex)]))
            prim.SetCustomDataByKey('calibrated_scene_reconstruction:physical_connection_basis',record['basis'])
            paths.append(path)
        result.append({'indices':indices.tolist(),'numerical_attachments':paths,
                       'physical_rigid_owner':owner,'backend':'individual'})
    return result
