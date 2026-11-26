"""Check declared furniture fronts against explicitly annotated source views.

A symmetric silhouette cannot distinguish a cabinet's doors from its back.
The agent must identify front parts and photographs that actually show them.
This checks their exported facing, not object dimensions or photo realism.
"""
from pathlib import Path
import numpy as np
from calibrated_scene_reconstruction.scene.review.environment import digest


def facing(front,body_centre,part_centre,camera):
    front=np.asarray(front,float);front=front/np.linalg.norm(front)
    delta=np.asarray(camera)-part_centre
    return float((np.asarray(part_centre)-body_centre)@front),float(delta@front/max(1e-12,np.linalg.norm(delta)))


def audit(asset,checks,camera_path,image_root):
    from pxr import Usd,UsdGeom
    if not isinstance(checks,list):raise ValueError('Front-view checks must be a list')
    if not checks:return {'pass':True,'checks':[],'input_sha256':{}}
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();C=np.load(camera_path);names=list(map(str,C['names']))
    inputs={str(Path(p).resolve()):digest(p) for p in [camera_path,Path(__file__).resolve()]};rows=[]
    for spec in checks:
        entity=spec['entity'];axis=np.asarray(spec['local_front'],float);frames=spec['frames'];parts=spec['front_parts'];limit=spec.get('minimum_facing_cosine',.1)
        if (axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-6
                or len(set(frames))<2 or not set(frames)<=set(names) or not parts or not .05<=limit<=.5):
            raise ValueError('Explicit front axis/parts, two registered front photographs and bounded facing tolerance required')
        roots=[p for p in stage.Traverse() if p.GetName()==entity and UsdGeom.Xformable(p)]
        if len(roots)!=1:raise ValueError('Front-view check needs one exported entity root: '+entity)
        references=spec.get('reference_parts')
        if references is not None and (not isinstance(references,list) or not references or set(references)&set(parts)):
            raise ValueError('Distinct explicit rear reference parts required')
        root=roots[0];matrix=np.asarray(cache.GetLocalToWorldTransform(root));front=(np.r_[axis,0]@matrix)[:3];body=[];face=[];found=set();rear=[];rear_found=set()
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):continue
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
            world=(np.c_[points,np.ones(len(points))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:,:3];body.append(world)
            matched=set(str(prim.GetPath()).split('/'))&set(parts)
            if matched:face.append(world);found|=matched
            matched=set(str(prim.GetPath()).split('/'))&set(references or [])
            if matched:rear.append(world);rear_found|=matched
        if set(parts)!=found:raise ValueError('Declared front parts absent from exported furniture')
        if references is not None and set(references)!=rear_found:raise ValueError('Declared rear reference parts absent from exported furniture')
        vertices=np.concatenate(rear if references is not None else body);body_centre=(vertices.min(0)+vertices.max(0))/2
        vertices=np.concatenate(face);part_centre=(vertices.min(0)+vertices.max(0))/2;views=[]
        for frame in frames:
            image=Path(image_root)/frame;inputs[str(image.resolve())]=digest(image)
            offset,cosine=facing(front,body_centre,part_centre,C['centre'][names.index(frame)])
            views.append({'frame':frame,'front_offset_m':offset,'camera_facing_cosine':cosine,'pass':offset>1e-4 and cosine>=limit})
        rows.append({'entity':entity,'reference_parts':references,'views':views,'pass':all(v['pass'] for v in views)})
    return {'pass':all(r['pass'] for r in rows),'checks':rows,'input_sha256':inputs,
            'scope':'Declared front parts must be on the stated front and face the annotated source cameras. Manual completeness, source/free-view and physical review still required.'}
