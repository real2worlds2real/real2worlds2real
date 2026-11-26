"""Check agent-declared part geometry on the actual exported background.

These are consistency checks, not a chair template or an appearance approval.
The agent identifies surfaces, their front, and source-supported relations.
"""
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest


def axes(spec):
    front=np.asarray(spec['front_axis'],float);lateral=np.asarray(spec['lateral_axis'],float)
    if (front.shape!=(3,) or lateral.shape!=(3,) or not np.isfinite([front,lateral]).all()
            or min(np.linalg.norm(front),np.linalg.norm(lateral))<1e-8):
        raise ValueError('Finite nonzero front and lateral axes required')
    front/=np.linalg.norm(front);lateral/=np.linalg.norm(lateral)
    if abs(front@lateral)>1e-5:raise ValueError('Front and lateral axes must be perpendicular')
    return front,lateral,np.cross(lateral,front)


def check(surface, support, spec):
    front,lateral,_=axes(spec);surface=np.asarray(surface,float)
    if surface.ndim!=2 or surface.shape[1]!=3 or len(surface)<12 or not np.isfinite(surface).all():
        raise ValueError('Actual finite surface vertices required')
    along=surface@lateral;lo,hi=np.percentile(along,[1,99]);span=hi-lo;middle=(lo+hi)/2
    if span<.02:raise ValueError('A resolved lateral surface profile is required')
    central=abs(along-middle)<=span*.1
    left=along<=lo+span*.1;right=along>=hi-span*.1
    if not central.any() or not left.any() or not right.any():raise ValueError('Centre and both surface edges must be sampled')
    depth=surface@front;centre_depth=float(np.median(depth[central]))
    if spec['kind']=='concavity':
        limit=float(spec['minimum_depth_ratio'])
        if not np.isfinite(limit) or not .005<=limit<=.3:raise ValueError('Bounded declared concavity required')
        values=[float((np.median(depth[side])-centre_depth)/span) for side in [left,right]]
        return {'pass':bool(min(values)>=limit),'edge_forward_ratio':values,'minimum_depth_ratio':limit}
    if spec['kind']=='rear_support':
        support=np.asarray(support,float);up=np.asarray(spec.get('up_axis',[0,0,1]),float)
        if (support.ndim!=2 or support.shape[1]!=3 or not len(support) or not np.isfinite(support).all()
                or up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-8):
            raise ValueError('Actual finite rear support and up axis required')
        up/=np.linalg.norm(up)
        if max(abs(up@front),abs(up@lateral))>1e-5:raise ValueError('Profile axes must be perpendicular')
        bottom,top=np.percentile(surface@up,[1,99]);height=support@up
        shared=(height>=bottom+.002)&(height<=top+.002)&(abs(support@lateral-middle)<span*.2)
        if not shared.any():raise ValueError('Rear support must actually reach this surface')
        allowance=float(spec.get('maximum_front_overlap_m',.001))
        if not np.isfinite(allowance) or not 0<=allowance<=.003:raise ValueError('Small explicit front overlap allowance required')
        intrusion=float(np.max(support[shared]@front)-np.max(depth[central]))
        return {'pass':intrusion<=allowance,'front_intrusion_m':intrusion,'maximum_front_overlap_m':allowance}
    raise ValueError('Known agent-declared surface relation required')


def audit(asset,checks):
    from pxr import Usd,UsdGeom
    if not isinstance(checks,list):raise ValueError('Explicit shape profile checks required')
    if not checks:return {'pass':True,'checks':[],'input_sha256':{}}
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();rows=[]
    for spec in checks:
        roots=[p for p in stage.Traverse() if p.GetName()==spec['entity'] and UsdGeom.Xformable(p)]
        if len(roots)!=1:raise ValueError('One explicit background entity root required')
        root=roots[0];inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)))
        def points(names):
            if not isinstance(names,list) or not names or len(set(names))!=len(names):raise ValueError('Distinct exported mesh part names required')
            found=set();arrays=[]
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdGeom.Mesh):continue
                matched=set(str(prim.GetPath()).split('/'))&set(names)
                if not matched:continue
                found|=matched;local=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
                transform=np.asarray(cache.GetLocalToWorldTransform(prim))@inverse
                arrays.append((np.c_[local,np.ones(len(local))]@transform)[:,:3])
            if found!=set(names):raise ValueError('Declared profile part absent from exported geometry')
            return np.concatenate(arrays)
        surface=points(spec['surface_parts']);support=points(spec['support_parts']) if spec['kind']=='rear_support' else None
        rows.append({'entity':spec['entity'],'specification':spec,**check(surface,support,spec)})
    return {'pass':all(row['pass'] for row in rows),'checks':rows,
            'input_sha256':{str(Path(p).resolve()):digest(p) for p in [asset,Path(__file__)]},
            'scope':__doc__.strip()}
