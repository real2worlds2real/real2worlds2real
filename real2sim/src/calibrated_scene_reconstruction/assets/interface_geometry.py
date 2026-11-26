"""Read-only clearance at explicitly reviewed rigid mating surfaces."""
import hashlib
import numpy as np
from calibrated_scene_reconstruction.assets.ray_triangles import hits as _hits


def measure(origins, directions, triangles, *, maximum_gap_m):
    p,n,t=map(lambda x:np.asarray(x,float),[origins,directions,triangles])
    if (p.ndim!=2 or p.shape[1:]!=(3,) or n.shape!=p.shape or not len(p)
            or t.ndim!=3 or t.shape[1:]!=(3,3) or not len(t)
            or not all(np.isfinite(x).all() for x in [p,n,t])
            or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6)
            or not np.isfinite(maximum_gap_m) or maximum_gap_m<=0):
        raise ValueError('Reviewed finite mating-surface rays, triangles and positive gap limit required')
    gaps=[]
    for point,direction in zip(p,n):
        distances,ids=_hits(t,point-maximum_gap_m*direction,direction)
        outward=np.cross(t[ids,1]-t[ids,0],t[ids,2]-t[ids,0])
        lengths=np.linalg.norm(outward,axis=1)
        valid=(lengths>1e-15)&(outward@direction<-.5*lengths)
        candidates=distances[valid]
        gaps.append(float(candidates[0]-maximum_gap_m) if len(candidates) else None)
    return {'pass':all(g is not None and -maximum_gap_m<=g<=maximum_gap_m for g in gaps),
            'gap_m':gaps,'maximum_allowed_gap_m':float(maximum_gap_m),
            'scope':'Finite declared mating-surface rays, actual geometry only; independent penetration, cloth seam, contact and retention checks still required'}


def from_poses(parts, plans, poses):
    """Evaluate unchanged physical meshes in measured rigid poses."""
    by_name={p['name']:p for p in parts};rows={}
    for plan in plans:
        if not plan.get('basis'):raise ValueError('Mating interface needs a source/physical basis')
        first,second=plan['first'],plan['second']
        if any(by_name[k]['kind']!='rigid' for k in [first,second]):raise ValueError('Rigid mating surfaces required')
        p,r=poses[first];q,s=poses[second]
        if 'faces' in plan:
            face=by_name[first]['F']
            if hashlib.sha256(np.asarray(face,dtype='<i8').tobytes()).hexdigest()!=plan['topology_sha256']:
                raise ValueError('Mating-surface source topology changed; review and bind new anchors')
            triangles=by_name[first]['V'][face[np.asarray(plan['faces'],int)]]
            local=triangles.mean(1);direction=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        else:
            local,direction=np.asarray(plan['points']),np.asarray(plan['normals'])
        origins=local@r.T+p
        normals=direction@r.T
        v=by_name[second]['V']@s.T+q
        rows[plan['id']]=measure(origins,normals,v[by_name[second]['F']],maximum_gap_m=plan['maximum_gap_m'])
    return {'pass':bool(rows) and all(row['pass'] for row in rows.values()),'checks':rows}
