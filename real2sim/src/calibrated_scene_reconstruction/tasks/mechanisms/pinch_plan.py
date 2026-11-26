"""CAD-only antipodal contact proposals; native control/contact proof is required.

Contact points and normals must come from actual source collision geometry.
No cloth geometry, state targets, attachments or material fitting enter this
kinematic planner. It does not establish a friction grasp by itself.
"""
import numpy as np
from scipy.optimize import least_squares


def pad_geometry(transforms, contacts):
    """Measure source contact references in supplied actual rigid-link frames.

    This is not a contact force or grasp certificate. In particular, the
    distance between two chosen pad references is not a minimum mesh gap.
    """
    if len(contacts) != 2:
        raise ValueError('Two source pad contacts required')
    points, normals = [], []
    for contact in contacts:
        tf = np.asarray(transforms[contact['link']], float)
        point, normal = np.asarray(contact['point'], float), np.asarray(contact['normal'], float)
        if (tf.shape != (4, 4) or point.shape != (3,) or normal.shape != (3,)
                or not all(np.isfinite(v).all() for v in (tf, point, normal))
                or not np.allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(tf[:3, :3]), 1.)
                or not np.isclose(np.linalg.norm(normal), 1.)):
            raise ValueError('Finite rigid link frames and unit pad normals required')
        points.append(tf[:3, :3] @ point + tf[:3, 3])
        normals.append(tf[:3, :3] @ normal)
    delta = points[1] - points[0]
    return {'reference_point_gap_m': float(np.linalg.norm(delta)),
            'signed_normal_gap_m': float(delta @ normals[0]),
            'opposition_dot': float(normals[0] @ normals[1]),
            'points': np.asarray(points).tolist(),
            'scope': 'Actual source pad reference geometry only; not minimum surface separation, contact force or task approval'}


def propose(model,contacts,seed_state,*,closed_gap=.0001,open_gap=.020,attempts=16,search_bounds=None):
    if len(contacts)!=2 or not 0<closed_gap<open_gap<.1 or not 1<=attempts<=64:
        raise ValueError('Two source contacts, bounded gaps and search budget required')
    by_child={j.child:j for j in model.joints};selected=set()
    for contact in contacts:
        child=contact['link']
        while child!=model.root:
            j=by_child[child]
            if j.kind!='fixed':selected.add(j.name)
            child=j.parent
    joints=[j for j in model.movable if j.name in selected];names=[j.name for j in joints]
    lo=np.array([j.lower for j in joints]);hi=np.array([j.upper for j in joints])
    search_bounds={} if search_bounds is None else search_bounds
    if not isinstance(search_bounds,dict) or not set(search_bounds)<=set(names):
        raise ValueError('Search bounds must name actual participating source joints')
    for name,interval in search_bounds.items():
        limits=np.asarray(interval,float);index=names.index(name)
        if (limits.shape!=(2,) or not np.isfinite(limits).all()
                or not lo[index]<=limits[0]<limits[1]<=hi[index]):
            raise ValueError('Proposal search bounds must stay within actual source joint limits')
        lo[index],hi[index]=limits
    seed=np.clip([seed_state.get(n,0.) for n in names],lo+1e-8,hi-1e-8)
    def frames(q):
        state={**seed_state,**dict(zip(names,q))};tf=model.transforms(state);result=[]
        for contact in contacts:
            t=tf[contact['link']];result.extend([t[:3,:3]@contact['point']+t[:3,3],t[:3,:3]@contact['normal']])
        return result
    a,n,b,k=frames(seed);centre=(a+b)/2
    def residual(q):
        a,n,b,k=frames(q)
        return np.r_[(b-a-closed_gap*n)*1000,(n+k)*100,((a+b)/2-centre)*2,(q-seed)*.001]
    rng=np.random.default_rng(251);best=None
    for attempt in range(attempts):
        fit=least_squares(residual,seed if attempt==0 else rng.uniform(lo,hi),bounds=(lo,hi),max_nfev=1000)
        if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
    a,n,b,k=frames(best.x);centre=(a+b)/2;axis=n-k;axis/=np.linalg.norm(axis)
    down=centre-axis*(centre@axis)
    if np.linalg.norm(down)<1e-6:raise ValueError('Contact configuration cannot define an overhead wrist fixture')
    down/=np.linalg.norm(down);side=np.cross(down,axis)
    rotation=np.diag([1.,-1.,-1.])@np.stack([axis,side,down],axis=1).T
    offset=-rotation@centre
    def opening(q):
        pa,na,pb,nb=frames(q)
        # The open fingers must bracket the actual material. Full antipodal
        # normals are essential at the closed friction test, while an empty
        # preshape can rotate its contact surfaces to reach that opening.
        return np.r_[(pa-(centre-axis*open_gap/2))*1000,(pb-(centre+axis*open_gap/2))*1000,
                     (na-n),(nb-k),(q-best.x)*.001]
    opened=least_squares(opening,best.x,bounds=(lo,hi),max_nfev=2000)
    def describe(q):
        pa,na,pb,nb=frames(q)
        return {'gap_m':float((pb-pa)@axis),'normal_dot':float(na@nb),
                'normal_z':[(rotation@na)[2],(rotation@nb)[2]],
                'midpoint_shift_m':float(np.linalg.norm((pa+pb)/2-centre))}
    return {'proposal_only':True,'native_validated':False,'task_approved':False,
            'open_state':{**seed_state,**dict(zip(names,opened.x.tolist()))},
            'closed_state':{**seed_state,**dict(zip(names,best.x.tolist()))},
            'R':rotation.tolist(),'p':offset.tolist(),
            'diagnostics':{'closed':describe(best.x),'open':describe(opened.x),
                           'joint_search_bounds':{n:np.asarray(v,float).tolist() for n,v in search_bounds.items()},
                           'source_joint_limits_unchanged':True,
                           'wrist_height_over_contact_m':float(offset[2])}}
