"""Bounded initial-condition hypotheses, never runtime response controllers."""
import numpy as np
from scipy.spatial.transform import Rotation
from .contract import require
from .derived import place


def apply(stage,actors,specs,parameters,*,prefix=''):
    """Move explicitly grouped free responses once before native reset/settling.

    Angles use radians, translations metres. Spec order defines composition
    when parameters share a group. Relative meshes/mass properties/constraints
    within each group stay unchanged. Priors, bounds and source evidence belong
    to the process plan; native source/contact gates still decide acceptance.
    """
    from pxr import UsdGeom,UsdPhysics
    rows={}
    for name,spec in specs.items():
        require(name in parameters and np.isfinite(parameters[name]),'Finite declared initial pose parameter required')
        roots=[prefix+p for p in spec['roots']];targets=set(spec['bodies']);kind=spec['kind']
        require(roots and len(roots)==len(set(roots)) and all(stage.GetPrimAtPath(p) for p in roots)
                and kind in {'rotation','translation'},'Explicit native initial pose roots and transform kind required')
        require(not any(a!=b and a.startswith(b+'/') for a in roots for b in roots),'Initial pose roots must not overlap')
        covered={k for k,v in actors.items() if any(v['path']==p or v['path'].startswith(p+'/') for p in roots)}
        require(covered==targets and covered and all(actors[k]['kind']!='static' for k in covered),
                'Initial pose group must exactly cover its declared dynamic bodies, without supports')
        paths={actors[k]['path'] for k in covered}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):continue
            ends=[prim.GetRelationship(r).GetTargets() for r in ['physics:body0','physics:body1']]
            connected=[str(p) for e in ends for p in e]
            if paths.intersection(connected):
                require(all(e and all(str(p) in paths for p in e) for e in ends),
                        'Initial pose group has an external or world joint; use a model-specific initializer')
        reference=spec.get('reference_body');frame=np.eye(4)
        if reference:
            require(reference in actors and reference not in covered,'Reference body must remain outside the initial pose group')
            frame=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(actors[reference]['path']))).T
        axis=np.asarray(spec['axis'],float)
        require(axis.shape==(3,) and np.isfinite(axis).all() and np.linalg.norm(axis)>1e-10,
                'Finite nonzero initial pose axis required')
        axis=frame[:3,:3]@axis;axis/=np.linalg.norm(axis);H=np.eye(4);value=float(parameters[name])
        if kind=='rotation':
            pivot=np.asarray(spec['pivot'],float)
            require(pivot.shape==(3,) and np.isfinite(pivot).all(),'Explicit finite rotation pivot required')
            pivot=frame[:3,:3]@pivot+frame[:3,3];H[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix();H[:3,3]=pivot-H[:3,:3]@pivot
        else:H[:3,3]=axis*value
        cache=UsdGeom.XformCache();originals={p:np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T for p in roots}
        if value!=0:
            for path,old in originals.items():place(stage,path,H@old)
        rows[name]={'value':value,'unit':'rad' if kind=='rotation' else 'm','bodies':sorted(covered),
                    'roots':roots,'world_transform':H.tolist(),'scope':'Applied once before reset and free settling; no runtime response targets or new constraints.'}
    return rows
