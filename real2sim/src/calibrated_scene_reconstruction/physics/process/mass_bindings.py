"""Explicit rigid mass priors/estimates with consistent rotational inertia."""
import numpy as np
from .contract import require


def apply(stage,actors,specifications,parameters,*,reserved=()):
    from pxr import UsdPhysics,Gf
    used=set(reserved);rows=[]
    for name,item in specifications.items():
        value=parameters[name];mode=item.get('mode','scale');names=item['bodies']
        require(mode in {'scale','absolute'} and np.isfinite(value) and value>0 and names,
                'Declare positive rigid mass values and an explicit binding mode')
        for body in names:
            require(body in actors and body not in used and actors[body]['kind']=='rigid',
                    'Rigid mass bindings must be nonoverlapping and separate from the primary mass parameter')
            used.add(body);path=actors[body]['path'];prim=stage.GetPrimAtPath(path)
            require(prim.HasAPI(UsdPhysics.RigidBodyAPI),'A mass target must be a native rigid actor')
            api=UsdPhysics.MassAPI(prim);old=api.GetMassAttr().Get();inertia=api.GetDiagonalInertiaAttr().Get()
            require(old is not None and old>0 and inertia is not None and np.isfinite(inertia).all() and np.all(np.asarray(inertia)>0),
                    'Explicit positive source mass/inertia priors are required before scaling')
            mass=float(old*value if mode=='scale' else value);ratio=mass/old
            expected=np.asarray(inertia,float)*ratio
            api.GetMassAttr().Set(mass);api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*expected.tolist()))
            rows.append({'parameter':name,'body':body,'path':path,'mode':mode,'source_mass_kg':float(old),
                         'requested_mass_kg':mass,'requested_principal_inertia':expected.tolist()})
    return rows


def audit(stage,rows):
    from pxr import UsdPhysics
    return bool(rows) and all(np.isclose(UsdPhysics.MassAPI(stage.GetPrimAtPath(row['path'])).GetMassAttr().Get(),row['requested_mass_kg'],rtol=1e-6)
        and np.allclose(UsdPhysics.MassAPI(stage.GetPrimAtPath(row['path'])).GetDiagonalInertiaAttr().Get(),row['requested_principal_inertia'],rtol=1e-6,atol=1e-15) for row in rows)
