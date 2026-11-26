"""Explicit rigid mass moments from original closed source geometry.

Collider cooking and asynchronous automatic inertia calculation must not define
the inertia of a separately authored source model. This retains the declared
uniform-material hypothesis, not an identified real density distribution.
"""
import hashlib
import json
import numpy as np
import trimesh
from pxr import Gf,UsdGeom,UsdPhysics

KEY='calibrated_scene_reconstruction:source_mass_moments'


def author(prim,vertices,faces,mass):
    v=np.asarray(vertices,float);f=np.asarray(faces,int)
    # Generated metric models normally encode size in vertices. Preserve the
    # legacy path for externally scaled instances until its native body-frame
    # convention has independent qualification.
    a=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))[:3,:3]
    if not np.allclose(a@a.T,np.eye(3),atol=1e-6):return None
    solid=trimesh.Trimesh(v,f,process=True)
    if not solid.is_volume:raise ValueError('Source mass moments need an outward closed physical solid')
    record={'centre':solid.center_mass.tolist(),'inertia_per_kg':(solid.moment_inertia/solid.volume).tolist(),
            'source_geometry_sha256':hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest(),
            'basis':'Uniform density on original closed source geometry; density remains the declared material prior. No added stabilization inertia.'}
    prim.SetCustomDataByKey(KEY,json.dumps(record))
    return update(prim,mass)


def update(prim,mass):
    stored=prim.GetCustomDataByKey(KEY)
    if stored is None:return None
    if isinstance(mass,bool) or not np.isfinite(mass) or mass<=0:raise ValueError('Positive declared rigid mass required')
    record=json.loads(stored);centre=np.asarray(record['centre'],float)
    tensor=np.asarray(record['inertia_per_kg'],float)*mass
    if (centre.shape!=(3,) or tensor.shape!=(3,3) or not np.isfinite(centre).all()
            or not np.isfinite(tensor).all() or not np.allclose(tensor,tensor.T,atol=1e-15)):
        raise ValueError('Finite symmetric source mass moments required')
    from scipy.spatial.transform import Rotation
    values,axes=np.linalg.eigh(tensor)
    if values.min()<=0:raise ValueError('Positive physical source inertia required')
    if np.linalg.det(axes)<0:axes[:,-1]*=-1
    q=Rotation.from_matrix(axes).as_quat();api=UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass));api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,centre)))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,values)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
    return {**record,'mass_kg':float(mass),'inertia':tensor.tolist()}
