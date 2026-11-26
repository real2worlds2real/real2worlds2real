"""Equivalent fixed numerical components; never a real object articulation.

No collider, drive, world pin or added mechanical degree of freedom. Full
aggregate mass, COM and inertia include actual parallel-axis contributions.
"""
import json
import numpy as np
from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics

KEY='calibrated_scene_reconstruction:rigid_partition'
OWNER='calibrated_scene_reconstruction:partition_owner'
WELD='calibrated_scene_reconstruction:partition_weld'


def metadata(prim):
    value=prim.GetCustomDataByKey(KEY)
    return json.loads(value) if value else None


def _inertia(api,I):
    from scipy.spatial.transform import Rotation
    values,axes=np.linalg.eigh(I)
    if values.min()<=0:raise ValueError('Positive physical principal inertia required')
    if np.linalg.det(axes)<0:axes[:,-1]*=-1
    q=Rotation.from_matrix(axes).as_quat()
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,values)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))


def update(stage,owner,mass,*,linear_damping=None,angular_damping=None):
    prim=stage.GetPrimAtPath(owner);info=metadata(prim)
    if info is None:return
    if info['version']!=2:raise ValueError('Rebuild legacy numerical partition rather than changing its inertia convention')
    for i,path in enumerate([owner,*info['carriers']]):
        body=stage.GetPrimAtPath(path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Missing fixed numerical rigid component')
        api=UsdPhysics.MassAPI.Apply(body);api.CreateMassAttr(float(info['weights'][i]*mass))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,info['component_centres'][0] if i==0 else [0.,0.,0.])))
        _inertia(api,np.array(info['component_inertia_per_kg'][i])*mass)
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        if linear_damping is not None:rb.CreateLinearDampingAttr(float(linear_damping))
        if angular_damping is not None:rb.CreateAngularDampingAttr(float(angular_damping))
    info['source_mass_kg']=float(mass);prim.SetCustomDataByKey(KEY,json.dumps(info))


def create(stage,owner,part,count,*,anchors):
    from calibrated_scene_reconstruction.assets.rigid_partition import from_surface
    prim=stage.GetPrimAtPath(owner)
    if metadata(prim) is not None:raise ValueError('A physical owner can be partitioned only once')
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
        raise ValueError('Actual dynamic source rigid required')
    parent=prim
    while parent:
        if parent.HasAPI(UsdPhysics.ArticulationRootAPI):raise ValueError('Partitions in articulations need separate backend qualification')
        parent=parent.GetParent()
    mass=UsdPhysics.MassAPI(prim).GetMassAttr().Get()
    p=from_surface(part['V'],part['F'],mass,count,anchors=anchors)
    local=UsdGeom.Xformable(prim).GetLocalTransformation();matrix=np.asarray(local)
    if not np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6):raise ValueError('Bake source metric scale first')
    rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    position=rb.GetSolverPositionIterationCountAttr().Get();velocity=rb.GetSolverVelocityIterationCountAttr().Get()
    frames=[];welds=[]
    for i,anchor in enumerate(p['centres'][1:]):
        path=owner+'_bond_frame_'+str(i);frame=UsdGeom.Xform.Define(stage,path).GetPrim()
        origin=Gf.Matrix4d(1.);origin.SetTranslate(Gf.Vec3d(*map(float,anchor)))
        UsdGeom.Xformable(frame).MakeMatrixXform().Set(origin*local)
        UsdPhysics.RigidBodyAPI.Apply(frame);api=PhysxSchema.PhysxRigidBodyAPI.Apply(frame)
        api.CreateSolverPositionIterationCountAttr(position);api.CreateSolverVelocityIterationCountAttr(velocity)
        api.CreateSleepThresholdAttr(0.);api.CreateStabilizationThresholdAttr(0.)
        frame.SetCustomDataByKey(OWNER,owner)
        weld_path=owner+'_bond_weld_'+str(i);weld=UsdPhysics.FixedJoint.Define(stage,weld_path)
        weld.CreateBody0Rel().SetTargets([owner]);weld.CreateBody1Rel().SetTargets([path])
        weld.CreateLocalPos0Attr(Gf.Vec3f(*map(float,anchor)));weld.CreateLocalPos1Attr(Gf.Vec3f(0.))
        weld.CreateLocalRot0Attr(Gf.Quatf(1.));weld.CreateLocalRot1Attr(Gf.Quatf(1.))
        weld.CreateCollisionEnabledAttr(False);weld.GetPrim().SetCustomDataByKey(WELD,owner)
        frames.append(path);welds.append(weld_path)
    info={'version':2,'carriers':frames,'welds':welds,'source_centre':p['source_centre'].tolist(),
        'source_inertia_per_kg':p['source_inertia_per_kg'].tolist(),'source_volume_m3':p['source_volume_m3'],
        'source_mass_kg':float(mass),'weights':p['weights'].tolist(),
        'component_centres':p['centres'].tolist(),'component_inertia_per_kg':(p['inertias']/mass).tolist(),
        'mechanical_degrees_of_freedom_added':0,
        'basis':'Ordinary fixed numerical components located at real bond points; aggregate mass/COM/full inertia preserved including parallel axes.'}
    prim.SetCustomDataByKey(KEY,json.dumps(info));update(stage,owner,mass)
    return frames


def verify(stage,owner):
    from pxr import Usd
    from scipy.spatial.transform import Rotation
    from calibrated_scene_reconstruction.assets.rigid_partition import parallel_axis
    from calibrated_scene_reconstruction.assets.inertia_metric import relative_energy_error
    prim=stage.GetPrimAtPath(owner);info=metadata(prim)
    if not info:return None
    paths=info['carriers']
    if info.get('version')!=2 or not paths or len(paths)!=len(info['welds']):raise ValueError('Invalid numerical rigid partition')
    cache=UsdGeom.XformCache();matrix=np.asarray(cache.GetLocalToWorldTransform(prim));c=np.array(info['source_centre'])
    masses=[];inertias=[];centres=[]
    for i,path in enumerate([owner,*paths]):
        body=stage.GetPrimAtPath(path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
            raise ValueError('Numerical components must be actual dynamic rigid bodies')
        api=UsdPhysics.MassAPI(body);kg=float(api.GetMassAttr().Get());masses.append(kg)
        com=np.array(api.GetCenterOfMassAttr().Get(),float)
        expected_com=np.array(info['component_centres'][0]) if i==0 else np.zeros(3)
        if not np.allclose(com,expected_com,atol=1e-7):raise ValueError('Numerical rigid component COM changed')
        centre=np.array(info['component_centres'][i]);centres.append(centre)
        q=api.GetPrincipalAxesAttr().Get();R=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
        I=R@np.diag(np.array(api.GetDiagonalInertiaAttr().Get(),float))@R.T
        expected_I=np.array(info['component_inertia_per_kg'][i])*info['source_mass_kg']
        if relative_energy_error(expected_I,I)>2e-5:
            raise ValueError('Numerical component inertia changed')
        inertias.append(I+kg*parallel_axis(centre-c))
        if not np.isclose(kg,info['weights'][i]*info['source_mass_kg'],rtol=1e-6):raise ValueError('Numerical mass fraction changed')
        if i:
            if body.GetCustomDataByKey(OWNER)!=owner or any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(body)):
                raise ValueError('Numerical frame must be collider-free with its physical owner')
            actual=np.asarray(cache.GetLocalToWorldTransform(body))
            if not np.allclose(actual[:3,:3],matrix[:3,:3],atol=1e-6) or not np.allclose(actual[3,:3],centre@matrix[:3,:3]+matrix[3,:3],atol=1e-6):
                raise ValueError('Numerical frame moved away from its real bond anchor')
            weld=UsdPhysics.FixedJoint.Get(stage,info['welds'][i-1])
            if (not weld or list(map(str,weld.GetBody0Rel().GetTargets()))!=[owner]
                    or list(map(str,weld.GetBody1Rel().GetTargets()))!=[path]
                    or weld.GetPrim().GetCustomDataByKey(WELD)!=owner
                    or not np.allclose(weld.GetLocalPos0Attr().Get(),centre,atol=1e-7)
                    or not np.allclose(weld.GetLocalPos1Attr().Get(),0.,atol=1e-7)
                    or weld.GetLocalRot0Attr().Get()!=Gf.Quatf(1.) or weld.GetLocalRot1Attr().Get()!=Gf.Quatf(1.)
                    or weld.GetJointEnabledAttr().Get() is False):
                raise ValueError('Missing or changed actual fixed numerical weld')
            if any(weld.GetPrim().HasAPI(UsdPhysics.DriveAPI,axis) for axis in ('linear','angular')):
                raise ValueError('Numerical weld cannot introduce a drive')
    mass=info['source_mass_kg']
    if not np.isclose(sum(masses),mass,rtol=1e-6) or not np.allclose(np.average(centres,axis=0,weights=masses),c,atol=1e-8):
        raise ValueError('Numerical partition changed source mass or COM')
    if relative_energy_error(np.array(info['source_inertia_per_kg'])*mass,np.sum(inertias,axis=0))>2e-5:
        raise ValueError('Numerical partition changed source full inertia')
    return info
