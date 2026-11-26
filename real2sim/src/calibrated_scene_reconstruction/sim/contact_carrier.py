"""Restore requested adjacent-link contact without adding mechanical freedom.

PhysX filters adjacent articulation links. A fixed, coincident collision carrier
is a non-adjacent link to the other body. Original visual geometry is retained;
colliders are copied exactly and disabled on the visual owner. The two welded
components share the original mass, COM and full inertia, rather than adding
stabilization mass or an invisible obstacle. Native qualification is separate.
"""
import hashlib
import json

import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,PhysxSchema

KEY='calibrated_scene_reconstruction:contact_carrier'
OWNER='calibrated_scene_reconstruction:contact_carrier_owner'


def metadata(prim):
    value=prim.GetCustomDataByKey(KEY)
    return json.loads(value) if value else None


def _signature(prim):
    # Visibility and collisionEnabled deliberately differ. Physics materials
    # are bound to both copies when physical priors are applied.
    excluded={'visibility','physics:collisionEnabled'}
    def digest(value):
        array=np.asarray(value)
        if array.dtype.kind in 'bifcu':
            return [array.dtype.str,array.shape,hashlib.sha256(array.tobytes()).hexdigest()]
        return str(value)
    values={a.GetName():digest(a.Get()) for a in prim.GetAttributes()
            if a.GetName() not in excluded and not a.GetName().startswith('primvars:')}
    relations={r.GetName():list(map(str,r.GetTargets())) for r in prim.GetRelationships()}
    return hashlib.sha256(json.dumps([prim.GetTypeName(),values,relations],sort_keys=True).encode()).hexdigest()


def _moments(prim,mass,record):
    from scipy.spatial.transform import Rotation
    values,axes=np.linalg.eigh(np.array(record['inertia_per_kg'])*mass)
    if values.min()<=0:raise ValueError('Positive source inertia required for contact carrier')
    if np.linalg.det(axes)<0:axes[:,-1]*=-1
    q=Rotation.from_matrix(axes).as_quat();api=UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass));api.CreateCenterOfMassAttr(Gf.Vec3f(*record['centre']))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,values)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))


def update(stage,owner,mass,*,linear_damping=None,angular_damping=None):
    prim=stage.GetPrimAtPath(owner);info=metadata(prim)
    if info is None:return None
    if isinstance(mass,bool) or not np.isfinite(mass) or mass<=0:raise ValueError('Positive aggregate source mass required')
    from .source_moments import KEY as source_key
    record=json.loads(prim.GetCustomDataByKey(source_key))
    for path in [owner,info['carrier']]:
        body=stage.GetPrimAtPath(path)
        if not body:raise ValueError('Missing fixed contact carrier')
        _moments(body,mass/2,record)
        api=PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        if linear_damping is not None:api.CreateLinearDampingAttr(float(linear_damping))
        if angular_damping is not None:api.CreateAngularDampingAttr(float(angular_damping))
    info['source_mass_kg']=float(mass);prim.SetCustomDataByKey(KEY,json.dumps(info))
    return info


def create(stage,owner):
    prim=stage.GetPrimAtPath(owner)
    if metadata(prim):return verify(stage,owner)
    from .source_moments import KEY as source_key
    from .rigid_partition import metadata as other_partition
    if (not prim.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            or prim.GetCustomDataByKey(source_key) is None
            or other_partition(prim)):
        raise ValueError('Contact carriers require a metric source rigid with explicit mass moments and no other partition')
    if any(p!=prim and p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
        raise ValueError('Nested physical owners require a separate contact-carrier construction')
    colliders=[p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)
               and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    if not colliders:raise ValueError('Requested adjacent contact needs actual source colliders')
    carrier=owner+'_contact_carrier'
    if stage.GetPrimAtPath(carrier):raise ValueError('Contact carrier path is already occupied')
    frame=UsdGeom.Xform.Define(stage,carrier)
    frame.MakeMatrixXform().Set(UsdGeom.Xformable(prim).GetLocalTransformation())
    frame.CreateVisibilityAttr('invisible');UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
    api=PhysxSchema.PhysxRigidBodyAPI.Apply(frame.GetPrim());source=PhysxSchema.PhysxRigidBodyAPI(prim)
    api.CreateSolverPositionIterationCountAttr(source.GetSolverPositionIterationCountAttr().Get())
    api.CreateSolverVelocityIterationCountAttr(source.GetSolverVelocityIterationCountAttr().Get())
    api.CreateEnableCCDAttr(source.GetEnableCCDAttr().Get());frame.GetPrim().SetCustomDataByKey(OWNER,owner)
    roots={str(p.GetPath()).removeprefix(owner+'/').split('/')[0] for p in colliders}
    layer=stage.GetRootLayer()
    for name in roots:
        if not Sdf.CopySpec(layer,owner+'/'+name,layer,carrier+'/'+name):
            raise ValueError('Source collider subtree could not be copied exactly')
    pairs=[]
    for original in colliders:
        clone=stage.GetPrimAtPath(carrier+str(original.GetPath()).removeprefix(owner))
        if _signature(original)!=_signature(clone):raise ValueError('Contact carrier changed source collision geometry')
        UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
        clone.SetCustomDataByKey('calibrated_scene_reconstruction:mass_source_surface',False)
        pairs.append([str(original.GetPath()),str(clone.GetPath())])
    weld_path=owner+'_contact_weld';weld=UsdPhysics.FixedJoint.Define(stage,weld_path)
    weld.CreateBody0Rel().SetTargets([owner]);weld.CreateBody1Rel().SetTargets([carrier])
    weld.CreateLocalPos0Attr(Gf.Vec3f(0.));weld.CreateLocalPos1Attr(Gf.Vec3f(0.))
    weld.CreateLocalRot0Attr(Gf.Quatf(1.));weld.CreateLocalRot1Attr(Gf.Quatf(1.));weld.CreateCollisionEnabledAttr(False)
    info={'version':1,'carrier':carrier,'weld':weld_path,'collider_pairs':pairs,
          'source_mass_kg':float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()),
          'mechanical_degrees_of_freedom_added':0,'source_geometry_changed':False}
    prim.SetCustomDataByKey(KEY,json.dumps(info));update(stage,owner,info['source_mass_kg'])
    return verify(stage,owner)


def verify(stage,owner):
    prim=stage.GetPrimAtPath(owner);info=metadata(prim)
    if info is None:return None
    from .source_moments import KEY as source_key
    if info['version']!=1 or info.get('mechanical_degrees_of_freedom_added')!=0:
        raise ValueError('Unsupported contact carrier construction')
    record=json.loads(prim.GetCustomDataByKey(source_key));frame=stage.GetPrimAtPath(info['carrier'])
    if (not prim.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            or not frame or frame.GetCustomDataByKey(OWNER)!=owner or not frame.HasAPI(UsdPhysics.RigidBodyAPI)
            or UsdPhysics.RigidBodyAPI(frame).GetKinematicEnabledAttr().Get()):
        raise ValueError('Contact carrier must be a real dynamic fixed component')
    cache=UsdGeom.XformCache()
    if not np.allclose(cache.GetLocalToWorldTransform(frame),cache.GetLocalToWorldTransform(prim),atol=1e-7):
        raise ValueError('Authored carrier frame does not coincide with source owner')
    from scipy.spatial.transform import Rotation
    moment_errors=[]
    for body in [prim,frame]:
        api=UsdPhysics.MassAPI(body);mass=float(api.GetMassAttr().Get());q=api.GetPrincipalAxesAttr().Get()
        R=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
        actual=R@np.diag(api.GetDiagonalInertiaAttr().Get())@R.T
        expected=np.array(record['inertia_per_kg'])*mass
        relative_error=float(np.linalg.norm(actual-expected,ord=2)/np.linalg.norm(expected,ord=2))
        moment_errors.append(relative_error)
        if (not np.isclose(mass,info['source_mass_kg']/2,rtol=1e-6)
                or not np.allclose(api.GetCenterOfMassAttr().Get(),record['centre'],atol=1e-7)
                or relative_error>1e-6):
            raise ValueError(f'Contact carrier changed aggregate source mass, COM or inertia; relative tensor error={relative_error:g}')
    weld=UsdPhysics.FixedJoint.Get(stage,info['weld'])
    if (not weld or list(map(str,weld.GetBody0Rel().GetTargets()))!=[owner]
            or list(map(str,weld.GetBody1Rel().GetTargets()))!=[info['carrier']]
            or weld.GetJointEnabledAttr().Get() is False or weld.GetExcludeFromArticulationAttr().Get()
            or not np.allclose(weld.GetLocalPos0Attr().Get(),0.) or not np.allclose(weld.GetLocalPos1Attr().Get(),0.)
            or weld.GetLocalRot0Attr().Get()!=Gf.Quatf(1.) or weld.GetLocalRot1Attr().Get()!=Gf.Quatf(1.)):
        raise ValueError('Contact carrier needs an intact zero-motion articulation weld')
    if any(weld.GetPrim().HasAPI(UsdPhysics.DriveAPI,axis) for axis in ['linear','angular']):
        raise ValueError('A numerical collision weld cannot introduce a drive')
    if any(p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
           for p in Usd.PrimRange(prim)):
        raise ValueError('Source owner has duplicate active colliders outside its carrier')
    actual={str(p.GetPath()) for p in Usd.PrimRange(frame) if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False}
    if actual!={b for _,b in info['collider_pairs']}:raise ValueError('Contact carrier collider ownership changed')
    for a,b in info['collider_pairs']:
        x,y=stage.GetPrimAtPath(a),stage.GetPrimAtPath(b)
        if (not x or not y or UsdPhysics.CollisionAPI(x).GetCollisionEnabledAttr().Get() is not False
                or _signature(x)!=_signature(y)
                or not np.allclose(cache.GetLocalToWorldTransform(x),cache.GetLocalToWorldTransform(y),atol=1e-7)):
            raise ValueError('Contact carrier no longer matches original source colliders')
    return {**info,'maximum_relative_inertia_error':max(moment_errors)}


def author_requested(stage,root,joints,*,protected_owners=()):
    """Keep contact requests effective and preserve explicit disabled pairs."""
    carriers={};owners=[];protected=set(protected_owners)
    for joint in joints:
        if joint.get('collision_enabled',False):
            endpoints=[root+'/'+joint[key] for key in ['child','parent']]
            owner=next((path for path in endpoints if path not in protected),None)
            if owner is None:
                raise ValueError('Both contacting joint links own physical surface bonds; this combination needs a separately qualified collision/root adapter')
            owners.append(owner)
    for owner in owners:carriers[owner]=create(stage,owner)
    for joint in joints:
        if joint.get('collision_enabled',False):continue
        a,b=[root+'/'+joint[k] for k in ['parent','child']]
        x=carriers[a]['carrier'] if a in carriers else a
        y=carriers[b]['carrier'] if b in carriers else b
        if (x,y)!=(a,b):UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(x)).CreateFilteredPairsRel().AddTarget(y)
    return carriers


def audit_requested(stage,root):
    """A saved USD collision flag alone is not evidence of effective contact."""
    prim=stage.GetPrimAtPath(root);joints=[];graph={};roots=[]
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):roots.append(p)
        if not p.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(p);a,b=joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()
        if len(a)!=1 or len(b)!=1:continue
        a,b=str(a[0]),str(b[0]);joints.append((joint,a,b))
        graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    result=[]
    for joint,a,b in joints:
        if joint.GetCollisionEnabledAttr().Get() is not True:continue
        if joint.GetExcludeFromArticulationAttr().Get():
            result.append({'joint':str(joint.GetPath()),'route':'maximum_coordinate','pair':[a,b]});continue
        components={a};todo=[a]
        while todo:
            for other in graph.get(todo.pop(),set())-components:components.add(other);todo.append(other)
        matched=[]
        for p in roots:
            path=str(p.GetPath())
            connected=path in components or all(Sdf.Path(c).HasPrefix(p.GetPath()) for c in components)
            if p.IsA(UsdPhysics.Joint):
                endpoints=UsdPhysics.Joint(p).GetBody1Rel().GetTargets()
                connected |= any(str(e) in components for e in endpoints)
            if connected:matched.append(p)
        if not matched or any(PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get() is not True for p in matched):
            raise ValueError('Requested articulation contacts require enabled component self-collision')
        owners=[metadata(stage.GetPrimAtPath(path)) for path in [a,b]]
        if not any(owners):raise ValueError('Adjacent articulation contacts are filtered; a USD flag cannot enable them')
        pair=[]
        for path,info in zip([a,b],owners):
            if info:verify(stage,path)
            pair.append(info['carrier'] if info else path)
        for x,y in [pair,pair[::-1]]:
            parent=stage.GetPrimAtPath(x)
            while parent:
                if parent.HasAPI(UsdPhysics.FilteredPairsAPI):
                    if any(Sdf.Path(y).HasPrefix(target) for target in UsdPhysics.FilteredPairsAPI(parent).GetFilteredPairsRel().GetTargets()):
                        raise ValueError('A collision filter suppresses requested source contacts')
                parent=parent.GetParent()
        result.append({'joint':str(joint.GetPath()),'route':'fixed_source_collision_carrier',
                       'logical_pair':[a,b],'native_pair':pair,'mechanical_degrees_of_freedom_added':0})
    return result
