"""Native passive elastic connection for qualified material discretizations.

This is a numerical constitutive element, not a claim of a physical hinge.
Its approximation, operating range and material response need independent QA.
"""
import json
import numpy as np


def author(stage,path,parent,child,anchor,frame,stiffness,damping,*,articulation_root=None):
    from pxr import Gf,UsdPhysics
    q=np.asarray(frame,float);p=np.asarray(anchor,float)
    k=np.asarray(stiffness,float);c=np.asarray(damping,float)
    if (q.shape!=(4,) or p.shape!=(3,) or k.shape!=(3,) or c.shape!=(3,)
            or not np.isfinite(np.r_[q,p,k,c]).all() or not np.isclose(q@q,1.,atol=1e-6)
            or np.any(k<=0) or np.any(c<0) or parent==child):
        raise ValueError('Real endpoints, common metric anchor, unit frame and passive finite SI coefficients required')
    for endpoint in (parent,child):
        if not stage.GetPrimAtPath(endpoint).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Elastic element endpoints must be actual rigid material cells')
    if articulation_root is None:
        # Regular PhysX D6 joints share a swing drive for Y/Z. Independent
        # values have undefined behavior in107.3, even if one-axis tests pass.
        if not np.isclose(k[1],k[2],rtol=1e-6,atol=0.) or not np.isclose(c[1],c[2],rtol=1e-6,atol=0.):
            raise ValueError('Regular D6 swing axes require identical stiffness and damping; anisotropic material needs qualified reduced-coordinate controls')
    elif not stage.GetPrimAtPath(articulation_root).HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError('Reduced-coordinate material connection requires an actual native articulation root')
    joint=UsdPhysics.Joint.Define(stage,path)
    joint.CreateBody0Rel().SetTargets([parent]);joint.CreateBody1Rel().SetTargets([child])
    # Source material cells share the same asset coordinate frame. A caller
    # with independently transformed cells must convert these anchors first.
    for side in (0,1):
        getattr(joint,f'CreateLocalPos{side}Attr')(Gf.Vec3f(*map(float,p)))
        getattr(joint,f'CreateLocalRot{side}Attr')(Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:]))))
    for axis in ['transX','transY','transZ']:
        limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
        # USD/PhysX denotes a locked axis by low > high, not a zero-width
        # limited axis. The distinction matters in articulation parsing.
        limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
    for index,axis in enumerate(['rotX','rotY','rotZ']):
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
        drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
        drive.CreateStiffnessAttr(float(k[index]*np.pi/180))
        drive.CreateDampingAttr(float(c[index]*np.pi/180));drive.CreateMaxForceAttr(float('inf'))
    joint.CreateCollisionEnabledAttr(False)
    joint.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:elastic_joint',json.dumps({
        'stiffness_Nm_per_rad':k.tolist(),'damping_Nm_s_per_rad':c.tolist(),
        'role':'Passive numerical material element; not a physical articulation',
        'angular_usd_units':'Torque/degree and torque/(degree/second)',
        'articulation_root':articulation_root,
        'active_targets':False}))
    return joint
