"""Explicit kinetic initialization for source-stationary rigid/PBD responses."""
import numpy as np

from .contract import require


def validate_actors(stage,actors):
    """Reject unsupported kinetic initializers before any state is changed."""
    from pxr import UsdGeom,UsdPhysics
    paths={a['path'] for a in actors.values()}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            targets=[str(p) for rel in ['physics:body0','physics:body1'] for p in prim.GetRelationship(rel).GetTargets()]
            require(not paths.intersection(targets),'Jointed stationary responses require an explicit model-owned kinetic initializer')
    pending=[]
    for name,binding in actors.items():
        prim=stage.GetPrimAtPath(binding['path']);kind=binding['kind']
        if kind=='rigid':
            require(prim.HasAPI(UsdPhysics.RigidBodyAPI),'Native rigid schema required for kinetic initialization')
        else:
            schemas=' '.join(prim.GetAppliedSchemas())
            require(kind in {'deformable','particles'} and any(s in schemas for s in ['ParticleCloth','ParticleSet'])
                    and bool(UsdGeom.PointBased(prim)),
                    'Non-PBD stationary responses require an explicit model-owned kinetic initializer')
        pending.append((name,prim,kind))
    require(bool(pending),'Stationary initialization requires response actors')
    return pending


def stationary(stage,actors):
    """Clear inherited motion before reset; positions and mechanisms unchanged.

    Jointed or volumetric responses need their model's initializer. An absent
    policy preserves legacy behavior. Prescribed drivers/static supports are
    excluded by the caller, so their source trajectories remain model-owned.
    """
    from pxr import Gf,UsdGeom,UsdPhysics,Vt
    pending=validate_actors(stage,actors)
    rows={}
    def vector(value):return None if value is None else np.asarray(value,float).tolist()
    for name,prim,kind in pending:
        if kind=='rigid':
            body=UsdPhysics.RigidBodyAPI(prim)
            before={'velocity':vector(body.GetVelocityAttr().Get()),'angularVelocity':vector(body.GetAngularVelocityAttr().Get())}
            body.CreateVelocityAttr(Gf.Vec3f(0));body.CreateAngularVelocityAttr(Gf.Vec3f(0))
            good=bool(np.all(np.asarray(body.GetVelocityAttr().Get())==0) and np.all(np.asarray(body.GetAngularVelocityAttr().Get())==0))
        else:
            points=UsdGeom.PointBased(prim);old=points.GetVelocitiesAttr().Get();count=len(points.GetPointsAttr().Get())
            before={'velocity_count':0 if old is None else len(old),
                    'maximum_speed':None if old is None or not len(old) else float(np.linalg.norm(np.asarray(old,float),axis=1).max())}
            points.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros((count,3),np.float32)))
            good=bool(np.all(np.asarray(points.GetVelocitiesAttr().Get())==0))
        rows[name]={'kind':kind,'inherited':before,'authored_zero_before_reset':good}
    require(rows and all(row['authored_zero_before_reset'] for row in rows.values()),'Stationary initial kinetic state was not bound')
    return {'mode':'source_stationary','pass':True,'bodies':rows,
        'scope':'Authored native response velocities cleared before reset under a source-stationary initial-condition review. No pose locks or constraints added; native settling and source/contact checks remain separate.'}
