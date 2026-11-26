"""Affine geometry hypotheses for one rigid core with native particle sheets.

The original USD is untouched. Geometry, density priors, native rest lengths,
and local attachment coordinates change together; final native QA is required.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import require


def apply(stage,root_path,core_path,core_local_affine,*,reason):
    from pxr import Usd,UsdGeom,UsdPhysics,Vt,Gf
    root=stage.GetPrimAtPath(root_path);core=stage.GetPrimAtPath(core_path)
    affine=np.asarray(core_local_affine,float);L=affine[:3,:3];det=float(np.linalg.det(L))
    require(root and core and reason and affine.shape==(4,4) and np.isfinite(affine).all()
            and np.allclose(affine[3],[0,0,0,1]) and det>0,
            'Explicit positive affine geometry hypothesis and source rationale required')
    prims=list(Usd.PrimRange(root));rigid=[p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    require(len(rigid)==1 and rigid[0]==core and not any(p.IsA(UsdPhysics.Joint) for p in prims),
            'Use a representation-specific geometry loop for multiple rigid links or joints')
    meshes=[p for p in prims if p.IsA(UsdGeom.Mesh)]
    require(meshes and not any('Deformable' in str(p.GetMetadata('apiSchemas')) for p in meshes),
            'This geometry helper supports mesh rigid cores and particle cloth, not FEM')
    cache=UsdGeom.XformCache();C=np.asarray(cache.GetLocalToWorldTransform(core)).T
    world=C@affine@np.linalg.inv(C)
    def local_change(prim):
        M=np.asarray(cache.GetLocalToWorldTransform(prim)).T
        return np.linalg.inv(M)@world@M,M
    def transform(points,A):return np.asarray(points,float)@A[:3,:3].T+A[:3,3]
    api=UsdPhysics.MassAPI(core);mass=float(api.GetMassAttr().Get());com=np.asarray(api.GetCenterOfMassAttr().Get(),float)
    diag=np.asarray(api.GetDiagonalInertiaAttr().Get(),float);q=api.GetPrincipalAxesAttr().Get()
    require(mass>0 and com.shape==diag.shape==(3,) and (diag>0).all() and q is not None,
            'Explicit rigid mass, COM and positive principal inertia required')
    R=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix();I=R@np.diag(diag)@R.T
    covariance=.5*np.trace(I)*np.eye(3)-I
    require(np.linalg.eigvalsh(covariance).min()>=-1e-12,'Source inertia does not represent a physical mass distribution')
    changed=det*L@covariance@L.T;new_I=np.trace(changed)*np.eye(3)-changed
    eigen,basis=np.linalg.eigh(new_I)
    require((eigen>0).all(),'Affine hypothesis produced invalid rigid inertia')
    if np.linalg.det(basis)<0:basis[:,0]*=-1
    quat=Rotation.from_matrix(basis).as_quat();api.GetMassAttr().Set(mass*det)
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*transform([com],affine)[0].tolist()))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*eigen.tolist()))
    api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(quat[3]),Gf.Vec3f(*quat[:3].tolist())))
    cloth_rows=[]
    for prim in meshes:
        A,M=local_change(prim);points=np.asarray(prim.GetAttribute('points').Get(),float)
        new=transform(points,A);prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(new.astype(np.float32)))
        UsdGeom.Mesh(prim).GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([new.min(0),new.max(0)],np.float32)))
        normals=prim.GetAttribute('normals').Get()
        if normals is not None and len(normals):
            N=np.asarray(normals,float)@np.linalg.inv(A[:3,:3]);N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-20)
            prim.GetAttribute('normals').Set(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
        velocities=prim.GetAttribute('velocities')
        if velocities and velocities.Get() is not None:
            values=np.asarray(velocities.Get(),float)@A[:3,:3].T
            velocities.Set(Vt.Vec3fArray.FromNumpy(values.astype(np.float32)))
        rest=prim.GetAttribute('physxParticle:restPoints')
        if rest and rest.Get() is not None:
            old=np.asarray(rest.Get(),float);new_rest=transform(old,A)
            indices=np.asarray(prim.GetAttribute('physxParticle:springIndices').Get(),int)
            lengths=prim.GetAttribute('physxParticle:springRestLengths');old_lengths=np.asarray(lengths.Get(),float)
            before=transform(old,M);after=transform(new_rest,M)
            old_dist=np.linalg.norm(before[indices[:,0]]-before[indices[:,1]],axis=1)
            new_dist=np.linalg.norm(after[indices[:,0]]-after[indices[:,1]],axis=1)
            require(indices.ndim==2 and indices.shape[1]==2 and len(old_lengths)==len(indices)
                    and (old_dist>0).all(),'Explicit valid native spring topology/rest state required')
            rest.Set(Vt.Vec3fArray.FromNumpy(new_rest.astype(np.float32)))
            lengths.Set(Vt.FloatArray.FromNumpy((old_lengths*new_dist/old_dist).astype(np.float32)))
            F=np.asarray(prim.GetAttribute('faceVertexIndices').Get()).reshape(-1,3)
            def area(V):return .5*np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1).sum()
            ratio=float(area(after)/area(before));cloth_mass=UsdPhysics.MassAPI(prim);old_mass=float(cloth_mass.GetMassAttr().Get())
            require(np.isfinite(ratio) and ratio>0 and old_mass>0,'Positive particle areal mass prior required')
            cloth_mass.GetMassAttr().Set(old_mass*ratio)
            cloth_rows.append({'path':str(prim.GetPath()),'area_ratio':ratio,'old_mass':old_mass,
                               'new_mass':old_mass*ratio,'springs':len(indices)})
    bonds=0
    for prim in prims:
        if prim.GetTypeName()!='PhysxPhysicsAttachment':continue
        for index in [0,1]:
            targets=prim.GetRelationship('actor'+str(index)).GetTargets()
            require(len(targets)==1 and targets[0].HasPrefix(root.GetPath()),
                    'Do not deform an assembly attached to an external actor')
            actor=stage.GetPrimAtPath(targets[0]);A,_=local_change(actor)
            attr=prim.GetAttribute('points'+str(index));values=attr.Get()
            require(values is not None,'Explicit local attachment coordinates required')
            attr.Set(Vt.Vec3fArray.FromNumpy(transform(values,A).astype(np.float32)))
        bonds+=1
    return {'rigid_core':core_path,'rigid_mass_before':mass,'rigid_mass_after':mass*det,
            'volume_ratio':det,'meshes':len(meshes),'particle_sheets':cloth_rows,'attachments':bonds,
            'reason':reason,'native_checked':False,
            'scope':'Uniform rigid density and particle areal density priors retained. Native prestrain ratios retained; existing spring stiffness/damping remain explicit material priors. No runtime pose constraints added.'}
