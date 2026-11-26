"""Versioned native geometry hypotheses; source assets are never edited."""
from pathlib import Path

import numpy as np

from .contract import digest,require


def create_layer(source,output):
    from pxr import Usd
    source,output=Path(source).resolve(),Path(output).resolve()
    require(source.is_file() and not output.exists(),'New derived layer and existing reviewed source required')
    stage=Usd.Stage.CreateNew(str(output));stage.GetRootLayer().subLayerPaths=[str(source)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    stage.GetRootLayer().customLayerData={'process_geometry_source':str(source),'process_geometry_source_sha256':digest(source),
        'scope':'Derived geometry/initial-state hypothesis; source unchanged. New geometry and native checks required.'}
    return stage


def place(stage,path,world_pose):
    from pxr import UsdGeom,Gf
    prim=stage.GetPrimAtPath(path);require(bool(prim),'Missing initial assembly: '+path)
    parent=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
    xform=UsdGeom.Xformable(prim);xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d((np.linalg.inv(parent)@np.asarray(world_pose)).T.tolist()))


def copy_component(stage,donor,path,*,world_alignment=None):
    """Propose a whole component from a compatible hypothesis, including physics.

    Keep render meshes, colliders, mass and flexible attachments together. Paths
    must retain their identities; callers review external relationship targets.
    Combining individually useful parts does not prove combined contact safety.
    """
    from pxr import Sdf,UsdGeom
    original=donor.GetPrimAtPath(path);destination=stage.GetPrimAtPath(path)
    require(original and destination and original.GetTypeName()==destination.GetTypeName(),
            'Matching component identities/types required for a hypothesis merge')
    alignment=np.eye(4) if world_alignment is None else np.asarray(world_alignment,float)
    require(alignment.shape==(4,4) and np.isfinite(alignment).all()
            and np.allclose(alignment[3],[0,0,0,1])
            and np.allclose(alignment[:3,:3].T@alignment[:3,:3],np.eye(3),atol=1e-7)
            and np.linalg.det(alignment[:3,:3])>0,'Rigid component frame alignment required')
    desired=alignment@np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(original)).T
    layer=stage.GetRootLayer();Sdf.CreatePrimInLayer(layer,Sdf.Path(path).GetParentPath())
    require(Sdf.CopySpec(donor.Flatten(),path,layer,path),'Could not copy complete component hypothesis')
    place(stage,path,desired)
    return {'path':path,'donor':donor.GetRootLayer().realPath,'world_pose':desired.tolist(),
            'native_checked':False}


def restore_initial_state(stage,bodies,recorded,*,world_from_source,index=0):
    """Prepare one native initial snapshot, retaining rest models and dynamics.

    This writes an initial USD hypothesis only. It supplies no runtime driver,
    does not change rest lengths or material properties, and must be followed
    by ordinary free native settling and source validation for each trial.
    """
    from pxr import UsdGeom,UsdPhysics,Vt
    from scipy.spatial.transform import Rotation
    require(index==0,'Use a recorded initial state; later responses are not initialization evidence')
    G=np.asarray(world_from_source,float);inverse=np.linalg.inv(G);rigid=[];points=[]
    for name,item in bodies.items():
        if item['kind']=='static':continue
        prim=stage.GetPrimAtPath(item['path']);require(prim,'Missing initial actor: '+name)
        value=np.asarray(recorded[name+'.00000'],float)
        require(np.isfinite(value).all(),'Nonfinite native initial actor state')
        if item['kind']=='rigid':
            require(value.shape==(7,) and prim.HasAPI(UsdPhysics.RigidBodyAPI),'Native rigid initial pose required')
            M=np.eye(4);M[:3,:3]=Rotation.from_quat(value[[4,5,6,3]]).as_matrix();M[:3,3]=value[:3]
            place(stage,item['path'],inverse@M);rigid.append(name)
        else:points.append((name,prim,value))
    point_counts={}
    for name,prim,value in points:
        require(value.shape==np.asarray(prim.GetAttribute('points').Get()).shape
                and value.ndim==2 and value.shape[1]==3,'Native initial point topology changed')
        M=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        source=value@inverse[:3,:3].T+inverse[:3,3]
        local=(source-M[:3,3])@np.linalg.inv(M[:3,:3]).T
        prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        UsdGeom.Mesh(prim).GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([local.min(0),local.max(0)],np.float32)))
        prim.GetAttribute('normals').Block();point_counts[name]=len(local)
    return {'rigid_actors':rigid,'point_actors':point_counts,'sample':0,'native_checked':False,
            'scope':'One initial snapshot only; free bodies, native rest states, masses, materials and attachment definitions preserved. Caller retains its reviewed initial velocity policy. Fresh free settling/source/contact validation required.'}


def deform_body_meshes(stage,path,body_local_transform):
    """Apply the same geometry transformation to render and collision meshes."""
    from pxr import Usd,UsdGeom,UsdPhysics,Vt
    body=stage.GetPrimAtPath(path);cache=UsdGeom.XformCache()
    root=np.asarray(cache.GetLocalToWorldTransform(body)).T;count=0
    for prim in Usd.PrimRange(body):
        if prim!=body and prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Transform each independent body separately')
        if not prim.IsA(UsdGeom.Mesh):continue
        local=np.linalg.inv(root)@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        change=np.linalg.inv(local)@np.asarray(body_local_transform)@local
        points=np.asarray(prim.GetAttribute('points').Get());points=points@change[:3,:3].T+change[:3,3]
        prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points,np.float32)))
        UsdGeom.Mesh(prim).GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)],np.float32)))
        prim.GetAttribute('normals').Block();count+=1
    require(count>0,'No native/render mesh geometry transformed')
    return count


def stretch_uniform_cylinder(stage,path,bottom,top,radius,length_ratio):
    """Source-reviewed solid circular-shaft prior; preserve density on resize.

    Other shapes/material partitions need their own mass model. Render and
    collision geometry share the affine change; COM and principal inertia are
    rebuilt in body coordinates rather than leaving the old inertia behind.
    """
    from pxr import UsdPhysics,Gf
    from scipy.spatial.transform import Rotation
    bottom,top=np.asarray(bottom,float),np.asarray(top,float);axis=top-bottom;length=np.linalg.norm(axis)
    require(bottom.shape==top.shape==(3,) and np.isfinite(bottom).all() and np.isfinite(top).all()
            and length>0 and np.isfinite(radius) and radius>0 and np.isfinite(length_ratio) and .5<=length_ratio<=2,
            'Finite reviewed uniform-cylinder geometry and bounded length ratio required')
    axis/=length;new_top=bottom+axis*length*length_ratio
    change=np.eye(4);change[:3,:3]+=(length_ratio-1)*np.outer(axis,axis);change[:3,3]=bottom-change[:3,:3]@bottom
    api=UsdPhysics.MassAPI(stage.GetPrimAtPath(path));old_mass=api.GetMassAttr().Get()
    require(old_mass is not None and old_mass>0,'Uniform density requires a positive declared mass prior')
    meshes=deform_body_meshes(stage,path,change);mass=float(old_mass)*length_ratio
    u=np.cross(axis,np.eye(3)[np.argmin(abs(axis))]);u/=np.linalg.norm(u);v=np.cross(axis,u)
    q=Rotation.from_matrix(np.column_stack([u,v,axis])).as_quat()
    perpendicular=mass*(3*radius**2+(length*length_ratio)**2)/12
    api.GetMassAttr().Set(mass);api.CreateCenterOfMassAttr(Gf.Vec3f(*((bottom+new_top)/2).tolist()))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(perpendicular,perpendicular,mass*radius**2/2))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3].tolist())))
    return {'bottom':bottom,'top':new_top,'length_ratio':length_ratio,'mass_prior':mass,'meshes_changed':meshes,
            'basis':'Uniform solid circular cylinder density prior, not measured mass; native contact/source checks required.'}


def change_radial_rings(stage,path,*,center,old_radius,new_radius,heights,tolerance=1e-6):
    """Refine a source-reviewed meridian ring shared by render and SDF geometry."""
    from pxr import UsdGeom,Vt
    prim=stage.GetPrimAtPath(path);points=np.asarray(prim.GetAttribute('points').Get(),float).copy()
    radial=points[:,:2]-np.asarray(center);radius=np.linalg.norm(radial,axis=1)
    selected=(abs(radius-old_radius)<tolerance)&np.any(abs(points[:,2,None]-np.asarray(heights)[None,:])<tolerance,axis=1)
    require(selected.sum()>=16,'Source radial profile was not found; do not deform unrelated vertices')
    points[selected,:2]=np.asarray(center)+radial[selected]*new_radius/old_radius
    prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points,np.float32)))
    prim.GetAttribute('normals').Block()
    UsdGeom.Mesh(prim).GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)],np.float32)))
    return int(selected.sum())


def contact_patch_prior(stage,paths,*,radius,minimum_radius,maximum_radius,reason):
    """Declare finite-contact torsional friction without geometry or pose locks.

    Patch size is an uncertain material/contact prior, bounded by reviewed part
    geometry. It is not a rolling joint, an external restoring force, or proof
    that an initial arrangement is stable. Native source checks remain required.
    """
    from pxr import Sdf,UsdPhysics
    require(reason and np.isfinite([radius,minimum_radius,maximum_radius]).all()
            and 0<=minimum_radius<=radius<=maximum_radius and maximum_radius>0,
            'Contact patch must be bounded by reviewed local contact geometry')
    prims=[stage.GetPrimAtPath(path) for path in paths]
    require(prims and all(p and p.HasAPI(UsdPhysics.CollisionAPI)
            and p.GetMetadata('apiSchemas') is not None
            and 'PhysxCollisionAPI' in p.GetMetadata('apiSchemas').GetAppliedItems() for p in prims),
            'Contact patch targets must already be native PhysX collision shapes')
    rows=[]
    for prim in prims:
        values={}
        for attr,value in [('physxCollision:torsionalPatchRadius',radius),
                           ('physxCollision:minTorsionalPatchRadius',minimum_radius)]:
            a=prim.CreateAttribute(attr,Sdf.ValueTypeNames.Float,custom=False);a.Set(float(value))
            values[attr]=float(a.Get())
        rows.append({'path':str(prim.GetPath()),'values':values,'maximum_radius_m':maximum_radius,'reason':reason})
    return rows


def tint_materials(stage,material_paths,linear_gain):
    """Retain texture detail while revising an explicitly selected albedo prior."""
    from pxr import Usd,UsdShade,Gf
    gain=np.asarray(linear_gain,float);require(gain.shape==(3,) and np.isfinite(gain).all() and np.all(gain>0),'Finite positive linear material gains required')
    changed=0
    for path in material_paths:
        prim=stage.GetPrimAtPath(path);require(bool(prim),'Missing reviewed material: '+path)
        for shader in Usd.PrimRange(prim):
            if not shader.IsA(UsdShade.Shader) or shader.GetAttribute('info:id').Get()!='UsdUVTexture':continue
            attr=shader.GetAttribute('inputs:scale');value=attr.Get()
            require(value is not None,'Texture scale convention must be explicit')
            value=np.asarray(value,float);value[:3]*=gain;attr.Set(Gf.Vec4f(*value.tolist()));changed+=1
    require(changed>0,'No material textures retinted')
    return changed


def set_diffuse_prior(stage,material_paths,linear_rgb):
    """Use a bounded diffuse prior when an unpatterned material atlas bakes light.

    This deliberately changes an authored material hypothesis; it must be
    source-reviewed. Geometry, roughness, mass and collision data are untouched.
    """
    from pxr import Usd,UsdShade,Gf
    color=np.asarray(linear_rgb,float);require(color.shape==(3,) and np.all((color>=0)&(color<=1)),'Physically bounded diffuse prior required')
    changed=0
    for path in material_paths:
        material=stage.GetPrimAtPath(path);require(bool(material),'Missing material hypothesis: '+path)
        for prim in Usd.PrimRange(material):
            if prim.IsA(UsdShade.Shader) and prim.GetAttribute('info:id').Get()=='UsdPreviewSurface':
                diffuse=UsdShade.Shader(prim).GetInput('diffuseColor');diffuse.DisconnectSource();diffuse.Set(Gf.Vec3f(*color.tolist()));changed+=1
    require(changed>0,'No supported diffuse shaders found')
    return changed
