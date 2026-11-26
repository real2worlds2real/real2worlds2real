"""Compose observed GS appearance without replacing native physics geometry."""
import hashlib
from pathlib import Path

import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
from calibrated_scene_reconstruction.assets.topology import closed_oriented_surface


def reordered_uv(payload,world,counts,indices):
    """Remap only identical oriented faces after exporter face reordering.

    Blender may enumerate the same sphere triangles in a different order on a
    rebuild. UVs are face-varying, so accepting the geometry hash alone or just
    ignoring a mismatch would put textures on the wrong faces.
    """
    if not {'world','counts','indices'}<=set(payload):raise ValueError('Crop mask geometry changed')
    old_world=np.asarray(payload['world']);world=np.asarray(world)
    if old_world.shape!=world.shape or not np.allclose(old_world,world,atol=1e-6,rtol=0):
        raise ValueError('Crop mask geometry changed')
    faces={};offset=0;old_indices=payload['indices'];old_uv=payload['uv']
    def key(face):
        face=list(map(int,face))
        if len(face)<3 or len(set(face))!=len(face):raise ValueError('Crop mask geometry changed')
        i=face.index(min(face));return tuple(face[i:]+face[:i])
    for n in payload['counts']:
        face=old_indices[offset:offset+n];signature=key(face)
        if signature in faces:raise ValueError('Crop mask geometry changed: duplicate faces')
        faces[signature]=(face,old_uv[offset:offset+n]);offset+=n
    output=[];offset=0
    for n in counts:
        face=indices[offset:offset+n];signature=key(face)
        if signature not in faces:raise ValueError('Crop mask geometry changed')
        old,uv=faces.pop(signature);lookup={int(v):i for i,v in enumerate(old)}
        output.extend(uv[[lookup[int(v)] for v in face]]);offset+=n
    if faces:raise ValueError('Crop mask geometry changed')
    return np.asarray(output)


def apply_gap_mask(stage, prim, declaration):
    """Keep solid collision geometry, but expose GS-covered plane texels."""
    if declaration.get('mesh_parts'):
        ancestor=prim;matched=False
        while ancestor and str(ancestor.GetPath()).startswith('/World/Environment/'):
            matched |= ancestor.GetName() in declaration['mesh_parts'];ancestor=ancestor.GetParent()
        # Standalone meshes may be bound before composing the Environment root.
        matched |= prim.GetName() in declaration['mesh_parts']
        if not matched:return False
    # Scene policy validates the coverage declaration before preparing runtime
    # inputs. This adapter only consumes coordinates and verifies the payload.
    with Path(declaration['path']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=declaration['sha256']:
            raise ValueError('GS coverage mask changed')
    mesh=UsdGeom.Mesh(prim)
    if not mesh:raise ValueError('Gap coverage is only supported on explicit mesh surfaces')
    points=np.asarray(mesh.GetPointsAttr().Get(),float)
    transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim),float)
    world=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
    if declaration.get('one_sided_closed'):
        if not closed_oriented_surface(world,mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get()):
            raise ValueError('Layered coverage requires a closed consistently oriented solid')
        # Complete outward-facing rear surfaces stay native for unseen views;
        # their inward backfaces must not block a captured front-side opening.
        mesh.CreateDoubleSidedAttr(False)
        # RTX requires its geometry flag as well as the USD doubleSided flag.
        # The environment renderer enables culling only after isolation checks.
        prim.CreateAttribute('singleSided',Sdf.ValueTypeNames.Bool).Set(True)
    if declaration.get('uv_path'):
        with Path(declaration['uv_path']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=declaration['uv_sha256']:
                raise ValueError('GS crop UV payload changed')
        h=hashlib.sha256()
        for values in (np.rint(world*1e6),mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get()):
            h.update(np.asarray(values,dtype='<i8').tobytes())
        with np.load(declaration['uv_path']) as payload:
            uv=payload['uv'] if h.hexdigest()==declaration['geometry_key'] else reordered_uv(
                payload,world,np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int),np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int))
        if uv.shape!=(len(mesh.GetFaceVertexIndicesAttr().Get()),2) or not np.isfinite(uv).all():
            raise ValueError('Finite per-face-vertex crop UVs required')
        interpolation=UsdGeom.Tokens.faceVarying
    else:
        origin=np.asarray(declaration['origin'],float);axes=np.asarray(declaration['axes'],float);extent=np.asarray(declaration['extent_m'],float)
        if origin.shape!=(3,) or axes.shape!=(2,3) or extent.shape!=(2,) or np.any(extent<=0):
            raise ValueError('Prepared plane coordinates required')
        uv=(world-origin)@axes.T/extent;interpolation=UsdGeom.Tokens.vertex
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar('hybrid_gap_st',Sdf.ValueTypeNames.TexCoord2fArray,interpolation).Set(uv.tolist())
    # Subset materials remain separate. Replacing only a mesh-level material
    # either drops their appearance or leaves their opaque faces over the GS.
    targets=[s.GetPrim() for s in UsdGeom.Subset.GetGeomSubsets(mesh,UsdGeom.Tokens.face,'materialBind')]
    if UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]:targets.insert(0,prim)
    if not targets:raise ValueError('Gap mesh needs explicit bound surface materials')
    for target in targets:_bind_gap_material(stage,target,declaration)
    return True


def _bind_gap_material(stage,prim,declaration):
    """Clone each effective material network, preserving subset assignments."""
    old,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    original=old.ComputeSurfaceSource()[0] if old else None
    if not original or original.GetIdAttr().Get()!='UsdPreviewSurface':
        raise ValueError('Gap mesh needs an explicit USD Preview Surface material')
    # Make a per-mesh material. Editing a shared floor material would change
    # unrelated native objects and invalidate the declared ownership boundary.
    root='/World/HybridGapMaterials/m'+str(prim.GetPath()).replace('/','_')
    material=UsdShade.Material.Define(stage,root)
    # A material network cannot rely on shader connections inside another
    # material. Copying only Surface dropped image/normal/UV graphs in RTX.
    # An internal reference composes the complete network with remapped paths.
    if str(old.GetPath())==root:
        source=material.GetPrim().GetCustomDataByKey('r2s2r_gap_source_material')
        if not source:raise ValueError('Cannot reapply a gap material without its original network')
        old=UsdShade.Material(stage.GetPrimAtPath(source))
    material.GetPrim().GetReferences().SetReferences([Sdf.Reference(primPath=old.GetPath())])
    material.GetPrim().SetCustomDataByKey('r2s2r_gap_source_material',str(old.GetPath()))
    source_shader=old.ComputeSurfaceSource()[0]
    shader=UsdShade.Shader(stage.GetPrimAtPath(source_shader.GetPath().ReplacePrefix(old.GetPath(),material.GetPath())))
    if not shader:raise ValueError('Complete cloned material network is missing its surface')
    reader=UsdShade.Shader.Define(stage,root+'/HybridGapUV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('hybrid_gap_st')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    texture=UsdShade.Shader.Define(stage,root+'/HybridGapCoverage');texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(declaration['path']))
    texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
    for axis in ['S','T']:texture.CreateInput('wrap'+axis,Sdf.ValueTypeNames.Token).Set('clamp')
    texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    texture.CreateOutput('r',Sdf.ValueTypeNames.Float)
    mode=declaration.get('compositing','cutout')
    if mode not in ('cutout','depth_backing','solid_backing'):raise ValueError('Unknown prepared plane compositing mode')
    if mode in ('depth_backing','solid_backing'):
        if mode=='depth_backing' and not declaration.get('radiance'):raise ValueError('Depth backing requires bound capture radiance')
        # The physical plane supplies depth behind foreground GS. A hole mask
        # cannot infer per-ray GS transmittance: it can open black gaps where a
        # near-floor Gaussian is slightly below the calibrated surface. Keep a
        # source-textured solid backing instead of deleting foreground field.
        opacity=shader.CreateInput('opacity',Sdf.ValueTypeNames.Float);opacity.DisconnectSource();opacity.Set(1.)
    else:
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(),'r')
    # Coverage expresses ownership, not a translucent physical surface. RTX's
    # transparent mesh pass can otherwise composite the floor over foreground
    # NuRec geometry. Alpha testing keeps ordinary depth ordering and avoids a
    # glass-like replacement of the captured radiance.
    shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(.5)
    if declaration.get('radiance'):
        radiance=declaration['radiance'];path=Path(radiance['path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=radiance['sha256']:
                raise ValueError('Gap radiance changed')
        image=UsdShade.Shader.Define(stage,root+'/HybridGapRadiance');image.CreateIdAttr('UsdUVTexture')
        image.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
        image.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
        image.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        image.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
        for name,value in [('diffuseColor',(0.,0.,0.)),('specularColor',(0.,0.,0.))]:
            inp=shader.CreateInput(name,Sdf.ValueTypeNames.Color3f);inp.DisconnectSource();inp.Set(value)
        for name,value in [('roughness',1.),('ior',1.),('metallic',0.),('clearcoat',0.)]:
            inp=shader.CreateInput(name,Sdf.ValueTypeNames.Float);inp.DisconnectSource();inp.Set(value)
        shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(),'rgb')
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return True


def add_observed_context(stage,native_root,context):
    import omni.kit.app
    from .splat_stage import NUREC_SCHEMA,PATH
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(NUREC_SCHEMA,True)
    xf=UsdGeom.Xform.Define(stage,PATH)
    UsdGeom.Xformable(xf).AddTransformOp().Set(Gf.Matrix4d(np.asarray(context['matrix']).T.tolist()))
    if not xf.GetPrim().GetReferences().AddReference(context['asset']):raise ValueError('Cannot compose observed context')
    visible=set(context['native_entities']);seen=set();hidden=[]
    ownership=context.get('appearance_ownership');masks=context.get('gap_masks',{})
    for prim in Usd.PrimRange(native_root):
        if prim.HasAPI(UsdLux.LightAPI):
            attr=UsdLux.LightAPI(prim).GetIntensityAttr()
            attr.Set(float(attr.Get())*context.get('native_light_gain',1.))
        if prim.IsA(UsdLux.DomeLight):
            # Native illumination must not brighten the radiance field's holes.
            prim.CreateAttribute('visibleInPrimaryRay',Sdf.ValueTypeNames.Bool).Set(False)
        entity=prim.GetAttribute('userProperties:r2s2r_entity').Get()
        ancestor=prim.GetParent()
        while not entity and ancestor and str(ancestor.GetPath()).startswith(str(native_root.GetPath())):
            entity=ancestor.GetAttribute('userProperties:r2s2r_entity').Get();ancestor=ancestor.GetParent()
        if entity:seen.add(entity)
        if ownership and entity:
            if entity not in ownership:raise ValueError('Native entity has no appearance owner: '+entity)
            role=ownership[entity]['role']
            if role=='gap_fill' and prim.IsA(UsdGeom.Mesh):
                declaration=masks[entity]
                if declaration.get('kind') in ('source_crop','surface_atlas'):
                    key=str(prim.GetPath())
                    if key not in declaration['meshes']:raise ValueError('Crop mask missing native mesh: '+key)
                    declaration=declaration['meshes'][key]
                if declaration.get('kind')!='solid_backing':apply_gap_mask(stage,prim,declaration)
        if entity and entity not in visible and prim.IsA(UsdGeom.Imageable):
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar('hideForCamera',Sdf.ValueTypeNames.Bool,UsdGeom.Tokens.constant).Set(True)
            # Visibility is portable across RTX versions and ordinary USD
            # viewers. hideForCamera alone is not respected by every consumer.
            # USD visibility does not disable authored collision APIs.
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
    if not visible<=seen:raise ValueError('Observed context names absent native geometry: '+str(visible-seen))
    if ownership and set(ownership)!=seen:raise ValueError('Appearance owners do not match actual native entities')
    return hidden
