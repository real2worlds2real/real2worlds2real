"""Preserve agent PBR regions on the original physical triangles."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from calibrated_scene_reconstruction.sim.usd import add_mesh, color_material, textured_material


def surface_material(stage, path, props, *, look=None, look_base=None):
    """Preserve transmission separately from cutout opacity and baked albedo."""
    transmission = float(props.get('transmission', 0.))
    opacity, ior = float(props.get('opacity', 1.)), float(props.get('ior', 1.5))
    if not np.isfinite([transmission, opacity, ior]).all() or not 0 <= min(transmission, opacity) or max(transmission, opacity) > 1 or ior < 1:
        raise ValueError('Invalid optical material parameters')
    roughness, metallic = props.get('roughness', .5), props.get('metallic', 0.)
    from calibrated_scene_reconstruction.assets.subsurface_optics import omni_parameters
    scattering = (omni_parameters(props['subsurface'], UsdGeom.GetStageMetersPerUnit(stage))
                  if props.get('subsurface') is not None else None)
    if transmission > 0 or scattering is not None:
        mat = color_material(stage, path, props['color'], roughness=roughness, metallic=metallic,
                             opacity=opacity*(1-.85*transmission), ior=ior)
        mdl = UsdShade.Shader.Define(stage, path+'/Transmission')
        mdl.SetSourceAsset(Sdf.AssetPath('OmniSurface.mdl'), 'mdl')
        mdl.SetSourceAssetSubIdentifier('OmniSurface', 'mdl')
        for name, kind, value in [
                ('enable_specular_transmission', Sdf.ValueTypeNames.Bool, True),
                ('specular_transmission_weight', Sdf.ValueTypeNames.Float, transmission),
                ('specular_transmission_color', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*props['color'])),
                ('specular_reflection_roughness', Sdf.ValueTypeNames.Float, float(roughness)),
                ('specular_reflection_ior', Sdf.ValueTypeNames.Float, ior),
                ('diffuse_reflection_weight', Sdf.ValueTypeNames.Float, 1-transmission),
                ('diffuse_reflection_color', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*props['color'])),
                ('metalness', Sdf.ValueTypeNames.Float, float(metallic)),
                ('geometry_opacity', Sdf.ValueTypeNames.Float, opacity)]:
            mdl.CreateInput(name, kind).Set(value)
        if scattering is not None:
            for name, value in scattering.items():
                if isinstance(value, bool):
                    kind = Sdf.ValueTypeNames.Bool
                elif isinstance(value, list):
                    kind, value = Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*value)
                else:
                    kind = Sdf.ValueTypeNames.Float
                mdl.CreateInput(name, kind).Set(value)
            mdl.CreateInput('subsurface_transmission_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*props['color']))
            mat.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:subsurface_optics_requires_source_review', True)
            if transmission == 0 and look:
                # An opaque scattering asset may retain its reviewed surface
                # colour texture. Transmitting surfaces still require the
                # separate optical-print masks below.
                path = str(look_base / look['albedo'])
                gain = float(look.get('gain', 1.))
                if not np.isfinite(gain) or gain <= 0:
                    raise ValueError('Positive finite optical texture gain required')
                for name in ['diffuse_reflection_color', 'subsurface_transmission_color']:
                    mdl.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(gain, gain, gain))
                    mdl.CreateInput(name+'_image', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(path))
        from calibrated_scene_reconstruction.assets.optical_print import declaration
        printing = declaration(look, look_base) if look else None
        if printing:
            for name, key in [('diffuse_reflection_weight_image', 'opaque_mask'),
                              ('specular_transmission_weight_image', 'transmission_mask'),
                              ('diffuse_reflection_color_image', 'albedo')]:
                mdl.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(printing[key])))
                if name.endswith('weight_image'):
                    # MDL default reads image alpha; grayscale PNG has alpha1.
                    mdl.CreateInput(name+'_alpha_mode', Sdf.ValueTypeNames.Int).Set(1)
        mat.CreateSurfaceOutput('mdl').ConnectToSource(mdl.ConnectableAPI(), 'out')
        mat.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:transmission_weight', transmission)
        mat.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:preview_transmission_is_approximate', True)
        # Source RGB behind a transmitting/scattering surface is not albedo.
        return mat
    if look:
        return textured_material(stage, path, look_base/look['albedo'], roughness=roughness, metallic=metallic,
                                 tint=(float(look.get('gain', 1.)),)*3)
    return color_material(stage, path, props['color'], roughness=roughness, metallic=metallic, opacity=opacity)


def bind_surface_materials(stage, mesh, part, *, look=None, look_base=None):
    """Material subsets preserve one unchanged physical/deformable mesh."""
    indices = part.get('face_material')
    names, materials = part.get('material_names', []), part.get('materials', {})
    groups = sorted(set(np.asarray(indices).tolist())) if indices is not None and materials else [None]
    for group in groups:
        keep = np.asarray(indices) == group if group is not None else np.ones(len(part['F']), bool)
        props = {**part['material'], **(materials.get(names[group], {}) if group is not None else {})}
        path = str(mesh.GetPath()).replace('/Objects/', '/Looks/')+f'_material_{group}'
        mat = surface_material(stage, path, props, look=look, look_base=look_base)
        prim = mesh.GetPrim()
        if len(groups) > 1:
            subset = UsdGeom.Subset.Define(stage, str(mesh.GetPath())+f'/material_{group}')
            subset.CreateElementTypeAttr('face')
            subset.CreateFamilyNameAttr('materialBind')
            subset.CreateIndicesAttr(np.flatnonzero(keep).tolist())
            prim = subset.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def add_surface(stage, path, part, *, look=None, look_base=None):
    mesh = add_mesh(stage, path, part['V'], part['F'], part.get('UV'),normals=part.get('normals'))
    bind_surface_materials(stage, mesh, part, look=look, look_base=look_base)
    return mesh
