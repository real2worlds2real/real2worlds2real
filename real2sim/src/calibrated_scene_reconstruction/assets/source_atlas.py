"""Attach source appearance to unchanged mesh faces while retaining part paths.

Partial mode photographs only completely observed polygon interiors. Remaining
faces retain their original materials; this does not infer unseen photographs.
"""
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _key(points):
    values = [tuple(v) for v in np.rint(np.asarray(points)*1e6).astype('int64')]
    start = min(range(3), key=lambda i: values[i])
    return tuple(values[start:]+values[:start]), start


def observed_triangle(uv, mask, complete=True):
    h, w = mask.shape; p = np.asarray(uv)*[w, -h]+[0, h]
    lo = np.maximum(0, np.floor(p.min(0)).astype(int)); hi = np.minimum([w-1, h-1], np.ceil(p.max(0)).astype(int))
    if np.any(hi < lo):
        return False
    x, y = np.meshgrid(np.arange(lo[0], hi[0]+1)+.5, np.arange(lo[1], hi[1]+1)+.5)
    a, b, c = p; den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(den) < 1e-10:
        return False
    u = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
    v = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
    inside = (u >= -1e-7)&(v >= -1e-7)&(u+v <= 1+1e-7)
    return bool(inside.any() and (np.all if complete else np.any)(mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1][inside] > 128))


def _constant_pbr(material):
    shader = material.ComputeSurfaceSource()[0]
    if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':return None
    values = {}
    for name, default in [('diffuseColor', None), ('roughness', .5), ('metallic', 0.), ('ior', 1.5)]:
        attr = shader.GetInput(name)
        if attr and attr.HasConnectedSource():return None
        value = attr.Get() if attr else default
        if value is None:return None
        values[name] = np.asarray(value, float) if name == 'diffuseColor' else float(value)
    if shader.GetInput('emissiveColor') and (shader.GetInput('emissiveColor').HasConnectedSource()
            or np.max(np.asarray(shader.GetInput('emissiveColor').Get() or [0,0,0])) > 0):return None
    # Preserve richer original materials on unknown pixels instead of silently
    # discarding their normal, coat, transparency or connected specular maps.
    for name, default in [('normal', (0.,0.,1.)), ('clearcoat', 0.), ('opacity', 1.)]:
        attr = shader.GetInput(name)
        if attr and (attr.HasConnectedSource() or not np.allclose(attr.Get(), default)):
            return None
    workflow = shader.GetInput('useSpecularWorkflow')
    if workflow and workflow.Get() == 1:
        specular = shader.GetInput('specularColor')
        if specular and specular.HasConnectedSource():return None
        values['specularColor'] = np.asarray(specular.Get() if specular else [.04]*3, float)
    else:
        f0 = ((values['ior']-1)/(values['ior']+1))**2
        values['specularColor'] = f0*(1-values['metallic'])+values['diffuseColor']*values['metallic']
        values['diffuseColor'] *= 1-values['metallic']
    return values


def _pixel_material(stage, root, values, radiance, mask):
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, root); shader = UsdShade.Shader.Define(stage, root+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(values['roughness'])
    reader = UsdShade.Shader.Define(stage, root+'/UV'); reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('captured_surface_st')
    result = reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    for name in ['diffuseColor', 'specularColor']:
        colour = values[name]; tex = UsdShade.Shader.Define(stage, root+'/'+name); tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(mask).resolve())))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
        tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, -colour), 0.))
        tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, colour), 1.))
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(result)
        shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
    tex = UsdShade.Shader.Define(stage, root+'/Photo'); tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(radiance).resolve())))
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(result)
    shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface', Sdf.ValueTypeNames.Token))
    return mat


def attach(stage, proof, geometry, radiance, partial=True, path_prefixes=None, pixel_blend=False, output=None):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    from scipy.spatial import cKDTree

    if _sha(geometry) != proof['mesh_sha256'] or _sha(radiance) != proof['radiance_sha256']:
        raise ValueError('Source atlas payload changed')
    mask = None
    if partial:
        if _sha(proof['observed_mask']) != proof['observed_mask_sha256']:
            raise ValueError('Source observation mask changed')
        mask = np.asarray(Image.open(proof['observed_mask']))
        if mask.ndim != 2:
            raise ValueError('Scalar source observation mask required')
    mixed_radiance = None
    mixed_mask = None
    if pixel_blend:
        import tifffile
        if not partial or output is None:raise ValueError('Partial source mask and new texture directory required for pixel blending')
        output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
        colour = tifffile.imread(radiance)
        if colour.shape != (*mask.shape, 3):raise ValueError('Calibrated radiance and observed mask must align')
        mixed_radiance = output/('observed_'+_sha(radiance)[:16]+'.tiff')
        tifffile.imwrite(mixed_radiance, np.where(mask[...,None] > 128, colour, 0).astype('float32'), photometric='rgb')
        mixed_mask = output/('observed_mask_'+proof['observed_mask_sha256'][:16]+'.png')
        shutil.copyfile(proof['observed_mask'], mixed_mask)
    atlas = Usd.Stage.Open(str(geometry)); cache = UsdGeom.XformCache(); reports = []
    for k, source in enumerate(proof['source_parts']):
        target_path = source['path']
        if path_prefixes is not None:
            old, new = path_prefixes
            if not target_path.startswith(old+'/'):
                raise ValueError('Source part is outside its declared asset root')
            target_path = new+target_path[len(old):]
        prim = stage.GetPrimAtPath(target_path); mesh = UsdGeom.Mesh(prim)
        if not mesh:
            # Blender's generated data-block suffix may change after another
            # owner is added. The authored assembly part must stay identical;
            # vertex/oriented-triangle checks below still decide compatibility.
            parent = stage.GetPrimAtPath(str(Sdf.Path(target_path).GetParentPath()))
            if parent and source.get('part') == parent.GetName():
                candidates = [p for p in Usd.PrimRange(parent) if p.IsA(UsdGeom.Mesh)]
                if len(candidates) == 1:
                    prim = candidates[0]; mesh = UsdGeom.Mesh(prim)
        imported = UsdGeom.Mesh(atlas.GetPrimAtPath('/CapturedSurface/part_'+str(k)))
        if not mesh or not imported:
            raise ValueError('Preserve the exact original mesh part paths')
        vertices = np.asarray(mesh.GetPointsAttr().Get(), float)
        world = (np.c_[vertices, np.ones(len(vertices))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
        captured = np.asarray(imported.GetPointsAttr().Get(), float)
        distance, mapping = cKDTree(world).query(captured)
        if np.max(distance) > 2e-5:
            raise ValueError('Source atlas changed mesh vertices')
        faces = np.asarray(imported.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
        uv = np.asarray(UsdGeom.PrimvarsAPI(imported).GetPrimvar('st').ComputeFlattened(), float)
        lookup = {}
        for face in faces:
            key, start = _key(world[mapping[face]])
            if key in lookup:
                raise ValueError('Ambiguous duplicate source triangles')
            lookup[key] = np.roll(uv[face], -start, axis=0)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        photo_uv, photographed, fully_observed = [], [], []
        offset = 0
        for parent, count in enumerate(counts):
            face_uv = np.zeros((count, 2)); assigned = np.zeros(count, bool)
            compatible = True; observed = True; any_observed = False
            for j in range(1, count-1):
                local_corners = np.array([0, j, j+1]); face = indices[offset+local_corners]
                key, start = _key(world[face])
                if key not in lookup:
                    raise ValueError('Source atlas must retain every oriented original triangle')
                coords = np.roll(lookup.pop(key), start, axis=0)
                for corner, value in zip(local_corners, coords):
                    if assigned[corner] and not np.allclose(face_uv[corner], value, rtol=0, atol=1e-6):
                        compatible = False
                    face_uv[corner] = value; assigned[corner] = True
                observed &= not partial or observed_triangle(coords, mask)
                any_observed |= not partial or observed_triangle(coords, mask, complete=False)
            photo_uv.extend(face_uv.tolist() if compatible else np.zeros_like(face_uv).tolist())
            photographed.append(bool(compatible and (any_observed if pixel_blend else observed)))
            fully_observed.append(bool(compatible and observed))
            offset += count
        if lookup:
            raise ValueError('Source atlas added triangles')
        # Keep every original point, polygon, index, normal and prior UV exactly.
        # An atlas seam inside a polygon falls back to its existing PBR material.
        original = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        materials = [original]*len(counts)
        subsets = UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, 'materialBind')
        for subset in subsets:
            material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            for index in subset.GetIndicesAttr().Get():
                materials[index] = material
        for subset in subsets:
            stage.RemovePrim(subset.GetPath())
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('captured_surface_st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying')
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(photo_uv, 'float32')))
        root = str(prim.GetParent().GetPath())+'/_CapturedAppearance/m'+hashlib.sha256(str(prim.GetPath()).encode()).hexdigest()[:16]
        mat = UsdShade.Material.Define(stage, root); shader = UsdShade.Shader.Define(stage, root+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        for name in ['diffuseColor', 'specularColor']:
            shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        for name, value in [('ior', 1.), ('roughness', .95), ('metallic', 0.), ('clearcoat', 0.)]:
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        reader = UsdShade.Shader.Define(stage, root+'/UV'); reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('captured_surface_st')
        tex = UsdShade.Shader.Define(stage, root+'/Texture'); tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(radiance).resolve())))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput('result', Sdf.ValueTypeNames.Float2))
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
        mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface', Sdf.ValueTypeNames.Token))
        groups = {}; converted = {}; applied = []
        for index, captured_face in enumerate(photographed):
            target = materials[index]
            if captured_face and pixel_blend:
                values = _constant_pbr(target)
                if values is not None:
                    identity = str(target.GetPath())
                    if identity not in converted:
                        converted[identity] = _pixel_material(stage, root+'_mixed_'+str(len(converted)), values,
                                                             mixed_radiance, mixed_mask)
                    target = converted[identity]
                elif fully_observed[index]:target = mat
            elif captured_face:target = mat
            applied.append(captured_face and target != materials[index])
            if not target:
                raise ValueError('Unobserved source faces need an original native material')
            groups.setdefault(str(target.GetPath()), []).append(index)
        for index, (target, triangles) in enumerate(groups.items()):
            subset = UsdGeom.Subset.CreateGeomSubset(mesh, 'appearance_'+str(index), UsdGeom.Tokens.face, triangles, 'materialBind', UsdGeom.Tokens.partition)
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(target)))
        reports.append({'path': str(prim.GetPath()), 'source_path':source['path'],
                        'faces': len(counts), 'photographed_faces': int(sum(applied)),
                        'fully_photographed_faces': int(sum(fully_observed)), 'pixel_blend': pixel_blend,
                        'part_path_preserved': True, 'oriented_world_triangles_unchanged': True})
    if not any(row['photographed_faces'] for row in reports):
        raise ValueError('No completely observed native faces can use this source atlas')
    return {'partial_faces': partial, 'parts': reports, 'photorealism_approved': False}
