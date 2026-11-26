"""Check declared texture tiling against actual exported world-space UVs.

This checks an agent's UV contract, not real object dimensions or material truth.
It catches a small material patch stretched across an entire room. Appearance
and source/free-view review remain separate requirements.
"""
import numpy as np
from pathlib import Path
from calibrated_scene_reconstruction.scene.review.environment import digest


def triangle_scale(points,uv):
    points,uv=np.asarray(points,float),np.asarray(uv,float)
    delta=(uv[1:]-uv[0]).T
    if abs(np.linalg.det(delta))<1e-12:return None
    mapping=(points[1:]-points[0]).T@np.linalg.inv(delta)
    return np.linalg.norm(mapping,axis=0)


def audit(asset,checks,assets):
    from pxr import Usd,UsdGeom,UsdShade
    if not isinstance(checks,list):raise ValueError('Texture scale checks must be a list')
    if not checks:return {'pass':True,'checks':[],'scope':'No texture scale contracts declared; no appearance approval.'}
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();records=[]
    for spec in checks:
        logical=spec['asset'];expected=np.asarray(spec['tile_m'],float);tolerance=spec.get('relative_tolerance',.1)
        if (logical not in assets or expected.shape!=(2,) or not np.isfinite(expected).all() or np.any(expected<=0)
                or not .01<=tolerance<=.25):raise ValueError('Declared texture asset, positive metric repeat and bounded tolerance required')
        wanted=digest(assets[logical]);area=bad=0.;samples=0;scales=[]
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):continue
            material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            surface=material.ComputeSurfaceSource()[0] if material else None
            plug=surface.GetInput('diffuseColor') if surface else None
            if not plug or not plug.HasConnectedSource():continue
            pending=[plug.GetConnectedSource()[0]];seen=set();textures=[]
            while pending:
                node=pending.pop();key=str(node.GetPath())
                if key in seen:continue
                seen.add(key);shader=UsdShade.Shader(node.GetPrim())
                if shader.GetIdAttr().Get()=='UsdUVTexture':
                    value=shader.GetInput('file').Get();path=Path(value.resolvedPath or value.path)
                    if path.is_file() and digest(path)==wanted:textures.append(shader)
                for item in node.GetInputs():
                    if item.HasConnectedSource():pending.append(item.GetConnectedSource()[0])
            for texture in textures:
                connection=texture.GetInput('st').GetConnectedSource()
                if not connection:raise ValueError('Declared metric texture lacks an explicit UV reader')
                reader=UsdShade.Shader(connection[0].GetPrim())
                if reader.GetIdAttr().Get()!='UsdPrimvarReader_float2':raise ValueError('Texture scale audit needs explicit untransformed UV coordinates')
                name=reader.GetInput('varname').Get();mesh=UsdGeom.Mesh(prim);uv_prim=UsdGeom.PrimvarsAPI(prim).GetPrimvar(str(name))
                uv=np.asarray(uv_prim.ComputeFlattened(),float);interpolation=uv_prim.GetInterpolation()
                P=np.asarray(mesh.GetPointsAttr().Get(),float);P=(np.c_[P,np.ones(len(P))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:,:3]
                indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int);counts=mesh.GetFaceVertexCountsAttr().Get();offset=0
                if interpolation not in ('faceVarying','vertex','varying'):raise ValueError('Per-vertex or per-corner UV coordinates required')
                for count in counts:
                    face=indices[offset:offset+count];tex=uv[offset:offset+count] if interpolation=='faceVarying' else uv[face];offset+=count
                    for k in range(1,count-1):
                        tri=[0,k,k+1];points=P[face[tri]];weight=np.linalg.norm(np.cross(points[1]-points[0],points[2]-points[0]))/2
                        if weight<1e-10:continue
                        scale=triangle_scale(points,tex[tri]);area+=weight;samples+=1
                        if scale is None or np.max(abs(scale/expected-1))>tolerance:bad+=weight
                        if scale is not None:scales.append(scale)
        fraction=bad/area if area else 1.
        records.append({'asset':logical,'tile_m':expected.tolist(),'triangles':samples,'area_m2':float(area),
                        'bad_area_fraction':float(fraction),'pass':bool(samples>0 and fraction<=.1),
                        'observed_uv_repeat_m_median':np.median(scales,axis=0).tolist() if scales else None})
    return {'pass':all(r['pass'] for r in records),'checks':records,
            'scope':'Declared UV repeat only, with at most10% area tolerance for bevel projection. Not measured material scale or photorealism acceptance.'}
