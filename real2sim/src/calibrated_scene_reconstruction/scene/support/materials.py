"""Preserve embedded Blender GLB image materials as fingerprinted USD texture assets."""
from pathlib import Path
import json
import struct
import trimesh
from calibrated_scene_reconstruction.core.paths import rel


def sampler(path, material_name):
    """Read the GLB sampler that trimesh's PBRMaterial does not preserve."""
    with Path(path).open('rb') as stream:
        header=stream.read(20)
        if len(header)!=20:raise ValueError('Truncated support GLB')
        magic,version,total,length,kind=struct.unpack('<5I',header)
        if magic!=0x46546c67 or version!=2 or kind!=0x4e4f534a or length>total-20:
            raise ValueError('A GLB2 JSON material chunk is required')
        document=json.loads(stream.read(length))
    materials=[m for m in document.get('materials',[]) if m.get('name')==material_name]
    if len(materials)!=1:raise ValueError('Textured support material must have an unambiguous source name')
    index=materials[0]['pbrMetallicRoughness']['baseColorTexture']['index']
    texture=document['textures'][index]
    spec=document['samplers'][texture['sampler']] if 'sampler' in texture else {}
    modes={10497:'repeat',33071:'clamp',33648:'mirror'}
    try:return [modes[spec.get('wrap'+axis,10497)] for axis in ['S','T']]
    except KeyError as error:raise ValueError('Unsupported GLB texture sampler') from error


def extract(path, directory):
    from .blender import sha
    visual = trimesh.load(path, force='scene', process=False)
    result = []
    for index, node in enumerate(visual.graph.nodes_geometry):
        _, key = visual.graph[node]
        material = getattr(visual.geometry[key].visual, 'material', None)
        image = getattr(material, 'baseColorTexture', None)
        if image is None:
            continue
        validate_texture_coordinates(visual.geometry[key])
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory/f'part_{index}.png'
        image.save(destination)
        result.append({'part': index, 'path': rel(destination), 'sha256': sha(destination),
                       'wrap':sampler(path,material.name)})
    return result


def validate_texture_coordinates(mesh):
    """A spatial image must map to surface area, not one repeated texel.

    Solid-colour images remain valid with constant UVs. This is a transfer
    integrity check, not an appearance approval or a new unwrap operation.
    """
    import numpy as np
    image=getattr(getattr(mesh.visual,'material',None),'baseColorTexture',None)
    if image is None:return
    uv=np.asarray(getattr(mesh.visual,'uv',None),float)
    if uv.shape!=(len(mesh.vertices),2) or not np.isfinite(uv).all():
        raise ValueError('Textured support requires finite source vertex UVs')
    pixels=np.asarray(image.convert('RGB'))
    if np.ptp(pixels.reshape(-1,3).astype(float),axis=0).max()<=1:return
    triangles=uv[np.asarray(mesh.faces)]
    a,b=triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
    area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    if not (area>1e-12).any():
        raise ValueError('Spatial texture collapsed to zero-area UVs; repair source/Boolean channel transfer')
