"""Validate explicit real surface-to-rigid root connections, without simulation."""
import numpy as np


def validate(value, parts, vertices=None, faces=None):
    if not isinstance(value,list):raise ValueError('Surface bindings must be a list')
    links={p['name']:p for p in parts};seen=set()
    for item in value:
        if (not isinstance(item,dict) or set(item)-{'surface','rigid','vertices','basis','clamp'}
                or not {'surface','rigid','vertices','basis'}<=set(item)
                or not isinstance(item['basis'],str) or not item['basis'].strip()):
            raise ValueError('Surface bindings need actual links, root vertices and construction basis')
        surface=links.get(item['surface'],{});rigid=links.get(item['rigid'],{})
        if (surface.get('kind')!='cloth' or surface.get('backend','particle') not in {'surface_fem','particle'}
                or rigid.get('kind')!='rigid' or item['surface'] in seen):
            raise ValueError('One explicit native cloth root bond to a real rigid link is supported')
        indices=np.asarray(item['vertices'])
        if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
                or len(set(indices))!=len(indices) or indices.min()<0
                or indices.max()>=surface['vertex_count']):
            raise ValueError('Root indices must refer to distinct actual surface nodes')
        if vertices is not None and faces is not None:
            a,n,b,count=(surface[k] for k in ('vertex_start','vertex_count','face_start','triangle_count'))
            binding(np.asarray(vertices)[a:a+n],np.asarray(faces)[b:b+count]-a,indices,np.eye(4))
        clamp=item.get('clamp')
        if clamp is not None:
            if (not isinstance(clamp,dict) or set(clamp)!={'minimum_width_m','minimum_area_m2'}
                    or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in clamp.values())):
                raise ValueError('Finite clamp footprint needs explicit positive physical width and area bounds')
            if vertices is not None and faces is not None:
                from .bond_footprint import measure
                footprint=measure(np.asarray(vertices)[a:a+n],np.asarray(faces)[b:b+count]-a,indices,**clamp)
                if not footprint['pass']:raise ValueError('Actual clamped material footprint is below its declared physical bounds')
        seen.add(item['surface'])
    return value


def binding(points, faces, indices, local_to_world):
    points=np.asarray(points,float);faces=np.asarray(faces);indices=np.asarray(indices)
    matrix=np.asarray(local_to_world,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min()<0 or faces.max()>=len(points)
            or indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
            or len(set(indices))!=len(indices) or indices.min()<0 or indices.max()>=len(points)
            or matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[:3,3],0) or not np.isclose(matrix[3,3],1)
            or np.linalg.det(matrix[:3,:3])<=0):
        raise ValueError('Finite metric geometry, distinct real root indices and a valid USD row-vector frame required')
    inverse=np.linalg.inv(matrix)
    local=points[indices]@inverse[:3,:3]+inverse[3,:3]
    if not np.allclose(local@matrix[:3,:3]+matrix[3,:3],points[indices],atol=1e-10):
        raise ValueError('Attachment root frame fails initial coincidence')
    bonded=np.flatnonzero(np.isin(faces,indices).any(axis=1))
    if not len(bonded) or len(bonded)==len(faces):
        raise ValueError('Root bond must retain an unfiltered free physical surface')
    return {'indices':indices.tolist(),'local_positions':local.tolist(),
            'filtered_root_faces':bonded.tolist(),'unfiltered_free_faces':len(faces)-len(bonded)}



def validate_islands(rest, bindings, parts):
    links={p['name']:p for p in parts}
    for name,material in rest.items():
        if 'components' not in material:continue
        backend=links[name].get('backend','particle')
        if backend not in {'surface_fem','particle'}:
            raise ValueError('Disconnected material islands need an explicit native surface backend')
        if backend=='particle':
            # Only explicit component-owned springs are permitted. Native
            # auto-spring generation could connect separate free sheets.
            owner={i:k for k,c in enumerate(material['components']) for i in c}
            springs=material.get('springs',{})
            if (set(springs)!={'stretch','shear','bend'} or any(not pairs for pairs in springs.values())
                    or any(owner.get(a)!=owner.get(b) for pairs in springs.values() for a,b in pairs)):
                raise ValueError('Particle islands require explicit springs with no inter-sheet bridges')
        bond=next((b for b in bindings if b['surface']==name),None)
        if bond is None or any(not set(c)&set(bond['vertices']) for c in material['components']):
            raise ValueError('Every FEM island needs a real connection to the common rigid receptacle')
