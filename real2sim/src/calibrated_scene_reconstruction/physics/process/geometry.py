"""Freeze solid/rest geometry across candidates, while allowing particle states."""
import hashlib

import numpy as np


def signature(stage, actors, definitions):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.XformCache()
    h = hashlib.sha256()

    def rounded(value):
        values = np.round(np.asarray(value, dtype=np.float64), 6)
        return np.where(values == 0, 0., values).tobytes()  # Canonicalize signed zero.
    roots = {name: binding['path'] for name, binding in actors.items()}
    for name in sorted(actors):
        binding = actors[name]
        if binding['kind'] == 'particles':
            continue  # Particle quantity/initial placement is state, not solid/rest geometry.
        root = stage.GetPrimAtPath(binding['path'])
        M = np.asarray(cache.GetLocalToWorldTransform(root)).T
        inverse = np.eye(4)
        if definitions[name]['motion'] != 'static':
            u, _, vt = np.linalg.svd(M[:3,:3])
            R = u@vt
            inverse[:3,:3] = R.T
            inverse[:3,3] = -R.T@M[:3,3]
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if any(other != name and (path.startswith(p+'/') or path == p)
                   for other,p in roots.items()):
                continue
            if not prim.IsA(UsdGeom.Boundable):
                continue
            relative = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            h.update((name+path+prim.GetTypeName()).encode())
            h.update(rounded(relative))
            for attrname in ('points','faceVertexCounts','faceVertexIndices','size','radius','height','width','length','axis'):
                value = prim.GetAttribute(attrname).Get()
                if value is None:
                    continue
                h.update(attrname.encode())
                if isinstance(value,str):
                    h.update(value.encode())
                else:
                    h.update(rounded(value))
    return h.hexdigest()
