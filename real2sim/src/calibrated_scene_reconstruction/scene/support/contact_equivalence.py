"""A material-only support revision does not alter an object's contact fit."""
import hashlib
import numpy as np
from calibrated_scene_reconstruction.core.paths import resolve


def matches(saved,current):
    if saved==current:return True
    if not isinstance(saved,dict) or not isinstance(current,dict):return False
    if any(saved.get(k)!=current.get(k) for k in ['support','top_z']):return False
    a,b=saved.get('fix',{}),current.get('fix',{})
    if not a or not b or any(a.get(k)!=b.get(k) for k in ['xy','yaw_deg']):return False
    try:
        meshes=[]
        for fix in [a,b]:
            path=resolve(fix['mesh'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=fix['mesh_sha256']:return False
            with np.load(path) as data:meshes.append((data['V'].copy(),data['F'].copy()))
        return all(np.array_equal(x,y) for x,y in zip(*meshes))
    except (OSError,ValueError,KeyError):return False
