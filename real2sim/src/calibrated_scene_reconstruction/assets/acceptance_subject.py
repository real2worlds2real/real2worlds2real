"""Immutable physical-asset identities for source and task acceptance."""
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_digest(path):
    """Ignore appearance-only changes, retaining exact physical/material data."""
    h=hashlib.sha256()
    with np.load(path) as data:
        for name,dtype in [('V','<f8'),('F','<i8')]:
            array=np.ascontiguousarray(data[name],dtype=dtype)
            h.update(name.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
        for name in ['links_json','joints_json','cloth_rest_json']:
            h.update(name.encode());h.update(digest(json.loads(str(data[name])) if name in data else None).encode())
        if 'volume_initial_json' in data:
            h.update(b'volume_initial_json');h.update(digest(json.loads(str(data['volume_initial_json']))).encode())
    return h.hexdigest()


def subject(session,obj,mesh,case):
    fit=session.scene_graph['dynamic_objects'][obj].get('fit') or {}
    simulation={k:v for k,v in session.config('simulation.json',{}).items() if k!='basis'}
    from calibrated_scene_reconstruction.core.paths import resolve
    companions={}
    for name,spec in session.scene_graph['dynamic_objects'].items():
        if name==obj:continue
        fix=spec.get('fix') or {};request=spec.get('fit') or {}
        if fix.get('mesh'):
            companions[name]={'physical_mesh':mesh_digest(resolve(fix['mesh'])),
                'requested_numerics':{key:request.get(key) for key in ['sdf','link_physics']}}
    geometry_checks=session.config('agent/geometry_checks.json',{})
    extra={'reviewed_geometry':geometry_checks} if geometry_checks else {}
    interfaces=session.config('agent/interface_checks.json',{})
    if interfaces:extra['reviewed_interfaces']=interfaces
    return digest({'object':obj,'case':case,'physical_mesh':mesh_digest(mesh),**extra,
                   'companion_physical_assets':companions,
                   'task':session.config('task.json',{}),'simulation':simulation,
                   'capture_inputs':session.config('inputs.json',{}),
                   'mechanisms':session.config('agent/mechanisms.json',{}),
                   'physical_priors':session.config('agent/physics_priors.json',{}),
                   'requested_numerics':{key:fit.get(key) for key in ['sdf','link_physics']}})
