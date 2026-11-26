"""Actual opaque support triangles occlude contained-object source predictions.

Masking every pixel labelled as the container would hide oversized objects.
Depth ordering instead preserves those errors while handling a real front rim.
"""
import json
import numpy as np
import trimesh


def load(session,obj):
    request=session.scene_graph['dynamic_objects'][obj]['fit']
    enabled=request.get('source_support_occlusion',False)
    if type(enabled) is not bool:raise ValueError('Source support occlusion must be explicit boolean')
    if not enabled:return None
    from .support_seed import inputs
    provenance=inputs(session,obj)
    if not provenance:raise ValueError('Source support occlusion requires a current candidate support round')
    spec=request['support_seed'];root=session/'objects'/spec['object']/f"round_{spec['round']}"
    fitted=json.loads((root/'fit.json').read_text())
    from .evidence import placed
    with np.load(root/'mesh.npz') as data:
        materials=json.loads(str(data['materials_json']))
        if any(v.get('transmission',0)>.01 or v.get('opacity',1)<.99 for v in materials.values()):
            raise ValueError('Opaque support context cannot treat transparent materials as solid occluders')
        mesh=placed(data['V'],data['F'],fitted['pose'])
    return {'mesh':mesh,'inputs':provenance,'object':spec['object'],'round':spec['round'],
            'parent_source_geometry_pass':fitted.get('ok') is True and fitted.get('metrics',{}).get('ok') is True}


def compose(mesh,cameras,target=None):
    target=np.ones(len(mesh.faces),bool) if target is None else np.asarray(target,bool)
    if target.shape!=(len(mesh.faces),):raise ValueError('One ownership flag per actual candidate triangle required')
    context=cameras.get('_source_context')
    if context is None:return mesh,target
    parent=context['mesh']
    return trimesh.util.concatenate([mesh,parent]),np.r_[target,np.zeros(len(parent.faces),bool)]


def report(context):
    return {k:v for k,v in context.items() if k!='mesh'}
