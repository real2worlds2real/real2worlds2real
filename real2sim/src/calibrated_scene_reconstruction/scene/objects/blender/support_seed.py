"""Candidate support surfaces for objects contained by other reconstructed objects.

This initializes the source search. It never certifies contact, support or a
parent asset; final native checks must evaluate both actual geometries.
"""
import json
import numpy as np


def inputs(session, obj):
    spec=session.scene_graph['dynamic_objects'][obj]['fit'].get('support_seed')
    if spec is None:return {}
    if (not isinstance(spec,dict) or not {'object','round'}<=set(spec)
            or set(spec)-{'object','round','surface'}
            or spec['object']==obj or spec['object'] not in session.scene_graph['dynamic_objects']
            or type(spec['round']) is not int or spec['round']<1
            or spec.get('surface','uppermost') not in {'uppermost','interior_floor'}):
        raise ValueError('Support seed needs another reconstructed object and a positive candidate round')
    root=session/'objects'/spec['object']/f"round_{spec['round']}"
    from .store import digest
    paths=[root/name for name in ['mesh.npz','fit.json','input.json']]
    if not all(p.is_file() for p in paths):raise ValueError('Incomplete candidate support round')
    captured=json.loads((root/'input.json').read_text())['inputs']
    for name in ['parsing/posed_frames/cameras.npz','splat/align.json']:
        if captured.get(name)!=digest(session/name):
            raise ValueError('Candidate support belongs to another metric camera frame')
    return {str(p.relative_to(session.dir)):digest(p) for p in paths}


def top(vertices, faces, xy, *, surface='uppermost'):
    """Select actual upward ray intersections; a container floor is explicit.

    An inserted assembly's bounding-box centre can project over the rim even
    while its lowest endpoint is inside. The floor hypothesis uses the lowest
    upward intersection, never a filled bounding box or a fabricated cavity.
    This only seeds a candidate; placement and clearance still need validation.
    """
    if surface not in {'uppermost','interior_floor'}:
        raise ValueError('Unknown candidate support surface')
    tri=np.asarray(vertices,float)[np.asarray(faces,int)]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    good=normal[:,2]>1e-12
    tri=tri[good]
    if not len(tri):raise ValueError('No upward candidate support faces')
    basis=np.stack([tri[:,1,:2]-tri[:,0,:2],tri[:,2,:2]-tri[:,0,:2]],axis=2)
    bary=np.linalg.solve(basis,(np.asarray(xy)-tri[:,0,:2])[...,None])[...,0]
    inside=(bary.min(1)>=-1e-8)&(bary.sum(1)<=1+1e-8)
    heights=tri[:,0,2]+bary[:,0]*(tri[:,1,2]-tri[:,0,2])+bary[:,1]*(tri[:,2,2]-tri[:,0,2])
    if not inside.any():raise ValueError('Source seed lies outside the candidate support surface')
    return float(heights[inside].min() if surface=='interior_floor' else heights[inside].max())


def pose(session,obj,xy,yaw):
    provenance=inputs(session,obj)
    if not provenance:return None
    spec=session.scene_graph['dynamic_objects'][obj]['fit']['support_seed']
    root=session/'objects'/spec['object']/f"round_{spec['round']}"
    fitted=json.loads((root/'fit.json').read_text())
    from .evidence import placed
    with np.load(root/'mesh.npz') as data:
        mesh=placed(data['V'],data['F'],fitted['pose'])
    height=top(mesh.vertices,mesh.faces,xy,surface=spec.get('surface','uppermost'))
    return {'xy':list(map(float,xy)), 'yaw_deg':float(yaw),'local_top':height,'support_z':height,
            'contact':'candidate_parent_surface_unverified', 'support_seed':spec,
            'support_seed_inputs':provenance}
