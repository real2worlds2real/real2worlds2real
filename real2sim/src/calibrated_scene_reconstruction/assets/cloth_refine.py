"""Exact midpoint refinement for surface-FEM contact convergence experiments.

Preserves the piecewise-linear surface, intrinsic material and natural shape.
This is offline construction only. Native material/contact controls must be
repeated; particle spring laws cannot be transferred by this operation.
"""
from copy import deepcopy
import numpy as np

from .cloth_rest import validate, triangle_edges
from .cloth_curvature import adjacency
from .surface_refinement import uniform

CORNER_WEIGHTS=np.array([
    [[1,0,0],[.5,.5,0],[.5,0,.5]],
    [[.5,.5,0],[0,1,0],[0,.5,.5]],
    [[.5,0,.5],[0,.5,.5],[0,0,1]],
    [[.5,.5,0],[0,.5,.5],[.5,0,.5]]],float)


def subdivide(points,faces,chart,*,backend):
    if backend!='surface_fem':
        raise ValueError('Particle spring laws need separate resolution calibration')
    if any(k in chart for k in ['bend_memory','bend_angles','spring_model']):
        raise ValueError('Refine explicit spring/angle laws separately; do not silently discard them')
    points=np.asarray(points,float);faces=np.asarray(faces)
    validate(chart,points,faces)
    edges=triangle_edges(faces);offset=len(points)
    lookup={tuple(map(int,e)):i+offset for i,e in enumerate(edges)}
    def mids(a,b):return np.array([lookup[tuple(sorted((int(x),int(y))))] for x,y in zip(a,b)])
    a,b,c=faces.T;ab,bc,ca=mids(a,b),mids(b,c),mids(c,a)
    result,refined,_,_=uniform(points,faces,[])
    def interpolate(values):
        values=np.asarray(values,float)
        return np.r_[values,values[edges].mean(1)]
    material=deepcopy(chart);material['points']=interpolate(chart['points']).tolist()
    if 'bend_reference' in material:
        material['bend_reference']['points']=interpolate(chart['bend_reference']['points']).tolist()
    old_shear={tuple(sorted(e)) for e in chart['springs']['shear']}
    kinds={'stretch':set(),'shear':set()}
    for orig,mid,other in [(np.c_[a,b],ab,np.c_[bc,ca]),(np.c_[b,c],bc,np.c_[ca,ab]),(np.c_[c,a],ca,np.c_[ab,bc])]:
        for pair,node,inner in zip(orig,mid,other):
            kind='shear' if tuple(sorted(map(int,pair))) in old_shear else 'stretch'
            kinds[kind].update(tuple(sorted(map(int,e))) for e in [(pair[0],node),(node,pair[1]),inner])
    pairs,shared=adjacency(refined);bend=set();membrane=kinds['stretch']|kinds['shear']
    for (i,j),edge in zip(pairs,shared):
        opposite=tuple(sorted((next(int(v) for v in refined[i] if v not in edge),
                               next(int(v) for v in refined[j] if v not in edge))))
        if opposite not in membrane:bend.add(opposite)
    material['springs']={k:sorted(v) for k,v in {**kinds,'bend':bend}.items()}
    if 'components' in material:
        components=[]
        for component in chart['components']:
            owned=set(component)
            components.append(sorted(owned|{i+offset for i,(a,b) in enumerate(edges)
                if int(a) in owned and int(b) in owned}))
        material['components']=components
    validate(material,result,refined)
    return result,refined,material,{'vertices_before':len(points),'vertices_after':len(result),
        'triangles_before':len(faces),'triangles_after':len(refined),
        'surface_changed':False,'intrinsic_material_changed':False,'native_approved':False,
        'scope':'Exact offline subdivision; new native mass/contact/material response must be checked.'}


def interpolate_corners(values):
    """Transfer per-face UVs or other linear corner attributes, retaining seams."""
    values=np.asarray(values)
    if values.ndim!=3 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise ValueError('Finite per-face triangle corner attributes required')
    return np.einsum('kij,fjc->fkic',CORNER_WEIGHTS,values).reshape(-1,3,values.shape[2])
