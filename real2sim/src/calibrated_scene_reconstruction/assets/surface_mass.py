"""Audit uniform native node masses against a homogeneous sheet hypothesis.

The installed surface backend's uniform node model cannot preserve the masses
of disconnected sheets with unequal area per node. Native material parameters
alone do not fix this discretization error. Split bodies or remesh by area.
"""
import numpy as np


def distribution(points,faces,total_mass,*,maximum_component_error=.05):
    p=np.asarray(points,float);f=np.asarray(faces,int)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f)
            or f.min()<0 or f.max()>=len(p) or not np.isfinite(total_mass) or not total_mass>0
            or not 0<=maximum_component_error<=1):
        raise ValueError('Finite material geometry and positive total mass required')
    tri=p[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if not np.isfinite(area).all() or min(area)<=0 or not np.isfinite(p).all():raise ValueError('Nondegenerate finite material triangles required')
    parent=np.arange(len(p))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for face in f:
        for a in face[1:]:parent[find(int(a))]=find(int(face[0]))
    groups={}
    for i in range(len(p)):groups.setdefault(find(i),[]).append(i)
    rows=[];density=total_mass/area.sum()
    for indices in groups.values():
        selected=np.isin(f,indices).all(1);a=float(area[selected].sum())
        if a<=0:raise ValueError('Unused material nodes would receive nonphysical native mass')
        native_mass=total_mass*len(indices)/len(p);expected=density*a
        rows.append({'vertices':len(indices),'area_m2':a,'native_uniform_mass_kg':native_mass,
                     'homogeneous_material_mass_kg':expected,'relative_mass_error':float(abs(native_mass/expected-1)),
                     'effective_areal_density_kg_m2':float(native_mass/a)})
    uniform=p.mean(0);physical=(tri.mean(1)*area[:,None]).sum(0)/area.sum()
    return {'pass':all(x['relative_mass_error']<=maximum_component_error for x in rows),
            'model':'Uniform native node masses, separately qualified backend model; not direct per-node mass readback',
            'components':rows,'maximum_component_error':maximum_component_error,
            'uniform_node_centroid':uniform.tolist(),'homogeneous_area_centroid':physical.tolist(),
            'centroid_discretization_error_m':float(np.linalg.norm(uniform-physical)),
            'scope':'Independent-component mean mass consistency only. Within-sheet centroid error is reported, not certified; load/contact response remains mandatory.'}
