"""Equivalent coincident rigid masses for a fixed numerical body partition.

Every component has the same COM and normalized inertia. Welding them adds no
mechanical freedom and preserves all rigid mass moments. Native weld error and
coupled material response still require independent numerical qualification.
"""
import numpy as np


def properties(mass,centre,inertia,count,*,root_fraction=.5):
    c=np.asarray(centre,float);I=np.asarray(inertia,float)
    if (not np.isfinite(mass) or mass<=0 or c.shape!=(3,) or I.shape!=(3,3)
            or not np.isfinite(c).all() or not np.isfinite(I).all() or not np.allclose(I,I.T,atol=1e-14)
            or np.linalg.eigvalsh(I).min()<=0 or type(count) is not int or not 1<=count<=4096
            or not 0<root_fraction<1):
        raise ValueError('Positive rigid mass/inertia and a bounded partition required')
    weights=np.r_[root_fraction,np.full(count,(1-root_fraction)/count)]
    return {'masses':mass*weights,'centres':np.tile(c,(count+1,1)),
            'inertias':I[None]*weights[:,None,None],'weights':weights,
            'mechanical_degrees_of_freedom_added':0}


def parallel_axis(offset):
    q=np.asarray(offset,float)
    return np.eye(3)*float(q@q)-np.outer(q,q)


def anchored_properties(mass,centre,inertia,anchors):
    """Equivalent welded components whose force points are their own COMs.

    Retain aggregate COM and full inertia, including all parallel-axis terms.
    Reduce numerical frame mass until the residual physical root is positive;
    never add inertia to make a poor representation appear stable.
    """
    a=np.asarray(anchors,float);c=np.asarray(centre,float);I=np.asarray(inertia,float)
    if a.ndim!=2 or a.shape[1]!=3 or not len(a) or not np.isfinite(a).all():raise ValueError('Actual finite bond anchors required')
    properties(mass,c,I,len(a))
    # Allocate the largest well-conditioned feasible fraction to the numerical
    # frames. Tiny frame masses make the star of fixed constraints ill-scaled;
    # the admissibility test below still preserves every physical mass moment.
    fraction=.5
    for _ in range(30):
        centres=np.r_[((c-fraction*a.mean(0))/(1-fraction))[None],a]
        weights=np.r_[1-fraction,np.full(len(a),fraction/len(a))]
        shifts=sum(w*mass*parallel_axis(x-c) for w,x in zip(weights,centres))
        root=(1-fraction)*I-shifts
        if np.linalg.eigvalsh(root).min()>=.25*np.linalg.eigvalsh(I).min():break
        fraction/=2
    else:raise ValueError('Cannot preserve source mass moments with these numerical anchors')
    inertias=np.concatenate([root[None],np.tile((I*fraction/len(a))[None],(len(a),1,1))])
    return {'masses':mass*weights,'centres':centres,'inertias':inertias,'weights':weights,
            'mechanical_degrees_of_freedom_added':0}


def from_surface(vertices,faces,mass,count,*,anchors=None):
    import trimesh
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    if not mesh.is_volume:raise ValueError('Physical source partition requires an actual closed solid')
    I=np.asarray(mesh.moment_inertia)*mass/mesh.volume
    result=(properties(mass,mesh.center_mass,I,count) if anchors is None
            else anchored_properties(mass,mesh.center_mass,I,anchors))
    if len(result['masses'])!=count+1:raise ValueError('Numerical anchor count changed')
    return {**result,'source_volume_m3':float(mesh.volume),'source_centre':np.asarray(mesh.center_mass),
            'source_inertia_per_kg':I/mass}
