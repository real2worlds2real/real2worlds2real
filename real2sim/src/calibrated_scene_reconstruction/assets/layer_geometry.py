"""Explicit material-layer geometry hypotheses; never native attachments."""
import numpy as np


def extend_displacements(displacements,faces,selected):
    """Harmonically extend a partial geometry proposal across material seams.

    Abruptly moving only panel interiors can manufacture large strains at the
    last selected row. Interpolate the remaining source-indexed material nodes
    instead of leaving them fixed. This is a proposal, never a native pin or
    evidence that the prescribed layer spacing is physically correct.
    """
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve
    from .cloth_rest import triangle_edges
    values=np.asarray(displacements,float);selected=np.asarray(selected)
    if (values.ndim!=2 or values.shape[1:]!=(3,) or not len(values)
            or not np.isfinite(values).all() or selected.dtype.kind!='b'
            or selected.shape!=(len(values),) or not selected.any()):
        raise ValueError('Finite node proposals and an explicit nonempty material selection required')
    edges=triangle_edges(faces)
    if not len(edges) or edges.min()<0 or edges.max()>=len(values):
        raise ValueError('Material edges must reference the proposed nodes')
    graph=coo_matrix((np.ones(2*len(edges)),
                     (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),
                    shape=(len(values),len(values))).tocsr()
    count,labels=connected_components(graph,directed=False)
    if len(np.unique(labels[selected]))!=count:
        raise ValueError('Every material component needs an explicit proposal anchor')
    result=values.copy();unknown=~selected
    if unknown.any():
        laplacian=diags(np.asarray(graph.sum(1)).ravel())-graph
        result[unknown]=spsolve(laplacian[unknown][:,unknown].tocsc(),
                               -laplacian[unknown][:,selected]@values[selected])
    if not np.isfinite(result).all():raise ValueError('Nonfinite material proposal interpolation')
    return result


def coherent_displacements(displacements,labels):
    """Share a geometry proposal across explicitly corresponding layers.

    Labels describe agent-reviewed construction correspondence, not measured
    bonding or runtime constraints. Each node has exactly one material label.
    Relative vectors within a label remain unchanged by this initialization.
    """
    values,labels=np.asarray(displacements,float),np.asarray(labels)
    if (values.ndim!=2 or values.shape[1:]!=(3,) or not len(values) or labels.shape!=(len(values),)
            or labels.dtype.kind not in 'iu' or not np.isfinite(values).all()):
        raise ValueError('Finite node proposals and one integer construction label per node required')
    _,inverse=np.unique(labels,return_inverse=True)
    sums=np.zeros((int(inverse.max())+1,3));counts=np.bincount(inverse)
    np.add.at(sums,inverse,values)
    return sums[inverse]/counts[inverse,None]


def bundle(points,groups,normal,spacing,*,gain=1.):
    """Close independently proposed layers around their shared midsurface.

    Each row lists corresponding material nodes in physical layer order.
    The caller supplies source/construction evidence for those correspondences.
    Tangential locations are retained; this does not glue layers or change the
    intrinsic metric. Full strain, crossing/thickness and source checks must
    reject an inadmissible result. No native state is ever written here.
    """
    points,groups,normal=np.asarray(points,float),np.asarray(groups),np.asarray(normal,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or groups.ndim!=2 or groups.shape[1]<2
            or groups.dtype.kind not in 'iu' or not groups.size or groups.min()<0 or groups.max()>=len(points)
            or len(np.unique(groups))!=groups.size or normal.shape!=(len(groups),3)
            or not np.isfinite(points).all() or not np.isfinite(normal).all()
            or not np.allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-7)
            or not np.isfinite([spacing,gain]).all() or spacing<=0 or not 0<=gain<=1):
        raise ValueError('Disjoint explicit material bundles, unit normals and finite spacing required')
    values=points[groups];centre=values.mean(1)
    normal_coordinate=((values-centre[:,None])*normal[:,None]).sum(2)
    target=(np.arange(groups.shape[1])-(groups.shape[1]-1)/2)*spacing
    # Preserve the observed side ordering, including reflected material charts.
    sign=np.where(normal_coordinate[:,-1]>=normal_coordinate[:,0],1.,-1.)
    correction=(target[None]*sign[:,None]-normal_coordinate)*gain
    result=points.copy();result[groups]=values+correction[:,:,None]*normal[:,None]
    return result,{'proposal_only':True,'task_approved':False,'positions_in_native_written':False,
                   'material_bundles':len(groups),'layer_count':groups.shape[1],
                   'maximum_displacement_m':float(abs(correction).max()),
                   'intrinsic_material_changed':False,'spacing_m':float(spacing)}
