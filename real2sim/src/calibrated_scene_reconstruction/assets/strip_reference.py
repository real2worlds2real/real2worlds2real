"""Isometric single-axis natural-curvature proposals for a material grid.

An independent construction prior supplies segment angles. This helper does
not identify a crease from a captured shape or combine incompatible fold axes.
"""
import numpy as np


def extruded(u,v,segment_angles):
    """Return a grid whose every triangle preserves the rectangular metric.

    Angles (radians) specify each longitudinal segment's tangent in Y/Z.
    Integrate discrete unit tangents, rather than sampling a continuous arc,
    so finite elements retain exact edge lengths at any resolution.
    """
    u,v,angles=map(lambda a:np.asarray(a,float),(u,v,segment_angles))
    if (u.ndim!=1 or v.ndim!=1 or len(u)<2 or len(v)<2 or angles.shape!=(len(v)-1,)
            or not all(np.isfinite(a).all() for a in [u,v,angles])
            or (np.diff(u)<=0).any() or (np.diff(v)<=0).any()):
        raise ValueError('Increasing finite material axes and one tangent angle per longitudinal segment required')
    dyz=np.diff(v)[:,None]*np.c_[np.cos(angles),np.sin(angles)]
    yz=np.r_[np.zeros((1,2)),np.cumsum(dyz,axis=0)]
    points=np.empty((len(v),len(u),3));points[:,:,0]=u
    points[:,:,1:]=yz[:,None,:]
    return points.reshape(-1,3)


def crossed_proposal(u,v,across_angles,length_angles):
    """Initialize crossed fold histories, without promising an isometry.

    Independently extruded folds generally introduce a cross-metric term at
    intersections. Project and validate against the original membrane chart
    before considering these points as a native natural reference. Inputs
    describe construction hypotheses, never captured contact-imposed poses.
    """
    u,v,a,b=map(lambda x:np.asarray(x,float),(u,v,across_angles,length_angles))
    # Reuse the single-axis input checks and unit-tangent integration.
    uxz=extruded([0.,1.],u,a).reshape(len(u),2,3)[:,0,1:]
    vyz=extruded([0.,1.],v,b).reshape(len(v),2,3)[:,0,1:]
    q=np.empty((len(v),len(u),3));q[:,:,0]=uxz[:,0]
    q[:,:,1]=vyz[:,0,None];q[:,:,2]=uxz[None,:,1]+vyz[:,1,None]
    return q.reshape(-1,3)


def reflected_extrusion(cross_section,distances,axis,fold_index):
    """Exact zero-thickness return of an extruded cross-section.

    A reflection across a material row preserves every triangle's metric even
    when that row is a folded cross-section. The ideal reference may overlap;
    native cooking, filtering and actual finite-thickness dynamics still need
    validation. Asymmetric row sampling can avoid duplicate reference nodes
    without changing the continuous material dimensions.
    """
    cross,distances,axis=map(lambda x:np.asarray(x,float),(cross_section,distances,axis))
    if (cross.ndim!=2 or cross.shape[1:]!=(3,) or len(cross)<3
            or distances.ndim!=1 or len(distances)<3 or axis.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [cross,distances,axis])
            or not np.isclose(np.linalg.norm(axis),1.) or (np.diff(distances)<=0).any()
            or type(fold_index) is not int or not 0<fold_index<len(distances)-1
            or not np.allclose(np.diff(cross,axis=0)@axis,0.,atol=1e-10)):
        raise ValueError('Finite orthogonal cross-section, unit axis, increasing rows and interior fold index required')
    along=np.where(np.arange(len(distances))<=fold_index,distances,2*distances[fold_index]-distances)
    return (cross[None]+along[:,None,None]*axis).reshape(-1,3)
