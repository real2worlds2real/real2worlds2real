"""Source-mask centreline proposals in calibrated cameras.

This is an initializer for narrow straight parts, never a replacement for full
mesh, source silhouette, semantic or physical checks. Callers must establish
part identity and direct visibility independently; transmitted/occluded pixels
cannot provide a straight-ray centreline observation.
"""
import numpy as np


def image_axis(mask):
    """PCA axis of one independently reviewed visible straight-part mask."""
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype.kind!='b':raise ValueError('Boolean source mask required')
    yy,xx=np.nonzero(mask)
    if len(xx)<16:raise ValueError('At least16 directly observed part pixels required')
    pixels=np.column_stack([xx,yy]).astype(float);centre=pixels.mean(0)
    values,axes=np.linalg.eigh(np.cov(pixels.T))
    if values[1]<=1e-8 or values[1]/max(1e-8,values[0])<9:
        raise ValueError('Mask is not a resolved slender straight-part proposal')
    tangent=axes[:,1];normal=np.array([-tangent[1],tangent[0]])
    line=np.r_[normal,-normal@centre]
    along=(pixels-centre)@tangent;across=(pixels-centre)@normal
    return {'line':line,'centre':centre,'tangent':tangent,
            'endpoints':centre+np.quantile(along,[.005,.995])[:,None]*tangent,
            'half_width_px':float(np.quantile(np.abs(across),.95)),
            'elongation':float(np.sqrt(values[1]/max(1e-8,values[0]))),
            'pixels':int(len(xx))}


def interpretation_plane(line,intrinsic,world_from_camera,centre):
    """The observed image line backprojects to a plane through camera centre."""
    line,intrinsic,rotation,centre=map(lambda v:np.asarray(v,float),(line,intrinsic,world_from_camera,centre))
    if (line.shape!=(3,) or intrinsic.shape!=(3,3) or rotation.shape!=(3,3) or centre.shape!=(3,)
            or not all(np.isfinite(v).all() for v in [line,intrinsic,rotation,centre])):
        raise ValueError('Finite calibrated camera and homogeneous image line required')
    normal=rotation@intrinsic.T@line;length=np.linalg.norm(normal)
    if length<1e-10:raise ValueError('Degenerate image line')
    normal/=length
    return normal,float(normal@centre)


def triangulate(planes,*,anchor,axis_sign=None):
    """Least-squares world line from corresponding independent source planes.

    anchor fixes only the otherwise arbitrary point along the line. It is not
    an added observation or a known dimension. Return residuals, not a pass.
    """
    planes=list(planes);anchor=np.asarray(anchor,float)
    if len(planes)<2 or anchor.shape!=(3,) or not np.isfinite(anchor).all():
        raise ValueError('Two independent source planes and finite anchor required')
    normals=np.asarray([p[0] for p in planes],float);offsets=np.asarray([p[1] for p in planes],float)
    if normals.shape!=(len(planes),3) or not np.isfinite(normals).all() or not np.isfinite(offsets).all():
        raise ValueError('Finite source planes required')
    lengths=np.linalg.norm(normals,axis=1)
    if (lengths<1e-10).any():raise ValueError('Degenerate source plane')
    normals/=lengths[:,None];offsets/=lengths
    _,singular,vt=np.linalg.svd(normals,full_matrices=True)
    if len(singular)<2 or singular[1]<1e-4:raise ValueError('Insufficient multiview line parallax')
    direction=vt[-1]
    if axis_sign is not None:
        sign=np.asarray(axis_sign,float)
        if sign.shape!=(3,) or not np.isfinite(sign).all() or np.linalg.norm(sign)<1e-10:raise ValueError('Finite nonzero axis sign required')
        if direction@sign<0:direction=-direction
    design=np.vstack([normals,direction]);rhs=np.r_[offsets,direction@anchor]
    point=np.linalg.lstsq(design,rhs,rcond=None)[0]
    return {'point':point,'direction':direction,'plane_residual_m':normals@point-offsets,
            'angular_residual':normals@direction,'plane_singular_values':singular}


def at_height(point,direction,height):
    point,direction=np.asarray(point,float),np.asarray(direction,float)
    if point.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(np.r_[point,direction,height]).all() or abs(direction[2])<1e-8:
        raise ValueError('Finite nonhorizontal world line and height required')
    return point+direction*((height-point[2])/direction[2])


def refine_visible_segments(records, *, point, direction, robust_pixels=1., max_evaluations=200):
    """Fit the axis to actual observed segment endpoints, without extrapolation.

    A short occluded segment must not acquire the angular leverage of a long
    hypothetical projected rod. Every record retains its two observed pixels;
    the source-image pixel scale controls only robust initialization, never an
    acceptance tolerance. Full actual-mesh source checks remain mandatory.
    """
    from scipy.optimize import least_squares
    point,direction=np.asarray(point,float),np.asarray(direction,float)
    if (point.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(np.r_[point,direction]).all()
            or abs(direction[2])<1e-8 or not np.isfinite(robust_pixels) or robust_pixels<=0
            or type(max_evaluations) is not int or not 10<=max_evaluations<=1000 or len(records)<2):
        raise ValueError('Finite nonhorizontal axis, visible records and bounded refinement required')
    observed=[]
    for row in records:
        ends=np.asarray(row['endpoints'],float);K=np.asarray(row['K'],float)
        R=np.asarray(row['R_world_cam'],float);C=np.asarray(row['centre'],float)
        if (ends.shape!=(2,2) or K.shape!=(3,3) or R.shape!=(3,3) or C.shape!=(3,)
                or not all(np.isfinite(value).all() for value in [ends,K,R,C])
                or np.linalg.norm(ends[1]-ends[0])<1):
            raise ValueError('Each axis observation needs two resolved native pixels and a camera')
        observed.append((ends,K,R,C))
    height=point[2]
    def unpack(q):return np.r_[q[:2],height],np.r_[q[2:],1.]
    def residual(q):
        p,d=unpack(q);ends=np.stack([p,p+d*.1]);values=[]
        for pixels,K,R,C in observed:
            homogeneous=((ends-C)@R)@K.T;line=np.cross(homogeneous[0],homogeneous[1])
            norm=np.linalg.norm(line[:2])
            if norm<1e-12:values.extend([1e6,1e6]);continue
            values.extend(np.c_[pixels,np.ones(2)]@(line/norm))
        return np.asarray(values)
    initial=np.r_[point[:2],direction[:2]/direction[2]]
    fit=least_squares(residual,initial,loss='soft_l1',f_scale=robust_pixels,
                      x_scale='jac',max_nfev=max_evaluations)
    p,d=unpack(fit.x);d/=np.linalg.norm(d)
    return {'point':p,'direction':d,'pixel_residuals':residual(fit.x).reshape(-1,2),
            'initial_pixel_residuals':residual(initial).reshape(-1,2),
            'success':bool(fit.success),'evaluations':int(fit.nfev),
            'robust_pixels':float(robust_pixels),'proposal_only':True,
            'scope':'All actual visible segment endpoints; no extrapolated source evidence or geometry/task approval.'}
