"""Compression-driven geometric buckling seeds; never runtime state control."""
import numpy as np
import trimesh


def propose(rest,points,faces,*,wavelength,gain=1.,phase=0.):
    """Perturb a compressed sheet to escape a flat projection's zero gradient.

    A planar compressed element has no first-order length gradient in its
    normal direction. Small geometric wrinkle hypotheses can initialize a
    subsequent constrained solve. This is not a material law or identified
    texture; original source, thickness, intersections and native checks stay
    mandatory. The immutable intrinsic chart is never replaced by this pose.
    """
    rest,points,faces=np.asarray(rest,float),np.asarray(points,float),np.asarray(faces)
    if (rest.shape!=points.shape or rest.ndim!=2 or rest.shape[1:]!=(3,)
            or not np.isfinite(rest).all() or not np.isfinite(points).all()
            or not np.isfinite([wavelength,gain,phase]).all() or wavelength<=0 or not 0<gain<=1):
        raise ValueError('Finite matching material/state points and bounded wrinkle construction required')
    _,_,axes=np.linalg.svd(rest-rest.mean(0),full_matrices=False)
    uv=(rest-rest.mean(0))@axes[:2].T
    if abs((rest-rest.mean(0))@axes[2]).max()>1e-8:
        raise ValueError('Wrinkle seed requires a planar intrinsic chart')
    material=uv[faces];delta=np.stack([material[:,1]-material[:,0],material[:,2]-material[:,0]],-1)
    current=points[faces];deformation=np.stack([current[:,1]-current[:,0],current[:,2]-current[:,0]],-1)@np.linalg.inv(delta)
    stretches,directions=np.linalg.eigh(deformation.transpose(0,2,1)@deformation)
    compression=np.maximum(0.,1-np.sqrt(np.maximum(stretches[:,0],0.)))
    area=abs(np.linalg.det(delta))/2
    direction=directions[:,:,0];tensor=np.einsum('n,ni,nj->ij',area*compression,direction,direction)
    _,vectors=np.linalg.eigh(tensor);axis=vectors[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    nodal=np.zeros(len(rest));weights=np.zeros(len(rest))
    for column in range(3):
        np.add.at(nodal,faces[:,column],compression*area/3)
        np.add.at(weights,faces[:,column],area/3)
    if (weights<=0).any():raise ValueError('Every material node must own positive area')
    amplitude=gain*wavelength/np.pi*np.sqrt(nodal/weights)
    normal=np.asarray(trimesh.Trimesh(points,faces,process=False).vertex_normals).copy()
    # Project a patch direction into the local normal line continuously.
    # Choosing a hard normal sign creates a displacement seam where a curled
    # surface turns perpendicular to the patch direction. n*(n.dot(axis)) is
    # unchanged by normal reversal and fades to zero at that ambiguous turn.
    _,_,spatial_axes=np.linalg.svd(points-points.mean(0),full_matrices=False)
    normal*=(normal@spatial_axes[2])[:,None]
    wave=np.sin(2*np.pi*(uv@axis)/wavelength+phase)
    result=points+(amplitude*wave)[:,None]*normal
    return result,{'proposal_only':True,'task_approved':False,'intrinsic_metric_changed':False,
        'wavelength_m':float(wavelength),'gain':float(gain),'phase_rad':float(phase),
        'material_compression_axis':axis.tolist(),'maximum_normal_seed_m':float(abs(amplitude*wave).max()),
        'scope':'Compression-based initialization only; not equilibrium, source appearance, native or task approval'}
