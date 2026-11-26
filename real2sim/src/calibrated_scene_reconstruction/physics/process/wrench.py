"""Finite source-bounded contact wrench geometry, without response targets."""
import numpy as np

from .contract import require


def circular_side_contact(center,direction,radius,lateral_offset=0.):
    """Contact point on an explicitly upright circular-side hypothesis.

    A positive lateral offset produces negative vertical torque for a force
    along direction. Other support/contact shapes need a different model.
    """
    center=np.asarray(center,float);direction=np.asarray(direction,float)
    require(center.shape==direction.shape==(3,) and np.isfinite(center).all() and np.isfinite(direction).all()
            and np.isclose(np.linalg.norm(direction),1.) and abs(direction[2])<1e-8
            and np.isfinite(radius) and radius>0 and np.isfinite(lateral_offset) and abs(lateral_offset)<radius,
            'Finite horizontal unit force and a bounded upright circular contact prior required')
    tangent=np.array([-direction[1],direction[0],0.])
    return center-direction*np.sqrt(radius**2-lateral_offset**2)+tangent*lateral_offset


def body_side_contact(position,rotation,center_local,direction,radius,lateral_offset=0.):
    """Circular body-local side; retain contact geometry when the body tilts."""
    position,rotation,center_local,direction=map(lambda v:np.asarray(v,float),
                                               (position,rotation,center_local,direction))
    require(rotation.shape==(3,3) and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
            and np.linalg.det(rotation)>0 and position.shape==center_local.shape==direction.shape==(3,)
            and np.isfinite(np.r_[position,center_local,direction]).all(),
            'A rigid native body frame and finite local circular contact center are required')
    local=rotation.T@direction;local[2]=0
    require(np.linalg.norm(local)>1e-8,'Axial loads need a different contact model')
    local/=np.linalg.norm(local)
    point=circular_side_contact(center_local,local,radius,lateral_offset)
    return position+rotation@point


def circular_grip(position,rotation,center_local,normal_local,radii,heights,magnitude):
    """Two inward radial forces on explicitly reviewed circular side sections.

    Equal opposing forces have zero net force. Different heights create a
    finite couple without a joint, pose target or unbounded free torque. This
    is a source-conditioned unmeasured grip model, not a simulated human hand.
    """
    position,rotation,center,normal,radii,heights=map(lambda v:np.asarray(v,float),
        (position,rotation,center_local,normal_local,radii,heights))
    require(rotation.shape==(3,3) and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
            and np.linalg.det(rotation)>0 and position.shape==center.shape==normal.shape==(3,)
            and np.isfinite(np.r_[position,rotation.ravel(),center,normal]).all(),
            'Finite rigid body frame and radial grip geometry required')
    require(abs(normal[2])<1e-8 and np.linalg.norm(normal)>1e-8 and radii.shape==heights.shape==(2,)
            and np.isfinite(np.r_[radii,heights,magnitude]).all() and np.all(radii>0) and magnitude>=0,
            'Explicit circular section radii/heights and nonnegative inward force required')
    normal=normal/np.linalg.norm(normal);offsets=np.array([normal*radii[0],-normal*radii[1]])
    offsets[:,2]+=heights;points=(offsets+center)@rotation.T+position
    forces=np.array([-normal,normal])*magnitude
    return points,forces@rotation.T
