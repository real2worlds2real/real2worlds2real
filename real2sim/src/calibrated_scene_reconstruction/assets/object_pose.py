"""Shared object orientation for source fitting, native authoring and checks."""
import numpy as np


def support_rotation(normal):
    """Minimal proper rotation from canonical+Z to a verified support normal."""
    n=np.asarray(normal,float)
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isclose(n@n,1.) or n[2]<=.2:
        raise ValueError('An upward unit support normal is required')
    v=np.cross([0.,0.,1.],n)
    x,y,z=v;K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+K+(K@K)/(1+n[2])


def rotation(pose):
    a=np.radians(pose.get('yaw_deg',0.))
    yaw=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    frame=np.asarray(pose.get('support_rotation',np.eye(3)),float)
    if (frame.shape!=(3,3) or not np.isfinite(frame).all()
            or not np.allclose(frame.T@frame,np.eye(3),atol=1e-8)
            or not np.isclose(np.linalg.det(frame),1.,atol=1e-8)):
        raise ValueError('Object support orientation must be a proper rotation')
    return frame@yaw
