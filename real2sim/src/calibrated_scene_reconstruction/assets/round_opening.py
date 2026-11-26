"""Round planar mouth metadata measured from an agent's actual hollow mesh."""
import numpy as np


def measure(vertices,*,floor_z,tolerance_m=1e-6):
    v=np.asarray(vertices,float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
            or not np.isfinite([floor_z,tolerance_m]).all() or tolerance_m<=0):
        raise ValueError('Finite axisymmetric hollow-body mesh and floor plane required')
    rim=float(v[:,2].max());top=v[abs(v[:,2]-rim)<=tolerance_m,:2]
    centre=(v[:,:2].min(0)+v[:,:2].max(0))/2
    r=np.linalg.norm(top-centre,axis=1)
    if len(top)<6 or r.min()<=tolerance_m or np.ptp(r)<=4*tolerance_m or not floor_z<rim:
        raise ValueError('Two distinct nonzero planar rim rings required; solid caps are not mouths')
    inner=top[r<=r.min()+tolerance_m]
    if len(inner)<3:raise ValueError('Resolved circular inner rim required')
    a=np.c_[2*inner,np.ones(len(inner))];b=np.sum(inner**2,axis=1)
    coefficients=np.linalg.lstsq(a,b,rcond=None)[0];centre=coefficients[:2]
    radius=float(np.sqrt(coefficients[2]+centre@centre))
    if abs(np.linalg.norm(inner-centre,axis=1)-radius).max()>tolerance_m:
        raise ValueError('Source inner rim is not a circle; provide an appropriate aperture representation')
    return {'xy':centre.tolist(),'radius':radius,'rim_z':rim,'depth':float(rim-floor_z)}
