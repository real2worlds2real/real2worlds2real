"""Compare native jointed-link poses with their baked source initial state.

Metric link meshes share an asset coordinate frame. Their captured joint
coordinate is encoded by the local joint frames, rather than extra link
transforms. Common world placement and free fall therefore cancel here.
"""
import numpy as np
from .mechanisms import joint_frames,revolute_position


def measure(joints,poses,*,translation_tolerance_m=1e-4,rotation_tolerance_rad=1e-3):
    tolerances=np.asarray([translation_tolerance_m,rotation_tolerance_rad],float)
    if not np.isfinite(tolerances).all() or (tolerances<=0).any():
        raise ValueError('Finite positive initial articulation tolerances required')
    rows=[]
    for joint in joints:
        pa,ra=map(lambda x:np.asarray(x,float),poses[joint['parent']])
        pb,rb=map(lambda x:np.asarray(x,float),poses[joint['child']])
        for p,r in [(pa,ra),(pb,rb)]:
            if (p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all()
                    or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5)
                    or not np.isclose(np.linalg.det(r),1.,atol=1e-5)):
                raise ValueError('Actual finite proper rigid poses required')
        relative=ra.T@rb
        angle=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
        translation=float(np.linalg.norm(pb-pa))
        row={'parent':joint['parent'],'child':joint['child'],'type':joint['type'],
             'translation_error_m':translation,'rotation_error_rad':angle,
             'pass':bool(translation<=translation_tolerance_m and angle<=rotation_tolerance_rad)}
        if joint['type']=='revolute':
            row['authored_coordinate_rad']=float(joint.get('initial_position',0.))
            row['native_coordinate_rad']=revolute_position(ra,rb,*joint_frames(joint))
        rows.append(row)
    return {'pass':all(r['pass'] for r in rows),'joints':rows,
            'translation_tolerance_m':float(translation_tolerance_m),
            'rotation_tolerance_rad':float(rotation_tolerance_rad),
            'scope':'Native initialization parity only; source, articulation path, contact and task checks remain required'}
