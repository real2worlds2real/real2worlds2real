"""Isaac5.1 uniform particle-cloth mass cooking compatibility.

The integration normalizes uniform masses using sequential float32 sums. Dense
sheets can therefore miss a declared total by over0.1%. Correct the authoring
input before cooking, never runtime node masses or the physical mass prior.
Actual native readback remains mandatory; this is not a substitute for it.
"""
import numpy as np
from calibrated_scene_reconstruction.assets.cloth_rest import area


def authoring_mass(target,rest,faces):
    if isinstance(target,bool) or not np.isfinite(target) or target<=0:
        raise ValueError('Positive finite desired cloth mass required')
    area(rest,faces)
    points=np.asarray(rest,np.float32);faces=np.asarray(faces)
    tri=points[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    values=np.float32(.5)*np.sqrt(np.sum(cross*cross,axis=1,dtype=np.float32))
    total=np.cumsum(values,dtype=np.float32)[-1];count=len(points)
    base=np.float32(total/np.float32(count))
    accumulated=np.cumsum(np.full(count,base,np.float32),dtype=np.float32)[-1]
    if not np.isfinite([base,accumulated]).all() or min(base,accumulated)<=0:
        raise ValueError('Native float32 cloth mass normalization is ill-conditioned')
    def cooked(value):
        node=np.float32(base*np.float32(np.float32(value)/accumulated))
        return float(np.float32(1/np.float32(1/node)))*count
    value=float(target)
    original=cooked(value)
    for _ in range(2):value*=target/cooked(value)
    if not .95<value/target<1.05 or abs(cooked(value)/target-1)>1e-5:
        raise ValueError('Unvalidated native mass normalization correction; retain failed geometry')
    return {'desired_mass_kg':float(target),'authoring_mass_kg':float(np.float32(value)),
            'predicted_native_mass_kg':cooked(value),'uncorrected_native_mass_kg':original,
            'correction_factor':value/target,'backend':'Isaac5.1 uniform particle cloth',
            'runtime_node_masses_changed':False,'native_readback_required':True}
