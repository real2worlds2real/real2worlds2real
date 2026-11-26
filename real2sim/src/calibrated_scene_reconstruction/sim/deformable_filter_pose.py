"""Material bind pose for coincident simulation and display meshes.

Installed Isaac5.1 / OmniPhysics107.3 rejects selfCollisionFilterPose at
runtime and uses bindPose instead. These callers simulate and display one
mesh, so a material bind pose is sufficient. Separate graphical meshes must
receive corresponding bind poses too. Constitutive restShapePoints and the
actual initial points remain independent and unchanged.
"""
import hashlib
import json
import numpy as np

from calibrated_scene_reconstruction.assets.deformable_filter_pose import material_pose


def author(prim,reference,cells,*,separation_m):
    from pxr import Vt
    points,report=material_pose(reference,cells,separation_m=separation_m)
    instance='materialSelfCollision';purpose='bindPose'
    if not prim.ApplyAPI('OmniPhysicsDeformablePoseAPI',instance):
        raise RuntimeError('Explicit native self-collision filtering pose unavailable')
    prefix='deformablePose:'+instance+':omniphysics:'
    authored=points.astype('float32')
    if (not prim.GetAttribute(prefix+'points').Set(Vt.Vec3fArray.FromNumpy(authored))
            or not prim.GetAttribute(prefix+'purposes').Set(Vt.TokenArray([purpose]))):
        raise RuntimeError('Explicit native collision filtering pose authoring failed')
    if (not np.array_equal(np.asarray(prim.GetAttribute(prefix+'points').Get()),authored)
            or list(prim.GetAttribute(prefix+'purposes').Get())!=[purpose]):
        raise RuntimeError('Native collision filtering pose USD readback differs')
    report.update(instance=instance,reference_sha256=hashlib.sha256(np.ascontiguousarray(reference,dtype='float64').tobytes()).hexdigest(),
                  authored_pose_sha256=hashlib.sha256(authored.tobytes()).hexdigest(),usd_readback_pass=True,
                  native_pair_exclusions_read_back=False,purpose=purpose,
                  scope='Installed107.3 bind-pose fallback; shared simulation/display mesh only. USD readback is not a functional capability proof.')
    prim.SetCustomDataByKey('calibrated_scene_reconstruction:self_collision_filter_pose',json.dumps(report))
    return report


def audit(prim):
    """Require the material bind reference to survive stage serialization."""
    value=prim.GetCustomDataByKey('calibrated_scene_reconstruction:self_collision_filter_pose')
    if not value:raise ValueError('Explicit material self-collision filtering pose missing')
    report=json.loads(value);prefix='deformablePose:'+report['instance']+':omniphysics:'
    points=np.asarray(prim.GetAttribute(prefix+'points').Get(),dtype='float32')
    purposes=prim.GetAttribute(prefix+'purposes').Get()
    mesh_points=prim.GetAttribute('points').Get()
    if (points.shape!=(len(mesh_points),3) or not np.isfinite(points).all()
            or report.get('purpose')!='bindPose' or list(purposes or [])!=['bindPose']
            or hashlib.sha256(points.tobytes()).hexdigest()!=report['authored_pose_sha256']):
        raise ValueError('Material collision filtering pose changed or lost on stage reload')
    return {**report,'serialized_pose_readback_pass':True}
