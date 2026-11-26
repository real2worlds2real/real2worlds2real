"""Keep the observed top in place when changing geometry below it.

The generated mesh still uses its canonical bounding-box origin. Only the pose
parameterization changes; the actual transformed mesh receives all checks.
"""
import numpy as np


def material_anchor(mesh, material):
    names=list(mesh['material_names'])
    if material not in names:
        raise ValueError('Support pose anchor material missing from actual geometry')
    faces=mesh['F'][mesh['face_material']==names.index(material)]
    if not len(faces):raise ValueError('Support pose anchor material is empty')
    vertices=mesh['V'][np.unique(faces)]
    return (vertices[:, :2].min(0)+vertices[:, :2].max(0))/2


def anchored_pose(pose, reference, current):
    a=np.radians(pose['yaw_deg'])
    R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return {**pose,'xy':(np.asarray(pose['xy'])+R@(np.asarray(reference)-current)).tolist()}


def carry_pose(previous_pose,previous_anchor,new_pose,new_anchor):
    """Preserve a physical top anchor across different canonical mesh origins."""
    def rotation(pose):
        a=np.radians(pose['yaw_deg'])
        return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    world=np.asarray(previous_pose['xy'])+rotation(previous_pose)@np.asarray(previous_anchor)
    return {**new_pose,'xy':(world-rotation(new_pose)@np.asarray(new_anchor)).tolist()}
