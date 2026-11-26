"""Smooth source-mesh shape proposals around an explicitly supplied local axis.

Keeps topology and full fine structure. This is geometry authoring, not source,
collision, mass or task approval. The full generated mesh needs fresh checks.
"""
import numpy as np


def deform(vertices, *, origin, axis, radial_axis, transition_m, radius_m,
           radial_scale=1., depth_scale=1., advance_m=0., flare_m=0., tilt_deg=0.,
           tail_radial_scale=1., tail_depth_scale=1., tail_tilt_u_deg=0., tail_tilt_v_deg=0.,
           tail_offset_u_m=0., tail_offset_v_m=0.):
    vertices = np.asarray(vertices, float); origin = np.asarray(origin, float)
    axis = np.asarray(axis, float); radial_axis = np.asarray(radial_axis, float)
    values = [transition_m, radius_m, radial_scale, depth_scale, advance_m, flare_m, tilt_deg,
              tail_radial_scale, tail_depth_scale, tail_tilt_u_deg, tail_tilt_v_deg,
              tail_offset_u_m, tail_offset_v_m]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or origin.shape != (3,)
            or axis.shape != (3,) or radial_axis.shape != (3,)
            or not np.isfinite(vertices).all() or not np.isfinite(origin).all()
            or not np.isfinite(values).all() or min(transition_m, radius_m) <= 0
            or not .5 <= radial_scale <= 2 or not .5 <= depth_scale <= 2
            or not .3 <= tail_radial_scale <= 2 or not .5 <= tail_depth_scale <= 2
            or depth_scale > 2 * tail_depth_scale
            or abs(advance_m) > transition_m / 4 or abs(flare_m) > radius_m / 2
            or abs(tilt_deg) > 15 or not np.isclose(np.linalg.norm(axis), 1)
            or max(abs(tail_tilt_u_deg), abs(tail_tilt_v_deg)) > 10
            or max(abs(tail_offset_u_m), abs(tail_offset_v_m)) > transition_m/2
            or not np.isclose(np.linalg.norm(radial_axis), 1)
            or abs(axis @ radial_axis) > 1e-8):
        raise ValueError('Finite bounded shape parameters and orthonormal local axes required')
    offset = vertices - origin; depth = offset @ axis
    radial = offset - depth[:, None] * axis
    t = np.clip((depth + transition_m) / transition_m, 0, 1)
    weight = t * t * (3 - 2 * t)
    new_depth = tail_depth_scale * depth + weight * ((depth_scale - tail_depth_scale) * depth + advance_m
                                 + flare_m * np.sum(radial * radial, axis=1) / radius_m ** 2)
    result = origin + radial * (tail_radial_scale + weight * (radial_scale - tail_radial_scale))[:, None] + new_depth[:, None] * axis
    # Rotate the far tail as a rigid straight member; only its attachment blends.
    # Independent source regions constrain its direction as well as its length.
    tail = tail_radial_scale * radial + (tail_depth_scale * depth)[:, None] * axis
    rotated = tail.copy()
    for direction, angle in ((radial_axis, tail_tilt_u_deg),
                             (np.cross(axis, radial_axis), tail_tilt_v_deg)):
        theta = np.radians(angle)
        rotated = (rotated * np.cos(theta) + np.cross(direction, rotated) * np.sin(theta)
                   + (rotated @ direction)[:, None] * direction * (1 - np.cos(theta)))
    shift = tail_offset_u_m * radial_axis + tail_offset_v_m * np.cross(axis, radial_axis)
    result += (1 - weight)[:, None] * (rotated - tail + shift)
    # Whole-object rigid tilt preserves straight shafts outside the transition.
    theta = np.radians(tilt_deg); k = radial_axis
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rotation = np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * cross @ cross
    result = result @ rotation.T
    lower, upper = result.min(0), result.max(0)
    result -= np.r_[(lower[:2] + upper[:2]) / 2, lower[2]]
    return result
