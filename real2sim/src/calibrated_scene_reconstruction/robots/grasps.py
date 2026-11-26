"""Grasp families: TCP waypoints (pre-grasp, grasp, lift) for a target, expressed in the task frame (sim.taskframe).

A candidate is {"family", "params", "R", "pre", "grasp", "lift", "close"}; R is the world rotation of the TCP link,
built from the spec's `tcp_axes` (which link axis is the approach / palm normal, the fingers, the closing direction).

power_lateral  dexterous hands (Vega + Sharpa's former grasp): the hand comes from the robot's right of the object, palm
               facing it, fingers pointing forward, thumb up; offsets as in the former apps/episode.py
antipodal      parallel grippers: top-down (approach down, closing across the narrow side) and side grasps (approach
               forward at a fraction of the height) when the object fits the gripper's max width, else rim pinches
               (cups, bowls: the wall nearest the robot, from above)
"""
import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def _axis(spec_axis):
    sign = -1.0 if spec_axis.startswith("-") else 1.0
    return AXES[spec_axis[-1]], sign


def tcp_rotation(tcp_axes, world_dirs):
    """World rotation whose link axes named in `world_dirs` ({"approach": v, ...}, keys of tcp_axes) point along the
    given world vectors. Two named axes fix the rotation; the third follows from the right-hand rule."""
    cols = {}
    for key, v in world_dirs.items():
        if key in tcp_axes and len(cols) < 2:
            i, s = _axis(tcp_axes[key])
            cols[i] = s * np.asarray(v, float) / np.linalg.norm(v)
    if len(cols) < 2:
        raise ValueError(f"need two of {list(world_dirs)} in tcp_axes {tcp_axes}")
    (i, a), (j, b) = sorted(cols.items())
    k = 3 - i - j
    R = np.zeros((3, 3))
    R[:, i], R[:, j] = a, b
    R[:, k] = np.cross(R[:, (k + 1) % 3], R[:, (k + 2) % 3])
    return R


def power_lateral(tf, p_obj, ext, arm):
    """Former Vega grasp: (palm gap, palm-centre offset ahead, grasp height fraction) grid, closure 1.4."""
    axes = arm["tcp_axes"]
    R = tcp_rotation(axes, {"approach": tf.left, "fingers": tf.forward, "thumb": tf.up})
    r = 0.5 * max(ext[0], ext[1])
    z_min = float(arm["ee"].get("min_grasp_height", 0.0))  # a large palm held thumb-up reaches below the grasp point
    out = []
    for gap in (0.012, 0.0):
        for ahead in (0.07, 0.05, 0.03, 0.09):
            for zf in (0.5, 0.35, 0.65):
                g = p_obj + tf.vec(right=r + gap, forward=-ahead, up=max(zf * ext[2], z_min))
                out.append({"family": "power_lateral", "params": [gap, ahead, zf, 1.4], "R": R,
                            "pre": g + tf.vec(right=0.10, forward=-0.04, up=0.06), "grasp": g,
                            "lift": g + tf.vec(forward=-0.05, up=0.14), "close": 1.4})
    return out


def antipodal(tf, p_obj, ext, arm):
    axes = arm["tcp_axes"]
    max_w = float(arm["ee"].get("max_width", 0.08))
    out = []
    # top-down: closing along the object's narrower horizontal extent (world x or y), a few depths below the top
    for name, closing, width in (("x", np.array([1.0, 0, 0]), ext[0]), ("y", np.array([0, 1.0, 0]), ext[1])):
        if width > 0.9 * max_w:
            continue
        R = tcp_rotation(axes, {"approach": -tf.up, "closing": closing})
        for depth in (0.02, 0.035, 0.05):
            g = p_obj + tf.up * max(ext[2] - depth, 0.5 * ext[2])
            out.append({"family": "antipodal_top", "params": [name, depth], "R": R, "pre": g + tf.up * 0.12,
                        "grasp": g, "lift": g + tf.up * 0.14, "close": 1.0})
    # rim: wider than the gripper (cups, bowls) -> pinch the wall nearest the robot from above, closing radially
    if min(ext[0], ext[1]) > 0.9 * max_w:
        r = 0.5 * abs(np.dot([ext[0], ext[1], 0], np.abs(tf.forward)))
        R = tcp_rotation(axes, {"approach": -tf.up, "closing": tf.forward})
        for depth in (0.015, 0.03):
            for inset in (0.008, 0.015):
                g = p_obj + tf.vec(forward=-(r - inset), up=max(ext[2] - depth, 0.5 * ext[2]))
                out.append({"family": "antipodal_rim", "params": [depth, inset], "R": R, "pre": g + tf.up * 0.12,
                            "grasp": g, "lift": g + tf.up * 0.14, "close": 1.0})
    # side: approach forward, closing horizontal across
    side_w = abs(np.dot([ext[0], ext[1], 0], np.abs(tf.left)))
    if side_w <= 0.9 * max_w:
        R = tcp_rotation(axes, {"approach": tf.forward, "closing": tf.left})
        for zf in (0.5, 0.35, 0.65):
            r = 0.5 * abs(np.dot([ext[0], ext[1], 0], np.abs(tf.forward)))
            g = p_obj + tf.vec(forward=-0.6 * r, up=zf * ext[2])
            out.append({"family": "antipodal_side", "params": [zf], "R": R, "pre": g + tf.vec(forward=-0.12, up=0.02),
                        "grasp": g, "lift": g + tf.vec(forward=-0.03, up=0.14), "close": 1.0})
    order = sorted(range(len(out)), key=lambda i: out[i]["family"] == "antipodal_side")  # stable: top, rim, side
    return [out[i] for i in order]


FAMILIES = {"power_lateral": power_lateral, "antipodal": antipodal}


def candidates(tf, p_obj, ext, arm, max_n=None):
    c = FAMILIES[arm["ee"]["grasp"]](tf, np.asarray(p_obj, float), np.asarray(ext, float), arm)
    return c[:max_n] if max_n else c
