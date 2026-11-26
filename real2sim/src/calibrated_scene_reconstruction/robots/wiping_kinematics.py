"""Read-only CAD FK/IK for planning while a live robot maintains a friction grasp.

No simulator handle, teleport or physics step is used by this planner.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.robots.articulated_model import from_usd


class WipingKinematics:
    def __init__(self, spec, names, T_world_root):
        self.model = from_usd(spec['usd'], points_per_link=8)
        self.spec, self.names = spec, list(names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.base = np.asarray(T_world_root, float)
        self.by_name = {j.name: j for j in self.model.movable}

    def poses(self, q):
        local = self.model.transforms({n: float(q[i]) for i, n in enumerate(self.names) if n in self.by_name})
        return {n: self.base @ tf for n, tf in local.items()}

    def tcp(self, q, side='right'):
        arm = self.spec['arms'][side]
        tf = self.poses(q)[arm['tcp_link']]
        return tf[:3, 3] + tf[:3, :3] @ arm['tcp_offset'], tf[:3, :3]

    def solve(self, p_target, R_target, q0, side='right', iters=220, tol=(.001, .015), damping=.035):
        arm = self.spec['arms'][side]
        columns = arm['ik_joints']
        ci = [self.idx[n] for n in columns]
        lower = np.array([self.by_name[n].lower for n in columns])
        upper = np.array([self.by_name[n].upper for n in columns])
        q = np.asarray(q0, float).copy()
        q[ci] = np.clip(q[ci], lower, upper)
        offset = np.asarray(arm['tcp_offset'])
        for _ in range(iters):
            tf = self.poses(q)
            hand = tf[arm['tcp_link']]
            p = hand[:3, 3] + hand[:3, :3] @ offset
            ep = np.asarray(p_target) - p
            er = Rotation.from_matrix(R_target @ hand[:3, :3].T).as_rotvec()
            error = (float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
            if error[0] < tol[0] and error[1] < tol[1]:
                break
            J = np.zeros((6, len(columns)))
            for i, name in enumerate(columns):
                joint = self.by_name[name]
                frame = tf[joint.parent] @ joint.frame0
                axis = frame[:3, :3] @ joint.axis
                if joint.kind == 'revolute':
                    J[:3, i] = np.cross(axis, p - frame[:3, 3])
                    J[3:, i] = axis
                else:
                    J[:3, i] = axis
            J[3:] *= .6
            delta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), np.r_[ep, .6 * er])
            q[ci] = np.clip(q[ci] + np.clip(delta, -.15, .15), lower, upper)
        return q, error
