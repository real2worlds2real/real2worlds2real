"""Backend-independent robot view for one batched planning slot."""
import numpy as np
from scipy.spatial.transform import Rotation


class SlotRobot:
    """The planner's robot for one slot: kinematics served by BatchKinematics; static attributes (idx, limits, spec,
    names, body_names, mimic) from the main scene's Robot."""

    def __init__(self, kin, slot):
        self._kin, self._slot = kin, slot
        self._q, self._tf = None, None

    def __getattr__(self, name):
        return getattr(self._kin.base, name)

    def call(self, fn, *a, **k):
        """fn(*a, **k) in the main thread (anything that touches the stage)."""
        return self._kin.request(self._slot, "call", (fn, a, k))

    def solve_ik(self, side, p_target, R_target, q0, cols=None, iters=250, tol=(0.004, 0.03), damping=0.05,
                 rot_weight=0.6, low_weight_scale=1.0):
        q, err, tf = self._kin.request(self._slot, "ik", dict(
            side=side, p=np.asarray(p_target, float), R=np.asarray(R_target, float), q0=np.asarray(q0, float),
            cols=cols, iters=iters, tol=tol, damping=damping, rot_weight=rot_weight,
            low_weight_scale=low_weight_scale))
        self._q, self._tf = q, tf  # the articulation's state after the solve, as Robot.solve_ik leaves it
        return q, err

    def teleport(self, q):
        self._q, self._tf = np.asarray(q, float), None

    def sync(self):
        self._tf = self._kin.request(self._slot, "fk", self._q)

    def link_transforms(self):
        if self._tf is None:
            self.sync()
        return self._tf

    def link_pose(self, name):
        t = self.link_transforms()[self._kin.base.body_names.index(name)]
        return t[:3].copy(), Rotation.from_quat(t[3:7]).as_matrix()

    def tcp_pose(self, side=None):
        side = side or next(iter(self._kin.base.spec["arms"]))
        arm = self._kin.base.spec["arms"][side]
        p, R = self.link_pose(arm["tcp_link"])
        return p + R @ np.asarray(arm.get("tcp_offset", (0, 0, 0)), float), R
