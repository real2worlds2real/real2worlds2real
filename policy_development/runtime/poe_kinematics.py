"""Arm kinematics in numpy: product-of-exponentials FK with parameters (S, M) identified from a station's fk, DLS IK.

Generic code; the parameters (kin.npz) belong to a task's station frame (task spec station.kinematics_model).  A candidate
carries a copy of this file as kin.py next to its kin.npz."""
import numpy as np


def quat2mat(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def skew(w):
    w = np.asarray(w)
    O = np.zeros(w.shape[:-1] + (3, 3))
    O[..., 0, 1], O[..., 0, 2], O[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    O[..., 1, 0], O[..., 2, 0], O[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    return O


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-7:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-4:
        A = (R + np.eye(3)) / 2
        k = np.argmax(np.diag(A)); v = A[:, k] / np.sqrt(A[k, k])
        return v * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rot_err(Rt, R):
    """batched axis-angle of Rt R^T (N,3)."""
    E = Rt @ np.swapaxes(R, -1, -2)
    v = np.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], -1) / 2
    s = np.linalg.norm(v, axis=-1); c = (np.trace(E, axis1=1, axis2=2) - 1) / 2
    th = np.arctan2(s, c)
    f = np.where(s > 1e-9, th / np.maximum(s, 1e-12), 1.0)
    return v * f[:, None]


def exp6(S, th):
    """batched exp of screw S (6,) times angles th (N,) -> (N,4,4)."""
    w, v = S[:3], S[3:]
    W = skew(w)
    th = np.asarray(th)[:, None, None]
    R = np.eye(3) + np.sin(th) * W + (1 - np.cos(th)) * (W @ W)
    G = np.eye(3) * th + (1 - np.cos(th)) * W + (th - np.sin(th)) * (W @ W)
    T = np.tile(np.eye(4), (len(th), 1, 1))
    T[:, :3, :3] = R; T[:, :3, 3] = G @ v
    return T


def fk_poe(S, M, Q, all_frames=False):
    Q = np.atleast_2d(np.asarray(Q, np.float64))
    T = np.tile(np.eye(4), (len(Q), 1, 1))
    frames = []
    for i in range(len(S)):
        T = T @ exp6(S[i], Q[:, i])
        frames.append(T.copy())
    Tf = T @ M
    return (Tf, frames) if all_frames else Tf


def jacobian(S, M, Q):
    """spatial Jacobian (N,6,7) with rows (omega, v_s); converted to point velocity of the palm below."""
    Q = np.atleast_2d(Q)
    N = len(Q)
    J = np.zeros((N, 6, len(S)))
    T = np.tile(np.eye(4), (N, 1, 1))
    for i in range(len(S)):
        R, p = T[:, :3, :3], T[:, :3, 3]
        w = R @ S[i, :3]; v = R @ S[i, 3:] + np.cross(p, w)
        J[:, :3, i] = w; J[:, 3:, i] = v
        T = T @ exp6(S[i], Q[:, i])
    return J, T @ M


def ik_dls(S, M, seed, pos, R_target, iters=30, lam=0.02, rot_w=0.3, tol=1e-5):
    """damped least squares IK, batched. pos (N,3), R_target (N,3,3). Returns q (N,7), position error (N,)."""
    q = np.array(seed, np.float64, copy=True)
    for _ in range(iters):
        J, T = jacobian(S, M, q)
        p = T[:, :3, 3]
        ep = pos - p
        er = rot_err(R_target, T[:, :3, :3])
        # palm point linear velocity = v_s + w x p
        Jp = J[:, 3:, :] + np.cross(J[:, :3, :].transpose(0, 2, 1), p[:, None, :]).transpose(0, 2, 1)
        A = np.concatenate([Jp, rot_w * J[:, :3, :]], 1)
        e = np.concatenate([ep, rot_w * er], 1)
        At = A.transpose(0, 2, 1)
        H = A @ At + (lam ** 2) * np.eye(6)
        dq = (At @ np.linalg.solve(H, e[:, :, None]))[:, :, 0]
        n = np.abs(dq).max(1, keepdims=True)
        dq = dq * np.minimum(1.0, 0.3 / np.maximum(n, 1e-12))
        q = q + dq
        if np.abs(e).max() < tol:
            break
    T = fk_poe(S, M, q)
    return q, np.linalg.norm(pos - T[:, :3, 3], axis=1)
