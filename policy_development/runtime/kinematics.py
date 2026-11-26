"""Station-compatible fk / ik without the station: the arm's product-of-exponentials model identified from the station's fk
(kin.py + kin.npz of the site) behind the station's call signatures, so programs can run inside policy modules (and on the
real robot's policy worker).  numpy only."""
import importlib.util
from pathlib import Path

import numpy as np


def mat2quat(R):
    R = np.asarray(R, np.float64)
    out = np.zeros(R.shape[:-2] + (4,))
    for idx in np.ndindex(R.shape[:-2]):
        m = R[idx]
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            i = int(np.argmax(np.diag(m)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
            q = [0.0] * 4
            q[0] = (m[k, j] - m[j, k]) / s
            q[1 + i] = 0.25 * s
            q[1 + j] = (m[j, i] + m[i, j]) / s
            q[1 + k] = (m[k, i] + m[i, k]) / s
        q = np.asarray(q)
        out[idx] = q * (1 if q[0] >= 0 else -1)
    return out


class Kin:
    def __init__(self, directory):
        d = Path(directory)
        spec = importlib.util.spec_from_file_location("astra_kin", d / "kin.py")
        self.k = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.k)
        z = np.load(d / "kin.npz")
        self.S, self.M = z["S"], z["M"]

    def fk(self, q7):
        T = self.k.fk_poe(self.S, self.M, np.atleast_2d(np.asarray(q7, np.float64)))
        return T[:, :3, 3].astype(np.float32), mat2quat(T[:, :3, :3]).astype(np.float32)

    def ik(self, q7_seed, pos, quat, iters=30):
        seed = np.atleast_2d(np.asarray(q7_seed, np.float64))
        R = self.k.quat2mat(np.atleast_2d(np.asarray(quat, np.float64)))
        out = self.k.ik_dls(self.S, self.M, seed, np.atleast_2d(np.asarray(pos, np.float64)), R, iters=iters)
        q = out[0] if isinstance(out, tuple) else out
        p, _ = self.fk(q)
        return q.astype(np.float32), np.linalg.norm(p - np.atleast_2d(pos), axis=1).astype(np.float32)
