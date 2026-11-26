"""A stamped point-cloud ACT as a self-contained station `Policy` (copied into every ACT candidate as policy.py).

The candidate directory carries everything it needs: act/policy_stamped.pt (+ act/SHA256), the lab's model code in pkg/
(point_act_cvae.py, il/), the CUDA point-ops extension and the libstdc++ it needs.  So the same directory runs in the
station sandbox, in the sealed evaluation and in the real-robot station worker without any external path.
Computation as the lab ACT deployers: target29 = q29 + first-step ACT delta (delta-to-measured; hand inputs zeroed when
the stamp says hand=zero).
"""
import ctypes
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "pkg"
if (PKG / "libstdc++.so.6").exists():           # the point-ops extension needs a newer libstdc++ than the system's
    ctypes.CDLL(str(PKG / "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault("CUP_PKG", str(PKG))
sys.path[:0] = [str(PKG), str(PKG / "il")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from point_act_cvae import load_point_act_cvae  # noqa: E402
from point_features import build_point_features_torch  # noqa: E402
from pointnext_encoder import point_ops_backend  # noqa: E402

import pointnext_encoder  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FPS backends verified to give identical indices on the station clouds (11-11, 25/25 frames); the ball query is made
# identical below, so either may run either's model
EQUIVALENT = {"pointnet2_batch_cuda", "torch"}
_ball_query = pointnext_encoder.ball_query_indices


def _ball_query_official(xyz, query_xyz, neighbors, radius, *, order_invariant=False):
    """CPU ball query with the official kernel's semantics (pointnet2_batch ball_query_gpu.cu): the first `neighbors`
    points inside the radius in index order, empty slots filled with the first one, all zero when none.  The encoder's
    own CPU fallback is a radius-limited KNN, i.e. a different neighbour set than the CUDA kernel the model trained with."""
    if order_invariant or xyz.is_cuda:
        return _ball_query(xyz, query_xyz, neighbors, radius, order_invariant=order_invariant)
    neighbors = min(int(neighbors), xyz.shape[1])
    q, p = query_xyz.float().unsqueeze(2), xyz.float().unsqueeze(1)
    d = q - p
    d2 = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
    r = torch.tensor(float(radius), dtype=torch.float32)
    inside = d2 < r * r
    n = xyz.shape[1]
    ar = torch.arange(n)
    first_k = torch.where(inside, ar, ar + n).topk(neighbors, dim=-1, largest=False, sorted=True).values
    cnt = inside.sum(-1, keepdim=True)
    idx = torch.where(torch.arange(neighbors) < cnt, first_k, first_k[..., :1])
    return torch.where(cnt > 0, idx, torch.zeros_like(idx)).long()


pointnext_encoder.ball_query_indices = _ball_query_official


class Policy:
    def __init__(self, info):
        ck = HERE / "act" / "policy_stamped.pt"
        if hashlib.sha256(ck.read_bytes()).hexdigest() != (ck.parent / "SHA256").read_text().strip():
            raise RuntimeError("ACT checkpoint sha256 differs from act/SHA256")
        self.model, self.cfg, self.on, self.an, self.meta, _ = load_point_act_cvae(str(ck), str(DEV), allow_absolute=True)
        self.model.eval().requires_grad_(False)
        have, want = point_ops_backend(DEV), self.meta.get("point_ops_backend")
        if have != want and not {have, want} <= EQUIVALENT:
            raise RuntimeError(f"point ops backend {have} != training {want}")
        if self.meta.get("action_target") != "delta_to_measured_state":
            raise RuntimeError("expected a delta-to-measured ACT (stamped)")
        self.hand_measured = (self.meta.get("gear_input") or {}).get("hand", "measured") == "measured"

    @torch.inference_mode()
    def _delta(self, X, q29):
        """X (E, 3, N) clouds, q29 (E, 29) -> (E, 29) first-step deltas; all envs in one forward pass."""
        q29 = np.atleast_2d(q29)
        q = np.concatenate([q29[:, :7], q29[:, 7:] if self.hand_measured else np.zeros((len(q29), 22))], 1)
        q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
        pts = build_point_features_torch(X, X.new_zeros((X.shape[0], 4, 0)), layout="metadata8", num_tactile_fingers=5,
                                         include_visual=True, include_tactile=False)
        return self.an.unnormalize(self.model.predict(pts, self.on.normalize(q).clamp(-1, 1)).clamp(-1, 1))[:, 0].double().cpu().numpy()

    def reset(self, env):
        pass

    def act(self, obs):
        X = torch.as_tensor(np.asarray(obs["cloud"], np.float32).transpose(0, 2, 1).copy(), device=DEV)      # (E, 3, N)
        q29 = np.asarray(obs["q29"], float)
        return q29 + self._delta(X, q29)
