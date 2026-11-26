"""Residual policy = frozen base policy + a small learned, bounded, incremental correction (lab ResFiT recipe).

The correction is an offset in palm position (xy or xyz, applied through the identified arm model's IK around the base's
arm target, keeping its orientation) or in the 7 arm joints.  It is incremental: each step the actor moves the offset by at
most `step_frac * scale`, the offset stays within +-scale, and the current offset is part of the actor's state (an
absolute offset that the state cannot see was unlearnable in the lab).  Same code in training (residual_learner.py) and
deployment (as a candidate's policy.py).

Directory layout: policy.py (this file), config.json, actor.pt, base/ (a station policy module), kin/ (kin.py, kin.npz,
kinematics.py)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(m)
    return m


def call_act(policy, obs, envs):
    """policy.act restricted to `envs` when the policy supports it (stateful stages); stateless policies act on all."""
    import inspect
    if envs is not None and "envs" in inspect.signature(policy.act).parameters:
        return policy.act(obs, envs=envs)
    return policy.act(obs)


def mlp(i, o, h=256, out_act=None):
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)]
    if out_act:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Core:
    """Features and the application of the offset; identical in training and deployment."""

    def __init__(self, cfg: dict, kin_dir: Path):
        self.cfg = cfg
        self.kin = load_module("astra_res_kin", kin_dir / "kinematics.py").Kin(kin_dir)
        self.labels = {int(k): v for k, v in cfg["labels"].items() if v not in ("table", "hand")}
        self.dim = {"palm_xy": 2, "palm_xyz": 3, "arm_joints": 7}[cfg["action_space"]]
        self.obs_dim = 7 + 3 + 4 * len(self.labels) + 3 + self.dim + 1

    def start(self, obs_env):
        st = {}
        for lab, _ in self.labels.items():
            c = self._centroid(obs_env, lab)
            st[lab] = c
        return st

    @staticmethod
    def _centroid(o, lab):
        if "labels" not in o:
            return None
        m = np.asarray(o["labels"]) == lab
        if m.sum() < 15:
            return None
        return np.median(np.asarray(o["cloud"], np.float32)[m], axis=0)

    def features(self, o, offset, start, step):
        palm = np.asarray(o["palm_pos"], np.float32)
        f = [np.asarray(o["q29"], np.float32)[:7], palm]
        for lab in self.labels:
            c = self._centroid(o, lab)
            s = start.get(lab)
            if c is None:
                f.append(np.zeros(4, np.float32))
            else:
                f.append(np.r_[c - palm, (c[2] - s[2]) if s is not None else 0.0].astype(np.float32))
        f.append(np.asarray(o.get("tactile_n", np.zeros(3)), np.float32)[:3] / 20.0)
        f.append(np.asarray(offset, np.float32) / self.cfg["scale"])
        f.append(np.array([step / max(1, self.cfg["max_steps"])], np.float32))
        return np.concatenate(f)

    def step_offset(self, offset, a):
        s = self.cfg["scale"]
        return np.clip(offset + np.asarray(a) * self.cfg["step_frac"] * s, -s, s)

    def apply(self, base_q29, offset):
        q = np.asarray(base_q29, np.float32).copy()
        if self.cfg["action_space"] == "arm_joints":
            q[:7] += offset
            return q
        pos, quat = self.kin.fk(q[None, :7])
        tgt = pos[0].copy()
        tgt[: self.dim] += offset
        q7, _ = self.kin.ik(q[None, :7], tgt[None], quat)
        q[:7] = q7[0]
        return q


class Policy:
    def __init__(self, info, directory=None):
        d = Path(directory) if directory else HERE
        self.cfg = json.loads((d / "config.json").read_text())
        self.core = Core(self.cfg, d / "kin")
        self.base = load_module("astra_res_base", d / "base" / "policy.py").Policy(info)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = mlp(self.core.obs_dim, self.core.dim, out_act=nn.Tanh()).to(self.dev)
        self.actor.load_state_dict(torch.load(d / "actor.pt", map_location=self.dev))
        self.actor.eval()
        self.env = {}

    def reset(self, env):
        self.base.reset(env)
        self.env.pop(env, None)

    @torch.inference_mode()
    def act(self, obs, envs=None):
        base = np.asarray(call_act(self.base, obs, envs), np.float32)
        out = base.copy()
        for e in (range(len(base)) if envs is None else envs):
            o = {k: v[e] for k, v in obs.items()}
            st = self.env.get(e)
            if st is None:
                st = self.env[e] = {"offset": np.zeros(self.core.dim, np.float32), "start": self.core.start(o), "step": 0}
            s = torch.as_tensor(self.core.features(o, st["offset"], st["start"], st["step"]), device=self.dev)[None]
            a = self.actor(s)[0].cpu().numpy()
            st["offset"] = self.core.step_offset(st["offset"], a)
            st["step"] += 1
            out[e] = self.core.apply(base[e], st["offset"])
        return out
