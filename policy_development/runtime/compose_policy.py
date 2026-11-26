"""Stage composition: stage_a acts until the switch condition holds for `hold` consecutive steps, then stage_b acts for the
rest of the episode (stage_b.reset at the handover; no switching back).  Conditions are observation predicates:
  label_rise      {label, rise_m}            a labelled object rose rise_m above its start
  palm_near_label {label, dist_m, offset}    the palm is within dist_m of the object's centroid + offset
  tactile         {min_n}                    a fingertip force exceeds min_n
  step            {n}                        n steps elapsed
Directory layout: policy.py (this file), switch.json, stage_a/ and stage_b/ (station policy modules)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _policy(name, d, info):
    spec = importlib.util.spec_from_file_location(name, d / "policy.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(d))
    spec.loader.exec_module(m)
    return m.Policy(info)


def _call(policy, obs, envs):
    import inspect
    if "envs" in inspect.signature(policy.act).parameters:
        return policy.act(obs, envs=envs)
    return policy.act(obs)


def _centroid(o, lab):
    m = np.asarray(o["labels"]) == lab
    return None if m.sum() < 15 else np.median(np.asarray(o["cloud"], np.float32)[m], axis=0)


class Policy:
    def __init__(self, info, directory=None):
        d = Path(directory) if directory else HERE
        self.sw = json.loads((d / "switch.json").read_text())
        self.ids = {v: int(k) for k, v in (info.get("labels") or {}).items()}
        self.a = _policy("astra_stage_a", d / "stage_a", info)
        self.b = _policy("astra_stage_b", d / "stage_b", info)
        self.env = {}

    def reset(self, env):
        self.a.reset(env)
        self.b.reset(env)
        self.env.pop(env, None)

    def _cond(self, o, st):
        k = self.sw["kind"]
        if k == "step":
            return st["step"] >= self.sw["n"]
        if k == "tactile":
            return float(np.max(o.get("tactile_n", [0]))) >= self.sw["min_n"]
        c = _centroid(o, self.ids[self.sw["label"]])
        if c is None:
            return False
        if k == "label_rise":
            s = st["start"]
            return s is not None and c[2] - s[2] >= self.sw["rise_m"]
        if k == "palm_near_label":
            return np.linalg.norm(np.asarray(o["palm_pos"]) - (c + np.asarray(self.sw.get("offset", [0, 0, 0])))) <= self.sw["dist_m"]
        raise ValueError(k)

    def act(self, obs, envs=None):
        n = len(obs["q29"])
        out = np.asarray(obs["q29"], np.float32).copy()
        stage_a = []
        mine = set(range(n) if envs is None else envs)
        for e in range(n):
            if e not in mine:
                stage_a.append(None)
                continue
            o = {k: v[e] for k, v in obs.items()}
            st = self.env.get(e)
            if st is None:
                lab = self.ids.get(self.sw.get("label"))
                st = self.env[e] = {"stage": "a", "step": 0, "held": 0, "start": _centroid(o, lab) if lab is not None else None}
            if st["stage"] == "a":
                st["held"] = st["held"] + 1 if self._cond(o, st) else 0
                if st["held"] >= int(self.sw.get("hold", 1)):
                    st["stage"] = "b"
                    self.b.reset(e)
            st["step"] += 1
            stage_a.append(st["stage"] == "a")
        ia = [e for e in range(n) if stage_a[e] is True]
        ib = [e for e in range(n) if stage_a[e] is False]
        if ia:
            out[ia] = np.asarray(_call(self.a, obs, ia), np.float32)[ia]
        if ib:
            out[ib] = np.asarray(_call(self.b, obs, ib), np.float32)[ib]
        return out
