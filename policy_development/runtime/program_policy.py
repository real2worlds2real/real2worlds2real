"""A program (code as policy) behind the station `Policy` interface, so it can be a stage of a composition, the base of a
residual, or run by a policy worker.  Each env runs the program in its own thread exactly as the station's evaluation
runner does (program_runtime.py, the one implementation of the program contract); fk / ik come from the identified arm
model in this directory (kin.py, kin.npz) instead of the station.

Directory layout: policy.py (this file), program.py, program_runtime.py, kin.py, kin.npz, kinematics.py."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Policy:
    def __init__(self, info, directory=None):
        d = Path(directory) if directory else HERE
        self.info = info
        self.code = compile((d / "program.py").read_text(), "<program>", "exec")
        self.kin = _load("astra_kinematics", d / "kinematics.py").Kin(d)
        self.ProgramEnv = _load("astra_program_runtime", d / "program_runtime.py").ProgramEnv
        self.runs = {}

    def reset(self, env):
        r = self.runs.pop(env, None)
        if r is not None:
            r.stop()

    def act(self, obs, envs=None):
        """envs: only advance these envs (a composition calls a stage for the envs it controls)."""
        n = len(obs["q29"])
        out = np.asarray(obs["q29"], np.float32).copy()
        for e in (range(n) if envs is None else envs):
            ob = {k: v[e] for k, v in obs.items()}
            r = self.runs.get(e)
            if r is None:
                r = self.runs[e] = self.ProgramEnv(self.code, self.info, self.kin, ob).start()
            elif not r.finished:
                r.feed(ob)
            out[e] = r.next_action(out[e])
        return out
