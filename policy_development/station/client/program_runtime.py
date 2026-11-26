"""Programs (code as policy) on a station: the one implementation of the program contract.

A program is a script run once per episode per env, in its own thread, with these names available:
    get_task_context() -> dict (the station's info), get_observation() -> dict of this env's arrays,
    step(action) -> next observation dict (blocks until the controller advances the batch),
    fk(q7) -> (pos, quat), ik(q7_seed, pos, quat) -> (q7, err)   only where the station provides them,
    load_skill(name) -> a library skill (a namespace with `run(**params)`), where the workspace has a skill library,
    np; a program may set a top-level `result`.

A library skill is a directory <skills>/<name>/ with `skill.py`: a script in the program's dialect (the same names, plus
SKILL_DIR for its own files) that defines `run(**params)`; a program file whose last lines are
`if __name__ == "program": result = run()` is itself a valid skill.py.  The skill library is $STATION_SKILLS or ./skills.

The same class runs a program in the sealed evaluation (eval_runner.py), in training rollouts (rollout_runner.py), behind
the policy interface (program_policy.py) and on the real robot's station worker (`realtime=True`).  numpy only.
"""
import os
import queue
import threading
import types
from pathlib import Path

import numpy as np


def has_kinematics(info: dict) -> bool:
    """fk / ik are part of a station's program contract when the station says so; stations that predate the `kinematics`
    info key provide them exactly when they take 29-D joint targets (cup, peg; nut's 14-D action has none)."""
    if "kinematics" in info:
        return bool(info["kinematics"])
    return int(info.get("action_dim", 29)) == 29


def hold_action(info: dict, obs: dict) -> np.ndarray:
    """The 'stay still' action for every env of a batch: absolute joint targets hold the measured joints; a delta action
    (nut: finger command changes + palm descent) holds with zeros.  `info["hold"]` names the rule; stations that predate
    it imply it by the action size."""
    rule = info.get("hold") or ("measured_joints" if int(info.get("action_dim", 29)) == 29 else "zero")
    n = len(obs["q29"]) if "q29" in obs else len(next(iter(obs.values())))
    if rule == "measured_joints":
        return np.asarray(obs["q29"], np.float32).copy()
    return np.zeros((n, int(info["action_dim"])), np.float32)


class ProgramEnv:
    """One episode of a program in one env.  The program thread blocks in step() until the controller feeds the next
    observation.

    batch_index: the env's row when observations are whole batches (sealed evaluation, rollouts); None when they are
    already this env's arrays.  kin: an object with fk / ik (the station client, the identified arm model or the robot's
    kinematics), used only when the station's contract includes them.  realtime: the robot's variant - feed() drops
    observations the program has not consumed and next_action() takes the newest pending action, so a program that fell
    behind resumes on the newest frame instead of running ticks behind the robot."""

    def __init__(self, src, info, kin, obs0, batch_index=None, *, realtime=False, error_chars=500):
        self.info, self.kin, self.batch_index, self.realtime, self.error_chars = info, kin, batch_index, realtime, error_chars
        self.adim = int(info.get("action_dim", 29))
        self.to_prog, self.from_prog = queue.Queue(), queue.Queue()
        self.obs = obs0
        self.done = False
        self.error = None
        self.result = None
        self.finished = False
        self.code = compile(src, "<program>", "exec") if isinstance(src, str) else src
        self.thread = threading.Thread(target=self._main, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _view(self, obs):
        return obs if self.batch_index is None else {k: v[self.batch_index] for k, v in obs.items()}

    def _main(self):
        def step(action):
            if self.done:
                raise RuntimeError("episode over")
            self.from_prog.put(("action", np.asarray(action, np.float32).reshape(self.adim)))
            msg = self.to_prog.get()
            if msg is None:
                raise SystemExit
            self.obs = msg
            return self._view(msg)

        g = {"__name__": "program", "get_task_context": lambda: dict(self.info), "get_observation": lambda: self._view(self.obs),
             "step": step, "np": np}
        if self.kin is not None and has_kinematics(self.info):
            g.update(fk=self.kin.fk, ik=self.kin.ik)
        skills_root = Path(os.environ.get("STATION_SKILLS", "skills"))

        def load_skill(name):
            d = (skills_root / str(name)).resolve()
            if skills_root.resolve() not in d.parents or not (d / "skill.py").is_file():
                raise RuntimeError(f"no library skill {name!r}")
            ns = {k: v for k, v in g.items() if k != "result"}
            ns.update(__name__="skill", SKILL_DIR=str(d))
            exec(compile((d / "skill.py").read_text(), f"<skill {name}>", "exec"), ns)
            if not callable(ns.get("run")):
                raise RuntimeError(f"library skill {name!r} defines no run(**params)")
            return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})

        g["load_skill"] = load_skill
        try:
            exec(self.code, g)
            self.result = g.get("result")
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"[:self.error_chars]
        self.from_prog.put(("done", None))

    def next_action(self, hold, timeout=60.0):
        """The program's next action, or `hold` once it has finished.  Simulation: waiting longer than `timeout` ends the
        program (an error).  Real robot (realtime): a late program holds this tick and keeps running."""
        if self.finished:
            return hold
        if not self.realtime and not self.thread.is_alive() and self.from_prog.empty():
            self.finished = True
            return hold
        try:
            kind, a = self.from_prog.get(timeout=timeout)
        except queue.Empty:
            if self.realtime:
                self.late = True
                return hold
            self.error = f"program did not call step() within {timeout:g} s"
            self.finished = True
            return hold
        self.late = False
        if self.realtime:
            while kind == "action":       # after a late tick, skip the backlog: the newest action belongs to the newest frame
                try:
                    kind2, a2 = self.from_prog.get_nowait()
                except queue.Empty:
                    break
                kind, a = kind2, (a2 if kind2 == "action" else a)
        if kind == "done":
            self.finished = True
            return hold
        return a

    def feed(self, obs):
        if self.realtime:
            while True:                    # a program busy computing resumes on the NEWEST observation
                try:
                    self.to_prog.get_nowait()
                except queue.Empty:
                    break
        self.to_prog.put(obs)

    def stop(self):
        self.done = True
        self.to_prog.put(None)
