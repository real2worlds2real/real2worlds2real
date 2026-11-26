"""Twin station backend: a real2sim twin (a frozen reconstruction session of this repository) behind the station protocol,
so the policy-development harness and the baselines work on a reconstructed scene exactly as on a hand-authored task.

The task spec names the twin (`twin.session`, pinned in the binding by its freeze manifest qa/independent_freeze.json) and
an environment factory (`station.args`: --twin-session S --twin-factory <file.py>:<function>).  The factory builds the
scene from the frozen session (scene.twin.runtime_inputs.prepare_scene, sim.scene_stage, a tasks/* environment) and returns
one episode environment per station env, each following the calibrated_scene_reconstruction task-environment protocol:

    reset(case)          case: None (a fresh random training start) or a sealed start spec (int seed or dict)
    observe() -> dict    this env's station observation (numpy arrays without the env axis; keys of the task's interface)
    step(action)         one control step with this env's action
    measure() -> dict    the task state; "success" (bool) is the hidden judge, optional "failed" (str) ends an episode early
    info() -> dict       optional: text, labels, joint_names, action_dim, interface, hold, kinematics, observation_layout
    fk(q7) / ik(...)     optional (programs on stations whose contract includes them)

The backend keeps the station semantics: an episode ends on success held for `--twin-hold` steps, on a failure the
environment reports, or at max_steps; the client sees only success / termination reason / steps.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from server.backend import Refusal, TaskBackend


def verified_delivery(path: Path) -> dict:
    """The delivery record after re-hashing every file it pins: a station never serves a twin that changed after its
    acceptance.  (The same check as astra_station.twin_task.verify_delivery; an installed station root has no controller.)"""
    record = json.loads(path.read_text())
    try:
        from server.delivery_files import changed_files  # installed station root
    except ModuleNotFoundError:
        from astra_station.delivery_files import changed_files  # source checkout
    try:
        changed = changed_files(record)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if changed:
        raise SystemExit(f"the delivered twin changed after acceptance: {changed}")
    return record


def load_factory(spec: str):
    path, _, name = spec.partition(":")
    p = Path(path).expanduser().resolve()
    sys.path.insert(0, str(p.parent))
    s = importlib.util.spec_from_file_location(f"twin_factory_{p.stem}", p)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return getattr(mod, name)


class TwinStation(TaskBackend):
    tasks = ()
    runtime = "python"          # the factory starts the simulator itself (calibrated_scene_reconstruction.sim.application), like the bound factories

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--twin-session", default="", help="frozen real2sim session directory")
        parser.add_argument("--twin-delivery", default="", help="real2sim v1 or revision-bound v2 delivery record; its pinned "
                            "files are re-hashed before the station starts")
        parser.add_argument("--twin-factory", required=True, help="<file.py>:<function>(session, task, num_envs, run_dir) -> envs")
        parser.add_argument("--twin-task-spec", default="", help="the task spec (json) handed to the factory")
        parser.add_argument("--twin-max-steps", type=int, default=600)
        parser.add_argument("--twin-hold", type=int, default=1, help="consecutive successful steps that end an episode")
        parser.add_argument("--twin-partition-kind", default="", help="sealed partition file kind (default: the task name)")

    def __init__(self, args, num_envs):
        super().__init__(args, num_envs)
        self.task = args.task
        self.partition_kind = args.twin_partition_kind or args.task
        if args.twin_delivery:
            record = verified_delivery(Path(args.twin_delivery))
            session = Path(args.twin_session or record["source"].get("session") or record["source"]["run"])
        elif args.twin_session:
            session = Path(args.twin_session)
        else:
            raise SystemExit("a twin station needs --twin-session or --twin-delivery")
        if not args.twin_delivery and not (session / "qa" / "independent_freeze.json").is_file():
            raise SystemExit(f"{session} is not a frozen twin (no qa/independent_freeze.json)")
        spec = json.loads(Path(args.twin_task_spec).read_text()) if args.twin_task_spec else {"task_id": args.task}
        self.envs = list(load_factory(args.twin_factory)(session, spec, self.num_envs, Path(args.run)))
        if len(self.envs) != self.num_envs:
            raise SystemExit(f"the twin factory built {len(self.envs)} environments, the station needs {self.num_envs}")
        self.MAX, self.HOLD = int(args.twin_max_steps), max(1, int(args.twin_hold))
        meta = self.envs[0].info() if hasattr(self.envs[0], "info") else {}
        self.meta = {"interface": "state_vector", "hold": "zero", "kinematics": hasattr(self.envs[0], "ik"), **meta}
        self.steps = np.zeros(self.num_envs, np.int32)
        self.held = np.zeros(self.num_envs, np.int32)
        self.act = np.zeros(self.num_envs, bool)
        self.finished = {}
        self.last = [None] * self.num_envs

    @property
    def action_dim(self):
        return int(self.meta["action_dim"])

    @property
    def max_steps(self):
        return self.MAX

    def info(self, live):
        return {"task": self.task, "control_hz": 20, "max_steps": self.MAX, "labels": {}, **self.meta}

    def set_active(self, mask):
        self.act = mask.copy()

    def _start(self, r, case):
        self.envs[r].reset(case)
        self.steps[r] = 0
        self.held[r] = 0
        self.last[r] = self.envs[r].observe()

    def reset(self, rows, starts):
        for r, s in zip(rows, starts):
            self._start(r, s)

    def step(self, actions):
        done = np.zeros(self.num_envs, bool)
        for r, env in enumerate(self.envs):
            if not self.act[r]:
                continue                                  # inactive rows hold still: their action is not applied
            env.step(np.asarray(actions[r]))
            self.steps[r] += 1
            m = env.measure()
            self.held[r] = self.held[r] + 1 if m.get("success") else 0
            term = ("success" if self.held[r] >= self.HOLD else str(m["failed"]) if m.get("failed")
                    else "timeout" if self.steps[r] >= self.MAX else None)
            if term:
                self.finished[r] = {"success": term == "success", "term": term, "steps": int(self.steps[r])}
                done[r] = True
            self.last[r] = env.observe()
        return done

    def outcomes(self, done_rows):
        return {r: self.finished.pop(r) for r in done_rows if r in self.finished}

    def after_train_episodes(self, fresh_rows):
        for r in fresh_rows:                              # a new random training start
            self._start(r, None)

    def observe(self):
        keys = self.last[0].keys()
        out = {k: np.stack([np.asarray(o[k]) for o in self.last]) for k in keys}
        out.setdefault("q29", np.zeros((self.num_envs, 29), np.float32))
        out["step"] = self.steps.copy()
        out["active"] = self.act.copy()
        return out

    def fk(self, q7):
        if not self.meta.get("kinematics"):
            raise Refusal("fk / ik are not part of this twin station")
        return self.envs[0].fk(q7)

    def ik(self, q7_seed, pos, quat, iters):
        if not self.meta.get("kinematics"):
            raise Refusal("fk / ik are not part of this twin station")
        return self.envs[0].ik(q7_seed, pos, quat, iters)

    def first_reset(self):
        for r in range(self.num_envs):
            self._start(r, None)
