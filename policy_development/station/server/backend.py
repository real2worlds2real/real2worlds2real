"""The task backend contract of a station server.

A station serves one task through a fixed interface; everything task-specific sits behind this contract: the simulation
(an Isaac task, a real2sim twin, a numpy fake), its start distribution, its hidden judge, its observation and action
hooks and, optionally, the arm kinematics.  The generic server (core.py) owns everything else: the socket protocol,
sealed partitions and the coordinator token, the usage policy and training-episode budget, the episode ledger, which
envs are active.

Backends are named in a task spec's `station` block (`"backend": "<file.py>:<Class>"`, resolved against the task
repository) and constructed as Backend(args, num_envs); `args` is the server's argparse namespace, extended by the
backend's own `add_args`.
"""
from __future__ import annotations

import numpy as np


class Refusal(Exception):
    """A command the station does not offer (e.g. fk / ik on a station without an arm model): the client gets the message."""


class TaskBackend:
    """Base class; subclasses implement the methods that raise NotImplementedError."""

    #: argparse choices / identity of the task(s) this backend serves
    tasks: tuple = ()
    #: "isaac": the server launches Isaac Sim's app before importing the backend module; "python": nothing to launch
    runtime = "python"
    #: which sealed partition file the server reads (<sim dir>/sealed/<partition_kind>_partitions.json); tasks that share
    #: a start distribution share it (cup, cup_lift, cup_place, cup_reach -> "cup")
    partition_kind = ""

    def __init__(self, args, num_envs: int):
        self.args, self.num_envs = args, int(num_envs)
        self.read_sealed = None          # set by the server: () -> the sealed partition file (e.g. a backend's train rows)

    @classmethod
    def add_args(cls, parser) -> None:
        """Backend-specific command-line options (coordinator-only validation paths and the like)."""

    # ------------------------------------------------------------------ identity and interface
    def info(self, live: bool) -> dict:
        """Task facts for the client: text, control_hz, max_steps, labels, joint_names, action_dim, interface, hold,
        kinematics and any interface extras (observation_layout, limits).  live: a reset happened (observation sizes can
        be measured)."""
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def max_steps(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ episodes
    def set_active(self, mask: np.ndarray) -> None:
        """Rows whose current episode counts; inactive rows must hold still (their actions are ignored)."""

    def reset(self, rows: list, starts: list) -> None:
        """Reset these rows; each start is None (a fresh random training start) or a sealed start spec (partition file)."""
        raise NotImplementedError

    def validate_actions(self, actions: np.ndarray) -> str | None:
        """None if the batch is acceptable, else the error for the client."""
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            return f"actions must be finite, shape ({self.num_envs}, {self.action_dim})"
        return None

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Apply one control step; returns the (num_envs,) bool mask of rows whose episode ended on it."""
        raise NotImplementedError

    def outcomes(self, done_rows: list) -> dict:
        """{row: {"success": bool, "term": str, "steps": int, ...}} for the rows that just ended (the hidden judge)."""
        raise NotImplementedError

    def after_train_episodes(self, fresh_rows: list) -> None:
        """Training mode: these rows ended and continue with a new random start (backends whose env does not reset them
        itself do it here)."""

    def observe(self) -> dict:
        """The observation of every row as numpy arrays with a leading env axis (always `step` and `active` included)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ optional
    def fk(self, q7: np.ndarray) -> tuple:
        raise Refusal("fk / ik are not part of this station")

    def ik(self, q7_seed: np.ndarray, pos: np.ndarray, quat: np.ndarray, iters: int) -> tuple:
        raise Refusal("fk / ik are not part of this station")

    def validation_targets(self):
        """--check_act: the actions of the backend's own reference controller for the current batch (validates the
        control path end to end); None if the backend has none."""
        return None

    def first_reset(self) -> None:
        """Called once before serving (the env's initial reset)."""
