"""Optional timed interventions and durable execution evidence, without task templates."""
from __future__ import annotations

import copy
import json

import numpy as np


class ExperimentActions:
    """Execute native callbacks at integration boundaries, not wall-clock times.

    Event callbacks change persistent controls/constraints or apply state impulses.
    Per-step force application belongs in the runner's apply_control callback.
    Returned callback evidence must be a finite JSON object; it is a declaration,
    so callers should include effective runtime readback when available.
    """

    def __init__(self, experiment, dt_s, steps, callback, path):
        self.callback, self.path, self.dt = callback, path, dt_s
        self.schedule, self.events, self.completed = {}, {}, set()
        rows = experiment.get("interventions", [])
        if rows and callback is None:
            raise ValueError("Scheduled interventions require an explicit native apply_event callback")
        names = set()
        for event in rows:
            name, time = event.get("name"), event.get("time_s")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("Intervention names must be nonempty and unique")
            if not isinstance(event.get("source"), str) or not event["source"].strip():
                raise ValueError("Interventions need timing/control evidence in source")
            if isinstance(time, bool) or not isinstance(time, (int, float)) or not np.isfinite(time) or time < 0:
                raise ValueError("Intervention time must be finite and nonnegative")
            tick = round(time / dt_s)
            if tick > steps or not np.isclose(time, tick*dt_s, atol=1e-10, rtol=1e-10):
                raise ValueError("Intervention must lie on an integration boundary within the trial; do not silently round")
            retained = json.loads(json.dumps(event, allow_nan=False))
            self.schedule.setdefault(tick, []).append(retained)
            names.add(name)
        self.path.write_text("")

    def _write(self, row):
        encoded = json.dumps(row, allow_nan=False)
        with self.path.open("a") as stream:
            stream.write(encoded + "\n")
            stream.flush()

    def at(self, tick, invoke):
        """invoke(callback, event) supplies backend-specific native arguments."""
        if tick in self.completed:
            return
        self.completed.add(tick)
        for event in self.schedule.get(tick, []):
            row = {"kind": "intervention", "step": tick, "time_s": tick*self.dt, "event": event}
            self._write({**row, "status": "started"})
            try:
                evidence = invoke(self.callback, copy.deepcopy(event))
                if not isinstance(evidence, dict):
                    raise ValueError("apply_event must return a JSON object describing effective execution")
                self._write({**row, "status": "completed", "evidence": evidence})
            except Exception as exc:
                self._write({**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                raise
            self.events[event["name"]] = tick*self.dt

    def control(self, tick, evidence):
        """Optionally retain applied control evidence returned by a step callback."""
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValueError("apply_control evidence must be a JSON object or None")
            self._write({"kind": "control", "step": tick, "time_s": tick*self.dt,
                         "status": "applied", "evidence": evidence})
