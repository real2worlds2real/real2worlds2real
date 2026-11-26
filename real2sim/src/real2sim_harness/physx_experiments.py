"""PhysX Tensor API parameter binding and external-agent experiment execution.

Uses public NumPy-fronted rigid/articulation views. No Isaac Sim/Lab import,
launcher, backend switch or model session is started by importing this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .experiment_actions import ExperimentActions


PROPERTIES = {
    "static_friction": ("material_properties", 0),
    "dynamic_friction": ("material_properties", 1),
    "restitution": ("material_properties", 2),
    "joint_stiffness": ("dof_stiffnesses", None),
    "joint_damping": ("dof_dampings", None),
}


def _array(view, field):
    getter = getattr(view, "get_" + field, None)
    if not callable(getter):
        raise ValueError(f"Selected PhysX view does not support {field}")
    value = getter()
    if not isinstance(value, np.ndarray):
        raise ValueError("Use a NumPy-fronted PhysX tensor view or an explicit frontend adapter")
    return value.copy()


def bind_parameters(views, candidate, bindings):
    """Read-modify-write full tensors, preserving unselected bodies/shapes/DOFs."""
    if set(bindings) != set(candidate):
        raise ValueError("Each candidate parameter needs an explicit PhysX binding")
    touched, updates, rows = set(), {}, []
    for name, value in candidate.items():
        if not np.isfinite(value) or value < 0 or not bindings[name]:
            raise ValueError("Use nonnegative finite coefficients and nonempty bindings")
        for binding in bindings[name]:
            view_name, prop = binding["view"], binding["property"]
            if view_name not in views or prop not in PROPERTIES:
                raise ValueError("Unknown view or unsupported PhysX property")
            field, component = PROPERTIES[prop]
            key = view_name, field
            if key not in updates:
                updates[key] = _array(views[view_name], field)
            data = updates[key]
            expected_dim = 2 if component is None else 3
            if data.ndim != expected_dim or (component is not None and data.shape[-1] != 3):
                raise ValueError("PhysX tensor shape differs from the supported API contract")
            if not binding["indices"]:
                raise ValueError("Binding indices cannot be empty")
            for coord in binding["indices"]:
                if len(coord) != 2 or any(type(i) is not int or not 0 <= i < n for i, n in zip(coord, data.shape[:2], strict=True)):
                    raise ValueError("Select valid [body,shape] or [articulation,DOF] indices")
                index = tuple(coord) + (() if component is None else (component,))
                entry = view_name, field, index
                if entry in touched:
                    raise ValueError("Parameters cannot overwrite the same PhysX entry")
                touched.add(entry)
                data[index] = value
                rows.append({"parameter": name, "view": view_name, "property": prop,
                             "field": field, "index": list(index)})
    # Validate before any runtime mutation. In particular, do not allow a backend
    # to silently clamp inconsistent static/dynamic friction or restitution.
    for (view_name, field), data in updates.items():
        if not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("Invalid PhysX parameter tensor")
        if field == "material_properties" and (np.any(data[..., 1] > data[..., 0]) or np.any(data[..., 2] > 1)):
            raise ValueError("Require dynamic friction <= static friction and restitution <= 1")
        if not callable(getattr(views[view_name], "set_" + field, None)):
            raise ValueError("PhysX view has no matching parameter setter")
    for (view_name, field), data in updates.items():
        indices = np.arange(data.shape[0], dtype=np.int32)
        getattr(views[view_name], "set_" + field)(data, indices)
    actual = {key: _array(views[key[0]], key[1]) for key in updates}
    applied = {}
    for row in rows:
        value = float(actual[row["view"], row["field"]][tuple(row["index"])])
        if not np.isclose(value, candidate[row["parameter"]], rtol=1e-6, atol=1e-12):
            raise ValueError("PhysX runtime parameter readback differs from the candidate")
        row["value"] = value
        applied[row["parameter"]] = value
    return applied, rows


class PhysXExperimentRunner:
    """Run an agent-owned PhysX session with the common run_trial protocol.

    session_factory(experiment) is a context manager yielding a runtime with:
      views: name -> NumPy PhysX rigid/articulation view;
      metadata: backend='physx', version=<actual runtime version>;
      reset(experiment) -> True after restoring state/controllers/clock;
      step(dt_s): advances exactly one physics step, without forced rendering.
    observe(runtime, time_s, experiment) -> {track_id: coordinates}
    Optional apply_control(runtime, time_s, dt_s, experiment) supplies external
    actions. Runtime startup/assets/version choices stay outside this adapter.
    apply_event(runtime, event, experiment) returns execution evidence for optional
    scheduled interventions; no event implementation is imported from JSON.
    """

    def __init__(self, session_factory, observe, bindings, *, apply_control=None, apply_event=None):
        self.session_factory, self.observe = session_factory, observe
        self.bindings = json.loads(json.dumps(bindings))
        self.apply_control = apply_control
        self.apply_event = apply_event

    def __call__(self, request, directory):
        directory = Path(directory)
        experiment = request["experiment"]
        if experiment.get("target_backend", "physx") != "physx":
            raise ValueError("PhysX runner does not match the target asset backend")
        settings = experiment["solver"]
        frames, substeps = settings["frames"], settings["substeps"]
        sample_dt = float(settings["sample_dt_s"])
        if (type(frames) is not int or frames < 2 or type(substeps) is not int or substeps < 1
                or not np.isfinite(sample_dt) or sample_dt <= 0):
            raise ValueError("Choose frames>=2, substeps>=1 and positive sample_dt_s")
        dt = sample_dt / substeps
        actions = ExperimentActions(experiment, dt, (frames-1)*substeps, self.apply_event,
                                    directory / "actions.jsonl")
        templates = experiment["observations"]["tracks"]
        ids = [row["id"] for row in templates]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Declare at least one unique observation track")
        times, values, raw = [], {name: [] for name in ids}, {}
        with self.session_factory(experiment) as runtime:
            metadata = dict(runtime.metadata)
            if metadata.get("backend") != "physx" or not metadata.get("version"):
                raise ValueError("Runtime must report the actual PhysX backend and version")
            if runtime.reset(experiment) is not True:
                raise ValueError("Runtime must confirm reset of state, controls and clock")
            applied, entries = bind_parameters(runtime.views, request["candidate"], self.bindings)
            # Capture every supplied view. A view can expose rigid or articulation
            # state, allowing both asset kinds without a fixed task template.
            getters = {}
            for i, (name, view) in enumerate(runtime.views.items()):
                for field in ("transforms", "velocities", "root_transforms", "root_velocities",
                              "dof_positions", "dof_velocities", "link_transforms"):
                    if callable(getattr(view, "get_" + field, None)):
                        getters[f"view_{i}_{field}"] = (name, field)
            if not getters:
                raise ValueError("Runtime supplies no supported raw-state views")

            def snapshot(time_s):
                observed = self.observe(runtime, time_s, experiment)
                if set(observed) != set(ids):
                    raise ValueError("Observer must return exactly the declared track IDs")
                for key, value in observed.items():
                    data = np.asarray(value, float)
                    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
                        raise ValueError("Nonfinite/invalid observed PhysX state")
                    values[key].append(data.copy())
                for key, (name, field) in getters.items():
                    data = _array(runtime.views[name], field)
                    if not np.isfinite(data).all():
                        raise ValueError("Nonfinite PhysX state")
                    raw.setdefault(key, []).append(data)
                times.append(time_s)

            def intervene(tick):
                actions.at(tick, lambda callback, event: callback(runtime, event, experiment))

            try:
                intervene(0)
                snapshot(0.)
                for frame in range(1, frames):
                    for substep in range(substeps):
                        time_s = ((frame - 1) * substeps + substep) * dt
                        if self.apply_control:
                            evidence = self.apply_control(runtime, time_s, dt, experiment)
                            actions.control((frame-1)*substeps+substep, evidence)
                        runtime.step(dt)
                        intervene((frame-1)*substeps+substep+1)
                    snapshot(frame * sample_dt)
            finally:
                np.savez_compressed(directory / "states.npz", time_s=np.asarray(times),
                                    **{key: np.asarray(value) for key, value in raw.items()})
                atomic_json(directory / "state_views.json", getters)
                atomic_json(directory / "parameter_bindings.json", entries)
            # Controllers can overwrite gains/materials; a changed final binding
            # is a failed experiment, not a valid comparison of the candidate.
            for row in entries:
                value = float(_array(runtime.views[row["view"]], row["field"])[tuple(row["index"])])
                if value != row["value"]:
                    raise ValueError("PhysX coefficient changed during the experiment")
        tracks = [{"id": row["id"], "units": row["units"], "frame": row["frame"],
                   "source": "PhysX simulated state via experiment observer", "time_s": times,
                   "values": np.asarray(values[row["id"]]).tolist(),
                   "sample_state": ["observed"] * len(times)} for row in templates]
        for track, template in zip(tracks, templates, strict=True):
            if "observable" in template:
                track["observable"] = template["observable"]
        return {"rollout": {"schema_version": 1, "clock_source": "PhysX integration seconds", "tracks": tracks,
                            "events": actions.events},
                "receipt": {**metadata, "reset": True, "experiment_sha256": request["experiment_sha256"],
                            "applied_parameters": applied, "binding_evidence": "parameter_bindings.json",
                            "states_sha256": sha256_file(directory / "states.npz"),
                            "actions_sha256": sha256_file(directory / "actions.jsonl"), "frames": len(times), "dt_s": dt}}
