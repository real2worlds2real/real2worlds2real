"""Native Newton experiment adapter for agent-authored models and controls.

This module imports Newton only when executing a trial. Factories and callbacks
are ordinary Python under the external client's grants, never loaded from JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .experiment_actions import ExperimentActions


def bind_coefficients(model, candidate, bindings):
    """Assign selected scalar coefficient entries and read back actual values.

    Coordinates are explicit (e.g. [[shape_id]] or [[triangle_id, component]]).
    Mass/inertia require coupled updates and are deliberately not scalar bindings.
    Solver-specific meaning and support remain the experiment author's decision.
    """
    if set(candidate) != set(bindings):
        raise ValueError("Each candidate parameter needs explicit coefficient bindings")
    allowed = {"shape_material_mu", "shape_material_mu_torsional", "shape_material_mu_rolling",
               "shape_material_ke", "shape_material_kd", "shape_material_kf",
               "shape_material_restitution", "tri_materials", "edge_bending_properties",
               "tet_materials", "spring_stiffness", "spring_damping",
               "joint_target_ke", "joint_target_kd"}
    touched, updates, entries = set(), {}, []
    for name, value in candidate.items():
        if not np.isfinite(value) or value < 0 or not bindings[name]:
            raise ValueError("Coefficient candidates must be nonnegative with nonempty bindings")
        for binding in bindings[name]:
            field = binding["field"]
            if field not in allowed:
                raise ValueError(f"Unsupported coefficient field {field}; use a coupled custom adapter")
            array = getattr(model, field, None)
            if array is None:
                raise ValueError(f"Newton model has no coefficient array {field}")
            if field not in updates:
                updates[field] = array.numpy().copy()
            data = updates[field]
            if not binding["indices"]:
                raise ValueError("Binding indices cannot be empty")
            for coord in binding["indices"]:
                if len(coord) != data.ndim or any(type(x) is not int or not 0 <= x < n for x, n in zip(coord, data.shape, strict=True)):
                    raise ValueError("Coefficient index outside array dimensions")
                key = field, tuple(coord)
                if key in touched:
                    raise ValueError("Parameters cannot overwrite the same coefficient entry")
                touched.add(key)
                data[tuple(coord)] = value
                entries.append({"parameter": name, "field": field, "index": coord})
    for field, data in updates.items():
        getattr(model, field).assign(data)
    actual = {field: getattr(model, field).numpy() for field in updates}
    applied = {}
    for row in entries:
        value = float(actual[row["field"]][tuple(row["index"])])
        name = row["parameter"]
        if not np.isclose(value, candidate[name], rtol=1e-6, atol=1e-12):
            raise ValueError("Backend coefficient readback differs from requested value")
        row["value"] = value
        applied[name] = value
    return applied, entries


class NewtonExperimentRunner:
    """Fresh model/solver per candidate, controlled stepping, retained raw states.

    build_model(experiment, device) -> finalized Newton Model
    make_solver(model, solver_settings) -> Newton solver
    observe(model, state, time_s, experiment) -> {track_id: coordinate vector}
    initialize(model, state, control, experiment) -> optional initial conditions
    apply_control(model, state, control, time_s, dt_s, experiment) -> optional drive
    apply_event(model, state, control, event, experiment) -> execution evidence dict

    A single CollisionPipeline is used. Custom multi-solver coupling or fluid grid
    state requires a separate runner using the same run_trial receipt protocol.
    """

    def __init__(self, build_model, make_solver, observe, bindings, *, device="cpu",
                 initialize=None, apply_control=None, collision_factory=None, apply_event=None):
        self.build_model, self.make_solver, self.observe = build_model, make_solver, observe
        self.bindings = json.loads(json.dumps(bindings))
        self.device, self.initialize, self.apply_control = device, initialize, apply_control
        self.collision_factory = collision_factory
        self.apply_event = apply_event
        self._previous_model = None

    def __call__(self, request, directory):
        if request["experiment"].get("target_backend", "newton") != "newton":
            raise ValueError("Newton runner does not match the target asset backend")
        import newton
        import warp as wp

        directory = Path(directory)
        experiment = request["experiment"]
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
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError("Observe at least one unique track")
        times, observed, raw, contact_counts = [], {key: [] for key in ids}, {}, []
        with wp.ScopedDevice(self.device):
            model = self.build_model(experiment, self.device)
            if model is self._previous_model:
                raise ValueError("Build a fresh Newton model for every trial")
            self._previous_model = model
            applied, entries = bind_coefficients(model, request["candidate"], self.bindings)
            solver = self.make_solver(model, settings)
            pipeline = (self.collision_factory(model) if self.collision_factory else newton.CollisionPipeline(model))
            contacts = pipeline.contacts()
            current, following, control = model.state(), model.state(), model.control()
            if model.joint_count:
                newton.eval_fk(model, model.joint_q, model.joint_qd, current)
            if self.initialize:
                self.initialize(model, current, control, experiment)

            def snapshot(time_s):
                values = self.observe(model, current, time_s, experiment)
                if set(values) != set(ids):
                    raise ValueError("Observer must return exactly the declared track IDs")
                for key, value in values.items():
                    array = np.asarray(value, float)
                    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                        raise ValueError("Nonfinite/invalid observed simulator state")
                    observed[key].append(array.copy())
                for field in ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "joint_qd"):
                    array = getattr(current, field, None)
                    if array is not None:
                        data = array.numpy()
                        if not np.isfinite(data).all():
                            raise ValueError(f"Nonfinite Newton {field}")
                        raw.setdefault(field, []).append(data.copy())
                counts = {}
                for field in ("rigid_contact_count", "soft_contact_count"):
                    array = getattr(contacts, field, None)
                    if array is not None:
                        counts[field] = int(array.numpy()[0])
                contact_counts.append(counts)
                times.append(time_s)

            def intervene(tick):
                actions.at(tick, lambda callback, event: callback(model, current, control, event, experiment))

            try:
                intervene(0)
                snapshot(0.)
                for frame in range(1, frames):
                    for substep in range(substeps):
                        time_s = ((frame - 1) * substeps + substep) * dt
                        current.clear_forces()
                        if self.apply_control:
                            evidence = self.apply_control(model, current, control, time_s, dt, experiment)
                            actions.control((frame-1)*substeps+substep, evidence)
                        pipeline.collide(current, contacts)
                        solver.step(current, following, control, contacts, dt)
                        current, following = following, current
                        intervene((frame-1)*substeps+substep+1)
                    snapshot(frame * sample_dt)
            finally:
                # Also retain partial numeric evidence when a run fails. This is
                # not a warm-start checkpoint and does not establish acceptance.
                np.savez_compressed(directory / "states.npz", time_s=np.asarray(times),
                                    **{key: np.asarray(value) for key, value in raw.items()})
                atomic_json(directory / "contacts.json", contact_counts)
                atomic_json(directory / "parameter_bindings.json", entries)
            # Detect coefficient changes made by controllers/solvers after setup.
            for row in entries:
                value = float(getattr(model, row["field"]).numpy()[tuple(row["index"])])
                if value != row["value"]:
                    raise ValueError("Coefficient changed during trial; start a new experiment")
        tracks = [{"id": row["id"], "units": row["units"], "frame": row["frame"],
                   "source": "Newton simulated state via experiment observer",
                   "time_s": times, "values": np.asarray(observed[row["id"]]).tolist(),
                   "sample_state": ["observed"] * len(times)} for row in templates]
        for track, template in zip(tracks, templates, strict=True):
            if "observable" in template:
                track["observable"] = template["observable"]
        return {"rollout": {"schema_version": 1, "clock_source": "Newton integration seconds",
                            "tracks": tracks, "events": actions.events},
                "receipt": {"reset": True, "experiment_sha256": request["experiment_sha256"],
                            "applied_parameters": applied, "backend": "newton",
                            "version": str(newton.__version__), "warp_version": str(wp.__version__),
                            "solver": type(solver).__name__, "device": str(model.device),
                            "binding_evidence": "parameter_bindings.json",
                            "states_sha256": sha256_file(directory / "states.npz"),
                            "actions_sha256": sha256_file(directory / "actions.jsonl"),
                            "frames": len(times), "dt_s": dt}}
