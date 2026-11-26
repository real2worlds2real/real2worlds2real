# Physics inference from observed motion

Physical alignment has its own task-level
[calibration workflow](../skills/calibrate-observed-physics/SKILL.md), used with
the [common working methods](../skills/work-with-task-evidence/SKILL.md). It may
start from an existing asset through the native experiment APIs below, without
creating a Blender generation task. It remains optional for ordinary generation.

An external agent may calibrate a standalone asset or selected scene participants.
The agent proposes parameters and interprets results. The library supplies evidence
comparison and repeatable trial records; no agent, optimizer or simulator starts
implicitly. Force/torque sensing is not required by these interfaces.

Choose the backend from the asset's intended deployment and set
`experiment.target_backend` to `physx` or `newton`. Compute availability may change
which checks can run now, but does not justify silently calibrating a different
backend. Backend-specific coefficients and contact laws are not interchangeable.

## Compare retained observations and simulation

Existing tracker outputs, selected masks and measured video points can be prepared
with `prepare_motion_observations`; see [motion evidence](MOTION_OBSERVATIONS.md)
for source-matched frame/clock/pixel conventions. Perception remains agent-selected.

Call `real2sim-harness task call <run> compare_physics_observations --request <request.json>`
with the normal task-call envelope:

```json
{"request_path":"physics/compare.json"}
```

The workspace-relative `physics/compare.json` contains:

```json
{"schema_version":1,"observations":"observed.json","rollout":"simulated.json",
 "max_gap_s":0.05,"time_offset_s":0.0}
```

Each file has this shape (values below are synthetic schema examples):

```json
{"schema_version":1,"clock_source":"original video seconds; release at t=0",
 "tracks":[{"id":"target.center","units":"px","frame":"registered-camera-A",
 "source":"measured mask centroids; camera A",
 "time_s":[0.0,0.1,0.2],"values":[[10,20],[12,20],[13,20]],
 "sample_state":["observed","observed","observed"],"residual_scale":2.0}],
 "events":{"release":0.0,"stop":null}}
```

For rollout tracks, `observed` denotes an actual simulated sample, not a real-world
measurement; identify the simulator in `source`. Matching IDs, units, dimensions
and registered frame names are required. Frame names are assertions: the caller
must actually use the registered camera/frame. Coordinates can be image points,
metric positions or other Euclidean measurements. Do not compare quaternion or
wrapped-angle coordinates with Euclidean distance; first derive an appropriate
observable. The first version does not compute silhouette/video losses itself.

`sample_state` can be `observed`, `missing`, `interpolated`, `predicted`. Only
observed reference samples contribute. Simulator samples can be linearly
interpolated between adjacent valid samples within `max_gap_s`; gaps and endpoints
are never extrapolated. `time_offset_s` is added to simulator timestamps; justify
it independently. No time scaling or automatic best-alignment search is performed.

The result includes per-track distance curves, coverage and worst timestamps.
The search score is mean dimensionless Huber distance, normalized by each observed
track's explicit positive `residual_scale`. Events are reported in seconds
separately and do not currently enter that score. Missing expected events or
incomplete simulated coverage yields `complete:false, score:null`. `ok:true` means
comparison ran, not that the physics matches. No numeric pass threshold or physical
identifiability is inferred. Both source files are retained by the task tool.

## Agent-proposed experiments

Native Python API:

```python
from real2sim_harness.physics_inference import run_trial

report = run_trial(experiment, parameters, candidate, runner, trial_directory)
```

`parameters` is an agent-selected list, for example:

```json
[{"name":"contact.dynamic_friction","units":"1","target":"object/support pair",
  "bounds":[0.05,1.0],"rationale":"known-scale slide after release"}]
```

`candidate` maps exactly those names to values. Optional `log:true` provides
logarithmic scaling for numerical search, not a change of physical units.
`experiment` contains:

- `observations`: the complete observation document above.
- `initial_state`: declared pose, velocity, deformation and initialization recipe.
- `controls`: external motion/boundaries, release conditions and their evidence.
- `sources`: source fingerprints for geometry, camera, observations and controls.
- `solver`: backend settings, numerical configuration and seed where applicable.
- `comparison`: `max_gap_s` and optional `time_offset_s`.

These descriptions are passed to `runner(request, trial_directory)`. The runner
must reset the actual simulation and return `{"rollout":...,"receipt":...}`.
The receipt contains `experiment_sha256` from the request, `reset:true`,
`applied_parameters`, `backend`, and `version`. Obtain the latter fields from
effective runtime state. A callback that merely echoes the request cannot establish
correct parameter binding. The orchestration helper checks receipt consistency,
not backend internals. Retain raw states, videos and runtime logs in the trial
directory as needed. Inputs, receipt, rollout, comparison and failure reports are
retained; an existing trial directory cannot be overwritten.

Requested and actual values are retained separately. Receipt checks allow float32
rounding (`rtol=1e-6`, `atol=1e-12`), not arbitrary parameter changes.

The callback executes through the external client's native permissions. This API
does not grant host access or bypass task execution boundaries. There is no JSON
field that imports code or launches arbitrary commands through a task tool.

## Native Newton experiments

`real2sim_harness.newton_experiments.NewtonExperimentRunner` supplies a real simulator
callback. The agent provides ordinary Python functions for building its model,
choosing a compatible solver, observing state and optionally initializing/driving
the experiment. There is no fixed object, prescribed motion, optimizer or model
session. Example assembly:

```python
from real2sim_harness.newton_experiments import NewtonExperimentRunner

runner = NewtonExperimentRunner(
    build_model, make_solver, observe, bindings,
    device="cpu", initialize=initialize, apply_control=apply_control,
)
report = run_trial(experiment, parameters, candidate, runner, trial_directory)
```

Callbacks:

- `build_model(experiment, device)` returns a **new**, finalized Newton model each
  trial, using the existing asset loader or an agent-authored builder.
- `make_solver(model, settings)` returns the actual solver; it is constructed
  **after** coefficient binding so cached material data starts with current values.
- `observe(model, state, time_s, experiment)` returns `{track_id: coordinates}`
  for every declared observation track. Use registered camera projection for image
  coordinates, or extract metric state where observed metric evidence exists.
- Optional `initialize(model, state, control, experiment)` sets initial conditions.
- Optional `apply_control(model, state, control, time_s, dt_s, experiment)` applies
  supported external motion after force clearing and before collision/stepping.
  It must not prescribe the target response that calibration aims to explain.
- Optional `collision_factory(model)` replaces the default collision pipeline.

Add `frames`, `sample_dt_s`, `substeps` and actual solver options to
`experiment.solver`; `make_solver` must consume its declared solver options.
Explicit bindings map parameter names to coefficient arrays and scalar indices:

```python
bindings = {
    "contact.dynamic_friction": [
        {"field": "shape_material_mu", "indices": [[object_shape_id], [support_shape_id]]}
    ]
}
```

Check the selected solver's contact mixing rule. Binding both sides to one value
is appropriate only for the stated shared-contact hypothesis. This is not a
universal material-pair override. Two parameters cannot write the same entry.
Supported coefficient fields are listed in `bind_coefficients`; unsupported
fields/indices fail explicitly. Existence/readback alone does not establish that
every solver uses a field: check sensitivity and the selected solver's source.
Restitution, for example, requires an enabling solver configuration where relevant.
Mass/inverse mass/inertia are intentionally excluded from scalar binding: changing
them needs a coupled update or model rebuild through a custom runner.

Each trial retains `states.npz` (available body, particle and joint state arrays),
`contacts.json`, and `parameter_bindings.json`, including partial states if stepping
fails. These are numerical samples, not restart checkpoints. Receipt metadata
includes solver/backend versions and a state-file hash. Callback code, geometry
and control provenance still belong in the experiment's source records.

This adapter steps one model/solver/collision pipeline. Multi-solver coupling and
fluid grid states need a corresponding runner, not an unsupported solver swap.
Event extraction and rendering remain available to agent-authored analysis: wrap
the callback to add measured rollout events, or render retained states through an
appropriate renderer. No event times are copied from the reference automatically.

The opt-in `REAL2SIM_TEST_NEWTON_EXPERIMENTS=1` test uses an operator-installed
Newton runtime on CPU. It checks actual friction sensitivity, material readback
and identical reset using a synthetic free-sliding contact example. This validates
backend plumbing, not video-derived friction or real material accuracy.

## Native PhysX experiments

### Writing coupled mass properties into the asset

The PhysX bundle author accepts an optional backend-specific section in the
existing `physics_parameters.json` (also supported by `run_simulation` parameter
overrides):

```json
{"physx": {
  "friction_combine_mode": "max",
  "mass_properties": {
    "base": {
      "mass_kg": 1.0,
      "center_of_mass_m": [0.0, 0.0, 0.06],
      "inertia_kg_m2": [[0.003,0,0],[0,0.003,0],[0,0,0.0015]],
      "source": "Illustrative assumption, not a measured mass distribution"
    }
  }
}}
```

Use `base` for a rigid object's root or an articulation's base, and
`joint:<Blender joint object name>` for a moving link. Unknown bodies are rejected.
Each override supplies mass, COM, the full inertia tensor and its provenance
together, taking precedence over automatic mass/density for that body. Coordinates
are unscaled body-local axes in metres; the tensor is about the specified COM in
kg m². Positive principal moments, symmetry and the physical triangle inequality
are checked. The author writes principal axes and diagonal moments to USD MassAPI
and retains the source assertion. The rigid drop check also reads the actual mass,
COM and inertia back from PhysX. Articulated links still require corresponding
live adapter checks; successful authoring alone does not prove their runtime use.

This is asset authoring, not a scalar mass/inertia binding in the experiment
runner. An agent-owned session factory may rebuild a coupled mass distribution
for an experiment and retain actual readback. A single free slide does not identify
absolute mass. Collision hulls also need not describe the material distribution
of a hollow or filled object. Keep unsupported mass/COM assumptions explicit and
test their effect before attributing residuals to friction.

The combine-mode setting changes the object's material policy; verify the other
material and the backend's pair-combination rule as well. This `physx` section
is specific to the PhysX asset adapter and does not configure Newton mass models.
Changed geometry or parameters require fresh source-matched simulation and review.

### Optional source-solid mass computation

`real2sim_harness.solid_mass_properties.from_mesh(vertices, triangles,
density_kg_m3=..., source=...)` integrates an explicitly oriented, closed material
boundary in metres. It returns volume and a validated `mass_properties` record
containing mass, COM and full inertia about COM. Use the actual material boundary:
inner cavity faces point into the cavity, so empty space does not contribute
mass. These are material properties, not properties of a convex collision hull.

For separate materials, compute each disjoint component with its declared density
and use `combine(records, source=..., disjoint_material=True)` in a common body-local
frame. This applies the parallel-axis rule. Union overlapping solids upstream;
adding their records would double-count overlapping material. Topological closure
checks cannot establish absence of self-intersections or overlapping components.
Density remains a prior unless separately measured.

Write the resulting record into the corresponding body's `physx.mass_properties`
entry shown above. This optional computation does not change automatic density
fallbacks or infer physical truth from appearance. Check geometry, frame, scale
and density before authoring, and repeat source-matched target simulation with
actual mass/inertia readback after applying it.

### Running native trials

Revolute `joint_drive` gains use SI units: stiffness in N·m/rad, damping in
N·m·s/rad, and maximum force in N·m. The USD author converts angular gains to
degree-based USD coefficients; PhysX tensor readbacks remain radian-based.
Earlier exports wrote nonzero SI gains directly into USD, producing runtime
gains about 57.3 times too large. Re-export affected assets and verify actual
readback; zero-gain passive defaults are unchanged. Calibration writeback must
be checked on the exported asset, not only on a separately constructed model.

For contact between links of one articulated asset, the PhysX author accepts
`"physx": {"enabled_collision_pairs": [["joint:left", "joint:right"]]}`.
Body names use the same `base` / `joint:<Blender joint object name>` convention
as mass properties. The default empty list preserves internal collision filtering;
each named pair removes its symmetric exclusion and enables PhysX articulation
self-collision. Other internal pairs remain filtered. An explicitly selected pair
connected by a joint also enables that joint's collision flag. Review the authored
colliders before selecting contacts, then verify their actual response in the
target runtime; a valid pair list alone does not prove impulses occur. This
setting does not prescribe controls or implement a motion-specific experiment.

`real2sim_harness.physx_experiments.PhysXExperimentRunner` implements the same trial
protocol through the public PhysX Tensor API. It imports neither Isaac Lab nor
Isaac Sim automatically. An agent/operator supplies the session factory for the
installed runtime and target asset; the common protocol does not require a
particular Sim/Lab version.

```python
from real2sim_harness.physx_experiments import PhysXExperimentRunner

runner = PhysXExperimentRunner(session_factory, observe, bindings,
                               apply_control=apply_control)
report = run_trial(experiment, parameters, candidate, runner, trial_directory)
```

`session_factory(experiment)` is a context manager yielding a runtime with:

- `views`: names mapped to NumPy-fronted PhysX rigid-body/articulation tensor views.
- `metadata`: actual `backend:"physx"`, `version`, and optional runtime details.
- `reset(experiment)`: restores initial state, controllers and clock; returns `True`
  only when complete. A fresh stage/session per trial is a straightforward option.
- `step(dt_s)`: advances exactly the requested physical duration; it need not render.

`observe(runtime, time_s, experiment)` returns the declared track coordinates.
Optional `apply_control(runtime, time_s, dt_s, experiment)` issues external controls.
The session factory is responsible for loading the original asset, resolving view
indices, applying the numerical solver settings, and cleaning up even on failure.
Existing USD loading and runtime setup can be reused. A live Lab controller that
rewrites gains must be adapted so it cannot overwrite the calibrated coefficients.

Declare which point an initial linear velocity describes. PhysX rigid-body tensor
velocities use the COM in world coordinates, whereas observed poses may describe
the asset's body origin. With body-local COM `c`, body-to-world rotation `R` and
world angular velocity `omega`, convert using
`v_com = v_origin + cross(omega, R @ c)`. Retain the measured derivative window,
this conversion and runtime readback. A short noisy initialization window can
create directional trajectory errors that friction alone should not absorb.
For dynamic silhouettes, the native `posed_observations` renderer and
`shape_evidence.silhouette_metrics` can review each registered pose. The existing
multi-view shape fitter assumes a common object configuration; moving frames
cannot be treated as static views without justified object-frame registration.

Bindings select properties by explicit indices:

```python
bindings = {
    "object.dynamic_friction": [
        {"view":"object", "property":"dynamic_friction", "indices":[[0,0]]}
    ],
    "hinge.damping": [
        {"view":"cabinet", "property":"joint_damping", "indices":[[0,0]]}
    ],
}
```

Material indices are `[body_or_articulation, shape]`. Joint indices are
`[articulation, DOF]`. Supported properties are `static_friction`,
`dynamic_friction`, `restitution`, `joint_stiffness` and `joint_damping`.
Read-modify-write preserves unselected entries; actual runtime readback must
match the candidate within float32 precision. Restitution must be within [0,1]
and dynamic friction must not exceed static friction. If both friction values
should change together, bind both explicitly. Mass/inertia and other coupled
properties require additional adapter work.

These are material/drive coefficients, not universal effective-contact values.
Retain the opposing material, combination mode, contact settings and actuator
configuration. The public API documents that drive values may be set even when
no drive is present, in which case they do not affect dynamics: readback alone is
not proof of physical sensitivity. Verify the actual shape/DOF mapping, configured
drive mode and motion response. See the [PhysX Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html).

Raw supported view states are saved in `states.npz`; `state_views.json` maps array
names back to views/fields. Parameter evidence and receipts use the same conventions
as the Newton adapter. No rendered movie or acceptance is implied by a successful
trial. The observer contract also requires matching the physical/video observable.

Contract tests use explicit test doubles and do not establish a live PhysX run.
`scripts/check_physx_experiment.py --output <new-directory>` is an opt-in synthetic
contact/reset check to run inside a compatible Isaac Sim Python environment.
It starts Kit through `SimulationApp`, uses CPU dynamics and may still require
graphics hardware for application startup. It does not require Lab. Validate this
entrypoint against the operator's runtime before claiming live adapter support.

## Native experiment controls and optional timed interventions

An external coding agent can directly author model/session construction,
initialization, observation and per-step control callbacks. No command to read a
record or query a state is required: use native Python/state access and retained
files. This interface supplies execution and reproducible evidence, not another
agent or a fixed catalog of experiments.

For one-time changes, optionally include an `interventions` array in the experiment:

```python
experiment["interventions"] = [
    {"name": "release", "time_s": 0.5,
     "source": "release frame mapped onto the experiment clock",
     "payload": {"constraint": "authored_grip"}}
]
```

Pass a native `apply_event` callback to the selected runner:

- Newton: `apply_event(model, state, control, event, experiment)`.
- PhysX: `apply_event(runtime, event, experiment)`.

The callback implements the actual constraint/control/state operation and returns
a finite JSON object with execution evidence, preferably effective runtime
readback. Event names and payloads are agent-owned; there is no built-in release,
push or manipulation template. Runtime-specific mechanics remain the callback's
responsibility. Removing a grip must actually remove its physical effect rather
than only change a label. State impulses are permissible controlled experiments,
but should not be described as validated physical grasp/release mechanisms.

Times must coincide with integration boundaries within the trial. Off-grid times
fail rather than silently moving an observed event or changing integration steps.
Choose a justified experiment origin/step size or explicitly document timing
uncertainty. Events at zero run after initialization, before the initial sample;
later events run after the preceding integration step and before a sample at that
boundary. Events sharing a boundary run in their declared order. End-time events
can change the final state but provide no subsequent response evidence.

Each event executes once per trial. `actions.jsonl` records start, completion or
failure, requested event, executed step/time and callback evidence. Failed events
abort the trial and retain partial state and action records. Successful event
timestamps enter `rollout.events`; the receipt hashes the action log. These
timestamps describe executed interventions, not automatically detected response
events such as settling or impact. Derive response events from actual state or
contacts using an appropriate observer/custom runner.

Existing `apply_control` callbacks still execute before each integration step.
They may return a JSON object to retain applied control evidence, or `None` to
avoid logging every step. Newton clears forces before these callbacks: persistent
forces belong here, not in the event callback. Changes to persistent drive targets
or constraints, and explicitly declared state impulses, can use `apply_event`.
The agent chooses which controls/readbacks matter; callback output is not itself
independent proof of physical effect. Inspect resulting motion as well.

This is an offline trial executor. It does not add an interactive pause/resume
server, checkpoint the solver for warm starts, or automatically stop early at a
task-specific condition. An agent needing these can author a native runner under
the same request/receipt protocol. Restarting trials rebuilds/resets the runtime.

## Fixed-candidate validation across experiments

`experiment_validation.evaluate_candidate(cases, parameters, candidate, output)`
executes one agent-selected candidate with the existing trial runners. Use it
when repeating boilerplate across several motions would obscure which parameters
were actually validated. There is no new query tool, optimizer or model session.

Each case supplies `id`, `role` (`fit` or `validation`), `experiment` and a native
`runner`. Validation cases also supply `independence_evidence`, explaining the
separate observation/interval and whether it influenced parameter selection.
Each experiment can have different initial conditions, controls and observations;
the named candidate parameter values remain fixed. Keep parameter meaning and
units consistent across runners; a common scalar does not imply that PhysX and
Newton material coefficients have equivalent physical meaning.

```python
from real2sim_harness.experiment_validation import evaluate_candidate

report = evaluate_candidate([
    {"id": "original-motion", "role": "fit", "experiment": fit_experiment,
     "runner": fit_runner},
    {"id": "separate-motion", "role": "validation", "experiment": validation_experiment,
     "runner": validation_runner,
     "independence_evidence": "different capture not used to select this candidate"},
], parameters, chosen_candidate, "candidate-check")
```

The function records a suite request and one complete trial directory per case.
Fit and validation scores stay separate, and each case's events remain visible.
Missing comparison coverage prevents a complete role score; an execution failure
retains progress and aborts remaining cases. It does not turn a low mean error into
acceptance. Normalization scales and observation coverage affect interpretation,
so inspect per-case results rather than just the role mean.

Exactly identical observation documents cannot be labeled both fit and validation.
This check is deliberately limited: it cannot prove independent capture, detect
all overlapping intervals, or determine whether an agent already tuned against
validation results. If those results guide the next candidate, that sequence has
become development evidence; reserve another motion if a held-out claim matters.

## Interpreting comparison feedback

Per-track comparisons retain `residual_vector` in the declared coordinate order,
with sign **simulation minus observation**, plus `axis_bias`, `axis_rmse`,
`rms_distance` and `max_distance`. Missing/unusable samples remain null. Bias and
RMSE help distinguish persistent displacement from alternating errors, but do not
by themselves diagnose material parameters, camera errors or instability. These
are sample-weighted Euclidean statistics; no smoothing, clock warping or angular
wrap is applied. Rotations need an appropriate common observable/residual model.
The existing normalized Huber score and coverage requirements remain unchanged.

## Optional existing numerical search

`search(experiment, parameters, runner, optimizer, output, generations=N)` accepts
an optimizer with `ask()` and `tell(points, losses)`. The optimizer operates on the
unit box; the wrapper converts to the selected physical ranges, evaluates trials
sequentially, retains best-so-far results and stops on missing/failed evidence.
It allocates no compute resources. Trial count is the requested generations times
the optimizer's population size. The agent can instead call `run_trial` directly
for every candidate or implement a different analysis/search.

Pass an operator-installed ask/tell optimizer explicitly in a native script. The repository does not require a particular optimizer package. External fitting routines can provide initial hypotheses once their camera, scale and motion assumptions are checked.
See [motion geometry and fitting](MOTION_GEOMETRY_FITTING.md) for the optional
registered-plane projection, time-window selection and native slide-fitter bridge.

## Implementation boundary

The first slice provides video-measurement import, comparison, trial records, optional optimizer integration,
native Newton coefficient experiments, a PhysX tensor experiment adapter and agent guidance. The existing generation
and basic operability workflows remain independent. Dynamic tracking, inferred
manipulation boundaries, general live mass/inertia updates and arbitrary-video held-out validation are not supplied automatically. Coupled mass properties can be authored in the source asset and require fresh target-asset validation. Synthetic tests establish
interface behavior and backend sensitivity, not real-world material accuracy.
No particular simulator version is imposed by the protocol; adapters still need
verification against the installed runtime.

Run `scripts/check_physx_experiment.py` on the selected runtime to check friction sensitivity and reset; add `--articulation` to check driven-joint coefficient readback, response sensitivity and repeatability. See [5.1 compatibility](ISAAC_SIM_51.md) for setup. These checks do not establish physical calibration of an observed real object.
