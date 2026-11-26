# World-bound native vector tools

`bound_native.py` executes `collect`, `develop`, and `confirm` against a frozen
native binding. It never reads legacy Peg checkpoints, source demonstrations,
hard-coded CAD, or a controller during student evaluation. The imported factory
is pinned by its absolute path and source hash in the current work packet.

## Environment boundary

`build(binding, request, config)` returns an object with:

- `metadata()`: actual engine, world/task/backend/session/I/O identities, loaded
  frozen scene inputs and explicit verification/deployment limitations.
- `reset(case)`: initialize a contract-owned group/seed/stratum/repetition.
- `observe()`: the ordered finite observation vector declared by the binding.
- `controller(method, pinned_parameters)`: a callable used only in collection.
- `step(command)`: execute bounded commands and return `command`, `physics_steps`,
  and `teleport_used: false`. Reset teleportation is separate and disclosed.
- `measure()`: physical measurements plus the frozen outcome predicate.
- `close()`: release the native application.

The worker labels observation t with the command actually executed at t, records
the following observation, retains failed cases and errors, and only adds
successful expert rollouts to the BC manifest. It saves every attempt and trace.
Evaluation decodes the exact currently accepted dense checkpoint, with no expert
fallback. Cases are the entire frozen split product, including failures.

## Configuration

The `astra_bound_native_v1` configuration requires `backend_id`,
`simulation_only: true`, `factory`, `factory_file`, `engine`, `action_low`,
`action_high`, `max_steps`, `success_ticks`, `collectors`, `success_rule` and
factory-specific `environment`. All files, including this worker, the factory,
the training decoder and configuration, must be inputs in the registered package.

Collection request parameters are `{}` or `{"episodes_per_case": 1..16}`;
evaluation accepts only `{}`. The native configuration and contract, not an
agent-selected request, control success, world, splits, bounds and reset ranges.

`success_rule` is a conjunction of `{field, operator, target}` predicates.
Operators are `le`, `ge`, `abs_le`, and `eq`. The independent reviewer recomputes
this predicate from trace measurements and checks consecutive success ticks.
The same source-qualified physical criteria must be fixed before experimentation.

## Source-geometry insertion adapter

`bound_insertion_factory:build`, composing scene preparation with
`calibrated_scene_reconstruction.tasks.rigid.bound_insertion.InsertionEnvironment`, is an explicitly limited **pregrasp
terminal insertion** adapter. It obtains the moving object, receptacle, meshes,
contact settings, source robot base and opening from the current session. It
does not replace the source scene with a legacy benchmark or task CAD.

The adapter initializes a source-geometry object attached to the real simulated
articulated TCP, holds the source-pose receptacle as a disclosed fixture, executes
bounded IK/drive commands, then removes the object attachment and opens the hand.
Success requires independent moving-body measurements after a pinned settlement
interval. The separate invisible IK articulation supplies numerical kinematics;
it never teleports the physical robot during policy steps.
The learned action is Cartesian displacement only. A fixed measured-pose rule
triggers hand opening and fixture removal in both teacher and student execution;
the student does not learn grasp acquisition or a separate release action.

The environment configuration explicitly supplies:

- `scope: "pregrasp_insertion"`, `fixture_release_before_success: true`,
  `hold_receptacle: true`, optional `arm`.
- `object_rotation`, `object_long_axis`, `tcp_rotation`,
  `tcp_from_object_origin_m` (world axes): declared pregrasp initialization.
- `physics_hz`, `initial_offset_bounds_m` (3x2), `reset_settle_ticks`,
  `reset_position_tolerance_m`, `reset_rotation_tolerance_rad`.
- `hand_close_fraction`, `release_depth_m`, `release_radial_m`,
  `release_tilt_deg`, `release_settle_seconds`, `controller_target_depth_m`.
- Optional bounded `reset_ik_iterations` and `step_ik_iterations`.
- Optional `robot_initializer: "source_visual_estimate"` uses the existing
  `robot_visual_fit.json` estimated joints; its file hash and estimate type are
  recorded. Uncontrolled joints use that source initialization and their actual
  physical state is measured during control. The default
  `spec_home` remains available and is labeled as such.

Runtime metadata records controlled arm/hand and passive joints, original
robot drive settings, disabled-gravity and self-collision settings. These are
simulation assumptions, not evidence of real gravity compensation or safety.
The receptacle fixture is a native fixed joint authored at its original source
pose before the first physics step; source collision and material settings stay
unchanged. Pregrasp orientation candidates can be proposed with
`bound_native_pregrasp_scan.py`, which supplies only native IK evidence. The
selected initialization must still pass actual contact/release qualification.

Initialization can apply a bounded physical servo before the first recorded
policy observation, retaining every setup command and the actual start error.
It does not move the object pose to conceal drive error. During policy execution,
each IK solve starts from measured articulation positions, including passive
joints; a stale clone pose must not substitute for physical forward kinematics.
The numerical IK tolerance is smaller than the source opening clearance, so
valid sub-millimetre corrections are not silently treated as zero.

Isaac 5.1 can invalidate articulation wrappers when a temporary joint is
deleted. The adapter refreshes invalid wrappers at reset/control boundaries,
without resetting the world. It checks native joint positions, velocities, link
transforms and simulation time before/after refresh and rejects any change.
Failures retain the full Python stack and all preceding transitions, including
failures before the first policy step. Qualification must include multiple
distinct episodes within a single native process to exercise this lifecycle.

Its observation is the simulated object's horizontal centre and bottom relative
to the source opening, in metres. Its action is a world Cartesian TCP delta, in
metres. These observations are privileged: this is a simulation verification
profile, not a vision policy or a deployment candidate. Autonomous grasping,
image perception and real-world transfer require separate adapters and evidence.
The collector options are a Cartesian servo and a receding-horizon shooting
controller with an explicit nominal diagonal action-response model. Neither is
claimed to be a fully identified contact dynamics model.

## Review

`bound_native_review.review(binding_path, result, output_dir=...)` independently
verifies hashes, labels, timing, case coverage, policy I/O, lineage, numerical
success predicates, student-only execution and measured inference latency.
The native source/contact qualification is upstream and remains mandatory.
Unknown custom checks fail instead of being assigned the target value.

Supported extra check names are `coverage:declared_cases`,
`coverage:initial_offsets`, and `deployment:io_parity`,
`deployment:action_bounds`, `deployment:released_success`,
`deployment:simulation_only`. They describe actual evidence; a passing protocol
check does not assert a passing success rate. The outer controller separately
applies the frozen success-rate requirement and holdout seal.

CPU tests validate integrity and accounting with an explicitly numerical test
fixture. Native collection/training/evaluation receipts are required to claim a
physical end-to-end result.

## Native contact-report diagnostics

`real2sim/apps/scene/check_rigid_intrinsic.py` produces unaccepted candidate intrinsic
measurements for independent review. `--diagnose-contact-reporting` allows
sampling an unresolved exact unknown-face reporting warning while keeping every
native verdict false. It cannot approve a scene or policy.

An optional `--audit-contact-reporting` profile is restricted to inspected Isaac
5.1.0.0 / PhysX107.3.26 and current single-material collider schemas. It records
the engine's unknown-face material-report warning as a metadata limitation,
including the original messages/count, because GPU mesh-contact reduction can
omit triangle indices. Unsupported versions, per-face material subsets,
malformed material coefficients, any other NULL warning and native solver errors
remain failures. Unread native default coefficients are explicitly unknown and
unchanged. Native contact count, force/separation, penetration, source geometry
and released response remain independent requirements. For this compatibility
profile, compare native trajectories with contact reporting disabled and enabled.
Record the trajectory hashes and independent review.
