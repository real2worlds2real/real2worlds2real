# Real2Sim coding-agent entry

This is Part 1 of the repository (real-world observations to simulation assets); run its commands from this directory.
[The skill index](skills/README.md) gives the reading order: core principles, task workflows (twin, cousin, physics),
specialist skills, and the experience kept from earlier work.

For object/scene generation or observed-motion physics calibration requests,
read `skills/use-real2sim/SKILL.md` and select the relevant task workflow.
Use the current Codex/Claude conversation to perform the task. The `real2sim-harness task`
commands expose domain capabilities without starting another model session.
When changing a public entry point, update its usage instructions and verify the documented invocation.

Read the common methods at `skills/work-with-task-evidence/SKILL.md` at the start
of task work. Generation task initialization writes the common and selected task
workflow to `<run>/external/startup.md`; read it before modeling. Standalone
physics calibration uses the common methods and
`skills/calibrate-observed-physics/SKILL.md` directly with the native experiment
APIs; it does not require creating an unrelated generation task. The remaining
skills under `skills/` are optional specialists selected for the actual need.
Read task records and source directly when useful. Use your native image/video,
shell and delegation capabilities; repository evidence readers are optional aids.

Generated work belongs in the selected task workspace. Keep operator
configuration, credentials and unrelated personal files outside task records.
Your coding client's native permissions remain in force. Repository Blender
commands use a separate task execution boundary; that boundary does not constrain
commands you run directly in the outer client.

Each scene participant and cousin has its own task and acceptance. Delegate only
within user-authorized concurrency/resources. Do not run two writers against the
same task; domain mutations acquire a per-task lock. Complete source-matched
simulation and review before accepting. An image, model exit or successful build
alone does not establish an operable asset. Return genuine failures for repair.

Use the [agent module workflow](../end_to_end/docs/reconstruction.md) to combine these operations with calibrated capture, placement and policy development in the same run.

## Metric scene twins (the scene pipeline)

A scene in metres with the robot placed, reconstructed from a walk-around video, follows
[the scene workflow](docs/agent/workflow.md) and the capture / model / deliver skills. Its JSON tools are
`calibrated-scene-reconstruction schemas` and `calibrated-scene-reconstruction <tool> --request <json> --output <dir>` (the same tools as
`python -m calibrated_scene_reconstruction.agent.tool`); GPU stages run in the environments `setup/*.sh` build, with `PYTHONPATH=src`.

- Author session code in `configs/<session>/agent/`; runtime evidence goes to `sessions/<session>/`.
- The gates S0-S6 and the object and support loops (OBJ, SUP) are mandatory. Each stage keeps its own loop: proposal,
  execution, independent checks, review, refinement. An upstream change invalidates the evidence that depends on it.
- Scale comes from legal evidence in the capture: the calibration board or the robot. Withheld task dimensions never
  enter an independent reconstruction.
- Look at the actual images, motion, geometry and logs before writing a positive review. Hashes establish provenance,
  not truth.
- Never weaken a gate, a tolerance or a test to obtain a pass, and never overwrite a freeze: create a derivative
  session instead. A gate that cannot be passed honestly is a result to report.
- Simulator, GPU and real-robot validation are separate claims. Report exactly what was exercised.

## Experience and capability evolution

Before choosing reconstruction methods, consult the
[experience module index](skills/twin-model/references/experience/index.md) and the records that match the current
context (`calibrated-scene-reconstruction evolve recall outputs/real2sim_evolution <context.json>`). Procedures are reusable; earlier
dimensions, fits, poses and approvals are not evidence for a new capture.

When a mechanism, setting or task exceeds the current representation, procedure or tools, use
[real2sim-evolve](skills/real2sim-evolve/SKILL.md): diagnose the gap, research or prototype a solution, validate and
regress it, then register a scoped procedure or update a discoverable skill. Do not silently substitute an easier
task. Keep capability experiments in isolated directories with records under `outputs/real2sim_evolution/`; shared
changes follow the [layer boundaries](docs/scene/reconstruction.md) and need fresh evidence on the affected scenes.

## Delivery

An accepted twin is handed on with `real2sim-harness deliver` ([delivery record](docs/DELIVERY.md)). Robot training, policy
execution and real-robot work are not part of this directory: see `../policy_development/` and `../end_to_end/`.

## Framework development

Keep reusable computation, execution and evidence contracts in the framework.
Keep task-specific geometry, experiment policies and fitting scripts in task
workspaces. Update the affected tests and public documentation when changing an
interface. Validate changes in the relevant target runtime; unit tests alone do
not establish simulator compatibility.

When investigating generation failures, distinguish a broken/missing capability
from failure to use an available one. Repair demonstrated capability defects
with highest priority; stronger skill wording cannot compensate for lost
geometry or material data. When the capability works, improve its evidence-based
discovery, triggering and review obligations. Preserve the agent's structural
understanding and supported priors during tool use; do not reward image-score
gains that introduce unsupported shape changes. Attribute overfitting only when
the retained candidates and evidence demonstrate it.
