# Reconstruction of the entry and code boundary

This article describes the process of reconstructing a reviewable simulation world from raw footage. Robot demonstrations, strategy training, and data generation are not covered in this scope. For specific execution rules, see the [agent workflow](../agent/workflow.md), and for levels, see the [twin pipeline](twin_pipeline.md).

## workflow

The agent understands tasks and observations, selects representations, develops plans, uses tools, examines evidence, and returns to the module that requires correction. S0–S6 represent acceptance dependencies, and not all modules are required to execute this process only once. The success of tool execution does not mean that the agent has been reviewed or can be delivered.

When encountering new mechanisms, settings, or tools with insufficient capabilities, follow the [Capability Evolution Process](../agent/evolution.md): identify gaps → isolate experiments and implementations → conduct regression/integration and necessary native verifications → document applicable scenarios → update tools or skills → retrieve subsequent tasks for re-validation. This process also accumulates experience on "how to handle different mechanisms"; do not use the dimensions, fit, or acceptance criteria of old objects as evidence for new inputs.

| module | Current implementation | Input vs. Output |
| --- | --- | --- |
| Shooting, Cameras, and Scale | `scene/capture/` | Photography/Videos, Legal Calibration, Explicit Priorities → Camera and Metric World |
| support surface | `scene/support/` | Capturing observations, agent models → Supporting geometry, materials, contact surfaces, and review |
| operating object | `scene/objects/` | Agent Blender source code, component/property declarations → geometric, visual, and structural evidence per round |
| environment | `scene/background/` | GS, editable mesh or hybrid → Background assets, source and coverage checks |
| Physics related to the task | `physics/process/`, `tasks/` | Hypotheses and experiments proposed by the agent → Native execution records, parameters, and independent review |
| The entire process of delivery | `scene/review/`, `scene/twin/` | Current object, support surface, environment, and task evidence → levels, freeze list, USD/USDZ |

The operating object uses a Blender model written by the agent by default; templates and other fitting routes need to be explicitly selected. Scaling and evaluation still follow the original protocol: the actual dimensions of the task object or support surface are not used as independent reconstruction feedback. For physical processes that require replication, see [Physical Workflow](../physics/process_workflow.md).

## Entry points

First, prepare the session configuration and review the inputs and roles according to [capture skill](../../skills/twin-capture/SKILL.md).

```bash
pipeline/twin.sh --help
pipeline/twin.sh <session> --from <stage> --to <stage>
python -m calibrated_scene_reconstruction.agent.tool schemas
python -m calibrated_scene_reconstruction.agent.tool status --request request.json --output outputs/status
python -m calibrated_scene_reconstruction.agent.mcp_server
```

The `session` in the status request points to `sessions/<session>`. Other requests are based on the contract returned by `schemas`. The modeling source code for the same session is stored in `configs/<session>/agent/`, while the running rounds, reviews, evidence, and delivery are stored in `sessions/<session>/`. New modeling rounds should retain their own input and implementation fingerprints; after modifying the model, observations, or implementations, the old reviews cannot replace the new ones.

## layering

| layer | Contents | Internal dependencies |
| --- | --- | --- |
| `core/` | Multi-field shared arrays/corrections, file records, COLMAP reading, visibility, and resource tools | core; no domain import or Isaac |
| `assets/`, `robots/` | Asset Geometry, Material/Contact Contracts, Pure Robot Description and Computation | self and core |
| `scene/`, `physics/` | Reconstruction strategy, observation processing, parameter search, audit records | self, core, assets, robots |
| `sim/` | USD/PhysX/Isaac native implementation, state readback, and runtime adaptation | self, core, assets, robots |
| `tasks/` | Operation testing, task measurement; running the adapter after it is started on the backend using sim | Self and lower-level services, do not read scene policies in reverse |
| `agent/` | CLI/MCP, workflow operations and tool scheduling | Combinatorial domain services; do not import directly from sim |
| `apps/`, `calibrated_scene_reconstruction.apps` | Launchpad and integrable applications | Combines scene, physics, tasks, and simulation; domain layer does not import apps in reverse |

`calibrated_scene_reconstruction.apps` is a packaged application combination layer. It does not provide another set of model sessions or agent schedulers. For example, `apps.physics.process_native` combines physical search with native execution, `apps.reconstruction.candidate_scene` combines reconstruction input with scene runtime, and `apps.reconstruction.observation_match` combines source observations with actual task states.

Calculator for pure triangular distance, self-intersection, and fabric gap is located in `assets/`. General visibility and numerical solution adaptation are located in `core/`. These tools can use NumPy, Torch, or Warp, but they should not be classified under the Isaac Runtime just because they use a GPU. Native particle parameters are read back by `sim` and then compared at a higher level; higher-level components cannot replace the actual read-back values. The conversion of native materials in the environment is executed through independent backend commands, with the scene layer preserving its source code fingerprint and results.

## Compatibility and Verification

The entry points for the old modules are retained for forwarding, with mappings found in `src/calibrated_scene_reconstruction/compatibility.json`. The new code will prioritize using the official implementation path. The forwarding mechanism retains the import entries for existing session scripts, but the old physical/visual auditing mechanisms will no longer be valid after the changes. Source code evidence should be tied to the actual implementation, not just to unchanged forwarding files.
