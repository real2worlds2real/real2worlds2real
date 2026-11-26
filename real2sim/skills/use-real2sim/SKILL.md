---
name: use-real2sim
description: Route object/scene generation, functional cousins and observed-motion physics calibration to their task workflows in the current coding-agent session, using Real2Sim capabilities without a nested model session.
---

# Use Real2Sim in this conversation

Respect the user's backend choice throughout generation and review. With no
explicit choice, use PhysX by default for cloth, garments and soft bodies; use
Newton when the user requests it. Check adapter capability before execution and
make missing support explicit instead of silently switching engines.

You are the task agent. Use your existing reasoning, shell, file, media and
delegation capabilities. This entry does not log in to a model, relay prompts or
launch a second agent. Take images, video and natural-language constraints from
the user. Select the workflow from the requested outcome.

For an observation-driven task, start a [module plan](../../../end_to_end/docs/reconstruction.md). Use [reconstruct-object](../reconstruct-object/SKILL.md) per participant and [reconstruct-environment](../reconstruct-environment/SKILL.md) for the room. Select the native tools below as needed.

## Select the task workflow

Read [the common methods](../work-with-task-evidence/SKILL.md), then the relevant
task-level skill. These workflows form the second layer; their specialist
links supply concrete methods as needed. [The skill index](../README.md) lists
every layer, the experience kept from earlier work and the tools behind them.

| Requested outcome | Task workflow |
| --- | --- |
| One editable operable object twin | [generate-editable-object](../generate-editable-object/SKILL.md) |
| An interactive scene twin | [plan-scene-generation](../plan-scene-generation/SKILL.md) |
| A scene twin in metres with the robot placed, from a walk-around video | [twin-capture](../twin-capture/SKILL.md), then [twin-model](../twin-model/SKILL.md) and [twin-deliver](../twin-deliver/SKILL.md) |
| Functional object variants | [generate-functional-cousins](../generate-functional-cousins/SKILL.md) |
| Scene variants | [generate-scene-cousins](../generate-scene-cousins/SKILL.md) |
| Physical alignment to observed motion | [calibrate-observed-physics](../calibrate-observed-physics/SKILL.md) |

Physics calibration may be the primary task for an existing asset or a requested
follow-on to generation. Begin its observation/experiment workflow directly;
the native Python experiment APIs do not require a Blender generation run.
`task init` below creates object/scene generation workspaces, not a separate
physics-calibration mode. Existing generation tasks may retain observation
tool results there; calibrated asset changes return to their object/scene
source-matched acceptance. Ordinary generation does not require calibration.

The operator installs compatible Blender and physics runtimes following
`docs/INSTALLATION.md` and selects runtime adapters as needed. Inspect the actual
installation and existing task before installing or allocating resources. Native
commands run under the outer coding client's permissions. Repository Blender
execution remains task-scoped on Linux.

## Start or continue a task

```bash
real2sim-harness task init --reference /path/to/object.jpg \
  --objective 'Preserve its design and function; generate two functional cousins afterward.' \
  --blender /path/to/blender
```

Retain the returned run path. Read its `external/context.json` and
`external/startup.md`: these provide the common methods, selected task workflow,
input paths, workspace, skill catalog and required completion operation. Write
modeling and analysis code in that workspace. Source and all task records are
available for on-demand investigation under the outer client's actual grants.

Use repeated `--reference` flags for images/video, or no reference for an explicit
text-only concept. Original videos are retained without mandatory decoding or
thumbnail sampling. View them through your native abilities; optional frame
inspection/promotion is available when useful. Supply genuine camera/scale
evidence with `--measurements` when available; inferred scale remains uncertain.

To continue an existing task, read its retained context. Older managed runs can
be prepared with `real2sim-harness task adopt <run> --blender /path/to/blender`; their
source, attempts and acceptance history stay available. Task continuation uses
the current conversation and does not require a transport resume command.

A domain response may include `guidance_update` after framework guidance or
capability descriptions change. Read and apply the returned current instructions
and reconsider affected evidence. An acceptance request that first discovers
such an update is rejected before dispatch; retry only after that review. Existing
source and attempts remain available.

## Discover and use domain capabilities

```bash
real2sim-harness task schemas --run <run>
real2sim-harness task schemas run_blender --run <run>
real2sim-harness task call <run> run_blender --request <workspace>/build-request.json
```

Operations take JSON requests with the same field names used in specialist
skills. Read the exact schema for the selected operation rather than loading all
schemas. For example, a Blender build request is:

```json
{"script_path":"object.py","plan":"Named editable parts and materials","change_summary":"initial"}
```

`script_path` resolves inside the task workspace. The Python source defines
`build_scene()` and uses the representation metadata described by the selected
task/specialist skill. Commands return full reports, image paths and retained
operation records, including failed results. Open the files natively, inspect
logs and states, write your own analyses, revise and retest as appropriate.
Video/reference inspection wrappers remain optional. The required target-physics
and final acceptance contracts remain in force independently of the interface.

Finish with `accept_result` for an object or `accept_scene` for a scene only after
source-matched simulation and your visual/functional review. Supply an explicit
evidence-based reason; do not write acceptance files manually. Read the final
receipt and return the editable asset/runtime package and useful evidence paths.

## Scene participants and cousins

For requested physical alignment, follow the task-level
[calibrate-observed-physics](../calibrate-observed-physics/SKILL.md) workflow for
the asset or selected scene participants.

For a scene, initialize with `--scene` and an operator-owned `--scene-runner`
configuration. Follow `plan-scene-generation` and `compose-interactive-scene`.
`build_scene_background` returns Blender previews immediately. A
`build_scene_object` call prepares a child task and returns `needs_object_work`;
it starts no model. Work on that child in this conversation, or delegate the
child run and objective using the client's native subagents when resources and
user authorization permit. After child acceptance, repeat the operation with its
returned `asset_id` to package it for scene assembly. Scene-cousin participant
redesign follows the same handoff through `build_scene_cousin_object`.

After an accepted twin, prepare requested cousins:

```bash
real2sim-harness task cousins <accepted-twin-run> --prompt '<user cousin request>'
real2sim-harness task next <family>
```

The first command prepares a planner task; it does not generate a hidden plan.
Read the planner startup/context and use `submit_cousin_plan` to record the count
and variants inferred from the user request. Counts are never fixed by an
example. `task next` returns one retained variant task. Complete and accept that
variant, then call it again. The caller may delegate within authorized resources;
independent task directories preserve ownership. A planning-only family stays
planned until an explicit execution request (`task next <family> --execute`).
For authorized parallel work, `task next <family> --index N` prepares a specific
variant. Delegate distinct returned tasks; resources remain caller-managed.
Finally call `task next <family>` without an index to verify all final receipts
and family completion, including sibling diversity after concurrent generation.

The existing cousin contracts preserve function and articulated connectivity,
allow meaningful geometry/material variation, and preserve scene layout by
default. Explicit layout changes follow the user's request. Parent acceptance
does not certify a variant; each variant requires current simulation and review.
