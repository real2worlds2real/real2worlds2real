# Plan a reconstruction

Start at the repository root and source `env.sh`. Give `astra-e2e init` a task and its observations. The agent chooses operations after inspecting the inputs. An object or scene can be the final deliverable. A policy handover additionally needs the [station and campaign settings](chain.md).

```json
{
  "name": "open_container",
  "task": "Reconstruct the container and test opening its lid.",
  "observations": ["/captures/container.mp4"],
  "goal": "scene",
  "require": ["metric_scale"]
}
```

Use `require` for delivery claims that must have positive evidence. Available claims are `metric_scale`, `robot_placed` and `real_physics_calibrated`. Omit claims the task does not need. An estimated or unspecified value cannot satisfy a required claim. To require a family alongside the twin add e.g. `"cousins": 10`.

```sh
astra-e2e init --run /work/container --request request.json
astra-e2e catalog
astra-e2e next --run /work/container
astra-e2e plan --run /work/container --request plan.json
astra-e2e step --run /work/container --id views --arguments views.json
```

## Shared model revisions

Use `object.revise` for each participant or precise support and `environment.revise` for the room. Bind the revision step to the absolute working source directory. List the entry and the relative files needed to reconstruct the candidate. Include its geometry, material and physical declarations. An initial cap revision might use

```json
{"target":"cap","parent":null,"entry":"model.py","files":["model.py"],"checks":["object.test/contact"]}
```

The result supplies a revision digest. Inspect `next` for current heads and their uses. Edit the working source and submit a new revision with that digest as `parent`. Earlier snapshots survive working-file edits. Declared checks carry forward and need fresh evidence after a revision.

Bind build, fit and test steps to a native task or metric session. Their arguments are `target`, `revision`, `method` and `options`. Recall with `{"module":"object"}` or `{"module":"environment"}` returns shared instructions and method-to-tool bindings. Supply the native tool's arguments in `options`. The candidate supplies the script or manifest, and the build supplies the native attempt or scene ID.

| Operation | Execution |
|---|---|
| `object.from_template` | Start from a built-in rigid kit. Bind the recipe directory and supply `target`, `parent` and `recipe`. The recipe names a fitted kit and source-estimated `parameters`, with optional `fixed` values. The editable Blender model uses meters and +Z up. The calibrated worker keeps the kit’s native −Y front. The workspace worker rotates the model to +Y front. For a custom kit or another frame, use `object.revise` with an explicit `build_scene()`. Repeat source, material and collision checks. The original native kit renderer remains available for its specialized materials and collision recipes. |
| `object.build` | `blender` takes `plan` and `change_summary`. A rigid source can define workspace `build_scene()` or metric `PARAMS` and `build(params)`. For linked or flexible metric models, provide a `build_scene()` binding their parts, constraints and materials to workspace metadata. `metric_blender` and `support_blender` read matching source in the metric session. Install the active candidate in a mutable native configuration when needed. The adapter verifies its bytes. |
| `object.fit` | `metric` and `support` use the calibrated fitters. `posed` takes a workspace `request_path` for the observation recipe. Omit the model from the recipe. The adapter supplies the current program or NPZ surface. Use quality targets for a required projection check. |
| `object.adopt_fit` | Supply `target`, `revision` and the originating `fit_step`, bound to the same source as that fit. Inspect source overlays before adopting. A bounded fit updates the literal initial parameters and creates a new editable revision. Fixed values and placement retain their own evidence. |
| `object.test` | Select the mechanism and requested runtime. For `contact` and `deformable` bind the native request's `source_attempt_id` to the current build. `metric_task` runs the calibrated session's assembled task check. |
| `environment.build` | Choose `metric` for the configured native room stage, `blender` for authored geometry or `gaussian` for a field request. A room program taking parameters and context can use a `settings` member containing those inputs for a workspace build. |
| `environment.assemble` and `environment.test` | Assemble the native scene manifest and execute its configured simulator. Environment revision `bindings` can select members for `build/blender`, `build/gaussian`, `build/metric` and `assemble/scene`. A manifest background can name `from_model_step` in place of its path to consume that revision's built layer. |

Use native tools directly for specialist tasks, template fallback and legacy plans. When a plan includes model revisions, native delivery must include each current target. The delivery record exposes its editable inputs and the receipts for its required checks in `model_revisions`. The harness and the downstream station verify those files against their recorded hashes, including the report and media used to accept a workspace scene. See the [delivery contract](../../real2sim/docs/DELIVERY.md) for compatibility with earlier records. Workspace workers still apply their own file-access rules and dependency limits. A source-bound geometry check supplies no simulator or scale approval.

## Native tools and handoffs

A plan contains a reason and ordered steps. Each step binds an operation to an absolute native source directory. This example assumes the metric capture session and the destination scene task already exist.

```json
{
  "why": "Use the registered walk-around for placement and test the hinge in the composed scene.",
  "steps": [
    {"id":"views","operation":"exchange.views","source":"/repo/sessions/container","why":"Supply registered pixels to the object worker"},
    {"id":"place","operation":"exchange.placements","source":"/repo/sessions/container","why":"Place the reviewed hinge bundle using its fitted origin"},
    {"id":"assemble","operation":"workspace.assemble_scene","source":"/work/scene","why":"Preserve bodies, joints and colliders"},
    {"id":"simulate","operation":"workspace.run_scene_simulation","source":"/work/scene","why":"Exercise the assembled interaction"},
    {"id":"review","operation":"workspace.accept_scene","source":"/work/scene","why":"Inspect the native evidence"},
    {"id":"deliver","operation":"delivery.workspace","source":"/work/scene","why":"Pin the inspected scene and its acceptance"}
  ]
}
```

For `views.json` use e.g. `{"task":"/work/object","frames":[{"name":"0_00006.jpg"}],"verified":true}`. Use names from the current capture. The result names the actual manifest inside the object's native workspace. The agent can extend the plan with a fit or model revision after inspecting it. Supply returned paths and IDs to dependent calls. `workspace.init` takes the native runs parent as its source and returns the new task path. A plan can be revised as soon as a generated path becomes known.

| Need | Operations |
|---|---|
| Registered cameras, robot scale and support | Select applicable `session.stage_s*`, `session.support_*` and review tools. The metric session keeps its S0–S6, OBJ and SUP requirements. |
| Geometry, appearance and interaction | Select `session.object_*` and/or `workspace.*` tools for the object's parts and material. Fit and inspect against current observations. Use the user's simulator and its runtime checks. |
| Share pose or geometry | `exchange.views` retains images and optional mask, ignore mask and depth channels. It converts OpenCV camera centres and axes into world-to-camera matrices. `exchange.geometry` retains fitted mesh topology, model-to-world pose and available editable sources. |
| Place an accepted participant | `exchange.asset` rechecks a native task object and packages its complete runtime bundle. `exchange.placements` takes the fitted pose, bundle path and explicit rigid `bundle_to_model` registration for each object. It returns the manifest path for `workspace.assemble_scene`. Scene simulation and review follow assembly. |
| Observed-motion calibration | Select `session.physics_propose/run/review/attach` as needed, or the workspace observation and replay tools. A metric-session delivery reporting real-motion calibration pins its frozen process receipt and referenced evidence. |
| Background and cousins | Choose mesh or captured backgrounds according to the request. `workspace.cousins` and `family.variant` use the existing family planner and independent child tasks. Use `family.complete` to recheck the native participants. A requested count is checked against completed families of the delivered parent. |
| Delivery | End with `delivery.session` for a frozen metric scene or `delivery.workspace` for an accepted object/scene. The adapter identifies the native acceptance owner. Operations from both implementations can precede it. |

Each step records arguments, results and failures. Pin additional **files** that affected the decision using repeated `--evidence /absolute/file`. On revision the common current prefix is retained. Set `restart_from` to a step ID to retry it with different arguments. Changing the plan or a pinned file makes the affected suffix eligible for execution again. Use native review to evaluate appearance, behavior and task success. The agent can inspect the receipts returned by `next`.

After the last step run `astra-e2e advance --run /work/container`. The chain rechecks source identity, native acceptance, and the delivery level and required statements. For `goal: real_preparation`, follow `next` for the task binding, campaign, sealed evaluation and handover. Once a campaign exists, `astra-e2e campaign-step --run RUN` invokes one decision of the configured station agent. Use `astra-e2e verify --run RUN` to check a delivered twin again. The station remains responsible for its budgets and sealed partition. Real validation is operator-run.

## Skills and experience

Use `astra-e2e recall --run RUN --request query.json` before planning. A query can contain `query`, exact-match `context`, `skills_root` for personal skills, `evolution_root` for native reconstruction records, and `policy_tags` for station cards. For a native evolution lookup, give at least one context field. Add `--state /work/shared-methods` to recall or learn across runs. CLI memory defaults to the individual run.

After inspecting development evidence, use `astra-e2e learn --run RUN --state /work/shared-methods --request experience.json`. The entry includes `id`, `problem`, `context`, `procedure`, `applicability`, `limitations`, `negative_results`, `transfer_rationale`, `evidence` and `validation_runs`. A failed approach can be stored as a candidate. Transfer requires current completed validation runs on independent sources. Validated entries written through the chain also produce a versioned `SKILL.md`. Retrieval rechecks evidence. Withheld dimensions and sealed test results stay outside the development library.

The same operations are available through the [agent JSON service](unified_harness.md) as `chain_init`, `chain_next`, `chain_verify`, `chain_catalog`, `chain_plan`, `chain_step`, `chain_advance`, `chain_campaign_step`, `chain_recall` and `chain_learn`. The service uses its configured state directory for memory. Supply `--execute` to run native tools. Reuse a request ID when retrying a lost response.

Requests that explicitly pin the legacy `route` keep the original delivery contract. Existing native workspaces, freezes and station bindings remain readable.

For module-scoped experience, list semantic operations in `required_capabilities` and cite their current outputs. Promotion requires disjoint observation sets and evidence for every listed operation in two completed runs. A recall request with `module` still searches shared experience in `reconstruction`. In `validated_skills`, each `validated_scopes` entry identifies a pair of independent runs and the operations, methods, native tools and recorded runtime versions they both exercised for that module. Records without module evidence remain discoverable through shared experience. Validate the procedure on the current candidate and requested simulator before reuse.
