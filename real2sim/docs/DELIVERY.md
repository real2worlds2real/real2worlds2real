# Delivery record

An accepted twin leaves real2sim with one small JSON record, whichever route produced it. Policy development and the
end-to-end chain read this record and the files it pins; they do not need to know how the twin was built.

```bash
# a scene-pipeline session that passed S0-S6 and was frozen
real2sim-harness deliver --session sessions/<name> --output /path/to/<name>.delivery.json
# an accepted task workspace of the agent harness (scene if one was accepted, otherwise the object)
real2sim-harness deliver --run /path/to/run --output /path/to/<name>.delivery.json
# later: are the pinned files still the accepted ones?
real2sim-harness deliver --verify /path/to/<name>.delivery.json
```

The command reads files only. It starts no simulator and no model, and it never overwrites a record. A scene-pipeline session must sit at
`sessions/<name>` of this repository (a symbolic link is enough): its scene graph names the session's files as
`sessions/<name>/...`, and everything that builds the twin later resolves them from here, so `deliver` refuses a
session whose files it cannot find there.

| Field | Meaning |
|---|---|
| `schema` | Native delivery records use `astra_twin_delivery_v1`. End-to-end deliveries with model revisions use `astra_twin_delivery_v2`. |
| `route` | `scene_pipeline` (frozen session) or `task_workspace` (accepted run) |
| `level` | `scene` or `object` |
| `observation_sha256` | Digests of original inputs recorded for a delivered scene. Historical scenes may have no recorded hashes. |
| `task` | The task in words; target, goal and support when the route recorded them; the objects and their roles |
| `scene` | Path, sha256 and format of the delivered USD(Z); absent for a single object |
| `assets` | For an object: the accepted model files with their hashes. For a task-workspace scene: every file of the composed scene package (the scene file only references them) |
| `source` | The session and its configuration directory, or the run |
| `acceptance` | The route's own acceptance record, pinned by hash: the freeze manifest, or `final_scene.json` / `final.json` |
| `statements` | What that record states: metric scale, robot placed, physics calibrated against real motion. `null` means the route does not state it. |
| `recheck` | Commands for checking the frozen scene or auditing a task workspace's retained numerical evidence. Supply the original audit request for the workspace operation. |
| `model_revisions`, `model_revisions_sha256` | A v2 record identifies each target's editable files and the passing receipt for each declared check. The digest covers this complete mapping. |

For v2, each reader checks the receipt's target, revision, method and verdict, then rehashes its retained files.
Keep the original receipt directories along with the model and simulator output. Current workspace scene acceptances
also pin the exact runtime review, report, requested media, numerical evidence, runner settings and declared runtime
dependencies in `final_scene.json`. These pins remain available to a standalone station through the native record.

Historical v1 deliveries from the native tools keep their original pins. A pre-release v1 record with
`model_revisions` must be reissued to verify its checks. For an older workspace scene, inspect the intended runtime
trial and repeat `workspace.accept_scene` to pin it; select and run the matching model check before reacceptance.
For an open end-to-end run with current native acceptance, set `restart_from` to
the final delivery step in its plan, run `astra-e2e plan --run RUN --request PLAN.json`, then run
`astra-e2e step --run RUN --id DELIVERY_STEP --arguments empty.json` with `{}` in `empty.json`.
Use `astra-e2e next --run RUN` to identify and repeat an earlier step if its files are stale. For an already
advanced run, start a new chain with the original observations and replay the reconstruction plan and checks.
The returned delivery path names the new record.

For a task workspace, replay an evidence audit with the run directory and its request file:

```sh
real2sim-harness task call /path/to/run audit_task_evidence --request /path/to/request.json
```

A metric scene with attached observed-motion calibration carries its frozen process receipt and the referenced evidence files. A changed receipt or trial invalidates that calibration.

For a task-workspace scene, `statements.metric_scale` carries the accepted scene bundle's
`frame.scale_verified` value. A verified scale yields `true`, an explicitly unverified scale yields
`false`, and an older bundle without the declaration yields `null`.

The record restates the route's acceptance; it is not a review and adds no claim. A record without a `scene` (a single
object) cannot serve as a policy-development station: compose it into a scene first. Where `statements.robot_placed`
is not true (every task-workspace scene), the twin brings no robot interaction setup; the station's author declares one
for the environment factory and it is recorded as theirs, not as a statement of the twin.
[Worked factories for both routes](../examples/station/README.md).
