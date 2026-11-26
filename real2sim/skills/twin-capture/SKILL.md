---
name: twin-capture
description: Prepare and review scene-capture inputs, task roles, calibrated cameras and metric placement for an calibrated_scene_reconstruction reconstruction through S0 and S1.
---

# Capture and metric placement

For a new capture setting or a calibration/tool capability gap, use
[real2sim-evolve](../real2sim-evolve/SKILL.md) to retrieve experience and validate
an extension. Ordinary camera/scale iteration still follows this skill.

Use one new session per capture state. Keep existing delivered sessions unchanged.
For an explicitly independent reconstruction, use only its original capture,
legal calibration inputs and current workflow documentation. Do not read previous
session notes, fits, models or archived examples to initialize the reconstruction.

Write `configs/<name>/twin.json`, `task.json` and `perception.json` from the supplied
task and visible frames. The background choices are `zone`, `splat`, `mesh` and
`completion`, plus `blender` for an agent-authored surrounding environment.
The Blender route requires `environment_quality_version: 1`; see
[environment modeling](../../docs/scene/blender_environment.md).
With `zone`, `splat` or `blender`, only the support, robot and task objects are
meshes; a surrounding rig, cables and room belong to the background.
For a new full-environment reconstruction, set `environment_quality_version: 1`
in `twin.json`. S3 then requires a current whole-scene quality review, including
materials/texture scale, lighting, contact shadows, occlusion, layout and
captured/novel view consistency. Geometry or a training PSNR alone cannot pass it.

For new task captures set `mechanism_contract: 1` and
`physical_priors_contract: 1` in `task.json`. The modeling agent must declare all
physical links/mechanisms and estimate their material/physical priors, including
ordinary rigid objects and task-support contact properties. See
[task-object iteration](../twin-model/SKILL.md) and
[mechanism/physics contracts](../../docs/scene/object_mechanisms_plan.md).
Capture/metric placement uses legal calibration only; mass/material priors do
not supply task-object or desk scale targets.

Choose scale from available calibration: `board`, `robot_joints`, `robot_parts`,
then `prior` when no stronger source exists. Board specifications and robot URDF
dimensions are calibration inputs. Measured task-object or desk dimensions,
`references`, `task.json.sizes` and development evaluation results are forbidden
reconstruction inputs. Record scale uncertainty and the independent scale check.

For an explicitly no-joint Dexmate capture, select `robot_visual` and provide
`robot_visual.json` with the CAD identity and `state_input: "none"`. CAD home is
an initialization hypothesis. Never load supplied joint snapshots or a previous
session's pose, camera or geometry. This route remains under real-capture
validation: preserve failed attempts and abstain when its shared evidence gates
fail. See [visual robot and support iteration](../../docs/scene/visual_robot_support.md).

Low COLMAP reprojection error is not a calibration uncertainty bound. When
independently observed shapes or support geometry conflict across views, inspect
the retained self-calibration rounds before repeatedly changing object geometry.
`tools/run_splat.sh -m calibrated_scene_reconstruction.scene.background.splat.capture sessions/<name>
--calibration-audit --sample-offset <offset>` fits another raw-frame sampling in
QA and records comparison with the current camera. The offset must be smaller
than the calibration stride. This audit uses source image features only; do not
feed task dimensions, desired contact locations or task success into calibration.
Keep the current camera until the alternative has independent source review;
adopting another camera requires fresh poses, scale and dependent geometry
checks. A second sampling fit is still not a complete uncertainty estimate.

Review robot labels on full-resolution RGB: dark structural parts can be
omitted, and windows can contain a reflected or separate robot-shaped region.
Revise source-derived role prompts and acquire new mask suffixes, preserving
earlier observations. Independent background labels can resolve indirect-image
ambiguity under the existing bounded-unknown policy; review that observation
scope explicitly. Never fill or delete pixels from a projected CAD mismatch.
If only one initialization passes, continue the other independent lineages on
the same observations with `visual_render_fit --resume-from` and a bounded
iteration budget. The optional contour-distance term can improve optimization;
all original image, held-out depth and scale-spread checks still apply. Changed
observations require a new prepared round; never resume across that change.

Run `pipeline/twin.sh <name> data/<capture.mp4> --robots <robot>` through the draft.
Review `capture/color.json` and `capture/exposure/` before interpreting bright
parts. HLG/PQ video must use the shared color-managed reader, not direct 8-bit
OpenCV extraction. Inspect canonical and original-HDR exposure alternates at
full task-part resolution. Resolve each exposure review with visible evidence;
white masks alone cannot establish connectivity, holes or material behavior.
If detail remains clipped/ambiguous, inspect another source frame or retain the
uncertainty. Never darken a clipped 8-bit image and claim recovered evidence.
S0 binds the source-bound exposure review. Changed conversions require new
capture evidence and affected labels/fits to be reviewed again. Keep prior and
frozen captures unchanged. Brackets are one camera/frame, never extra views.
Inspect `parsing/overlay.png` and `qa_autograph.png`; correct the task target/goal,
their instance IDs and the support in `scene_graph.json`. Reject false robot-hand
detections and duplicate supports. An overlapping reference-image seed is not an
independent object measurement: obtain distinguishing calibrated walk views.
For workspace-only captures, review the graph's delivery scope before geometry
or native QA. Keep the task bodies, required props and their support; preserve
excluded room labels in a scope-review record. Room items used to register
cameras do not automatically become reconstruction targets. Hiding a room mesh
after runtime construction does not remove its build-time dependency.
Check that current floor points lie at z=0 after changes in scale. Correct a stale
vertical origin from that capture's floor plane while retaining the reviewed
gravity/yaw; never set camera or support height from a measured task dimension.

Inspect supplementary plane masks as well as the primary photo. A large floor
proposal overlapping independently labelled tabletop is ambiguous floor evidence,
not an observed ground plane. `placement_planes.json` can supply current-video
floor descriptions and support exclusions; preserve old masks before revisions.
For robot calibration, inspect occluder interiors: observed adhesive marks or a
reference sheet on a solid tabletop still occlude the robot behind that surface.
Use independently segmented descriptions in a new `occluder_suffix` when those
surface patches are omitted. Do not fill arbitrary holes, infer unknown labels
from CAD mismatch, or discard views simply because their fit is poor.

Continue with `--from poses --to support`. Review registration, model count,
coverage, board/robot overlays, support plane and every scale check. `S0` is the
capture/pose review; `S1` is the scale and metric-placement review. Reconstructed
camera poses must exist before a task model can pass its accuracy iteration.
If support alignment turns the world, review the redrafted graph and the resulting
camera/support overlays before proceeding.

For extra object photos, closeups or manipulation videos, inspect the media and
write `configs/<session>/inputs.json` with roles, identity, state, camera groups,
selected intervals/frames and legal scale source. Run
`python -m calibrated_scene_reconstruction.scene.capture.supplementary sessions/<session> --prepare`.
Keep moving-object intervals outside static scene reconstruction. Extra cameras
do not inherit primary intrinsics or metric poses. Inspect prepared role-specific
boards with the agent's `object_inputs` tool; calibrate scale through the usual
S1 route using metric evidence from the capture.

`--pose-workers N` optionally parallelizes CPU nearest-neighbour queries (default
1), retaining the same calibration objective and thresholds. For an explicit
resume or migration control, `--reuse-camera-model` uses this session's existing
COLMAP model and undistorted images; missing cache is an error. The corresponding
JSON fields are `pose_workers` and `reuse_camera_model`, and reused camera files
are fingerprinted. An independent reconstruction starts a new session without
copied camera caches and leaves reuse disabled.

Use `python -m calibrated_scene_reconstruction.agent.tool schemas` for the JSON contracts and `stage_s0`,
`stage_s1`, `record` and `status` for tool-driven execution. Stage success means
execution completed; it does not approve the review. Review records must be newer
than their stage run. See [the pipeline reference](../../docs/twin_pipeline.md)
for the full stage order and options.
