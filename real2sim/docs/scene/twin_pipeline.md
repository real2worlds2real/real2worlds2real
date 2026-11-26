# Twin pipeline: agent in the loop, recursive refinement

Use intrinsics and distortion coefficients calibrated for the camera, resolution and crop of your capture. Supply your `camera.json` for the session or run the board calibration step on your own recording. For a recorded robot pose, supply the joint readout matched to your capture.

How a twin is built from a capture. Cousins are out of scope here. [pipeline.md](pipeline.md) lists the modules
stage by stage; [object_review.md](object_review.md) holds the review's fixes and checks. This document fixes the
**process** those modules run in: the stages, the two options, and the loop every stage goes through.

The twin serves two goals, in this order:

- **a. the task**: the manipulated objects' size, shape and place -- on their support and relative to each other --
  as accurate as the capture allows;
- **b. the environment**: the room around the task as real as the capture allows.

## 1. One session per capture state

A session is one capture state: one day's arrangement of the scene, filmed in one or more videos (or photographed).
Captures of different days are different sessions, even of the same room: an object moved between them, and a twin
mixing them is wrong. The options below are settings of the session, not new sessions. A splat twin and a mesh twin
of the same capture share everything up to the environment stage and differ only there and after.

## 2. Inputs

| input | required | where |
| --- | --- | --- |
| the capture: a walk-around video (or several of the same state), or a photo (+ views) | yes | `pipeline/run_session.sh <name> <video \| photo>` |
| the robot's named joint command snapshot near capture | uncertain prior for the `robot_joints` scale | `configs/<name>/robot_state.json` (joint angles only) |
| task information | no | `configs/<name>/task.json` |
| the options | no (defaults below) | `configs/<name>/twin.json` |

**Task information** (`task.json`, all fields optional):

```jsonc
{
  "task": "insert the peg into the block's hole",        // what the robot will do, in words
  "objects": {                                            // the manipulated objects
    "peg":        {"prompt": "white peg", "role": "target", "near": [x, y]},   // near: about where, if known
    "hole_block": {"prompt": "white block with a hole", "role": "goal"}
  },
  "support": "the desk inside the aluminium frame"
}
```

Without it the agent decides: the things on the support in front of the robot are the task's objects, the task the
most likely one for them. It writes its decision into `task.json` with `"inferred": true` and lists it in the final
report for the user to confirm. Measured object sizes are not deployment inputs or acceptance criteria.
Configuration fields `references` and `task.json.sizes` are rejected before stages run. Independent development
evaluation is kept outside the deployment package.

**Options** (`twin.json`):

```jsonc
{
  "background": "mesh",            // splat | mesh | completion
  "scale": "robot_joints",         // board | robot_joints | robot_parts | prior
  "scale_check": "board",    // the independent source the scale is checked against (default: the best other)
  "task_look": "frames"            // kit (the fitted kit's flat colour, default) | frames (recon.object_texture)
}
```

`task_look: frames`: the task zone stage unwraps each fitted kit and bakes its texture from the frames its silhouettes
were found in (front-most texels inside the eroded silhouette, weighted by obliqueness and sharpness, the samples
nearest their median averaged; what no frame saw filled from round it); `qa/object_texture/<name>.jpg` shows the
frames beside the textured mesh, the close-ups show it in the twin.

| background | what lies beyond the task zone | needs |
| --- | --- | --- |
| `splat` | a Gaussian splat of the walk (NuRec), the task zone cut out of it; walls and floor as colliders | a video; COLMAP; 3DGRUT |
| `mesh` | walls, floor and ceiling from the walk frames' poses and depths; the room's objects reconstructed from them; nothing invented where no frame looked | a video; COLMAP |
| `completion` | the photo's room completed by a panorama model, the completion's objects reconstructed | a photo |

Default: `mesh` with a video, `completion` with a photo alone.

| scale | where the metre comes from | default check |
| --- | --- | --- |
| `board` | a calibration board's corners, triangulated over the views | another available robot calibration |
| `robot_joints` | legal robot CAD fitted to current images, jointly refining scale, root pose and uncertain commanded joints; abstains if evidence or uncertainty bounds fail | robot parts (diagnostic only) |
| `robot_visual` | CAD surfaces fitted while inferring articulation, without loading a supplied joint configuration; experimental, abstains on insufficient evidence | robot parts (diagnostic only) |
| `robot_parts` | parts of the robot of known width | board or joint calibration |
| `prior` | the depth models' own metric scale (MoGe-2, Depth Anything 3) | none: the report marks the scale unverified |

Default: `board` when a board config exists, else `robot_joints` when there is a readout, else `robot_parts`.
`prior` can be selected explicitly. Whatever sets the scale, the splat (if any) and the posed frames are brought into the same metric frame in
the same stage.

The explicit `robot_visual` route defaults to agent Blender support iteration.
Other routes can set `support_route: "agent_blender"` to include their table.
See [visual scale and task-support contracts](visual_robot_support.md) for inputs,
failure criteria, `SUP` reviews and the static collision surface.

`robot_joints` treats `robot_state.json` as an uncertain command snapshot, not
exact execution telemetry. It now shares the image/CAD backend with
`robot_visual`; the latter still never reads supplied joints. Optional
`robot_calibration.json` specifies the legal robot and uncertainty assumptions:

```json
{
  "robot": "vega_1p_sharpa_arm",
  "joint_uncertainty": {
    "sigma_deg": 3.0,
    "max_delta_deg": 12.0,
    "weight": 0.002,
    "floor_sigma_m": 0.02,
    "max_floor_offset_m": 0.06
  }
}
```

These defaults are modelling assumptions, not measured hardware accuracy.
`per_joint` may override `sigma_deg` and `max_delta_deg` by CAD joint name.
Named non-finger commands must be finite and within CAD limits. Missing or
ambiguous commands abstain; finger joints are explicitly unestimated and their
pixels do not supply calibration scale. Angular units are never applied to
prismatic joints. A bounded root-height offset models floor/contact error
separately from articulation. The backend compares nominal, plus-sigma and
minus-sigma starts and a fixed-command baseline. An execution-uncertainty bound
being reached, a joint outside CAD limits, failed image/sparse evidence, or
ambiguous scale blocks application. A legitimate physical joint stop alone is
not an uncertainty failure.
At least two independent refined starts must pass and agree in scale within
5%. No task size or task STL enters these checks. A static pose model does not
claim to estimate temporal servo lag or actual execution telemetry; inconsistent
video evidence remains a failure.

`splat/robot_joint_fit.json` records commanded/refined joint differences, the
fixed-command baseline, uncertainty assumptions and fingerprinted evidence.
`parsing/robot_visual_state.json` stores the applied inferred presentation pose;
the supplied command file is preserved. Unfrozen joint-scale sessions require
this verified anchor for S1/S6. Previously frozen deliveries retain their
historical review status and are not silently regenerated or recertified.
Command bounds may be revised in a new experimental round when current image
evidence reaches an artificial bound. This tests a different uncertainty model;
it does not change image/surface/ambiguity gates or establish measured servo
accuracy. A large fitted command correction may also indicate timing, zero-frame
or model mismatch, rather than mechanical following error alone.
For further convergence with unchanged evidence, `visual_render_fit --resume-from`
preserves each separate starting lineage and the fixed-command baseline. Resume
requires the same capture, observations, legal CAD and exact prior bytes; joint
and no-joint modes cannot seed one another. Changed uncertainty starts afresh.
New no-joint rounds use five CAD initialization poses: inferred, CAD home and
their quarter/midpoint/three-quarter interpolations. All candidates, including
failures, remain in the assessment; adding starts does not relax image/surface
or scale-ambiguity criteria. Older three-start packages retain their original
seed lists when assessed.
An initializer is copied into its immutable round input before fitting. The
result binds that snapshot, so publishing a later result to the same canonical
JSON filename cannot overwrite its own input evidence.

Robot and support fitting distinguish unobserved rectification padding from
negative image labels. Capture writes per-frame `_valid.png` maps and
`splat/rectification.json`; older captures can derive validity from their own
camera calibration. Validity follows OpenCV's complete remap footprint and never
classifies pixels by darkness. Camera metadata and validity maps enter input
fingerprints. Unknown legacy calibration retains the full image rather than
inventing missing pixels. The direct triangle-rendering backend uses dense
observed pixels and independent sparse surfaces even when occlusion leaves too
few exposed contour-proxy samples; the older contour-only backend still requires
those samples. Acceptance thresholds are unchanged. GS training also consumes
the geometric validity maps when invoked with a session.

Both robot routes respect `observations.robot_suffix`, `occluder_suffix` and
optional `background_suffix` in their calibration config. The background
protocol is experimental and requires independently labelled masks. The pipeline
acquires configured prompt/suffix revisions through `scene.capture.calibration_masks`;
it does not infer negative labels from fitted CAD. Raw identity/occlusion rejection
reasons exclude a frame from both image fitting and actual sparse-track votes.
`--reuse` accepts masks only when prompts, threshold, zone-overlap requirement
and dilation match their metadata. Changed settings require a new suffix so
earlier evidence remains intact. Completed workers and the actual assessment
rules have separate archived source fingerprints.

## 3. Stages

Task feasibility in every reconstruction route means **object-model feasibility**:
the reconstructed shapes support insertion, stacking, stable contact or meshing.
Robot grasping, IK, reachability and autonomous execution are not delivery
prerequisites. The default `placements` stage prepares a rendering pose from the
capture, without IK. Robot geometry and its placement relative to the table remain
part of reconstruction review. Optional robot data-generation stages require
`--robot-execution`; their outcomes do not define object-model task success.

Rendering placements marked `scope: presentation_only` retain the calibrated
scene during export and tours; they do not run robot-operation clearance that
could move furniture. `check_clearance.py --background zone` independently
checks task objects at rest on the actual task-zone support.

Task-relevant mechanisms and physical-property priors follow
[object_mechanisms_plan.md](object_mechanisms_plan.md). New captures require agent mechanism and physical-prior declarations
before fitting. S5 binds native results and nominal/low/high evidence to current
geometry, parameters and check code. S6 requires every applicable prior gate and
object/support loop. After delivery, `python -m calibrated_scene_reconstruction.scene.twin.freeze SESSION`
preserves accepted files; `--verify` checks them without rewriting the record.

| stage | does | depends on the options |
| --- | --- | --- |
| S0 capture | key frames, the photo frame; COLMAP poses of the walk (for `splat` and `mesh`) | background |
| S1 scale and frame | the metre, the floor, the support's top; the splat and the posed frames in that frame | scale |
| S2 task zone | the support's geometry; each task object's shape, size, place and colour; marks on the support | none |
| S3 environment | structure and room objects (`mesh`, `completion`) or the splat (`splat`) | background |
| S4 appearance | materials and colours of the support, the task objects, (`mesh`, `completion`) the room | background |
| S5 whole scene | penetration, physics at rest, the twin against every frame from where it was filmed | none |
| S6 delivery | the tour video, the single USDZ, opened alone in Isaac Sim | background (NuRec inside or not) |

S2 comes before S3 on purpose: the task zone is the same for every background, and a finding in S3 never changes it.

In `pipeline/run_session.sh` (a session with `twin.json`; `--from STAGE` resumes):

| stage | pipeline stages | gate |
| --- | --- | --- |
| S0 | capture (key frames), `calib` (the video's camera self-calibrated when there is no `camera.json`), `poses` (COLMAP and observed floor/support planes for robot anchors); the articulated image backend does not require Isaac robot-cloud generation | S0 |
| S1 | `perceive`, `multiview` (provisional depth coordinates for robot anchors), `graph`, `anchor` (`scene.capture.visual_render_pipeline` prepare/fit/assess for `robot_joints` and `robot_visual`; then `scene.capture.posed_frames select`) | S1 |
| S2 | `support` (`recon.splat_masks --role support / occluders`, `recon.support_carve`), `objects`, `align`, `audit`, `task_zone` (`recon.object_fit --from-config`, `recon.support_marks --from-config`, `apps/object_closeups.py`) | S2 |
| S3 | completion: `panorama`, `walls`, `completion_review`, `room_objects`; splat: `splat_masks`, `splat_train`, `splat_export`, `splat_review` | S3 |
| S4 | `layout`, `irradiance`, `albedo`, `surfaces`, `materials` | S4 |
| S5 | `clearance`, `object_review` (`apps/review_objects.py`, `apps/object_closeups.py --all`) | S5 |
| S6 | `deliver` (`apps/scene_tour.py`, `apps/export_usd.py`, `apps/pack_usdz.py`, `apps/view_usd.py`) | S6 |

A gate stops the run until `qa/review/<stage>_r<k>.json` says `"pass": true`; the message names what to review and
the stage to rerun from. The scale `robot_joints` needs no trained splat. Current
registered images and independently labelled sparse tracks supply evidence;
the old fixed-articulation trimmed ICP remains a diagnostic utility, rather than
the default deployment anchor. Robot-only routes do not load board scale.

## 4. The loop (every stage)

```
        +-------------------------------------------------------------+
        v                                                             |
  1 produce --> 2 automatic checks --> 3 agent review (checklist) --> 4 attribute each finding
                                                                      |   a config     -> fix, rerun this stage
                                                                      |   b upstream   -> rerun from that stage
                                                                      |   c method     -> change the method, rerun
                                                                      |   d check      -> add the check first, then a/b/c
                                                                      v
                                       5 pass: all checks pass, every checklist item has a verdict
                                       6 or: a known limitation, with its evidence
```

1. **Produce**: the stage's modules, from the session's configs only.
2. **Automatic checks**: each a number and a pass / fail against a stated threshold (section 5).
3. **Agent review**: the stage's checklist (section 5), item by item and object by object -- every object, not
   only the flagged ones. Each item gets a verdict: right, or a finding with its evidence (the frame and what it
   shows).
4. **Attribute** each finding to the layer at fault:
   - **a. config**: an input was wrong (a prompt took the wrong thing, a place was off, a decision of the agent's).
     Fix it in the config, with its evidence; rerun the stage.
   - **b. upstream**: the cause lies in an earlier stage (the scale makes every size wrong). Rerun from there.
   - **c. method**: the method cannot do it (SAM 3D completes a 30 px object from its prior). Change the method or add
     an opt-in one -- generic, never for this scene -- and rerun.
   - **d. check**: the agent saw it and no automatic check did (an upside-down sofa passed support, size and
     overlap). First add or fix the automatic check and confirm it flags the case, then fix the cause by a, b or c.
     The next capture then does not depend on the agent noticing.
5. **Pass** when every automatic check passes and every checklist item has a verdict without an open finding.
6. What the capture cannot tell (a hole no frame looks into, a colour every frame clips) is a **known limitation**:
   recorded with its evidence and with what the twin uses instead (an explicit unmeasured cavity-depth hypothesis; the colour's upper
   bound). It ends the loop for that item; the report lists it.

Each round is recorded in `sessions/<name>/qa/review/<stage>_r<k>.json`:

```jsonc
{"stage": "S2", "round": 1,
 "checks": {"peg": {"iou_median": 0.87, "frames": 19, "sectors": 8, "ok": true}, "...": {}},
 "review": {"peg": {"shape": "right", "orientation": "right", "size": "right", "place": "right", "colour": "right",
                    "surroundings": "finding: an L of blue tape at the block's corner"}},
 "findings": [{"item": "tape", "what": "missing", "evidence": "frames 0_00150, 0_00720", "layer": "a",
               "action": "support mark carved (S2)", "rerun_from": "S2"}],
 "limitations": [{"item": "hole_block", "what": "cavity floor not seen", "uses": "explicit unmeasured depth hypothesis, unchanged by task success"}],
 "pass": false}
```

Rules:

- **Every change goes through the pipeline.** A fix is a config entry with its evidence (`scene_graph.json` fixes,
  `task.json`, `twin.json`, the stage configs), applied by the stage that owns it. No script outside the pipeline
  edits a session's results or configs, and nothing is copied from another session.
- **A decision the agent makes twice becomes part of the pipeline**: a stage option, a default, or an automatic
  check.
- **Rerun from the stage that changed**, not from wherever is convenient: everything after it is rebuilt.
- **The stop condition is the record, not the agent's impression**: a stage passes when its record says so.

## 5. Checks and checklists per stage

### S0 capture

For small objects missed in the full capture image, `perception.json` can add
`regions`, for example:

```json
{"prompts": ["floor", "wooden desk", "humanoid robot"],
 "regions": [{"box": [1226, 639, 1301, 702], "prompt": "white gear",
              "label": "large white gear", "threshold": 0.2}]}
```

Each box is integer `[x0, y0, x1, y1]` in the current `capture/rgb.png`, after any
undistortion; choose it by inspecting that image. SAM runs on the crop and its
mask and box return to full-image coordinates in `parsing/`. `label` defaults to
`prompt`; the region threshold defaults to the label's entry in `thresholds`,
then 0.4. All detected instances in a region are kept, so review the overlay and
instance IDs. A region does not provide a mask, metric size or camera pose. Avoid
querying the same object in both `prompts` and a region unless duplicates are
intended. Changing regions requires rerunning from `perceive` and reviewing the
new graph IDs. This option does not change video frame selection.

- Checks: share of the frames COLMAP registers (>= 80 %); the task zone seen from >= 6 of 8 directions; one capture
  state (one day's videos).
- Checklist: the videos are of one arrangement (the objects stand where they stand in every video); the robot's
  readout belongs to this capture.

### S1 scale and frame

If the draft world selects the wrong floor-height layer, `splat.json` can set
`"align": {"photo_planes": true}`. The poses stage then selects floor and support
heights from reliable COLMAP tracks actually observed inside the reference
photo's floor mask and the task support's `faces.top` masks (at least 30 tracks
per surface), levels against that floor and puts it at z=0. Review those mask
IDs first. Without a board, the graph's draft support height provides only the
initial scale; `robot_joints` still requires the subsequent S1 robot anchor.
This option reads no measured support dimensions. Rerun `poses`; for an existing
model from this same capture, `--reuse-camera-model` keeps its registered cameras
and images while recomputing placement. Missing labelled tracks fail explicitly.

- Checks: the scale's residual (board RMS or the robot fit's median point distance);
  the floor plane fits the floor's points within 1 cm; the support top level. The independent calibration check
  (`scale_check`) is reported with its deviation. Inspect large discrepancies against the captured evidence for
  each calibration source; a source contradicted by the frames is a finding. Measured task-object dimensions
  are never read, compared or reported by the pipeline or used for gate acceptance. Development-only dimensional
  evaluation runs separately after the reconstruction products are frozen.
- Checklist: the robot (or the board) overlaid on the frames lands on it; the support's box overlaid on the frames
  lands on its edges.

### S2 task zone (goal a)

Before fitting a Blender object, inspect every cached observation for instance
identity. If SAM selected a robot hand, a different object or an unusable
occluded fragment, record an explicit reason under that object's
`fit.observation_rejections`, for example
`{"0_00420.jpg": "Robot hand, not the gear"}`. Rerun
`calibrated_scene_reconstruction.scene.objects.object_fit SESSION --from-config --observations-only --reuse`
to apply the review to that session's cached observations. Frame names must
exist and reasons must be nonempty. This review must not select observations
by model fit error. Preserve the original acquisitions and review images; new
Blender acquisitions also retain `<object>.acquired.npz`. The final fit cache
contains the reviewed set after triangulation checks, and each Blender round
fingerprints it. Changing observations invalidates earlier rounds; revise and
fit a new round rather than presenting it as a comparison on identical inputs.

- Checks, per task object:
  - its silhouette in the frames against the twin's projected outline: IoU median >= 0.85 over >= 8 frames from
    >= 4 directions;
  - its size within its class's range and its agreement with observed silhouettes; a deployed capture has no
    measured object size (see S1);
  - it rests on its support (physics) and sits on it (no gap, no penetration);
  - for the support: its box against its edges in the frames; its top covered by its visible top.
- Checklist, per task object, in its close-ups (the twin rendered with the cameras of the walk's nearest frames from 8
  directions, beside them):
  - shape: holes, handles, surface details, a base plate -- what the thing is, not only its box;
  - orientation; size; place on the frames (the outline);
  - colour;
  - what lies beside it that the twin lacks (tape, labels, cables on the support).
- Target: sizes within 5 mm, places within 1 cm (by the reprojection over the frames).
- Default method: the agent writes a Blender model, fits it to metric silhouettes, inspects evidence and revises it; template fitting requires explicit fallback selection. For background assets: a complex one SAM 3D from
  the best frames, or a library model, size and place still from the silhouettes; marks on the support
  (`recon.support_marks`); what no frame sees from the task (a goal's hole fits the target); a rig on the support (a
  camera frame: perception keeps it out as clutter) a `box_frame` kit fitted to whole frames (scene graph `rigs`),
  written as the support's furniture boxes.

### S3 environment (goal b)

- `mesh` and `completion`:
  - checks, per room object: upright (no legs at its top); on something (no gap > 1.2 cm, not sunk); not
    overhanging its support; its collider's top covered by its visible top for what holds things; no intersection;
    size within its class's range; no duplicate (two pieces over one place);
  - checks, structure: walls and floor against the frames' depth from where they were filmed;
  - checklist, every room object in its close-ups: class, shape, orientation, place, size; merged pieces (a desk
    with its monitor); obvious things missing; walls' and floor's place.
- `splat`:
  - checks: held-out frames' PSNR; the task zone cut cleanly (no floaters over the support); the splat against the
    walls' colliders;
  - checklist: the splat renders from the robot's views; nothing of the task zone left in the splat.

### S4 appearance

- Checks: each surface's and object's colour against the frames (on the channels that do not clip:
  `recon.frame_colour`); textures at their real size.
- Checklist: the photo against the twin rendered with the photo's camera; the close-ups of S2 and S3 for colour.

### S5 whole scene

- Checks: clearance (no penetration, every dynamic object at rest); the object review's automatic checks over every
  item; the twin against every frame from where it was filmed (depth and outlines).
- Checklist: the overview from the photo camera and from the robot's head; anything S2-S4 did not see.

### S6 delivery

- Checks: every applicable S0-S5 gate and every active OBJ loop has a current passing review; the single USDZ
  opens alone in Isaac Sim with no unresolved asset; NuRec inside it if and only if `splat`; the tour renders.
- Report: the options used, the scale and its independent calibration check, reconstructed task-object dimensions
  and placement with their image evidence, the limitations, and the agent's inferences to confirm. Measured
  task-object dimensions and development-only accuracy results remain outside this pipeline report.

## Default task-object route

Both `pipeline/twin.sh` and `pipeline/run_session.sh` use `agent_blender` for task objects across zone, mesh,
completion and splat backgrounds. Scene drafts assign `fit.model`; old kit-only requests in a new session are
converted into agent-model requests. Missing source stops reconstruction and writes `qa/model_requests.json`.
Blender remains the default modeling tool. If it is unsuitable for a part, the
agent may use a suitable external geometry/CAD tool or native physical
representation, record the reason, and bind generated data through the model
asset contract. Import/assembly in the existing harness does not require every
surface to originate from Blender operators. Final simulation and task checks
always run in Isaac Sim5.1 on the actual delivered representation, including
contacts, articulation and deformables. Other simulators and visual animations
cannot replace those checks. Unavailable tool support must remain explicit.
Observation acquisition initializes an unfitted agent model from its own source
geometry envelope and the registered reference photo's instance mask projected
onto the observed support plane. A legacy single-image mesh and textured layout
are not prerequisites. Missing photo/camera/instance evidence stops acquisition;
the source envelope is a hypothesis, not a measured size or accepted fit.
The same policy applies to the configuration-driven object-fit and Blender pipeline CLIs.

Use `--object-route template` for an explicit fallback, or set `fit.route` to `template` on an individual object.
A saved delivered asset is not rewritten by route conversion. Model-based delivery requires a current passing OBJ
review that matches the exported round and source hash. The complete S6 prerequisite checks still apply.

The iteration uses calibrated image evidence and the existing IoU/local-difference thresholds. Real task-object
dimensions remain exclusively in development evaluation after freezing the result.
## Observation acquisition controls

For visually similar objects whose calibrated poses have already been reviewed,
`fit.identity_from_pose: true` keeps acquisition at each object's pose instead of
reassigning its identity using competing text prompts. `fit.crop_min` specifies
the minimum crop width in pixels (default 320); the actual crop also grows with
projected object size. `fit.exclude_prompts` lists observed occluder classes such
as `robot hand`; these masks reject overlapping candidate instances and require
identity review themselves. These settings do not approve an object or change
the Blender numeric thresholds.

Run `object_fit --observations-only --reuse --refresh-observations --only NAME`
to reacquire one object's masks while keeping its current-session pose start and
leaving other caches unchanged. Archive the previous cache before an intentional
refresh, inspect all new masks, and record semantic rejections before fitting.
A mask acquisition changes a round's input fingerprint and requires a new fit
and review. This controls object observations within the existing registered
frames; it does not change capture-video sampling.

Photo texture projection uses the registered primary camera when posed-frame
cameras exist. Its depth adaptation is computed in memory against the observed
floor and actual camera height; bootstrap depth files and fitting observations
are not rewritten. This keeps surface baking consistent with calibrated scene
placement.

The Blender optimizer retains the lowest-loss numeric-passing candidate it has
evaluated; a later lower aggregate loss cannot replace it with a candidate that
violates an IoU, local-error or frame-count gate. When no candidate passes, the
lowest-loss failure is retained for diagnosis. Optimizer termination is reported
separately from the selected candidate's acceptance metrics.

For a presentation of the captured robot, set
`photo_robot.render_capture_state: true` in the reviewed graph. The tour and USD
export then use this session's fully matched named joint readout and fitted robot
root height. A mismatched robot spec or partial joint audit is an error. Root
clearance and pose uncertainty still need review; using the captured articulation
does not make the calibration exact.

`fit.sdf` can explicitly request collider precision, for example
`{"resolution": 512, "contact_offset": 0.0002, "rest_offset": 0.0}` for fine
contact features. The default remains256/2mm/0. Export validates and records the
request, runtime authors it on the same surface, and OBJ review records bind the
declared precision. This does not change image or task acceptance criteria.
Scene construction reuses an existing World PhysicsScene so its integration
step remains authoritative. Task settling uses the world's actual step duration.

USD and USDZ previews retain fixed linear exposure for `zone` and `splat`
backgrounds after reopening, matching the task-zone tour and closeups.

### Modeled support contact and fixed-camera delivery review

When a verified Blender task support exists, object fitting uses that model's top
face and fixes downward penetration to zero. Provisional local sparse-point planes
remain the fallback before a support model exists. The object round fingerprints
the support mesh, visual artifact and exported support specification; changing
that support makes the object round stale. Run the support iteration before the
final task-object iteration, then inspect settled assembly poses against image-fit
poses. Internal contact agreement is separate from measured-size accuracy.

Object initialization evaluates24 full-circle yaw hypotheses against every cached
observation before bounded local shape/pose fitting. Image-only whole-outline
success still requires separate visible aperture, mounting-hole and construction
review; an unobserved cavity must not be enlarged/deepened just to pass a task.

For an exported USD and a fixed review-camera manifest containing source-image
hashes, calibrated intrinsics, captured poses and explicitly novel offsets:

```bash
tools/run_isaac.sh apps/scene/review_usd_cameras.py \
  sessions/<name>/export/usd/twin_splat/<name>_twin_splat.usd \
  sessions/<name>/qa/background_diagnostics/cameras.json \
  --output sessions/<name>/qa/background_diagnostics/new_review
```

The output directory must be new. The app edits only the in-memory USD session
layer and does not step physics or change task geometry. Each comparison labels
the original, matched captured camera and unpaired novel cameras. Compare blur,
holes, occlusion ghosts and removed detail before approving a background. Novel
views have no paired ground truth; do not report source-photo PSNR for them. These
review views do not change video sampling or training-frame selection.

Generate a reusable review manifest from any session's existing metric cameras:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.review.cameras sessions/<name>
```

By default it samples up to four registered held-out views when training metadata
exists, or existing registered capture views otherwise. `--frames frame.jpg ...`
fixes explicitly reviewed anchors; `--offsets .15 .35` supplies novel display
translations. The manifest fingerprints source images, metric cameras and sparse
geometry. Reuse it across candidates; do not overwrite it. View generation never
changes reconstruction/training frame selection. Source raster aspect ratio is
retained, and the USD review app supports an off-centre principal point; skewed
intrinsics must first be rectified. New review artifacts for frozen deliveries
belong in a derivative session.

For captured-view export diagnostics, score only independently labelled background
pixels intersected with geometric rectification validity:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.review.background_scores \
  sessions/<name> sessions/<name>/qa/background_diagnostics/new_review \
  --output sessions/<name>/qa/background_diagnostics/new_scores
```

The existing `_sam.png` background masks are required by default; another reviewed
revision can be named with `--mask-suffix`. Scores and error panels exclude unknown
padding without classifying dark RGB as missing. Gradient energy diagnoses detail
loss; it is affected by noise, lighting and registration and cannot approve
photorealism. Captured-view scores never stand in for novel-view quality.

GS export now retains `splat/export/full_colmap.ply`, row-aligned
`selection_masks.npz`, engine snapshots and `selection.json`. These record task-zone,
visibility, faint/large, camera-neighbourhood and floater exclusions separately.
Export appearance thresholds `crop.min_opacity` and `crop.max_sigma_m` default to
the previous0.005 and0.5m. Changing them creates an export candidate for image
review, not a change to task dimensions or geometry acceptance. Free-space checks
are evaluated only on rows surviving earlier rules; restoring excluded rows in an
ablation does not give them a passing free-space result.

`review_usd_cameras.py --background-field file.nurec` can compare another NuRec
payload in the same coordinate frame using the exact same task scene and cameras.
The override exists only in the USD session layer. The review fingerprints the
resolved external payloads and textures as well as the USD and raw source images.

### Whole-scene photorealism gate for new environments

Set `environment_quality_version: 1` in a new full-environment `twin.json`.
This policy applies to GS, agent-Blender and hybrid environment twins/cousins.
It leaves older frozen deliveries under their recorded policy.

Before recording a passingS3, write an agent review request and run:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.review.environment \
  sessions/<name> environment_request.json
```

The request contains `kind: twin|cousin`, `backend: gs|blender|hybrid`, the
session's actual `render_review` JSON from `review_usd_cameras.py`, a nonempty
`candidate_inputs` list binding the current assets/configuration, and `findings`.
Its `checks` must cover every name in `scene.review.environment.CHECKS`: shape and
proportions, materials/texture scale, lighting/exposure, contact shadows,
occlusion/intersections, layout relationships, cross-view consistency,
blur/detail, holes/coverage, ghosts/floaters, task preservation and export
renderability. Each needs boolean `pass`, an explanation in `note`, and `evidence`
images from that candidate's render review. Multiple captured and novel renders
are required. Unresolved findings keep the result failed even when every other
criterion passes. Numerical scores cannot override a failed visual criterion.

Reviews append under `qa/environment_quality/round_*`. S3 binds the passing
review's digest; changed inputs, renders, materials or protected task/support/robot
state invalidate it and blockS6. Cousins with changed environment appearance,
geometry, properties or layout are judged for whole-scene coherence and
photorealism, not exact pixel matching to the original room. Frozen task state
remains protected. Legacy theme/target augmentation is not a substitute for
editable background cousins, and procedural office-kit boxes alone are not an
accepted photoreal agent-Blender environment.

Saved GS checkpoints can be compared using the same review-camera manifest:

```bash
tools/run_splat.sh -m calibrated_scene_reconstruction.scene.review.splat_checkpoints sessions/<name> \
  --candidates <saved-round-a> <saved-round-b> \
  --cameras qa/background_diagnostics/cameras.json \
  --output qa/background_diagnostics/new_checkpoint_comparison
```

Each saved round supplies `background_rounds/<label>/splat/train.json` with its
checkpoint and settings. Labels and camera offsets are generic; this comparison
changes neither the checkpoints nor camera/video selection. Native checkpoint
renders and final exported NuRec renders are separate evidence streams.
