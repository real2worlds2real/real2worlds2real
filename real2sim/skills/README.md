# Skills

The agent (ASTRA) reads these instructions and selects tools as the task needs them. Every directory here is one skill
(`<name>/SKILL.md` with `name` and `description`); `real2sim-harness task call ... search_skills / read_skill` and any coding
client that loads skills find them by that layout. The layers below are how they are meant to be read.

```text
L1 core principles ──> L2 task workflows ──> L3 specialist skills ──> tools, selected as needed
        ^                                                                    │
        └────────────── personal skills: experience distilled and reused ────┘
```

## L1 Core principles

| Skill | What it fixes |
|---|---|
| [use-real2sim](use-real2sim/SKILL.md) | Entry: which workflow a request needs; the agent's own conversation is the task agent. |
| [work-with-task-evidence](work-with-task-evidence/SKILL.md) | Task-driven choices and evidence-based checks: what counts as an observation, a prior or an assumption; what a score may and may not justify. |

## L2 Task workflows

| Outcome | Workflow |
|---|---|
| Shared object workflow | [reconstruct-object](reconstruct-object/SKILL.md) |
| Shared environment workflow | [reconstruct-environment](reconstruct-environment/SKILL.md) |
| Workspace object tools for a **twin**: editable, operable, checked in the target simulator | [generate-editable-object](generate-editable-object/SKILL.md) |
| **Twin** of a scene composed in a task workspace | [plan-scene-generation](plan-scene-generation/SKILL.md), [compose-interactive-scene](compose-interactive-scene/SKILL.md) |
| **Twin** of a scene from a walk-around video, in metres, with the robot placed, through the gates S0-S6 | [twin-capture](twin-capture/SKILL.md) → [twin-model](twin-model/SKILL.md) → [twin-deliver](twin-deliver/SKILL.md) |
| **Cousins** of an object: form varies, function is kept | [generate-functional-cousins](generate-functional-cousins/SKILL.md) |
| **Cousins** of a scene: layout, objects or both vary | [generate-scene-cousins](generate-scene-cousins/SKILL.md) |
| **Physics** calibrated to motion observed in a video | [calibrate-observed-physics](calibrate-observed-physics/SKILL.md) |

Select these capabilities in the same [agent module plan](../../end_to_end/docs/reconstruction.md). A task may need both calibrated robot placement and a workspace-authored interactive object. The final native acceptance owner writes the [delivery record](../docs/DELIVERY.md).

## L3 Specialist skills

| Group | Skills |
|---|---|
| Geometry & cameras | [inspect-video](inspect-video/SKILL.md), [segment-reference-views](segment-reference-views/SKILL.md), [align-reference-camera](align-reference-camera/SKILL.md), [recover-multiview-geometry](recover-multiview-geometry/SKILL.md), [use-vggt-prior](use-vggt-prior/SKILL.md), [fit-multiview-geometry](fit-multiview-geometry/SKILL.md), [measure-reference-geometry](measure-reference-geometry/SKILL.md), [recover-support-footprint](recover-support-footprint/SKILL.md), [prepare-gaussian-background](prepare-gaussian-background/SKILL.md) |
| Shape & appearance | [review-silhouette](review-silhouette/SKILL.md), [bake-reference-texture](bake-reference-texture/SKILL.md), [improve-visual-realism](improve-visual-realism/SKILL.md), [refine-deformable-surface](refine-deformable-surface/SKILL.md) |
| Experiment design | [simulate-mating-rigid-parts](simulate-mating-rigid-parts/SKILL.md), [simulate-articulated-cloth](simulate-articulated-cloth/SKILL.md), [simulate-deformable-behavior](simulate-deformable-behavior/SKILL.md), [simulate-volumetric-solids](simulate-volumetric-solids/SKILL.md), [simulate-slender-bodies](simulate-slender-bodies/SKILL.md), [simulate-granular-materials](simulate-granular-materials/SKILL.md), [simulate-particle-materials](simulate-particle-materials/SKILL.md), [simulate-impact-fracture](simulate-impact-fracture/SKILL.md) |
| Functional checks | [review-articulation](review-articulation/SKILL.md), [review-simulation](review-simulation/SKILL.md), [review-dynamic-contact](review-dynamic-contact/SKILL.md), [review-deformation](review-deformation/SKILL.md), [review-numerical-evidence](review-numerical-evidence/SKILL.md) |

## Personal skills: distilled experience

Built-in skills are maintained with the framework. Experience from finished work is kept separately and read back when
a new task matches it. It carries methods, never old dimensions, fits or approvals.

| Where | What |
|---|---|
| [real2sim-evolve](real2sim-evolve/SKILL.md) | When a task exceeds the current tools or procedure: diagnose the gap, prototype, validate, regress, then register. `calibrated-scene-reconstruction evolve propose / check / promote / recall` keeps the records. |
| [experience modules](twin-model/references/experience/index.md) | Reusable methods by mechanism: open vessels, rigid insertion, gears, threaded closures, hinged lids, liquids, flexible attachments and thin-surface depth. For rooms and task obstacles, see [spatial cousin review](reconstruct-environment/references/experience/spatial-cousin-review.md). Written with the [module template](twin-model/references/experience/module-template.md). |
| The caller's own skill folder | [work-with-task-evidence](work-with-task-evidence/SKILL.md) explains what to distill into the coding client's personal skills and what must stay task-specific. |

## Tools, selected as needed

| Tool | Reached through |
|---|---|
| Segmentation (SAM 3) | `task call segment_reference_views`; scene pipeline `stage_s1` (`perceive`) |
| Structure from motion (COLMAP) | `task call recover_video_cameras`, `register_metric_reconstruction`; scene pipeline `stage_s0` (`poses`) |
| Learned geometry prior (VGGT) | `task call run_reference_prior`, `align_reference_prior` |
| Blender modelling | `task call run_blender`; scene pipeline `object_blender_run` |
| Camera pose | `task call solve_reference_camera`, `calibrate_measurement_camera`, `align_multiview_camera_rig`; scene pipeline `stage_s0` (`calib`), `stage_s1` (`anchor`) |
| Mesh fitting | `task call fit_multiview_geometry`; scene pipeline `object_fit` |
| Texture baking | `task call bake_reference_texture`; scene pipeline `object_texture` |
| Gaussian-splat backgrounds (3DGS) | `task call run_gaussian_background`, `filter_gaussian_visibility`; scene pipeline `stage_s3` (`splat`) |
| PhysX | `task call run_simulation`, `run_rigid_contact_behavior`, `run_scene_simulation`; scene pipeline `stage_s5` |
| Newton | `task call run_deformable_behavior`, `run_strands`, `run_solids` with a Newton adapter |
| Contact tests | `task call run_rigid_contact_behavior`; scene pipeline `object_task_check`, `stage_s5` (clearance, rest) |
| Motion checks | `task call prepare_motion_observations`, `compare_physics_observations`; scene pipeline `physics_propose`, `physics_run`, `physics_review` |

`real2sim-harness task schemas --run <run>` and `calibrated-scene-reconstruction schemas` print the exact request contracts.
