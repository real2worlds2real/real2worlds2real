# Compatibility inventory

Legacy imports and command entry points remain available. Migrate callers and
training checkouts before removing them. Module identity aliases preserve private
symbols, mutable globals and historical serialized import names. Canonical source
paths, not wrapper files alone, must be bound in physical evidence fingerprints.

## Module forwarding

This table is generated from `src/calibrated_scene_reconstruction/compatibility.json`.

| Legacy path | Canonical implementation |
|---|---|
| `calibrated_scene_reconstruction.calib` | `calibrated_scene_reconstruction.scene.capture.calib` |
| `calibrated_scene_reconstruction.capture` | `calibrated_scene_reconstruction.scene.capture.capture` |
| `calibrated_scene_reconstruction.cousins` | `calibrated_scene_reconstruction.scene.cousins.variants` |
| `calibrated_scene_reconstruction.deformable.appearance` | `calibrated_scene_reconstruction.tasks.rope.appearance` |
| `calibrated_scene_reconstruction.deformable.lehome_look` | `calibrated_scene_reconstruction.tasks.lehome.appearance` |
| `calibrated_scene_reconstruction.deformable.letters` | `calibrated_scene_reconstruction.tasks.rope.letters` |
| `calibrated_scene_reconstruction.deformable.newton_record` | `calibrated_scene_reconstruction.sim.newton.recording` |
| `calibrated_scene_reconstruction.deformable.rope_mpc` | `calibrated_scene_reconstruction.tasks.rope.mpc` |
| `calibrated_scene_reconstruction.deformable.video` | `calibrated_scene_reconstruction.sim.newton.video` |
| `calibrated_scene_reconstruction.demo.adapters.droid` | `calibrated_scene_reconstruction.datagen.demo.adapters.droid` |
| `calibrated_scene_reconstruction.demo.frame` | `calibrated_scene_reconstruction.datagen.demo.frame` |
| `calibrated_scene_reconstruction.demo.functional` | `calibrated_scene_reconstruction.datagen.demo.functional` |
| `calibrated_scene_reconstruction.demo.io` | `calibrated_scene_reconstruction.datagen.demo.io` |
| `calibrated_scene_reconstruction.demo.retarget` | `calibrated_scene_reconstruction.datagen.demo.retarget` |
| `calibrated_scene_reconstruction.demo.sampler` | `calibrated_scene_reconstruction.datagen.demo.sampler` |
| `calibrated_scene_reconstruction.demo.segment` | `calibrated_scene_reconstruction.datagen.demo.segment` |
| `calibrated_scene_reconstruction.dr` | `calibrated_scene_reconstruction.scene.cousins.randomization` |
| `calibrated_scene_reconstruction.export.lerobot` | `calibrated_scene_reconstruction.datagen.export.lerobot` |
| `calibrated_scene_reconstruction.gpu` | `calibrated_scene_reconstruction.core.gpu` |
| `calibrated_scene_reconstruction.keyframes` | `calibrated_scene_reconstruction.scene.capture.keyframes` |
| `calibrated_scene_reconstruction.objects.blender_harness` | `calibrated_scene_reconstruction.scene.objects.blender.blender_harness` |
| `calibrated_scene_reconstruction.objects.blender_run` | `calibrated_scene_reconstruction.scene.objects.blender.blender_run` |
| `calibrated_scene_reconstruction.objects.contract` | `calibrated_scene_reconstruction.scene.objects.blender.contract` |
| `calibrated_scene_reconstruction.objects.evidence` | `calibrated_scene_reconstruction.scene.objects.blender.evidence` |
| `calibrated_scene_reconstruction.objects.export` | `calibrated_scene_reconstruction.scene.objects.blender.export` |
| `calibrated_scene_reconstruction.objects.fit` | `calibrated_scene_reconstruction.scene.objects.blender.fit` |
| `calibrated_scene_reconstruction.objects.mcp_server` | `calibrated_scene_reconstruction.agent.legacy_object_mcp` |
| `calibrated_scene_reconstruction.objects.physics` | `calibrated_scene_reconstruction.assets.material_mass` |
| `calibrated_scene_reconstruction.objects.pipeline` | `calibrated_scene_reconstruction.scene.objects.blender.pipeline` |
| `calibrated_scene_reconstruction.objects.store` | `calibrated_scene_reconstruction.scene.objects.blender.store` |
| `calibrated_scene_reconstruction.objects.texture` | `calibrated_scene_reconstruction.scene.objects.blender.texture` |
| `calibrated_scene_reconstruction.objects.tool` | `calibrated_scene_reconstruction.agent.object_tool` |
| `calibrated_scene_reconstruction.paths` | `calibrated_scene_reconstruction.core.paths` |
| `calibrated_scene_reconstruction.physics.video.visualise` | `calibrated_scene_reconstruction.physics.video.visualize` |
| `calibrated_scene_reconstruction.policy.openpi_lab01` | `calibrated_scene_reconstruction.policy.openpi` |
| `calibrated_scene_reconstruction.real2sim.data` | `calibrated_scene_reconstruction.physics.replay.data` |
| `calibrated_scene_reconstruction.real2sim.droid` | `calibrated_scene_reconstruction.physics.replay.droid` |
| `calibrated_scene_reconstruction.real2sim.loss` | `calibrated_scene_reconstruction.physics.replay.loss` |
| `calibrated_scene_reconstruction.real2sim.observe` | `calibrated_scene_reconstruction.physics.replay.observe` |
| `calibrated_scene_reconstruction.real2sim.optim` | `calibrated_scene_reconstruction.physics.replay.optim` |
| `calibrated_scene_reconstruction.real2sim.overlay` | `calibrated_scene_reconstruction.physics.replay.overlay` |
| `calibrated_scene_reconstruction.real2sim.params` | `calibrated_scene_reconstruction.physics.replay.params` |
| `calibrated_scene_reconstruction.real2sim.tabletop` | `calibrated_scene_reconstruction.physics.replay.tabletop` |
| `calibrated_scene_reconstruction.recon.align` | `calibrated_scene_reconstruction.scene.objects.align` |
| `calibrated_scene_reconstruction.recon.audit` | `calibrated_scene_reconstruction.scene.review.audit` |
| `calibrated_scene_reconstruction.recon.autograph` | `calibrated_scene_reconstruction.scene.twin.graph` |
| `calibrated_scene_reconstruction.recon.autoviews` | `calibrated_scene_reconstruction.scene.review.views` |
| `calibrated_scene_reconstruction.recon.bake` | `calibrated_scene_reconstruction.scene.background.bake` |
| `calibrated_scene_reconstruction.recon.board` | `calibrated_scene_reconstruction.scene.capture.board` |
| `calibrated_scene_reconstruction.recon.code_scene` | `calibrated_scene_reconstruction.scene.background.code_room` |
| `calibrated_scene_reconstruction.recon.colmap_points` | `calibrated_scene_reconstruction.scene.capture.colmap_points` |
| `calibrated_scene_reconstruction.recon.completion_review` | `calibrated_scene_reconstruction.scene.review.completion_review` |
| `calibrated_scene_reconstruction.recon.cousin_rooms` | `calibrated_scene_reconstruction.scene.cousins.rooms` |
| `calibrated_scene_reconstruction.recon.da3_views` | `calibrated_scene_reconstruction.scene.capture.da3_views` |
| `calibrated_scene_reconstruction.recon.frame_colour` | `calibrated_scene_reconstruction.scene.objects.frame_color` |
| `calibrated_scene_reconstruction.recon.furniture` | `calibrated_scene_reconstruction.scene.support.furniture` |
| `calibrated_scene_reconstruction.recon.inpaint` | `calibrated_scene_reconstruction.scene.background.inpaint` |
| `calibrated_scene_reconstruction.recon.inputs` | `calibrated_scene_reconstruction.scene.capture.inputs` |
| `calibrated_scene_reconstruction.recon.layout` | `calibrated_scene_reconstruction.scene.background.layout` |
| `calibrated_scene_reconstruction.recon.materials` | `calibrated_scene_reconstruction.scene.background.materials` |
| `calibrated_scene_reconstruction.recon.multiview` | `calibrated_scene_reconstruction.scene.capture.multiview` |
| `calibrated_scene_reconstruction.recon.object_fit` | `calibrated_scene_reconstruction.scene.objects.object_fit` |
| `calibrated_scene_reconstruction.recon.object_hull` | `calibrated_scene_reconstruction.scene.objects.object_hull` |
| `calibrated_scene_reconstruction.recon.object_refine` | `calibrated_scene_reconstruction.scene.objects.object_refine` |
| `calibrated_scene_reconstruction.recon.object_texture` | `calibrated_scene_reconstruction.scene.objects.object_texture` |
| `calibrated_scene_reconstruction.recon.objects` | `calibrated_scene_reconstruction.scene.objects.reconstruct` |
| `calibrated_scene_reconstruction.recon.pano_furniture` | `calibrated_scene_reconstruction.scene.background.pano_furniture` |
| `calibrated_scene_reconstruction.recon.panorama` | `calibrated_scene_reconstruction.scene.background.panorama` |
| `calibrated_scene_reconstruction.recon.perceive` | `calibrated_scene_reconstruction.scene.capture.perceive` |
| `calibrated_scene_reconstruction.recon.posed_frames` | `calibrated_scene_reconstruction.scene.capture.posed_frames` |
| `calibrated_scene_reconstruction.recon.retrieve` | `calibrated_scene_reconstruction.scene.objects.retrieve` |
| `calibrated_scene_reconstruction.recon.robot_carve` | `calibrated_scene_reconstruction.scene.capture.robot_carve` |
| `calibrated_scene_reconstruction.recon.robot_fit` | `calibrated_scene_reconstruction.scene.capture.robot_fit` |
| `calibrated_scene_reconstruction.recon.robot_scale` | `calibrated_scene_reconstruction.scene.capture.robot_scale` |
| `calibrated_scene_reconstruction.recon.room_objects` | `calibrated_scene_reconstruction.scene.background.room_objects` |
| `calibrated_scene_reconstruction.recon.setpieces` | `calibrated_scene_reconstruction.scene.objects.setpiece_observations` |
| `calibrated_scene_reconstruction.recon.splat` | `calibrated_scene_reconstruction.scene.background.splat.config` |
| `calibrated_scene_reconstruction.recon.splat_capture` | `calibrated_scene_reconstruction.scene.background.splat.capture` |
| `calibrated_scene_reconstruction.recon.splat_cousin` | `calibrated_scene_reconstruction.scene.background.splat.inventory` |
| `calibrated_scene_reconstruction.recon.splat_export` | `calibrated_scene_reconstruction.scene.background.splat.export` |
| `calibrated_scene_reconstruction.recon.splat_fix` | `calibrated_scene_reconstruction.scene.background.splat.fix` |
| `calibrated_scene_reconstruction.recon.splat_masks` | `calibrated_scene_reconstruction.scene.background.splat.masks` |
| `calibrated_scene_reconstruction.recon.splat_train` | `calibrated_scene_reconstruction.scene.background.splat.train` |
| `calibrated_scene_reconstruction.recon.structure_hull` | `calibrated_scene_reconstruction.scene.objects.structure_hull` |
| `calibrated_scene_reconstruction.recon.support_carve` | `calibrated_scene_reconstruction.scene.support.support_carve` |
| `calibrated_scene_reconstruction.recon.support_marks` | `calibrated_scene_reconstruction.scene.support.support_marks` |
| `calibrated_scene_reconstruction.recon.surfaces` | `calibrated_scene_reconstruction.scene.support.surfaces` |
| `calibrated_scene_reconstruction.recon.unseen_audit` | `calibrated_scene_reconstruction.scene.review.unseen_audit` |
| `calibrated_scene_reconstruction.recon.wall_completion` | `calibrated_scene_reconstruction.scene.background.wall_completion` |
| `calibrated_scene_reconstruction.recon.zone_check` | `calibrated_scene_reconstruction.scene.review.zone_check` |
| `calibrated_scene_reconstruction.resolution` | `calibrated_scene_reconstruction.scene.review.resolution` |
| `calibrated_scene_reconstruction.robots.articulation` | `calibrated_scene_reconstruction.sim.robot_articulation` |
| `calibrated_scene_reconstruction.scene.capture.articulated_model` | `calibrated_scene_reconstruction.robots.articulated_model` |
| `calibrated_scene_reconstruction.scene.objects.blender.physics` | `calibrated_scene_reconstruction.assets.material_mass` |
| `calibrated_scene_reconstruction.scene.objects.frame_colour` | `calibrated_scene_reconstruction.scene.objects.frame_color` |
| `calibrated_scene_reconstruction.scene.objects.particle_contact_binding` | `calibrated_scene_reconstruction.sim.particle_contact_binding` |
| `calibrated_scene_reconstruction.scenegraph` | `calibrated_scene_reconstruction.scene.graph` |
| `calibrated_scene_reconstruction.session` | `calibrated_scene_reconstruction.scene.session` |
| `calibrated_scene_reconstruction.sim.cloth_clearance` | `calibrated_scene_reconstruction.assets.cloth_clearance` |
| `calibrated_scene_reconstruction.sim.cloth_contact` | `calibrated_scene_reconstruction.assets.cloth_contact` |
| `calibrated_scene_reconstruction.sim.cloth_shell_clearance` | `calibrated_scene_reconstruction.assets.cloth_shell_clearance` |
| `calibrated_scene_reconstruction.sim.cudss` | `calibrated_scene_reconstruction.core.cudss` |
| `calibrated_scene_reconstruction.sim.hand_collision` | `calibrated_scene_reconstruction.assets.hand_collision` |
| `calibrated_scene_reconstruction.sim.object_review` | `calibrated_scene_reconstruction.scene.review.objects` |
| `calibrated_scene_reconstruction.sim.shared_gpu` | `calibrated_scene_reconstruction.core.shared_gpu` |
| `calibrated_scene_reconstruction.sim.state_visibility` | `calibrated_scene_reconstruction.core.geometry.state_visibility` |
| `calibrated_scene_reconstruction.sim.surface_distance` | `calibrated_scene_reconstruction.assets.surface_distance` |
| `calibrated_scene_reconstruction.sim.task_checks` | `calibrated_scene_reconstruction.tasks.rigid.task_checks` |
| `calibrated_scene_reconstruction.sim.taskframe` | `calibrated_scene_reconstruction.robots.taskframe` |
| `calibrated_scene_reconstruction.sim.templates` | `calibrated_scene_reconstruction.assets.templates` |
| `calibrated_scene_reconstruction.standpoints` | `calibrated_scene_reconstruction.robots.viewpoints` |
| `calibrated_scene_reconstruction.tasks.cloth2d_geometry` | `calibrated_scene_reconstruction.assets.cloth2d_geometry` |
| `calibrated_scene_reconstruction.tasks.fluid_surface` | `calibrated_scene_reconstruction.assets.fluid_surface` |
| `calibrated_scene_reconstruction.tasks.mechanisms.contact_control_profile` | `calibrated_scene_reconstruction.assets.contact_control_profile` |
| `calibrated_scene_reconstruction.tasks.mechanisms.hand_acceptance` | `calibrated_scene_reconstruction.apps.reconstruction.hand_acceptance` |
| `calibrated_scene_reconstruction.tasks.mechanisms.hand_cloth` | `calibrated_scene_reconstruction.assets.hand_cloth_metrics` |
| `calibrated_scene_reconstruction.tasks.mechanisms.hand_contact_isolation` | `calibrated_scene_reconstruction.assets.hand_contact_isolation` |
| `calibrated_scene_reconstruction.tasks.mechanisms.hand_motion` | `calibrated_scene_reconstruction.assets.hand_motion` |
| `calibrated_scene_reconstruction.tasks.mechanisms.material_grasp` | `calibrated_scene_reconstruction.assets.material_grasp` |
| `calibrated_scene_reconstruction.tasks.mechanisms.observation_match` | `calibrated_scene_reconstruction.apps.reconstruction.observation_match` |
| `calibrated_scene_reconstruction.tasks.mechanisms.particle_hand_acceptance` | `calibrated_scene_reconstruction.assets.particle_hand_acceptance` |
| `calibrated_scene_reconstruction.tasks.pinch_friction` | `calibrated_scene_reconstruction.assets.pinch_friction` |
| `calibrated_scene_reconstruction.tasks.rigid.observation_state` | `calibrated_scene_reconstruction.assets.observation_state` |
| `calibrated_scene_reconstruction.tasks.wet_contact` | `calibrated_scene_reconstruction.assets.wet_contact` |
| `calibrated_scene_reconstruction.tasks.wetting` | `calibrated_scene_reconstruction.assets.wetting` |
| `calibrated_scene_reconstruction.tasks.wipe_geometry` | `calibrated_scene_reconstruction.assets.wipe_geometry` |
| `calibrated_scene_reconstruction.twin` | `calibrated_scene_reconstruction.scene.twin.gates` |
| `calibrated_scene_reconstruction.twin_report` | `calibrated_scene_reconstruction.scene.twin.report` |
| `calibrated_scene_reconstruction.video` | `calibrated_scene_reconstruction.datagen.video` |
| `calibrated_scene_reconstruction.vidphys.ballistic` | `calibrated_scene_reconstruction.physics.video.ballistic` |
| `calibrated_scene_reconstruction.vidphys.beam` | `calibrated_scene_reconstruction.physics.video.beam` |
| `calibrated_scene_reconstruction.vidphys.carried` | `calibrated_scene_reconstruction.physics.video.carried` |
| `calibrated_scene_reconstruction.vidphys.fit` | `calibrated_scene_reconstruction.physics.video.fit` |
| `calibrated_scene_reconstruction.vidphys.incline` | `calibrated_scene_reconstruction.physics.video.incline` |
| `calibrated_scene_reconstruction.vidphys.mesh` | `calibrated_scene_reconstruction.physics.video.mesh` |
| `calibrated_scene_reconstruction.vidphys.meta` | `calibrated_scene_reconstruction.physics.video.meta` |
| `calibrated_scene_reconstruction.vidphys.model_inputs` | `calibrated_scene_reconstruction.physics.video.model_inputs` |
| `calibrated_scene_reconstruction.vidphys.pendulum` | `calibrated_scene_reconstruction.physics.video.pendulum` |
| `calibrated_scene_reconstruction.vidphys.perceive` | `calibrated_scene_reconstruction.physics.video.perceive` |
| `calibrated_scene_reconstruction.vidphys.raster` | `calibrated_scene_reconstruction.physics.video.raster` |
| `calibrated_scene_reconstruction.vidphys.readings` | `calibrated_scene_reconstruction.physics.video.readings` |
| `calibrated_scene_reconstruction.vidphys.scene` | `calibrated_scene_reconstruction.physics.video.scene` |
| `calibrated_scene_reconstruction.vidphys.sheet` | `calibrated_scene_reconstruction.physics.video.sheet` |
| `calibrated_scene_reconstruction.vidphys.soft` | `calibrated_scene_reconstruction.physics.video.soft` |
| `calibrated_scene_reconstruction.vidphys.stabilise` | `calibrated_scene_reconstruction.physics.video.stabilise` |
| `calibrated_scene_reconstruction.vidphys.stage` | `calibrated_scene_reconstruction.sim.video_physics_stage` |
| `calibrated_scene_reconstruction.vidphys.stickslip` | `calibrated_scene_reconstruction.physics.video.stickslip` |
| `calibrated_scene_reconstruction.vidphys.strain` | `calibrated_scene_reconstruction.physics.video.strain` |
| `calibrated_scene_reconstruction.vidphys.track` | `calibrated_scene_reconstruction.physics.video.track` |
| `calibrated_scene_reconstruction.vidphys.turntable` | `calibrated_scene_reconstruction.physics.video.turntable` |
| `calibrated_scene_reconstruction.vidphys.visualise` | `calibrated_scene_reconstruction.physics.video.visualize` |

## Explicit composition adapters

Legacy demo CLI injects the scene Session. The old LeHome auto-registration adapter
injects its policy client. Legacy sim scene entries prepare scene inputs; new
runtime code consumes prepared contracts. Archived frame/overlay/physics examples
retain imports for historical reproducibility, not independent source evidence.

Pure cooking settings now live in `assets.convex_cooking`; native USD cooking
operations remain in `sim.convex_cooking`, which also re-exports validation.
Array/wxyz conversion lives in `core.geometry.state_math`; task-check imports
remain compatible. Native particle readback lives in sim and portable comparison
in assets. The app injects particle-friction qualification into the native hand
control rather than importing scene review from the backend.

## Command forwarding

| Legacy entry | Canonical entry |
|---|---|
| `apps/check_clearance.py` | `apps/scene/check_clearance.py` |
| `apps/code_compare.py` | `apps/scene/code_compare.py` |
| `apps/command_tcp.py` | `apps/datagen/command_tcp.py` |
| `apps/episode.py` | `apps/datagen/episode.py` |
| `apps/export_usd.py` | `apps/scene/export_usd.py` |
| `apps/flythrough.py` | `apps/scene/flythrough.py` |
| `apps/frame_check.py` | `apps/scene/frame_check.py` |
| `apps/irradiance.py` | `apps/scene/irradiance.py` |
| `apps/lehome_run.py` | `apps/tasks/lehome_run.py` |
| `apps/newton_rope_letter.py` | `apps/tasks/rope_letter.py` |
| `apps/object_closeups.py` | `apps/scene/object_closeups.py` |
| `apps/pack_usdz.py` | `apps/scene/pack_usdz.py` |
| `apps/real2sim_replay.py` | `apps/physics/replay.py` |
| `apps/render_views.py` | `apps/scene/render_views.py` |
| `apps/review_objects.py` | `apps/scene/review_objects.py` |
| `apps/room_views.py` | `apps/scene/room_views.py` |
| `apps/scene_tour.py` | `apps/scene/scene_tour.py` |
| `apps/splat_envmap.py` | `apps/scene/splat_envmap.py` |
| `apps/task_check.py` | `apps/scene/task_check.py` |
| `apps/video_physics.py` | `apps/physics/video.py` |
| `apps/video_soft.py` | `apps/physics/soft.py` |
| `apps/view_usd.py` | `apps/scene/view_usd.py` |
| `apps/wall_views.py` | `apps/scene/wall_views.py` |
| `tools/audit_backhand.py` | `tools/datagen/audit_backhand.py` |
| `tools/build_frame_cache.py` | `tools/policy/build_frame_cache.py` |
| `tools/collect_place.py` | `tools/datagen/collect_place.py` |
| `tools/copy_configs.py` | `tools/datagen/copy_configs.py` |
| `tools/dp_official_server.py` | `tools/policy/dp_official_server.py` |
| `tools/export_dp_official.py` | `tools/policy/export_dp_official.py` |
| `tools/export_dp_official_deformable.py` | `tools/policy/export_dp_official_deformable.py` |
| `tools/export_openpi.py` | `tools/policy/export_openpi.py` |
| `tools/fail_states.py` | `tools/policy/fail_states.py` |
| `tools/grasp_stats.py` | `tools/datagen/grasp_stats.py` |
| `tools/openpi_run.py` | `tools/policy/openpi_run.py` |
| `tools/openpi_serve.py` | `tools/policy/openpi_serve.py` |
| `tools/run_newton.sh` | `tools/runtime/run_newton.sh` |
| `tools/setup_lehome.sh` | `tools/tasks/setup_lehome.sh` |
| `apps/newton_rope.py` | `apps/tasks/rope.py`; imports preserve the Newton backend module identity |

Root-level documentation redirects preserve old links. The legacy object MCP
retains its original tool names; the scene workflow MCP uses stage names and
`object_`/`support_` prefixes. Oversight is an additional provider-neutral CLI.

## Synchronize training environments

1. Preserve the training checkout's modifications and active outputs.
2. Synchronize code, configuration, schemas, assets required by the task and all
   forwarding modules together. Pin the intended source revision.
3. Select the active checkout explicitly through `PYTHONPATH`; reinstall editable
   packages only in the intended isolated environment, not a shared worktree's env.
4. Validate existing train/serve/evaluate commands in their actual environments,
   then migrate callers to canonical entries.

OpenPI, LeHome, Newton and Isaac runtime validation requires their own installed
stacks. Static import and CPU tests do not substitute for those checks. Existing
Lab01 class aliases and color/colour field compatibility remain; new code uses
generic names and current spelling without rewriting historical JSON fields.
