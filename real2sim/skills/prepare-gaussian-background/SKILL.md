---
name: prepare-gaussian-background
description: Prepare, train or refine optional Gaussian scene backgrounds using captured views, geometric cleanup, restored novel views and depth-consistent propagation before portable USD export.
---

Use when captured background appearance helps the scene. A compact editable room can remain entirely mesh. Training needs multi-view capture or video with reliable camera poses; a single photo is not measured multi-view coverage.

Call `run_gaussian_background` with a workspace JSON request and a time budget. See `docs/GAUSSIAN_BACKGROUNDS.md` and `examples/gaussian-prepare.json` for schemas. `prepare` runs CPU PLY editing and standard ParticleField export. `train` and `transcode` use an explicitly configured operator backend; missing configuration returns a retained error. Reuse existing trained assets when suitable.

Keep source-to-scene scale/rotation/translation explicit. Do not call predicted depth metric evidence. Scene-frame cameras use OpenCV (+Z forward) world-to-camera transforms and pinhole intrinsics; undistort beforehand. Frustum inclusion is a geometric diagnostic, not an occlusion, appearance or accuracy guarantee.

Remove Gaussians covering editable objects and their intended swept motion so captured copies do not remain behind them. Exclusion boxes act on conservative particle bounds; inspect removal and exposed holes before widening crops. Supports and collision meshes remain separate. Cropped regions are not automatically filled or observed.

When the target renderer exposes floaters near capture cameras, optional `crop.near_camera_m` removes particles whose finite sigma bounds intersect that camera neighborhood. Supply actual posed capture cameras in the same scene frame; the default is disabled. Choose the radius from observed artifacts, then review removed particles and legitimate nearby surfaces. This is geometric cleanup, not image-based artifact restoration or evidence of unseen geometry. Generative novel-view repair and consistent whole-background restyling are separate capabilities and are not provided by this filter.

Use the returned `background` entry with an ID in the scene manifest. It preserves high-order color, anisotropic scale and orientation. `particle_field` and `nurec_volume` require a matching renderer adapter; training/export success alone does not establish scene delivery. For Isaac Sim 5.1, transcode the prepared Gaussian source to `nurec_volume` using the configured 3DGRUT bridge, and select the 5.1 scene runner declaring that capability. Direct `particle_field` rendering requires a compatible newer runtime. When transcoding prepared `background.ply`, carry its source-to-scene transform once; when transcoding the already aligned ParticleField USD, preserve its transform and use an identity wrapper. Review the actual composed scene for duplicates, holes, floating particles and view coverage, with video when useful. Astra chooses refinements and the final verdict.

Optional `refine` requires an operator backend declaring that capability. Read the refinement section of `docs/GAUSSIAN_BACKGROUNDS.md`. Supply the current Gaussian PLY, actual registered capture observations and selected novel camera poses near capture coverage. The backend renders the model, proposes anchor repairs with official Fixer, preserves consistent anchor pixels using depth/occlusion tests, and optimizes the 3D model against both retained captures and generated pseudo views. It returns before/after previews and coverage, not just edited 2D images.

Default `geometry_mode: color_only` verifies geometry and opacity stay unchanged. Optional `joint_refine` allows small geometry/opacity updates and reports their effect; neither mode supplies unseen geometry truth or automatically guarantees a better reconstruction. Inspect original-view residuals, repaired novel views, coverage and final scene appearance. Poor proposals may be rejected or revised; the source model is retained. Model-estimated depth and generated repair pixels must not be called observations.

Optional `operation:"cousin"` edits an existing standard ParticleField USD/USDZ on CPU with global or regional SH color grading. This is appearance editing, not generative semantic restyling. Scene-cousin sessions preserve the parent's recognizable layout by default; spatial region transforms are available for layout changes requested by the user. See [scene cousins](../../docs/SCENE_COUSINS.md) for the schema, layout policy and functional-object branch. Full-mesh scenes can generate cousins without this tool.

When geometric crops leave floaters, optional `filter_gaussian_visibility` uses
actual source-matched renderer visibility or registered depth free-space votes.
Read [the contract](../../docs/RECONSTRUCTION_REFINEMENT.md#rendered-gaussian-visibility-and-free-space).
The renderer worker can export per-particle visibility and its exact source hash;
frustum inclusion cannot stand in for it. Occlusion must not count as evidence of
empty space. Review filtered candidates and holes before preparation/export.

The refinement adapter also supplies actual capture references to restoration
workers. Optional [Difix conditioning](../../docs/RECONSTRUCTION_REFINEMENT.md#reference-conditioned-restoration)
complements existing Fixer; either keeps depth-consistent propagation and optional
geometry freezing. Pick the configured backend for the observed failure, without
treating generated pixels or model-rendered depths as real observations.
