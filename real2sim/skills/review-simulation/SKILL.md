---
name: review-simulation
description: Check a saved Blender asset in PhysX or Newton when target-backend operability, contacts, joint control or cloth surface following need verification.
---

# Review target simulation

The user’s backend choice takes precedence. Default cloth, garments and soft
bodies to **PhysX** when no backend is specified; use Newton when explicitly
requested. Confirm what the selected adapter actually runs. The existing
`run_simulation` surface route and its VBD-specific settings below use Newton;
they do not constitute a PhysX cloth implementation. Missing adapter support
must be handled explicitly, without silently overriding the requested backend.

`run_simulation(attempt_id, steps, render_views, parameters_json, assumptions)` exports an unchanged saved attempt, loads it in the target backend and returns numerical checks plus optional Isaac RTX images. Supported assets are rigid bodies, revolute/continuous articulations, and one cloth control mesh with its visual followers. Liquid particle materials use `run_fluid`.

During modeling, choose this check when it resolves a real uncertainty. Rigid/articulated modeling and review stay in Blender; extra physics-feedback iterations are not a routine requirement. Cloth can benefit from an intermediate check when constraints, contact or detailed surface following are uncertain. Do not add physics-calibration work to the current visual/operability goal.

Final target-simulator acceptance is mandatory before delivery. Once the Blender result looks ready, call this tool with `render_views=True` and inspect both its numerical checks and actual target-rendered images. A numerical-only intermediate check does not replace this final review. If export, motion or appearance fails, read the diagnostics, repair the source or justified parameter assumptions, and submit again in the same agent session. Do not call `accept_result` until the unchanged final source has passed and its returned views satisfy the task. A rejected acceptance attempt leaves the session open for correction.

Use 60, 120, 240 or 480 steps. The articulated check exercises every joint while holding the others; it rejects stationary joints, poor target tracking and unrelated drift. The rigid check drops the object onto a support. The cloth check exercises source pin groups when supplied and verifies finite motion, contact, stretching, pin targets and visual-node correspondence. Test constraints do not permanently pin the delivered cloth.

Set `render_views=False` for a numerical check, or `True` when actual target-renderer appearance helps the decision. RTX startup and rendering are more expensive than a numeric check. The bundle and its file hashes remain available under the returned attempt-relative directory.

`parameters_json=None` uses source assumptions and defaults. To explore a concrete issue, supply a JSON object with partial overrides in the reported sections: `rigid` (mass_kg, density_kg_m3, enable_ccd), `material` (friction, restitution), `joint_drive` (stiffness, damping, max_force), `surface` (density, particle_radius, tri_ke, tri_ka, tri_kd, edge_ke, edge_kd), `solver` (substeps, iterations, self_contact), or `constraints` (use_source_pins). For example, `{"material":{"friction":0.7}}` changes friction while preserving other source defaults. Values remain assumed parameters; real-world calibration is not an acceptance requirement.


Read failures as actionable feedback. Inspect `failed_stage`, `conversion_validation` and diagnostic log tails first. Export, coordinate, loading or renderer failures may belong to integration or runtime; do not blindly change source geometry or physics to hide them. A motion failure also does not by itself determine the cause. Check conversion consistency before attributing it to the model. Fix demonstrated geometry, hierarchy or collision-metadata defects in the Blender script; adjust numerical or material defaults only for a reason supported by observed behavior. If a selected backend check fails, resolve it before acceptance. Numerical success establishes operability, while the rendered views still require visual judgment.

The conversion checks compare source and USD object transforms, metre/Z-up conventions, collider vertices/topology/rigid owners, joint axes/limits/anchors and normalized rotations, and cloth control geometry and binding provenance. Uniform scale is removed from joint orientation; nonuniform scale or shear on a joint must be baked into geometry. Visible curves/text and other non-mesh geometry must currently be converted to meshes before integration; export rejects them instead of dropping them silently.

For thin walls, cavities, insertion or nesting, read
[rigid-contact.md](rigid-contact.md). It covers compound versus SDF collision,
CPU/GPU compatibility, effective CCD and functional checks beyond dropping onto
a support. Choose these settings for the actual geometry, action and installed
backend; neither CPU nor SDF is a universal requirement.

When contact or occlusion changes during the operation, consult `review-dynamic-contact`: final snapshots and sparse samples can miss transient penetration. Review the final motion representation and route defects back for correction before calling it deliverable.

Preserve authored environment assumptions. `environment.gravity` is a three-component metre/second-squared vector; source cloth gravity weighting is recorded on export. A source zero-gravity bending probe is not an Earth-gravity settling experiment. Label the actual environment and keep later physical calibration separate. Surface VBD self-contact interaction/query distances and contact capacities are editable solver assumptions. The optional `constraints.lift_selection` selects all source pins or a positive/negative world-axis side for a hold-and-pull probe; other source pins remain fixed. Choose a test that demonstrates actual deformation rather than only translating all grip points together.

For unexplained flutter or contact chatter, compare the interaction distance with the control-mesh edge scale and the source collision distance. Cloth initialization now uses source collision metadata when available, otherwise a thin-surface estimate bounded by mesh scale; explicit `solver.self_contact_radius` and `self_contact_margin` remain overrides. A large fixed distance can repel nearby folds and destabilize a fine mesh. Disabling self-contact is a diagnostic comparison, not a repair for an asset that requires cloth contact. Blender damping coefficients are not mechanically equivalent to Newton material coefficients; raising damping indiscriminately can also destabilize the solver.

Finite states, modest stretch and a successful lift do not establish natural temporal behavior. For suspicious motion, a lift followed by an unforced hold can separate continuing excitation from residual jitter. Optional `constraints` overrides expose `lift_distance_m`, `lift_duration_seconds`, `lift_delay_seconds`, `quiet_start_seconds` and `quiet_speed_tolerance_m_s`. Fit the protocol within the selected step budget (one step is 0.01 s), start the quiet window after motion stops and choose a context-appropriate tolerance. A failed declared quiet check blocks this simulation result; it is not a reason to loosen the threshold without justification. Inspect the actual motion as well as these diagnostics.

If still-air resistance is part of the chosen model, `environment.air_drag_rate_per_second` enables the assumed force `F=-m*rate*v` inside Newton substeps. The default is zero; choose the model when justified, rather than applying it universally. It is neither wind nor a rendered-frame smoothing effect. Retain its parameter in the bundle, and use `configure_surface_environment(folder)` once after scene reset in a collaborator's scene so delivery matches validation. Distinguish assumed damping from real-world calibration.
