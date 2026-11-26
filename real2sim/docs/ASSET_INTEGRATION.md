# Downstream asset integration

Keep the editable `.blend` and generating Python together with the complete runtime configuration. Exports retain metre units, Z-up coordinates, semantic owners and explicit supported physics structures. Source changes invalidate prior acceptance evidence.

For [scene orchestration](SCENES.md), select the backend for the entire interacting task. Rigid/articulated, cloth and soft-body scenes default to PhysX. Fluid/strand tasks retain a Newton default for the shared interaction task. Explicit user backend choices take priority, and the selected adapter must verify the complete capability set. Per-object exports are materialized representations, not permanent engine assignments. Preserve the source descriptions for backend conversion, and require a compatible scene adapter instead of assuming cross-engine coupling. Shared image/video/geometry tools are reused by both scene and object modes.

| Type | Preserve with the asset | Loading / review interface |
| --- | --- | --- |
| Rigid / articulated | Visual USD and textures, PhysX USD, physics parameters, conversion validation and bundle manifest | `real2sim_harness.sim_asset_loader.asset_configs(folder)` returns normal Isaac Lab asset configs |
| Rigid contact assemblies (optional native review) | `contact_asset.json`, passive `runtime.json`, exact collision meshes, original source/GLB, source visual bindings, trajectory and movie | `real2sim_harness.rigid_contact_runtime.load_rigid_contact(configuration, cache_dir=..., builder=optional_builder)` |
| Cloth | Visual geometry, control mesh, surface binding, parameters and native motion | `asset_configs(folder)` plus `bind_surface_visuals(folder, stage)`; update followers after physics |
| Surface with fixed context (optional native review) | `surface_asset.json`, passive `runtime.json`, original source/GLB, control mesh, visual bindings, fixed analytic fixture descriptions, trajectory and movie | `real2sim_harness.deformable_runtime.load_deformable(configuration, builder=optional_builder)`; fixed probes are explicitly opt-in |
| Hybrid | Manifest, rigid hierarchy, cloth, native attachments, bindings, parameters and motion | `real2sim_harness.hybrid_runtime.load_hybrid_asset(folder)` |
| Strand | Strand configuration, centerline, original detailed meshes, endpoint geometry and native motion | `real2sim_harness.strand_runtime.load_strand_asset(configuration, builder=optional_builder)` |
| Volumetric solid | Closed source control, tetrahedral configuration, original detailed surfaces, bindings, material assumptions and native motion | `real2sim_harness.deformable_runtime.load_deformable(configuration, builder=optional_builder)`; `simulate-volumetric-solids` describes generation and review |
| Fluid | Seed material volumes, particle/material configuration, particle motion and reconstructed surfaces | Dedicated `simulate_particle_materials.py` path; inspect the `simulate-particle-materials` skill |
| Fracture | Intact and fractured geometry, closed piece metadata, PhysX release configuration and native trajectory | `simulate_physx_fracture.py` plus `package_verified_fracture.py`; controller logic is part of the asset |

Cloth visual following uses the actual control state, for example `binding.update(cloth.data.nodal_pos_w.torch[0].cpu().numpy())`. `configure_surface_environment(folder)` installs optional still-air drag once per shared world. Do not register a world force separately for every garment. Source pins and test grasps are distinct from a downstream robot grasp controller.

Strand endpoints are passive dynamic bodies by default; the demonstration driver explicitly selects prescribed motion. A compatible native builder can include other bodies while preserving their IDs. This establishes composability, not an already implemented robot manipulation policy.

The fracture controller releases prefractured bodies when its contact criterion is met. An initial USD without that logic will not independently break on impact. Fluid and hybrid final-state USDs are visualization snapshots, not complete runtime models. A rendered animation is evidence of one configured behavior, not a substitute for a reusable loader.

All materials, dimensions, mass/density, friction, damping, stiffness and supported controls remain adjustable. Current checks establish source integrity, conversion consistency and bounded basic motion; collaborators can add calibration and demanding contact tasks separately. Check the relevant skill and recorded support limits before extending a type.

Native rigid contact review is an explicit alternative for contact-dependent rigid assemblies. Its runtime configuration contains no demonstration wrench controls; the separate test configuration retains them as evidence. Static/held parts are declared in the configuration and can be reconfigured for a downstream task. Source materials can be replayed from the exact retained rigid states; the shared local-surface reviewer also supports those states. The ordinary PhysX rigid route remains the default. A successful native Newton review does not certify a PhysX adapter.

New contact bundles explicitly name `source.blend` as `appearance_source` and
the complete original-source replay movie as `appearance_review`. The `visual`
GLB remains a preview, with `preview_material_equivalence_verified: false`;
procedural Blender shading is not certified by that export. A target-renderer
delivery must additionally preserve and review its own material representation.
Scene packaging of native contact assets uses the standard USD visual exporter,
including its glass and normal-transfer checks. An arbitrary task-local exporter
does not acquire those checks automatically and cannot substitute a color-only
render for verified source-material transfer.

Optional native surface review converts one authored elastic surface and fixed source fixtures to the isolated VBD runtime. `runtime.json` removes test grasps and measurements; fixed analytic probes are installed only with `include_probes=True`. Keep the separate test configuration as evidence and attach downstream manipulation controls explicitly. Collision conversion currently supports world-axis-aligned boxes and spheres, with source geometry and conservative bounds retained for review. The [source-surface contract](../skills/simulate-deformable-behavior/source-surface.md) describes ownership, optional edge/face contacts and acceptance. A successful object review does not establish stack dispensing, plastic folds, absorption, dynamic rigid coupling or a mixed-scene adapter.
