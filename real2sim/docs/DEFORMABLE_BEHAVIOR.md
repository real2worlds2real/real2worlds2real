# Optional shell and solid behavior tools

## PhysX cloth construction

For garments and cloth, prefer PhysX unless the user chooses another backend.
`real2sim_harness.physx_cloth.resolve_cloth_settings(configuration)` resolves a metric
triangle surface, `material.density_kg_m3`, `material.thickness_m`, optional
`material.total_mass_kg`, `fps`, and optional `physx` overrides. It preserves total
mass across mesh refinement and derives contact offsets from median edge spacing.
These defaults are initial assumptions; boundary quality, layer separation,
manipulation and post-release stability still require validation.

After starting Isaac Sim and creating a meter-scale, Z-up physics world:

```python
from real2sim_harness.physx_cloth import create_cloth, resolve_cloth_settings

resolved = resolve_cloth_settings(configuration)
world.set_simulation_dt(physics_dt=resolved["dt_s"], rendering_dt=1/configuration["fps"])
cloth, particle_system, material, resolved = create_cloth(
    stage, configuration, prim_path="/World/Garment",
    simulation_owner=world.get_physics_context().prim_path,
)
world.scene.add(cloth)
world.reset()
```

The caller owns Kit startup, world stepping, support geometry and grasp controls.
The asset is passive: this constructor never pins particles or installs a demo
controller. Retain the resolved settings with the source and runtime evidence.
The `physx` block exposes substeps, position iterations, stretch/shear/bend
stiffness, spring/material damping, friction, contact/rest offsets, velocity limit
and self-collision options. Unknown fields and invalid offset relationships fail
instead of being silently ignored. Particle CCD is separate from rigid-body CCD.

This is a native Python construction API for the Isaac Sim 5.1 particle-cloth
interface. It does not change the Newton diagnostic tool below, perform visual
reconstruction or certify the garment. Later Sim releases may replace these APIs.
See [NVIDIA's particle-cloth documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/particles/particles.html)
for the contact-distance and spring model.

Externally executed surface trials can use
`source_deformable.retain_surface_evidence(store, attempt_id, folder, detail)`
to register the complete source-matched evidence inventory. The folder must be
inside the source attempt and satisfy the same hashes, numerical/contact checks,
full source-material replay and video requirements as other native surfaces.
PhysX validation records use `backend_id: "physx_particle_cloth"`, a
`runtime_version`, and the complete `resolved_physx_settings`. The passive runtime
retains those settings and omits test grasps. Registration does not execute a
simulation or replace the agent's visual review. The native scene exporter
preserves PhysX surfaces as PhysX bundles, including the passive settings and
source visual binding. Use the [5.1 cloth scene route](SCENES.md#physx-particle-cloth-scenes-on-sim-51)
to run the surface with rigid participants and support collisions; scene
acceptance requires fresh scene simulation and visual review.

## Newton diagnostics

Install `pip install -e '.[solid]'` for MeshPy/TetGen conversion. Native tests additionally need a separately installed compatible Newton/Warp runtime; the tested reference is Newton 1.7.0 / Warp 1.17.0. Blender 4.5 performs retained-state replay. These tools are independent of an Isaac Sim version. They extend diagnostics and do not change existing asset delivery gates.

Both tools take `request_path` (workspace-relative JSON) and `assumptions` (recorded text). The behavior tool additionally takes `render_video`: false returns sampled images; true renders and registers the full movie for `inspect_video` (install the `video` extra). Geometry is inline explicit arrays in a common metre world frame, Z up. Inputs, logs, motion and hashes are retained. The tools are available in object and scene generation sessions; Astra chooses whether they help.

## Tetrahedralization

`tetrahedralize_control_mesh` accepts `vertices` (N×3), `triangles` (M×3), `max_volume_m3` (positive), optional `quality` (1..3, default 1.4), `max_elements` (default 100000), and `max_surface_edge_m` (optional conforming midpoint refinement). Supply a closed manifold with consistent winding and no self intersections. Surface refinement preserves authored planar facets and source vertex IDs; TetGen may then insert or reorder vertices. Output `geometry.json` contains `vertices`, `triangles` and `tetrahedra`; indices refer to the output vertex table. Original vertex IDs are not a persistent grasp mapping across remeshing.

The converter checks manifold edges, degeneracies, signed tetrahedral volumes and the output boundary. It does not certify arbitrary input self-intersection freedom or material correctness. The host executes conversion in a bounded subprocess, since meshing difficult input may take time. Keep input surfaces reasonably simple and inspect the output.

## Native behavior request

`run_deformable_behavior` accepts:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `representation` | `surface` or `volume` |
| `geometry` | `vertices`, `triangles`; also `tetrahedra` for a solid. Solid triangles must match the tetrahedral boundary. |
| `frames`, `fps` | Defaults 120 and 24; 2..1440 and 1..120 integer bounds. |
| `material` | `youngs_modulus_pa`, `poisson_ratio`, `density_kg_m3`, `contact_radius_m`; additionally `thickness_m` for shells. Optional native `damping`, shell `bending_stiffness`, `bending_damping`. |
| `environment` | Optional `gravity` ([0,0,-9.81]), `floor_height` (0), `friction` (.5), `contact_stiffness` (2e5), `contact_damping` (.1), `air_drag_rate_per_second` (0, an explicit mass-proportional still-air force). |
| `solver` | Optional `substeps` (12, 1..128), `iterations` (16, 1..64), `self_contact` (true for shells, false for volumes). |
| `probes` | Optional list: unique `id`, `shape` (`sphere` or `box`), `position_m`, `radius_m` or full `size_m`, `keyframes`; optional `collision=false` for a negative control. |
| `grasps` | Optional list: unique, disjoint integer particle `indices`, `keyframes`, optional `release_seconds`. |
| `measurements` | Optional regional response definitions below. |
| `review` | Optional `maximum_stretch` (1.5), `floor_tolerance_m` (.003), `contact_tolerance_m` (.001). These are diagnostic sanity bounds, not measured tolerances. |

Keyframes are ordered `[seconds, dx, dy, dz]`, initially `[0,0,0,0]`, relative to the initial pose. Smooth interpolation stops at the last target. Particle grasps are actual explicit constraints; releasing restores dynamic particle flags. Probes stay kinematic. Their target trajectories are separate from the deformable's native response. Friction and mass remain editable assumptions.

Shell coefficients derive from plane-stress Lamé parameters times physical thickness; mass uses areal density `density_kg_m3 * thickness_m`. Default plate rigidity is `D = E h³ / (12 (1-ν²))` in N·m, optionally overridden with `bending_rigidity_nm`. Each hinge receives `D / dual_width`, because Newton multiplies its coefficient by edge length internally. This keeps the energy of corresponding hinge shapes consistent under uniform refinement. `bending_stiffness` instead supplies an explicit uniform native coefficient and cannot be combined with physical rigidity; `bending_damping` is also a native coefficient. The volumetric adapter passes 3D Lamé parameters to native stable Neo-Hookean VBD FEM and currently requires nonnegative volumetric lambda. These mappings initialize a model; they do not identify real material properties.

Optional `solver.self_contact_margin_m` sets the positive interaction distance for native self-contact; `solver.self_contact_gap_m` sets nonnegative additional detection reach. These are separate from the material particle radius used for rigid contact. When omitted they retain the existing defaults of twice that radius and that radius, respectively. Inspect thickness, mesh scale, folds and actual motion before overriding; a very small detection reach can miss contacts. Reducing false repulsion does not establish self-intersection freedom, and disabling self-contact is a diagnostic rather than a general repair.

For each measurement supply unique `id`, particle `indices`, nonzero `axis`, `baseline_window_seconds: [start,end]`, `response_window_seconds`, optional `minimum_response_m`; optionally `recovery_window_seconds` and `maximum_residual_m`. Every window must contain retained timestamps. The baseline is the mean regional centroid projection after settling; response is its maximum signed change in the response window; recovery residual is the absolute change of the recovery-window mean. Positive axis direction defines the expected motion. This does not measure force or prove contact.


`validation.json` includes finite-state, floor, intrinsic shell stretch/collapse or volumetric Jacobian checks; per-measurement pass/fail; mass; and hashes for `request.json` / `motion.npz`. A separate triangle-sphere audit detects contact between vertices and solid containment, beyond particle contact candidates. Non-sphere probes are explicitly listed as outside this geometric audit. The NPZ retains every frame's positions, velocities, body poses and times. Broad-phase contact candidate counts are explicitly not measured contact forces or full pair coverage. Numerical checks apply to retained frames, not all continuous motion, self-intersections or detailed visual mesh intersections. The host caps retained vertex-frame samples at 20 million.

CLI equivalents:

```bash
python -m real2sim_harness.deformable_tools --request meshing.json --output meshed
python scripts/simulate_deformable_behavior.py behavior.json --output evidence --device cuda:0
blender --background --factory-startup --python scripts/render_deformable_behavior.py -- evidence --mode preview
blender --background --factory-startup --python scripts/render_deformable_behavior.py -- evidence --mode render
```

The renderer verifies request/cache hashes and frames the full retained envelope. It displays the actual control boundary and physical fixtures; detailed source appearance binding and final deliverable acceptance are separate stages. `preview_report.json` returns seven images, while render mode retains every frame and an `animation_report.json` for encoding. Existing Blender cloth and rod asset routes continue to use their source-bound acceptance and video tools.

For the source-bound volumetric asset route, declare kind `solid` and use `run_solids`; see the `simulate-volumetric-solids` skill. It exports the closed Blender control, remaps authored selections after meshing, binds the evaluated detailed visual meshes, runs native FEM and requires fresh-process source-material replay before acceptance. The optional diagnostic tool above never sets this asset's acceptance fields.

## Reusable native loading

`real2sim_harness.deformable_runtime.load_deformable(configuration, device="cuda:0")` loads shell/solid geometry as passive dynamic particles by default. Demonstration grasps and probe bodies are absent. The diagnostic runner explicitly opts into `prescribed_motion=True, include_probes=True`; a video's test controller is not a permanently constrained asset.

An optional prepopulated Newton `builder` lets existing compatible rigid bodies, particles or robot construction share the finalized VBD world. `particle_ids` identifies the appended control particles without changing existing IDs. An external builder retains its gravity and gets no additional ground by default; callers may choose `include_ground=True`. The loader returns the finalized model, solver, states and collision pipeline; the caller owns controls and stepping. Optional `apply_deformable_environment(runtime, state)` adds the configured still-air force only to this asset's particles after forces are cleared. This interface does not supply a robot controller or certify a manipulation task.
