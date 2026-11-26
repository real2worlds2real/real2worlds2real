---
name: simulate-particle-materials
description: Generate and review flowing or viscous materials from Blender volumes using Newton MPM, with reconstructed liquid surfaces and retained motion evidence.
---

# Simulate particle materials

Use `run_fluid(attempt_id, frames, assumptions)` for a saved fluid attempt. It samples the source Blender volumes, advances Newton's implicit MPM solver, reconstructs changing surface meshes and renders seven actual motion samples using the source Blender materials. Choose 48, 72, 96, 120, 144 or 192 frames at 24 physical FPS. Only the final chosen result needs a full video; tool iterations return still samples. `render_motion_video` can render a final passed MPM candidate and return it to `inspect_video`.

Declare the usual scene contract with `real2sim_asset_kind="fluid"`. Give every material phase a separate closed mesh with one visual material and a JSON string property:

```python
import json
volume["real2sim_mpm_material"] = json.dumps({
    "density": 1000.0,
    "viscosity": 10.0,
    "friction": 0.0,
})
```

The tool keeps Newton's liquid-like defaults for omitted properties, including `young_modulus=1e15`, `poisson_ratio=0.3`, `tensile_yield_ratio=1.0`, `yield_pressure=1e12`, and zero shear yield stress. These are solver assumptions, not measured material claims. Viscosity is adjustable in Pa·s; explore it through observed flow rather than interpreting the default as a real material identity. Lowering the near-incompressible stiffness requires a new stability check.

Scene-level optional settings are `real2sim_mpm_particle_spacing` (default 0.0015 m), `real2sim_mpm_voxel_size` (default twice spacing), `real2sim_mpm_surface_voxel_size` (default 0.55 times spacing), `real2sim_mpm_substeps` (2), `real2sim_mpm_iterations` (60), and `real2sim_mpm_floor_z` (0 m). These control numerical resolution and execution, independently from the material appearance.

Use enough samples through a thin volume; fewer than 50 interior particles is rejected. The review budget is 600k candidate lattice points per volume and 180k particles total. Start with a modest resolution; increase it only to address a visible or numerical defect. The source shapes set the initial state. The solver supplies the subsequent motion.

Higher `real2sim_mpm_priority` volumes are sampled first; overlapping lower-priority volume samples are excluded. This permits nested material phases without initializing duplicate matter. Default priority is zero with stable name ordering. Prefer clearly separated interiors or explicit priorities for nested phases.

The default support is a static flat plane at `real2sim_mpm_floor_z`. For finite fixed axis-aligned supports, set `scene['real2sim_mpm_static_colliders']` to a JSON list such as `[{"id":"surface","size_m":[0.8,0.6,0.04],"center_m":[0,0,0.73]}]` (full dimensions and center in metres). `scene['real2sim_mpm_use_ground_plane']=False` disables the infinite plane. These boxes are imported as actual Newton MPM colliders and receive sampled particle-center penetration checks. Their visible meshes still need matching placement and dimensions. Geometry marked `real2sim_fixture=True` is not automatically an MPM collider. Rotated/curved containers, moving bodies and two-way dynamic rigid/fluid coupling need a further adapter; unsupported box fields are rejected. A static-support demonstration does not certify those capabilities.

The original Blender material is preserved on each reconstructed surface, so use physically coherent transmission, IOR, roughness and optional absorption/scattering nodes. Transparent liquids need thickness, suitable lighting, and a visible support to read well. Use actual geometric volumes; a flat alpha decal cannot define the particle interior. Follow the visual input for color, opacity, proportions and essential material phases.
Reconstruction changes topology and does not advect UV maps. Use homogeneous materials or generated/object/world-coordinate procedural detail for these surfaces instead of relying on the seed mesh's UV layout.

Inspect early, middle and final samples for spreading or settling, material separation, surface continuity, contact and readable highlights. Footprint ratios and height changes are quantitative evidence; a finite simulation alone does not prove the requested appearance or motion. If the result is wrong, revise the source volume, numerical settings, material parameters or visual materials and rerun. The tool retains source hashes, the particle trajectory, per-frame surface meshes and a fresh Blender-process replay check. A static GLB remains only a snapshot.

This uses Newton's [implicit MPM](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverImplicitMPM.html) and [particle-surface reconstruction](https://docs.nvidia.com/isaaclab/3.0/concepts/implicit-mpm), rather than a category-specific motion script.

For a requested curing/setting/heating appearance, an optional JSON `real2sim_mpm_transition` property on each volume provides a prescribed irreversible phase model:

```python
volume['real2sim_mpm_transition'] = json.dumps({
    'start_seconds': 1.0,
    'duration_seconds': 4.0,
    'height_origin_m': 0.0,
    'height_delay_seconds_per_m': 100.0,
    'material': {'viscosity': 50.0, 'yield_stress': 300.0},
    'appearance': {'base_color': [0.9, 0.9, 0.85, 1.0],
                   'roughness': 0.3, 'transmission': 0.0,
                   'subsurface_weight': 0.03, 'subsurface_scale': 0.0005}
})
```

The values above illustrate the schema, not a universal recipe. Choose timing, mechanical targets and appearance from the requested effect and actual feedback. Optional height delay starts lower material earlier; zero gives a spatially uniform schedule. A smooth progress field is carried irreversibly by particles. It changes actual Newton viscosity/yield parameters before stepping, and the same particle state is interpolated onto reconstructed surfaces to blend the original shader with the target Principled surface. Material-only or appearance-only transitions are possible; clearly state which one was demonstrated. Supported mechanical targets are viscosity, yield_stress, yield_pressure, young_modulus, friction and tensile_yield_ratio. Appearance also supports ior, coat_weight and coat_roughness.

This is an editable phenomenological stage model, not a heat-transfer PDE, temperature prediction, chemical kinetics or measured cooking time. A white appearance alone does not establish setting: inspect reported native material ranges and actual flow/motion. Keep distinct phases recognizable and look for premature freezing, shell-like artifacts, discontinuous whitening, volume loss or excessive opaque thickness. A high yield stress represents resistance to further flow, not a certified rigid body or calibrated soft-solid mechanics. Review the final movie when temporal continuity matters, then repair and repeat as needed.

A source material may contain an initially unconnected Principled node named
`real2sim_transition_target` for a procedural cooked/settled surface. Replay reuses
that node as the target; scalar appearance fields set its unlinked defaults,
while authored links supply spatial colour or relief. Use object/world or
normal-based patterns, not seed UVs. Without a template, replay creates the target
and retains any procedural source normal. Review these shader details alongside
the actual particle progression; they do not establish thermal mechanics.
