---
name: simulate-impact-fracture
description: Prepare closed editable material volumes and review impact-triggered native PhysX fragmentation, preserving exterior materials and generating distinct fracture interiors.
---

# Impact fracture

Declare kind `fracture`. This is a generic prefracture route for brittle objects, not calibrated ceramic/glass failure or dynamic crack propagation. `run_fracture(attempt_id, frames, assumptions)` accepts 144, 192 or 240 frames at 24 FPS. It cuts closed material geometry with the official Blender Cell Fracture add-on, tests actual PhysX contact-triggered release, then returns native motion evidence and material-preserving previews. Final acceptance requires a successful unchanged native cache. No explosion animation or hand-authored shard trajectories substitute for this.

Create exactly one closed, consistently outward, manifold MESH containing the complete material volume. A hollow vessel needs joined inner and outer walls, rim and floor, rather than an open surface. Preserve openings and handles. Separate open shading patches used by other collision adapters cannot serve as fracture volumes; return to their sealed source construction. Apply or retain evaluable modifiers. Surface appearance, UVs and exterior materials remain editable. Create a separate material for the exposed rough interior.

Tag the source mesh with JSON:

```python
obj['real2sim_fracture'] = json.dumps({
    'pieces': 48, 'seed': 7, 'site_noise': 0.15,
    'interior_material': 'unglazed_interior',
    'density': 2300.0,
    'impulse_threshold_ns': 0.08,
    'friction': 0.55, 'restitution': 0.08,
    'motion': {'drop_height': 0.45, 'rotation_degrees': [20,35,10]},
})
```

These are illustrative assumptions, not a recipe or real material measurements. Density is kg/m³; optional positive `mass_kg` overrides density. Drop height is metres added to the object's authored center. Rotation is XYZ degrees about that center. The fixed plane is Z=0. Choose a readable impact and enough settling time. Geometry should be near its natural support height before the drop offset. There are 8–128 input fracture sites; disconnected islands may create more shards, bounded at 200. Closed-shard volume coverage must match the closed source within 1.5%. Avoid decorative loose surface meshes that would remain suspended after fragmentation; include relevant geometry in the material body or omit immaterial details.

Before impact a single aggregate owns compound shard colliders. Actual contact impulse crossing the editable threshold releases all pieces. The adapter deliberately re-simulates that detected contact from the preceding unbroken substep, transferring rigid pose and velocity `v_i = v + omega × offset`; this preserves incoming motion rather than copying an already stopped aggregate. The report records this one-substep rewind. No radial impulse is added. Native PhysX convex decomposition, CCD, friction and gravity determine fragment motion and settling. This approximation releases all bonds at once; it does not localize cracks or support gradual bond damage. Material physics remain editable for downstream work.

Inspect source-to-shard volume agreement, breakup timing, momentum transfer, fragment scale, initial overlap, floor contact, speed bounds and settling. Successful finite states alone do not certify visual quality. For this topology-changing process, a complete candidate video is strongly useful: call `render_motion_video` once mature and choose chronological coverage with `inspect_video`, including dense frames near impact and the settled state. Look for premature cracks, floating intact surfaces, instant position jumps, disappearing shards, explosive separation, floor penetration and unrealistic rolling. Repair geometry or assumptions and rerun when needed. The agent makes the final judgment after simulator feedback.

Keep source `.blend`, `fractured.blend`, configuration, native pose cache, validation and source hashes together. These are editable native rigid fragments with an explicit release mechanism, not a completed robot cleanup controller. The source generation and fracture adapter are distinct stages. When seeded from an accepted synthetic asset, describe it as extending that existing asset; do not claim a new real-photo reconstruction.

Runtime integration keeps all shard actors dynamic from initialization. Before release their collision shapes and gravity are disabled; release activates those properties and writes the same pose/velocity into both USD and native tensors. This avoids a stale initial-kinematic set overriding the handoff. The first dynamic substep is audited for actor resets. Motion validation also bounds surface speed and native-inertia mechanical energy rather than rejecting small chips solely for high angular speed. Every retained visible mesh is checked against the analytic ground plane, and final native speeds must settle. These checks do not prove hidden mesh-to-mesh contact accuracy.

The integration uses the official Blender Cell Fracture add-on (`object_fracture_cell`, GPL-2.0-or-later). The [source and change record](../../vendor/cell-fracture/PROVENANCE.json) identify the local modifications. Runtime uses the installed Isaac Lab 3 PhysX stage and tensor APIs.
