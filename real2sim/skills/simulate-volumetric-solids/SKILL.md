---
name: simulate-volumetric-solids
description: Generate volumetric soft bodies such as foam, sponges or rubber using a closed Blender control, native tetrahedral FEM and source-bound material replay and acceptance.
---

# Volumetric soft bodies

Follow an explicit user backend choice. Otherwise default volumetric soft bodies
to **PhysX**. The `run_solids` implementation documented below currently uses
Newton VBD FEM; select it when Newton is requested, rather than treating tool
availability as a reason to change the target. A missing PhysX volume adapter
is an integration gap to address explicitly.

## Existing Newton route

Use asset kind `solid` for an explicitly modeled elastic volume, such as foam or a rubber part. Cloth, tissue and thin bags remain surface models. A cushion modeled as a homogeneous effective solid is an assumption about its cover and filling; do not call that physical identification. Keep reference-driven modeling in Blender, then call `run_solids(attempt_id, frames, assumptions)` for native review. Available lengths are 120/144/192/240 frames, default 24 FPS.

## Authoring contract

Declare normal scene contract fields with `real2sim_asset_kind='solid'`, front/up axes and metre units. The current adapter supports one homogeneous closed control per asset, with one or more detailed visible meshes. A control may contain multiple disconnected closed components sharing the material model; their interactions still need review.

Use a baked triangle-capable MESH control without modifiers or shape keys. Set `real2sim_sim_proxy=True` to keep it editable but hide it from visual export. Tag `real2sim_solid` with JSON:

```python
control['real2sim_solid'] = json.dumps({
    'material': {
        'youngs_modulus_pa': 30000.0, 'poisson_ratio': 0.2,
        'density_kg_m3': 100.0, 'contact_radius_m': 0.0015,
        'damping': 0.05,
    },
    'max_volume_m3': 1e-5,
    'max_surface_edge_m': 0.015,
    'visual_objects': ['visible_surface'],
})
```

Numbers illustrate an assumed model, not a recommended material for every object. Optional `quality` and `max_elements` bound tetrahedralization. `max_surface_edge_m` adds conforming midpoint refinement without changing authored planar facets. Contact features need adequate boundary resolution: tetrahedral volume alone does not guarantee that a small probe cannot penetrate between surface vertices. Use the returned triangle-sphere witnesses to diagnose this; do not loosen tolerances to hide a coarse contact boundary.

Every visible asset surface must appear in `visual_objects`. Use editable meshes without shape keys or another deformation solver. Subdivision and appearance modifiers are allowed when viewport/render settings match. Conversion binds the evaluated detailed mesh; replay bakes that evaluated topology locally and preserves its materials/UVs. The original `.blend`, control and modifiers stay editable. This avoids driving only a few coarse render vertices and losing a local indentation.

Optional `scene['real2sim_solid_behavior']` JSON accepts `environment`, `solver`, `probes`, `grasps`, `measurements`, `review`, and `fps`, using [the shared behavior schema](../../docs/DEFORMABLE_BEHAVIOR.md). Source grasp/measurement indices refer to the authored control vertices; conversion remaps them to tetrahedral particles and rejects a moved/missing correspondence. New mesher vertices are not silently treated as original selections. Choose a measured region genuinely under the tool rather than averaging most of the unaffected object.

## Review and delivery

The tool converts the closed boundary, binds the detailed visual surface, advances native Newton stable Neo-Hookean VBD FEM, and returns actual motion samples with the original material in another Blender process. Review the shape, response, recovery, contact and attachment details. Optional `inspect_surface` supports a native cache frame; `render_motion_video` exposes the full retained motion to `inspect_video`. Intermediate tests are optional; successful source-bound native evidence is required to accept a solid asset.

Finite states alone do not establish behavior. Inspect tetrahedral Jacobians/inversions, stretch and floor/probe clearance, and choose useful regional measurements. The shell/solid diagnostic skill describes control/negative-control reasoning. Failure returns remain available to guide source repair. Rebuild and rerun after changing geometry, material, controls or bindings; changed evidence invalidates acceptance.

Volumetric self-contact is off by default, consistent with the upstream native FEM examples. Applying particle-versus-boundary self-contact indiscriminately can treat interior particles as colliders when a volume is refined. This route does not certify volumetric self-contact; external collider contacts and element-validity checks remain active. A new task needing bulk self-contact requires a compatible, separately validated contact model rather than enabling shell contact settings blindly.

The retained native package contains the original source, explicit metre-space tetrahedra, material and control assumptions, detailed bindings, full motion and hashes. `load_deformable(configuration)` loads passive geometry by default; demonstration probes/grasps are opt-in. Compatible bodies can share a supplied Newton builder. This is an operable native asset interface, not a robot policy or a claim of calibrated real-world dynamics. A target scene still requires its own compatible materialization and interaction validation; object acceptance alone does not certify a mixed scene.
