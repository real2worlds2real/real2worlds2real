# Optional source-bound native surface review

Use this branch when an authored surface must interact with fixed support/obstacle geometry, including narrow passages. The ordinary cloth/plane check does not establish correct interaction with an unmodeled fixture. This branch is independent of an Isaac Sim release and uses the isolated native Newton runtime. Dynamic articulated rigid participants remain a separate hybrid/runtime adapter concern.

Call `run_deformable_behavior` with a workspace-relative request, explicit assumptions, `render_video`, and `record_asset_evidence`. False records diagnostics only. True selects this source-bound VBD review for final delivery and requires a saved `deformable` attempt plus a complete movie. It does not itself accept an asset.

Source request fields (in addition to the ordinary behavior material, solver, environment, controls and measurements):

- `source_attempt_id`: saved successful Blender attempt.
- `representation`: `surface`.
- `control_object`: one baked mesh, without modifiers/shape keys/another physics solver. Control vertex indices define grasp/measurement selections. It may be the visible source surface or a hidden `real2sim_sim_proxy` control.
- `visual_objects`: unique visible source mesh names driven by the control. Evaluated static detail modifiers are supported when viewport/render settings match. Coarse controls can drive detailed followers through retained barycentric bindings.
- `static_visual_objects`: the other visible fixed source meshes. Every visible source asset mesh must appear in exactly one list.
- `probes`: each fixed source object must have a corresponding probe with `source_object`, `id`, and `shape` (`box` or `sphere`). Positions/dimensions derive from evaluated source bounds. Boxes must be world-axis aligned; sphere bounds must have equal extents. This is an explicit analytic collision approximation, whose conservatism must be reviewed. Source-bound probes cannot have nonzero motion keyframes. `collision:false` explicitly denotes visual-only context; do not use it to hide required contacts. Additional diagnostic probes without source objects retain the ordinary protocol.
- `frames`, `fps`: output duration/sampling, not a substitute for physical substeps.

For example, the naming portion of a request can be:

```json
{
  "source_attempt_id": "attempt-001", "representation": "surface",
  "control_object": "Control", "visual_objects": ["DetailedSurface"],
  "static_visual_objects": ["SupportPanel"],
  "probes": [{"id":"support", "shape":"box", "source_object":"SupportPanel"}]
}
```

Add the full material/solver/controls and any task measurements; this fragment is not a complete physics request. Source geometry and all lengths use metre coordinates.

For thin surfaces near obstacle edges, consider `solver.full_surface_contact:true`. It enables native edge/face contacts and the solver-owned pipeline; particle-only collision can miss walls between nodes. Self-contact remains a separate choice. Contact parameters, thickness, bending and scale remain editable assumptions.

Measurements support optional `minimum_final_point_projection_m` and `maximum_final_point_projection_m` on a selected region and normalized axis. These check every selected point at the final state, in addition to the regional centroid response. A selected region can thereby be required to clear an independently defined support/exit plane. The triangle/box SAT audit checks retained surfaces beyond particle-only contacts. Neither is continuous-time certification or real material identification.

A source-bound review preserves source hashes, native states, bindings, original-material replay and the encoded complete movie. `runtime.json` removes demonstration grasps/measurements; fixed analytic probes remain available only when the caller explicitly includes them through `load_deformable(..., include_probes=True)`. Default loading installs no demonstration grasps. `surface_asset.json` inventories the reusable source/runtime representation.

Before delivery, inspect the returned original appearance and relevant movie intervals; use `inspect_surface(frame=...)` for native-state material/clay/grazing views when useful. Then `accept_result` runs the ordinary final gate. Source/binding/cache/movie changes invalidate prior evidence. A later ordinary `run_simulation` reselects that adapter and must satisfy its own checks. Plastic creases, stack separation, absorption and robot policies are not supplied by this elastic-surface route.
