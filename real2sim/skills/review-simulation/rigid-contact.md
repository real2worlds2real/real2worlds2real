# Thin walls, cavities and rigid contact

Read this when an operation depends on an opening, internal surface or thin
barrier, or when a rigid object passes a drop test but fails insertion or nesting.

## Choose the representation and execution separately

CPU/GPU selects where physics runs. Compound convex pieces/SDF selects how a
surface is represented for collision. Rendering is another choice: CPU physics
can still use GPU rendering. Supported combinations depend on the backend and
runtime version; do not infer them from the availability of a renderer.

| Geometry and operation | Useful starting point | Main limitation to check |
| --- | --- | --- |
| Simple solid body | Primitive or convex collider | Required openings must remain open |
| Regular hollow body with readily described walls and floor | Explicit compound convex pieces | Coverage, seams and agreement with the material volume |
| Detailed concave mesh that is awkward to partition | SDF where supported | Thin-feature resolution, mesh quality, contact count and memory |

Automatic convex decomposition may fill a cavity or smooth a narrow opening.
A larger hull count alone does not prove the interior is usable. Prefer the
simplest representation that preserves the contacts needed by the task. SDF is
an alternative, not an automatic accuracy upgrade.

For a small contact diagnostic, CPU physics is a reasonable baseline when it
offers needed features. GPU physics is useful for throughput and supported
complex workloads. Do not force one device across unrelated tasks. Inspect the
operator's configured adapter/device rather than assuming a physics-parameter
override changes the execution device.

For bundled object-conversion checks, the operator can select
`REAL2SIM_PHYSICS_DEVICE=cpu` or `cuda:<index>`; the default remains `cuda:0`.
Scene runners select their device independently (the example accepts
`--device cpu`). These physics selections do not switch off GPU RTX rendering.
Use resources authorized for the task; selecting a device does not allocate it.

## Source metadata retained by this adapter

- Compound pieces: author mesh objects with `real2sim_collision_proxy=True`,
  `hide_render=True` and `real2sim_collision_approximation="convexHull"`. Set the
  replaced visual mesh's `real2sim_collision=False`. Proxies retain geometry and
  rigid ownership while remaining separate from visual export.
  The current visual GLB/USD paths exclude `real2sim_collision_proxy` directly;
  no additional `real2sim_sim_proxy` workaround is needed. Inspect visual nodes and
  physical collider members separately after changing roles.
- SDF: set `real2sim_collision_approximation="sdf"` on the intended collision mesh.
  Optional `real2sim_collision_sdf_resolution` is an integer in 16..1024; the adapter
  defaults to 256. Grid spacing is approximately the longest mesh extent divided
  by resolution. Resolve the thinnest material rather than copying a resolution
  from another asset. Use watertight material-volume meshes with adequate
  tessellation. A cup can have a closed material volume and an open mouth.
- Per-mesh `real2sim_collision_contact_offset_m` controls how early contacts are
  considered; it must be positive and exceed `real2sim_collision_rest_offset_m`.
  The rest offset shifts the effective contact separation. Neither changes the
  visual mesh. Retain and disclose nonzero rest offsets rather than concealing
  an inaccurate collider by inflation or shrinkage.
- Optional `scene["real2sim_enable_ccd"]=True` initializes `rigid.enable_ccd` in the
  physical parameters. Explicit parameter overrides remain available. CCD helps
  prevent fast bodies crossing thin barriers between discrete steps; it does
  not correct an inaccurate collision shape.

These are PhysX adapter choices. Other backends need their own supported
representation. Keep mass, friction, dimensions and hidden wall thickness
editable and distinguish inferred assumptions from measured evidence.

## Confirm effective runtime behavior

Sweep-based CCD requires both body and physics-scene flags. The current GPU
dynamics adapter rejects a CCD request because the engine would disable it in
that execution mode; supported CPU physics runs enable the scene flag. This is
not a general claim that GPU physics cannot handle hollow objects or that every
PhysX version has identical restrictions.

Inspect effective scene/body properties and startup warnings. An authored flag
or a report listing requested paths alone does not establish that the engine
used the feature. A runtime warning that disables a requested setting requires
a supported configuration or an explicit failure, not a successful-feature
claim. Keep a change of device or collision strategy visible in the evidence.

Sweep-based CCD is not an unconditional improvement for dense thin-wall contact.
Its finite advance passes can discard part of a step, especially for very thin
shapes; reported velocities may then coexist with stalled or stuttering poses.
If this occurs, inspect intermediate material gaps and actual pose advance before
fitting physical parameters. Compare supported CCD settings or smaller-step
discrete contact while retaining the required colliders. Verify transient contact
again and record the effective time step; disabling CCD alone is not a repair.
The [PhysX CCD tuning and limitations](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html#continuous-collision-detection)
explain the advance-pass limitation; check the behavior of the installed runtime.

## Verify the operation, not just loading and settling

A normal drop establishes only the tested external support contact. If the
requested function depends on an interior, exercise insertion, nesting or
containment in a small fixture before rendering a long scene movie. Keep
geometry, initial placement, controls and success criteria consistent when
comparing configurations; change one relevant factor when feasible.

Inspect intermediate poses and retained physics states, including the contact
transition, as well as relevant visual views. A body can tunnel through a floor
and settle in a nonintersecting final pose. Where applicable, check floor
ordering, containment or separation as well as sampled material penetration,
achieved insertion depth and residual motion. Such invariants need to match the
geometry and motion; an upright centered fixture does not certify arbitrary
orientations. State sampling does not certify continuous-time nonpenetration.

Before insertion/containment probes, check the entire probe volume against every
relevant material piece, not just its center or one wall. For spherical starts
against convex pieces, use the conservative clearance audit in
[review-numerical-evidence](../review-numerical-evidence/SKILL.md) or equivalent
native evidence. Unresolved corner cases require diagnosis, not an invented
collision claim. A useful functional check should distinguish a usable opening
from a blocked control; an exterior drop alone cannot do this.

For an immediate kick or sideways escape, inspect the first retained states and
initial overlap before changing passive geometry or friction. A changed release
pose is a diagnostic intervention; preserve the intended operation and report it.

For slats, gaps or beveled supports, a nominal support plane and a body-center
footprint test cannot establish contact with actual material. Transform retained
collider/visual geometry with the declared quaternion convention and inspect
overlap against individual support solids at the relevant times. Keep a concrete
geometric witness and corresponding native frame. Exported-surface intersection
is not a measurement of the solver's cooked hull or contact forces.

If bodies respond but contact callbacks are empty, investigate instrumentation
before concluding that contacts are absent: inspect effective contact processing,
report APIs, subscriptions and pair filters in the installed runtime. Verify the
selected device implements the control API actually used and compare actual
readback with commands. Do not apply another engine's global setting or silently
switch devices. Missing instrumentation and missing physical response are
different findings; neither establishes successful contact verification.

Choose tolerances from task scale and intended use; a tolerance from one demo
is not a universal delivery bound. Keep failed attempts and avoid loosening a
bound, freezing an end pose, disabling a required collider, changing visible
dimensions or driving a prescribed trajectory merely to obtain a pass.
Simulation success with assumed parameters does not establish real-world
physics calibration. Fresh source-matched checks and visual review are needed
after a repair; existing acceptance does not transfer to changed source.

## Primary references

- [Omni Physics colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html): compound/SDF colliders, compatibility and contact offsets.
- [Collision behavior guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/collision_guide.html): thin-feature resolution, tessellation and contact tuning.
- [Isaac Sim Simulation Manager](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/api.html): CCD restrictions in GPU dynamics. Check the installed runtime when applying version-dependent guidance.
