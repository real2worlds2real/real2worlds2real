# Task motion, optical evidence and rigid contact diagnostics

Maintained reference for [the modeling skill](../SKILL.md). Apply the sections relevant to the current representation and task; preserve all applicable local gates.

When the user requests a task from its captured initial state, declare the
`task_motion` request in `render_motion_trace.py`'s manifest: bind the actual
result, state file, protocol, ordered samples and requested phases. A preparation
recording cannot substitute for that sequence. Preserve failed attempts as
failed; do not animate a missing grasp or lift. See the task-video contract in
[object mechanisms](../../../docs/scene/object_mechanisms_plan.md).

For transparent bodies, separate observed opaque print from transmitted scenery.
Never bake a table, tape or a reflected/mirrored rear label into the front-wall
albedo. The experimental `scene.objects.blender.optical_print` tool proposes a
front-owned dark-ink layer from agent-reviewed source regions and unchanged
physical triangles. Inspect source regions, masks and native calibrated renders;
its output is unapproved. A `fix.look.optical_print` layer needs complementary
maps and a resolved source/mesh/hash-bound review. Unseen lettering remains
unknown. Native opacity and source-ray visibility must agree; printing never
removes solid collision geometry. See `docs/scene/object_review.md`.

For an opaque label on transparent material, `optical_print` also supports
`mode: opaque_patch` with independently reviewed source `polygons_xy`. Preserve
the whole backing and source colours, not just dark lettering. Isolated native
material sweeps are static diagnostics, not scene or physics approval.
Before unforced rigid diagnostics, inspect every colliding independent part
pair, including loose contents against each other. Save the authored failure;
settling cannot repair the exported initial state retroactively. The shared
`rigid_initial_state` screen samples actual closed surfaces in both directions
and preserves hollow cavities; its finite sampling is not an exhaustive
intersection certificate. Optical rays through glass are hypotheses with
thickness/IOR uncertainty. Review source identity and camera/geometry fit
before inventing bends or retention to reconcile an impossible source state.

For repeated thin shafts, verify whole-line correspondences separately from
physical tips. `landmark_sets.triangulate_lines` fits image-space line residuals;
`triangulate_line_sets` proposes unordered axes from reviewed distinct segments,
including partial views. Review duplicate masks and cross-view identities in
source RGB. Keep per-view errors, the original full-object cache and all native
checks. An infinite-axis proposal identifies neither finite length nor support.
Use `landmark_sets.line_plane_uncertainty` to inspect how endpoint pixel noise
propagates to the axis at the actual contact plane. Low image residual alone
does not establish a precise hidden endpoint. Its local independent-noise
estimate excludes calibration error and correlated nearby frames; retain that
limitation and do not turn it into a gate relaxation.
For a visible circular edge, `source_circle.fit_circle` independently estimates
the opening from calibrated edge arcs, with per-view residuals and positive-depth
checks. Label the same physical boundary in each image, leave occluded arcs
unobserved, and distinguish the top opening from an inner sleeve or reflection.
Keep edge labels and camera hashes with the proposal. Inspect omitted-view
sensitivity separately; neither a circular hypothesis nor a small residual
proves that an unseen inner wall supplies a required contact.
When a functional boundary is dark or washed out, use
`scene.capture.exposure.detail_brackets` on the original HDR source, with the
capture calibration and explicit detail crops. Compare positive as well as
negative exposure where relevant. The QA manifest binds the video, calibration,
conversion and extracted pixels; canonical images and their labels stay intact.
Exposure variants are the same observation, not extra views. An unresolved dark
interior remains unknown; do not infer a retention feature from task success.

For loose contents, separate user-confirmed support facts from a chosen contact
family. A confirmed floor contact does not imply that every shaft must also
touch a particular rim or the floor perimeter. Allow source-supported free
floor positions and inter-object contacts before rejecting straight geometry.
If the input explicitly permits approximate poses of secondary loose parts,
record the scoped requirement in `task.json.acceptance` and stop optimizing
their exact image poses. Preserve count, approximate source-derived dimensions,
independent rigid bodies, valid contacts and the primary manipulation passage.
Unknown interior construction remains unknown. Keep previous source failures
and require separate unrelaxed vessel/target geometry checks; a whole-assembly
appearance waiver must not hide an inaccurate opening or manipulated object.
The optional `check_rigid_assembly.py --force-probe-plan` applies a bounded
translation force at the actual native centre of mass, then releases it
completely. An explicitly declared orientation tool adds bounded torque, never
a joint or prescribed pose; both forces and torques vanish during release.
`--companion-plan` adds one current rigid asset with a hash-bound, rigid-only
initial experimental placement. It cannot resize or deform the source mesh.
The runner records actual native states, applied wrenches and contact penetration. This diagnostic
does not certify removal, insertion, full-scene support or delivery by itself;
inspect actual clearance and the complete trajectory before interpreting it.
Check tool capacity against each sampled native mass before a gravity-compensated
trial. A saturated tool that cannot hold the object is not controlled insertion.
Apply forces at the native centre of mass, and record force-free release.
Monitor every relevant rigid pair, including contacts between unactuated loose
parts. Keep peak-time native poses for comparison with actual triangles; a
large discrepancy with reported contact depth needs a numerical/collider
investigation. Do not silently discard either measurement. Contact sensors
must not change the scene's stabilization settings.
Save every native rigid timestep and run `apps/scene/check_rigid_trajectory.py`
against the unchanged source solids. Video-rate frames and the largest reported
contact alone can miss short intersections elsewhere. Report spatial sampling
and tolerance; this remains finite sampling, not a continuous collision proof.
Convex cooking may simplify thin cross sections even when its returned vertices
are a subset of the source. Read back the cooked hull and measure surface error.
Coordinate conditioning may change neither world geometry nor metric cooking
limits. Do not split SDFs at material boundaries unless each resulting solid is
closed; a Boolean-unioned shaft/head often has open material subsets.
For tool tests, an optional minimum-jerk target, end hold and release ramp can
avoid artificial start/stop impacts. Keep finite wrench bounds, record the
actual response and require a fully unforced final interval.
For an orientation fixture, use the complete native COM inertia tensor in the
body frame and rotate its principal axes into world space for damping. A thin
object's smallest inertia is not a valid substitute for its transverse moments;
that can create an underdamped test tool. Validate the fixture separately, bound
all torques and keep all actuation zero after release. Record the actual tensor.
For a straight insertion, `tasks.rigid.insertion_corridor` can propose a path
past existing contents. Its capsule/envelope geometry is a proposal only:
validate the full manipulated surface, opening and native release. A failed
straight-down press alone does not prove an impossible task; conversely,
finding a free proxy corridor does not prove the actual asset can be inserted.
Check the head, handles and other attached geometry throughout the sweep, not
only the insertion shaft. `swept_point_clearance` can screen sampled full-surface
paths against enclosing capsules; it does not approve moving-contact dynamics.
Check the requested stopping pose against the complete source surface as well.
A tilted flat-ended rod can intersect the floor even when its tip centre is
above it. `stop_before_plane` shortens a proposed translation using all vertices
and a disclosed support half-space; it never edits geometry or prescribes body
motion. Reject initially intersecting setups and retain the actual contact and
force-free release checks after planning. Increasing solver precision cannot
repair a tool target that intentionally presses a rigid surface through a floor.
Keep all assay placement changes on the rigid body's transform. Baking a rigid
rotation/translation into collider vertices can change its native SDF grid even
though the visible shape is unchanged; such a probe does not qualify the
delivered collider. Preserve source vertices/faces and validate the actual body
pose mapping in recorded trajectories. Check native SDF/convex cooking fidelity
against the source surface, not only the requested USD attributes. A numerical
contact correction requires measured error, explicit separation bounds and fresh
all-body contact/release/task tests; it is not permission to loosen the gates.
`fit.source_pose_groups` can describe indistinguishable rigid members with one
independently acquired collection mask. Each group requires an exact input
relaxation `source_pose:<object>/<group>`, a named dimension check in the
required native criterion, unrelaxed body-source checks and actual-solid
screening. Group masks must never be copied into purported per-member labels.
Raw source failures remain saved; absent or stale functional proofs fail.
Complete the finite solids, screen all pairs, and test unforced native motion;
a better source fit alone cannot approve retention or equilibrium. Hidden wall
and floor construction remain uncertain, not knobs to tune to task success.
Include mutual contacts when source-resolved shafts approach one another.
Keep alternative contact graphs as separate hypotheses; neither a near-contact
distance nor a fitted contact graph proves force balance. Test existing material
prior cases before attributing drift to friction. A source-derived property
bound must be identified independently of task success, with geometry ambiguity
and the raw failed cases retained.
When repeated native trials drift despite near-contact geometry, use a simple
positive/negative native control to test the numerical representation first.
`assets.rigid_statics.point_contact_balance` can then diagnose a supplied contact
graph using gravity, explicit masses/centres and friction priors. It enforces
equal/opposite internal forces and admits no adhesion or holding moments.
Its polygonal friction cones and assumed contact points are approximations:
feasibility is necessary under those assumptions, not proof of equilibrium in
Isaac. Do not create missing contacts or tune material priors merely to make
the linear solve succeed. Keep the native source-hold and full task checks.

Support parts need their own visible evidence. Review mistaken source instances
before changing geometry; `fit.part_identity_review` binds corrections to source
RGB and archived mask hashes and makes those pixels unknown. New support rounds
check declared material parts separately with actual whole-support occlusion.
A tabletop aggregate score cannot approve wrongly placed legs. Small source
fragments retain diagnostics while search weighting uses observed pixel counts.
See `docs/scene/visual_robot_support.md` for selective shape fitting, bounded
multi-start search and fixed-resolution final checks.
Preserve the source texture sampler through GLB-to-USD transfer; clamping a
repeated wood texture can turn most of a table into a flat edge colour. When
choosing a repeated source patch, reject compact marks rather than painting
them out or repeating screws across the surface. Keep original RGB and patch
provenance, and review unique visible marks separately.
Use the actual modeled support in native task validation. A diagnostic slab
cannot certify contact with the exported table. Preserve pre-union Blender
solids in support rounds; `collision_representation: source_convex` may use
them only when every source solid is closed and convex. Split large convex
polytopes into exact cells for the native vertex limit, never convexify a
concave table or fill the space beneath it. Read back cooked geometry and test
actual support contacts. Native invalid-material contact warnings are failures,
even when the process exits successfully. Numerical cooking fallbacks require
a revised declared profile and a new trial before any acceptance.

For a finite-thread release failure, inspect the full saved pose/wrench trace,
actual torque saturation, lifting load, lateral drift and contact separations
before changing geometry or material. The external tool can use gravity
compensation and finite viscous damping with no axial position target. State
its net lifting force and derive a finite tangential torque budget from the
actual cap radius; this budget is not a measured object property. Scale explicit
guide damping to actual mass/inertia and timestep. A torque-limited stationary
interval may stop a failed diagnostic early, never approve release. Preserve
material uncertainty and rerun every physical-prior case after a tool revision.
