# PhysX garment workflow

Prefer PhysX particle cloth for garment generation and manipulation. Isaac Sim 5.1 provides `SingleClothPrim`, `SingleParticleSystem` and `ParticleMaterial`; confirm the installed APIs rather than assuming all later releases retain this interface. Use a configured garment adapter, or task-scoped native code when the existing domain tool does not expose that route. A Newton trajectory rendered in Isaac Sim is not a PhysX garment test.

Use `real2sim_harness.physx_cloth.resolve_cloth_settings` and `create_cloth` for passive construction in a caller-owned PhysX world; see [the native API](../../docs/DEFORMABLE_BEHAVIOR.md#physx-cloth-construction). Apply the resolved timestep to that world. The constructor does not run controls or certify a garment, and the existing `run_deformable_behavior` tool remains a Newton route.

For a larger scene, an accepted source-bound PhysX surface can use the [particle-cloth scene adapter](../../docs/SCENES.md#physx-particle-cloth-scenes-on-sim-51). It handles triangle surfaces without a garment-specific category test. Reuse the passive configuration and visual binding, keep diagnostic fixtures/grips separate, and perform fresh scene review. Do not treat cloth-node proximity to a collider as measured contact force.

Accept single images, multiple views or video as available. Recover observed shape, openings, seams and pattern placement; mark hidden surfaces and metric scale as assumptions when unsupported. Static appearance does not measure fabric stiffness or damping. Use informative motion for optional material calibration.

Use a uniformly sampled, consistently wound simulation surface with appropriate garment openings and seams. Keep render thickness and detail separate from physical collision distances. Initial panel separation, contact/rest offsets and mesh spacing must be mutually feasible; oversized collision distances can repel neighboring garment layers. Enable required self-contact and check its filtering.

When a few long edges dominate contact coverage, `deformable_geometry.refine_surface(..., adaptive=True)` splits them consistently across neighboring triangles while retaining source vertex indices. Rebuild the visual binding and recheck mass, contact and manipulation after changing the control mesh. Refinement alone does not guarantee collision-free motion.

Configure stretch, shear, bending, spring/material damping, collision distances, iteration count and timestep explicitly. Treat them as assumptions unless measured. Coefficients from another solver, unit system or mesh scale are not directly interchangeable. Keep measured or assumed fabric properties separate from numerical settings. Preserve total mass when changing mesh density, recompute contact distances for the new sampling, and check boundary triangles as well as the interior. Inspect the cooked spring classes on irregular triangle meshes: a very soft shear class can allow local distortion despite high stretch stiffness. A preset is an initial condition for validation, not a guarantee for another garment. The source asset and reusable runtime must retain the resolved parameters, timestep and their provenance; explicit user overrides remain available.

Keep grasp controls local and separate from the passive asset. Verify both constrained tracking and free-region response, then release the constraints and inspect settling. Review retained positions and velocities alongside video: persistent contact chatter or flutter can fail a garment test even when stretch and penetration checks pass. Diagnose geometry, contact settings, dissipation and solver convergence; do not smooth the cache, hide the unstable interval or freeze the garment to manufacture stability.

Before applying source-indexed grasps or computing deformation metrics, verify the mapping from authored vertices to runtime particles. Cooking can weld vertices and change both counts and indices. Retain and audit the mapping, including transformed positions, tolerance and coverage; remap topology and controls consistently. Account for motion already integrated during reset when comparing positions. A nearest-position match alone is ambiguous at coincident garment layers or seams.

Verify native view initialization and actual readback before diagnosing failed
rigid controls. Use [review-numerical-evidence](../review-numerical-evidence/SKILL.md)
for named-body command/readback tracking over the full command interval, not just
pre-motion states. A target tensor or authored transform is not runtime motion.
Fixed/support attachment demonstrates only that fixture; two-way rigid load
transfer needs its own enabled coupling and measured response.

Interpret stretch and contact diagnostics at the scale of the task. Locate an outlier and inspect rest/current lengths, affected area, duration and the distribution across the garment; a large ratio on one short edge and widespread elongation have different implications. Inspect suspected penetration in the corresponding geometry and frames, and examine post-release motion for persistent jitter. Define useful tolerances from the intended manipulation and material assumptions; an exploratory maximum-edge threshold is not a universal garment acceptance criterion. Preserve earlier failed checks and explain any revised criterion separately, rather than relabeling a failed report as passed.

For persistent strain at a constrained/free interface, inspect rest geometry,
attachment footprint and collision feasibility before another stiffness sweep.
Track boundary outliers and late residual velocity alongside typical stretch;
an improvement in one metric can worsen another. Keep an unisolated cause open.

Retain source hashes, runtime version, controls, actual states and numerical/visual reviews. Check lifting, release and folding when requested. Local vertex constraints test garment response; they do not validate a robot gripper's contact mechanics. Do not infer robot-policy success, calibrated fabric behavior or permanent crease retention from these tests. Thin-shell garment results do not establish volumetric soft-body behavior.
