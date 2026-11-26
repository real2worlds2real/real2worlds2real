# Astra scene orchestration

`plan-scene-generation` describes broad planning dependencies while leaving detailed order and tool choice to Astra. Natural-language operation targets, actions, parts and exclusions take priority over inferred roles or automatic scene-graph task suggestions. Participant workers receive the original scene request and full-scene reference context as well as their assigned crops/objective.

Scene mode works in an existing coding-agent conversation or the optional managed
session. The user supplies images/video and an objective; the agent chooses
analysis, optional geometry tools, environment generation, participant tasks,
assembly and review. Each participant retains its ordinary object acceptance.
The external entry returns child tasks to the current agent and starts no model;
the managed entry can launch object workers with its explicit authorization.
No credentials are embedded in scene packages.

```bash
real2sim-harness task init --scene --reference /path/to/scene.mp4 \
  --objective 'Reconstruct this workspace with an operable cabinet and cup.' \
  --scene-runner /path/to/my-simulator-adapter.json \
  --blender /path/to/blender
```

This is an initial integration interface. USD composition, task-level physics routing and the adapter feedback/repair protocol are implemented. General autonomous reconstruction quality is not yet established by an end-to-end real-scene benchmark. PhysX composition preserves rigid/articulated bundles and source-bound particle-cloth surfaces through the Sim 5.1 route below. The Newton materializer retains native rigid/articulated construction data, cloth topology/detailed bindings and hybrid attachment data for one shared solver; missing native data returns an explicit conversion request. Liquid bundles can also enter normal assembly with native MPM inputs and packed material sources. Rope scene binding and dynamic rigid/fluid or cloth/fluid coupling remain extension work. Astra constructs the scene using shared evidence tools and editable code. Optional Gaussian preparation/export and replaceable training/conversion are described in [Gaussian backgrounds](GAUSSIAN_BACKGROUNDS.md).

## Engine selection belongs to the interaction task

Reference inspection, video reading/frame promotion, COLMAP, VGGT priors, PnP and supplied camera/ChArUco evidence reuse the existing tools and implementation in both object and scene modes. They are not duplicated scene algorithms. Optional per-object dimension/surface acceptance stays within the participant's object run, where its geometry and axes are defined. Scene-only tools cover task physics planning, environment authoring, participant worker invocation, composition and scene runtime/acceptance.

`plan_scene_physics` defaults to PhysX for rigid/articulated, cloth and volumetric soft-body tasks. Fluid/strand tasks retain a Newton default for the whole interacting task. An explicit user backend choice takes priority; unavailable capabilities require a compatible adapter, not a silent engine change. It does not assign an engine independently to each object or assume automatic PhysX/Newton coupling. The adapter must verify the selected backend and all required contact capabilities. Fluid/rigid and deformable/fluid coupling are separate capabilities; a backend label is not sufficient evidence.

Keep editable geometry, collision descriptions, structure, physical assumptions and deformation bindings. A prior per-object backend does not permanently constrain scene use. Assembly returns `backend_conversion_required` and specific conversion requests when existing bundles need a different target backend. It never silently flattens them.

Mixed-scene execution takes priority over expanding the example collection. Preserve each asset's editable representation, import shared environment collisions, verify interaction and return state/image/video feedback to Astra for repair and final acceptance. Existing standalone object support does not establish mixed-scene support; that requires actual integrated runtime checks. Physical parameter identification remains outside this milestone.

### PhysX particle-cloth scenes on Sim 5.1

The [5.1 runner](../examples/scene-physx51-runner.json) can materialize one
source-bound particle-cloth surface alongside rigid/articulated participants
and static support boxes in the same PhysX world. This is a physical
representation, not an object-category-specific implementation. The route
requires Sim 5.1, CUDA physics and the host `fitting` extra for particle mapping.
It does not transfer a Newton trajectory into PhysX.

`package_native_for_scene` retains an accepted PhysX surface's passive
configuration, resolved material/solver settings, original visual USD and
control-to-visual binding. The bundle declares `backend: "physx"`,
`asset_kind: "deformable"`, `native_representation: "surface"`, and a retained
`cloth_configuration` JSON path. The same `assemble_scene`,
`run_scene_simulation` and `accept_scene` flow handles the result. Test grips
and diagnostic probes are not passive scene physics; fixed source fixtures
must be exported as separate participants. A Newton surface is not silently
relabelled as PhysX.

Without an explicit frequency override, the runner selects a common frequency
compatible with the source cloth settings and its default clock, preserving
the configured duration. Explicit `physics_hz` or `--physics-hz` remains
available and is recorded against the source timestep; it must be a multiple
of the cloth FPS within the constructor's substep range. Other resolved
cloth coefficients are retained. Placement transforms apply once, and a
bijection between authored and cooked particles is checked before updating
the original-material visual meshes.

The runtime retains every integration-step node position/velocity, source-to-
runtime indices, rigid states, resolved settings and rendered evidence. Checks
include finite states, maximum control-edge stretch and triangle/support-box
SAT over retained states. `--cloth-max-stretch` defaults to 2 and
`--cloth-penetration-tolerance` to 0.002 metres; choose tolerances for the task.
Live PhysX overlap queries report collider proximity at simulated cloth nodes.
These are geometric interaction witnesses, not measured contact forces;
inspect motion and use a collision-disabled control when attribution matters.
`--disable-cloth-rigid-collision` filters explicit particle-system/collider
pairs for that diagnostic. Such a run cannot pass the interaction checks.

This adapter currently accepts one cloth participant. Multiple interacting
cloth surfaces, cloth attachments or task-specific grasp controllers need an
extended adapter; it rejects unsupported manifest actions rather than ignoring
them. Arbitrary rigid-shape penetration, self-intersection and continuous-time
collision freedom are not certified by the support-box audit. Basic scene
checks do not replace visual review or verification of the requested operation.

### Native Newton mixed adapter

[scene-newton-runner.json](../examples/scene-newton-runner.json) configures `scripts/newton_scene_adapter.py` in the target rendering environment with a separate Newton 1.7/Warp 1.17 package overlay. Physics runs in an isolated subprocess; RTX replays its actual rigid poses and control nodes without stepping another physics engine. Source assets are unchanged. Each package includes `newton_bindings.json` describing materialization and explicitly leaves target simulation unverified until runtime checks pass.

Rigid/articulated inputs require retained `manifest.json`, `collision_meshes.json`, `physics_parameters.json` and `visual.usdc`. Cloth also requires `sim_mesh.json` and the source-matched detailed surface binding files; hybrid inputs retain their attachment/loop configuration. Placement is applied to physics and appearance once, with per-asset namespacing. This reconstructs Newton physics from native data; it does not reinterpret PhysX-specific USD schemas as Newton constraints. Missing data must be re-exported or supplied through another compatible adapter.

The current adapter uses one shared gravity, selects a common VBD iteration/substep budget, retains material/pin/air-drag inputs and keeps motion tests separate from passive delivery geometry. Declared support boxes become static collisions in that model. Arbitrary background collisions must be authored as participants or supported collision geometry, rather than inferred from appearance. The initial RTX replay route accepts mesh backgrounds; Gaussian rendering with this adapter requires a matching renderer hook. This does not restrict the separate PhysX Gaussian route.

VBD review requires `contact_capacity`; cloth adds `deformation_integrity`, and mixed rigid/cloth scenes add `rigid_deformable_contact`, in addition to loading, finite states and environment contact. Contact evidence reports actual nonzero solver forces between distinct asset IDs at sampled states, with explicit pair coverage. It does not certify every potential pair or a complete robot policy. Hybrid scenes additionally require joint/attachment checks. The adapter sizes contact storage for the assembled scene and reports per-body rigid/soft capacity checks; `--body-contact-capacity` is an explicit override. Insufficient capacity fails review. The negative-control flags disable environment collision or participant particle contact to test that missing interaction fails. The latter also disables a hybrid's internal particle contact and is diagnostic only. Failed physical checks retain rendered feedback and do not authorize acceptance.

Scene validation must exercise the requested cross-asset contacts, attachments and articulation after placement. Preserve source bindings and review continuous motion when needed; isolated asset checks do not certify a combined scene.

### MPM liquid scene adapter

Liquid workers still use Astra-authored Blender volumes, materials and editable MPM parameters. After successful object review, `build_scene_object` exports a native fluid bundle automatically: retained `fluid_configuration.json`, seed-preview `visual.usdc` and a packed `render_source.blend`. Re-exported construction must match the accepted source and cache. This packages construction data; it does not deliver a baked liquid animation as a substitute for physics.

The same `assemble_scene`, `run_scene_simulation` and `accept_scene` tools serve these bundles. [scene-fluid-runner.json](../examples/scene-fluid-runner.json) invokes `scripts/fluid_scene_adapter.py` with an operator-owned [MPM physics runner](../examples/mpm-physics-runner.json) and Blender executable. A container wrapper translates physics input/output paths just as in other adapters. The internal scene manifest is written by the agent; users do not need to supply it manually.

Particle positions are transformed into scene metres once. All fluid participants share one MPM execution, while their phase/material mappings retain independent asset IDs even when source object names repeat. Scene `supports` replace the standalone object's previous floor/colliders; there is no implicit infinite plane. The initial MPM environment adapter supports finite fixed axis-aligned boxes (including axis-permuting rotations). Other rotated collision geometry returns an explicit adapter gap. Uniform prescribed material transitions retain timing; spatial height-delay transitions require a compatible field frame.

Whole-scene replay imports visual mesh backgrounds at their scene transforms and preserves the liquid's packed Blender materials. Reconstructed surfaces come directly from the saved MPM state; replay advances no second physics solver. Requested video is continuous at the simulation sample rate, registered with the same video inspection tool, and passed back for Astra's final judgment. Ordinary iterations can return sparse review stills without requesting video.

Mesh backgrounds may explicitly include `editable_source`, a matching `.blend`
with packed images. Composition retains this file and its hash alongside the USD
background. Blender liquid replay uses the native source to preserve procedural
materials and authored world lighting that USD Preview Surface cannot represent.
The source must be visual-only, share the USD frame and depict the same geometry;
collision supports remain explicit. Other renderers continue using the USD.

MPM review requires `fluid_state`, `surface_integrity`, `static_environment_integrity` and observed support evidence, in addition to scene identity and finite states. Particle-center penetration is checked against the finite boxes, and native collider presence and per-asset support coverage are reported. These are basic sampled checks, not calibrated forces or complete surface-shell certification. Dynamic rigid/fluid and cloth/fluid coupling are not declared by this adapter: unsupported task capabilities reject startup rather than freezing those participants.

A scoped synthetic test moved/rotated an existing liquid asset and changed table height, then exercised actual assembly, MPM, full-scene rendering and final video review by Astra. This establishes the shared interface, not general reconstruction accuracy. The standalone liquid tool also retains its original optional `real2sim_mpm_static_colliders` interface.

## Scene manifest

Write a JSON file inside the agent workspace. Paths resolve relative to that file. Transforms are explicit 4×4 column-vector rigid transforms from asset-local metres into the scene frame. Arbitrary scale/shear/reflection is rejected: resize editable source and regenerate its physics/bindings when necessary.

```json
{
  "schema_version": 1,
  "frame": {
    "meters_per_unit": 1,
    "up_axis": "Z",
    "scale_source": "Estimated from the reference; no metric evidence supplied",
    "scale_verified": false
  },
  "backend": "auto",
  "objects": [{
    "id": "cabinet",
    "bundle": "/path/to/verified/asset-bundle",
    "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0.75],[0,0,0,1]]
  }],
  "backgrounds": [{
    "id": "room",
    "path": "/path/to/visual-environment.usdc",
    "format": "mesh",
    "observed": false,
    "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  }],
  "supports": [{
    "id": "worktop",
    "size_m": [2,1,0.1],
    "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0.7],[0,0,0,1]],
    "static_friction": 0.6,
    "dynamic_friction": 0.5,
    "restitution": 0
  }]
}
```

`objects` are complete asset bundles, never flattened GLBs. Assembly verifies their retained files/dependencies, preserves prim hierarchy, materials, joint links/limits and collision geometry, and relocates joints' world-side anchors under scene placement. Multiple references retain independent relationships. USD inputs must declare metres/Z-up and a transformable default root, or an explicit root-level `root_prim` for a background; adapt other frame conventions explicitly. Dynamic reset-transform stacks and instanceable inputs currently require an adapter.

Composition and `verify_scene` also resolve the packaged USD dependency graph.
Dependencies must be retained files inside the scene delivery, including the
outer file of nested USDZ members. A copied layer that still references its
original source through an absolute path fails this check even while that source
exists. Use portable relative references when authoring asset bundles. This
checks dependency closure; it does not replace fresh target-runtime loading or
interaction checks after delivery.

`supports` are static box collision proxies; `size_m` gives full extents about the transform's origin. They are invisible so the environment supplies appearance. Proxy geometry is an initial interface, not a substitute for detailed concave container collision meshes. Runtime adapters must import environment collisions into the selected solver.

### Required support responses

When a task requires particular rigid participants to rest on particular supports,
add explicit per-pair requirements to the scene manifest:

```json
{"support_requirements":[
  {"asset_id":"bottle","support_id":"shelf","release_height_m":0.04,
   "minimum_drop_m":0.02,"separation_tolerance_m":0.006,"max_late_speed_m_s":0.03}
]}
```

Choose bounds for the task; these example values are not universal thresholds.
Each pair must name an existing rigid asset and horizontal static support box.
This initially covers single rigid bodies, not articulated/deformable support
tests. Repeated assets must have one consistent release height for a scene trial.
The selected runner must advertise `rigid_support_response`; missing capability
rejects the request before startup. The PhysX example raises the selected body
once by its declared release height, then observes natural drop/settling. Source
placement stays unchanged. Check clearance before choosing this runtime fixture.

Runtime evidence must include every required pair's initial/final bottom,
support top, late speed, `footprint_inside:true` and method
`observed_free_drop_and_settle_on_declared_horizontal_support`. The host recomputes
drop/separation/speed bounds and rejects missing/failed relations even when global
environment contact passes. Footprint currently means the adapter's body-reference
point inside the support, not full contact-patch coverage. No force calibration
or continuous-time penetration guarantee is implied. No requirements means the
existing general contact review remains; it does not certify all scene objects.

With an explicit `backend: "physx"`, a support may set `friction_combine_mode`
to `average`, `min`, `multiply` or `max`. Omission preserves the backend default.
The mode is authored on the support's bound physics material and retained in the
portable USD. It controls how two material coefficients form an effective contact
coefficient; inspect both materials and their combination priority. This field is
rejected for other backend selections rather than silently ignored.

An optional `review_camera` can specify `eye_m`, `target_m`, `up` (default `[0,0,1]`) and `horizontal_fov_deg` (default 55). It is retained as an authored USD camera and in the scene request for adapters. Use it to review the intended scene context rather than allowing one small participant to fill the frame. The example adapter otherwise fits objects and declared supports together; renderer/view conventions remain adapter responsibilities.

For a fixed calibrated review view, supply `review_camera:{"calibration":...}`
using the existing observation convention:

```json
{"image_size":[640,480],
 "intrinsics":[[600,0,290],[0,620,250],[0,0,1]],
 "world_to_camera":[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]],
 "source":"Describe the camera calibration and metric pose evidence"}
```

The transform maps world metres to OpenCV camera coordinates (X right, Y down,
Z forward). Use undistorted images and their corresponding intrinsics. Skew,
nonzero distortion and calibrated camera paths are currently rejected. A
calibrated scene requires the adapter capability `calibrated_camera`; a field-of-view
adapter must not silently claim support.

The Sim 5.1 adapter enables the native OpenCV lens schema. Its calibrated route
renders an enclosing square-pixel sensor raster and explicitly samples the same
pinhole rays into the requested output raster. This avoids relying on standard
USD aperture attributes alone to reproduce unequal horizontal/vertical focal
lengths in RTX. Raw frames remain in `raw_camera/`; the runtime report retains
sensor calibration, output calibration and the sampling transform. This is
image resampling, not a scene-geometry correction. Review known projected points
in actual rendered images before adopting a new runtime or renderer mode.

The original calibrated image size is used by default. `--width` / `--height`
resize the calibrated output and scale intrinsics; supplying one preserves the
original aspect ratio. `--resolution` continues to control ordinary uncalibrated
views. Oversampling must fit within 4096 pixels per sensor dimension; excessive
requests fail explicitly. Camera parameter readback and successful rendering do
not establish visual acceptance of the scene.

`review_camera_path` optionally supplies ordered camera keys with `seconds` and the same view fields, starting at zero and using a constant field of view. The native renderer interpolates position/target/up smoothly at the actual retained simulation times. The operator must run enough simulation frames to cover the requested path; camera keys do not extend physics or authorize time scaling. Framing must be checked at the renderer's actual aspect ratio.

`review_render` optionally contains `authored_light_exposure` (−16..16 stops, default 0) and `default_lighting` (boolean, default true). The native RTX renderer adds the exposure offset to authored USD lights and optionally adds its review dome/key lights. This makes cross-renderer lighting differences adjustable without modifying asset geometry or the original Blender illumination. Judge actual output; a numerical gain alone does not establish material or illumination parity.

### Explicit shared-world review actions

The optional `actions` mapping names scene object IDs. A `joint_targets` action has `start_seconds` and `joints`, each with a joint name and ordered `[seconds, native-coordinate target]` keyframes. A `native_test` action imports the retained source test's wrenches, grasps and diagnostic probes, offset by `start_seconds`. Passive scene delivery does not implicitly install these tests.

For native rigid participants, an optional `controls` list replaces only the test wrenches using the existing validated native rigid control schema. This allows scene gravity/friction adaptation without changing accepted geometry. `hold_last_controls:true` keeps the terminal review grip with bounded forces and zero terminal velocity targets; it does not continue a lift indefinitely or freeze dynamic body poses. Source particle grasps and kinematic test probes retain their declared targets. Record controller assumptions separately from material calibration.

Shared-world review reapplies native source thread/separation checks, shell/solid integrity, regional response/recovery measurements and retained-frame analytic fixture checks at the correct action offset and source sample rate. Incomplete action windows fail. These checks supplement contact-force/capacity and articulation checks; they do not replace final visual judgment or establish real-world physical identification. Native diagnostic gravity is recorded separately from the scene's shared gravity.

The current VBD API exposes one particle-side contact material per model. The native adapter explicitly initializes it from an active self-contact participant, or the first native soft participant when none requires self-contact, and reports that choice. Per-shape coefficients remain retained; elasticity and air drag remain per asset. Conflicting active self-contact materials require a compatible adapter rather than silently choosing one. A shared contact material is a scene assumption and must not be described as exact preservation of every standalone contact experiment.

The native physics CLI optionally supports `--capture-graph` for CUDA solver-step replay. Time-dependent controls remain outside the captured solver graph and run every physical substep. Every substep contributes to retained capacity maxima. This is an execution optimization, not cached object motion; native behavior checks and final video review remain required.

Background `format` is `mesh`, `nurec_volume` or `particle_field`. Captured geometry and generated unseen completion must be distinguished with `observed`. Backgrounds are visual-only; duplicated target objects must be removed before use, and exposed regions and camera coverage reviewed. Gaussian appearance supplies no collision model. NuRec/ParticleField loading is capability-dependent, with no hardcoded Isaac Sim version in scene orchestration or USD assembly.

`build_scene_background` executes a workspace Blender file defining `build_scene()`, saves editable source and exports a visual USD environment including authored materials and lights. Captured background USD can be supplied instead. `build_scene_object` takes selected workspace image/video paths and an objective, returns a verified object bundle or a retained failure with a resumable asset ID. Its objective should retain original scene context and any available scale/pose evidence. Dedicated fluid/rod/hybrid/fracture outputs remain intact when their scene binding is not yet implemented.

## Portable composition

Install the optional `scene` extra in the host Python, separately from simulator Python. It uses OpenUSD without launching a simulation application. Object bundles and USD background dependencies are copied into the output. When a bundle's declared Blender source is available and matches its retained hash, it is also copied into `assets/<id>/editable/scene.blend`; the report says whether this editable source was included. The USD geometry/configuration remains available when an external library bundle has no retained Blender source. Keep generating scripts from the original object runs when distributing the parametric construction as well. Source inputs remain unchanged. The scene references complete assets through relative paths and reopens after original source directories are removed.

```bash
python scripts/compose_scene.py /path/to/scene.json --output /path/to/new-output
python scripts/compose_scene.py /path/to/new-output --verify
python scripts/run_scene_runtime.py /path/to/new-output --runner /path/to/my-simulator-adapter.json
```

Composition records source and package hashes and reports `simulation_verified:false`. Within a live Astra run, changed source inputs invalidate acceptance; an exported portable package can be verified independently of the original directories. Treat `request.json` and provenance reports as local evidence when sharing media: they can contain operator-local paths. They are run artifacts and excluded from repository publishing.

## Simulator adapter protocol

Supply an operator-owned JSON through `--scene-runner` or `REAL2SIM_SCENE_RUNNER`. This selects a concrete installed simulator and can target different Isaac Sim/Lab versions or another compatible runtime. No adapter command is taken from model-generated shell text. Arguments are executed directly without a shell; paths containing spaces remain single arguments.

```json
{
  "schema_version": 1,
  "capabilities": ["physx", "usd_mesh", "rigid", "articulated"],
  "argv": ["/path/to/runtime-python", "/path/to/my_scene_adapter.py",
           "--scene", "{scene}", "--output", "{output}", "--manifest", "{manifest}"]
}
```

An operator can also configure `routes` instead of a single `argv`: each route has an `id`, `capabilities` and `argv`. The host selects the first route satisfying every composed task requirement and the optional video request. This lets one local configuration expose PhysX, native Newton VBD and liquid MPM without asking users to pick a backend per asset. No compatible route returns the missing capabilities before execution. Astra's task/asset selection and backend plan determine the request; operator order controls preference among compatible installations. The selected route and root configuration hash are retained.

The adapter owns simulator startup and the solver context, including gravity. The assembly layer intentionally embeds no simulator physics scene. A Gaussian background additionally requires the matching `nurec_volume` or `particle_field` capability. Compare capabilities before startup; an unsupported representation returns an actionable failure. This protocol does not certify every named simulator version: each adapter must be smoke-tested in its actual environment.

`scripts/isaac_lab_scene_adapter.py` is an example for the current Lab API. It loads the complete scene, drives articulated participants, checks finite states and returns RTX images. Environment contact can be established by native PhysX reports or an actual free-drop-and-settle response on a declared horizontal support. The latter reports drop, final separation and late velocity explicitly; it is not a contact-force measurement. A `--disable-environment-collision` negative-control mode allows checking that this response fails without the support. The example is interchangeable, not an orchestration dependency. Contact coverage and interaction protocols should be specialized for the task; one contact somewhere in a scene does not prove every asset's support or every robot action succeeds.

Write `runtime_report.json` beneath `{output}`:

```json
{
  "schema_version": 1,
  "ok": true,
  "runtime": {"name": "your-actual-simulator", "version": "actual-installed-version"},
  "scene_sha256": "sha256-of-the-input-scene-usda",
  "checked_asset_ids": ["cabinet"],
  "checks": [
    {"name": "scene_loaded", "ok": true},
    {"name": "finite_state", "ok": true},
    {"name": "environment_contacts", "ok": true},
    {"name": "joint_motion", "ok": true}
  ],
  "images": ["overview.png"],
  "videos": ["interaction.mp4"]
}
```

The adapter must perform the reported checks, cover every asset ID, report actual runtime identity and provide rendered evidence. An optional `runtime_dependencies` list of absolute operator-config paths and SHA-256 values binds nested runner configuration to acceptance; the liquid adapter uses it for its physics runner. Changed dependencies invalidate the verdict. `joint_motion` is required for articulated scenes. Add detailed observations, thresholds, assumptions and failure scope inside checks. `ok:true` must not mean merely that import completed. This is an adapter contract, not a defense against a dishonest adapter. Physical parameter identification and complete robot task policies are outside this initial scope.

For rigid objects initially resting on a support, the example adapter can use
`--rigid-review-drop-m 0.03` for an explicit upward displacement followed by free
release. This optional runtime probe leaves saved assets and passive scene
placements unchanged. Its displacement and participant IDs are retained in the
runtime report. It supplies observable motion for the existing support-response
check when direct contact reports are unavailable; it does not relax the check
or prove contact-force calibration. Review the resulting motion and any nearby
obstacles before choosing this probe.

Adapters may return `state_files:[{"path":"states.npz","sha256":"..."}]` for
task-local numerical evidence. These files are checked and included in stale
acceptance detection alongside images/video. The example PhysX adapter retains
`rigid_states.npz`: `times_s`, `asset_ids`, and `body_poses_wxyz` with shape
`[time,asset,7]`, world metres and XYZ plus WXYZ orientation. These are actual
simulated poses for further task-specific analysis, not prescribed animation.
The separate `rigid_trajectories.npz` preserves the Sim 5.1 branch's per-object
arrays in XYZ plus XYZW order; it is not overwritten by the structured export.

## Feedback and acceptance

`run_scene_simulation` returns failed checks/logs and images as well as successful evidence. Videos are registered with the existing video tool; Astra can inspect intervals, exact frames and crops. Correct an asset, placement, environment collision or demonstrated adapter defect at the appropriate layer, then reassemble and retest. Original inputs and earlier scene attempts remain retained.

### Optional continuous scene video

Call `run_scene_simulation(scene_id, timeout_seconds, record_video=true)` when useful, or pass `--record-video` to `scripts/run_scene_runtime.py`. It is not enabled for every iteration. The configured runner must declare `scene_video` and consume `{review_request}`; unsupported requests return a capability failure before startup. [scene-video-runner.json](../examples/scene-video-runner.json) shows the example adapter arguments. Paths must be translated by the operator's wrapper when crossing a container boundary, just like the existing scene/output/manifest paths.

The host writes `review_request.json` containing `schema_version:1`, `record_video`, requested `fps:30` and `time_basis:"simulation_seconds"`. An adapter can return a complete encoded movie in `videos`, or an ordered `frame_sequence` in its runtime report:

```json
{"fps":30,"frames":["video_frames/frame_00000.png","video_frames/frame_00001.png"],
 "time_basis":"simulation_seconds"}
```

These must be actual consecutive captures at that sample rate, not sparse review snapshots. The example adapter captures at an integer divisor of its configured physics rate; more physics steps produce a longer movie. Its optional `--joint-excursion` changes the bounded operation-test excursion, retaining the original default when omitted. The host encodes captures on CPU using PyAV/libx264 and decodes every returned movie to check frames, timing and dimensions. Video and report hashes participate in current-evidence acceptance. `video_sources` returns stable IDs that work with `list_video_sources`, `inspect_video` and frame promotion; viewing remains Astra's choice.

Agents can pass `physics_hz` to `run_scene_simulation` (omit or use null for the runner default). The runtime must advertise `configurable_timestep` and consume `{review_request}`; unsupported adapters are rejected before launch. This per-call override preserves the configured physical duration by adjusting the step count. The host verifies the runtime timestep readback and binds the request hash to acceptance. This capability is implemented by the PhysX example; other backends need their own implementation.

The PhysX example adapter accepts `--physics-hz` (default `120`); its timestep is `1 / physics_hz`. For example, `--physics-hz 960 --steps 2880 --capture-fps 30` runs three seconds with a 30 FPS movie. Configure this in the operator runner used for final execution when an experiment requires smaller steps. Video FPS must divide the physics frequency. Action times remain in seconds and must lie exactly on a solver boundary; incompatible times are rejected rather than delayed. Runtime reports retain requested frequency, configured and runtime-read-back timestep, duration and capture timestamps. The first video capture follows the first physics step, so its time offset is recorded explicitly. Changing physics frequency does not implicitly preserve duration when `--steps` is unchanged.


`accept_scene` requires current successful composition, runtime checks, rendered evidence and an explicit Astra visual/functional verdict. Input/package/report/media/config hashes bind the result to the tested scene. Changed inputs or artifacts reject acceptance. Ordinary single-object runs retain their existing tools and acceptance path.

If new video or human feedback arrives after acceptance, use `real2sim-harness reopen-scene /path/to/run --reason 'Inspect the newly recorded scene video for transient intersections'`, then `real2sim-harness resume /path/to/run` with your own runtime/model authorization options. The previous final verdict and Codex report are archived, counters and generated assets remain, and there is no active `final_scene.json` until Astra accepts again. New evidence can justify an unchanged asset; demonstrated defects require appropriate repair and current simulation checks.

Each explicit `resume` grants a bounded invocation using the retained turn setting, or `resume --max-turns N` (1..30) to select that invocation's limit. Cumulative turn and call counters are retained; they do not count down the new invocation's turn allowance. Reopening review alone does not start model execution or grant an automatic retry.
