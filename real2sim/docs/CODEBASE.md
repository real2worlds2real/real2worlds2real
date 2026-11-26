# Codebase map

The default product is an independent **object generator**. `real2sim-harness generate`
builds one object; scene generation is an optional `--scene` mode. Object work
does not depend on generating a room. Video, multiview geometry, camera alignment,
metric evidence and skills are shared capabilities, not scene-only services.

The delivery boundary is an editable, operable object or scene that downstream
systems can load and interact with using compatible robots. Later physical
alignment can refine its parameters. Robot training, motion planning, grasp
strategy, policy execution and robot dataset production remain downstream
responsibilities, rather than planned subsystems of this framework.

## Entry points and ownership

| Concern | Main modules | Interface documentation |
| --- | --- | --- |
| CLI and task lifecycle | `cli.py`, `artifacts.py`, `steering.py` | README, `real2sim-harness --help` |
| External coding-agent entry | `external_agent.py`, `task_tools.py`, `capabilities.py` | `AGENTS.md`, `use-real2sim` skill; no model launch |
| Optional managed model session | `codex_backend.py` | [Model connection](MODEL_CONNECTION.md) |
| Replaceable authorization and transport | `providers.py`, `codex_transport.py`, `chatgpt_auth.py`, `chatgpt_plan.py` | [Model connection](MODEL_CONNECTION.md) |
| Source/task investigation access | `project_inspection.py` | [Task access](MODEL_CONNECTION.md#source-and-task-access) |
| Blender construction and appearance review | `blender.py`, `blender_*_harness.py`, `blender_surface.py` | Skills selected for the asset and uncertainty |
| Reference evidence | `video_review.py`, `vggt.py`, `multiview.py`, `camera.py`, `calibration.py`, `measurement.py`, `observation_tools.py` | [Measurements](MEASUREMENTS.md), [posed observations](OBSERVATION_TOOLS.md) |
| Optional object cousins | `cousins.py`, `cousin_contract.py` | README |
| Optional scene composition and review | `scene_agent.py`, `scene.py`, `scene_review.py`, `scene_runners.py` | [Scenes](SCENES.md) |
| Optional scene cousins and backgrounds | `scene_cousins.py`, `gaussian_background.py`, `gaussian_refinement.py` | [Scene cousins](SCENE_COUSINS.md), [Gaussian backgrounds](GAUSSIAN_BACKGROUNDS.md) |
| Shared evidence integrity and review | `sim_bundle.py`, `conversion_audit.py`, `motion_review.py`, `usd_surface_binding.py` | [Asset integration](ASSET_INTEGRATION.md) |

The internal Python namespace remains `real2sim_harness` for compatibility. It is not a
second framework or a requirement to use a particular account or machine.

The external entry leaves conversation management, native source/log/media
inspection and delegation to the coding client. It returns reports and file
paths from domain operations, with a per-task mutation lock and explicit child
handoffs. Schema and reference-computation ownership is independent of the
optional managed transport. It does not silently spawn another model for a
scene participant or cousin. User-authorized compute concurrency remains a
caller responsibility; separate tasks do not authorize unlimited resources.

## Physical representations

Object capabilities use reusable physical representations shared by generation,
simulation and delivery.

| Representation / mechanism | Source and runtime ownership | Scope |
| --- | --- | --- |
| Rigid bodies and joint articulations | `sim_review.py`, `sim_parameters.py`, `sim_asset_loader.py`, PhysX scripts | Standard rigid/contact and joint review. |
| Curves / rods | `strands.py`, `strand_contract.py`, `strand_runtime.py`, `strand_contacts.py` | Source-bound strands with optional endpoints and attachments; winding is a task test, not a certified knotting capability. |
| Thin shells / sheets | `deformable_behavior.py`, `deformable_runtime.py`, `source_deformable.py` | Native shell geometry, explicit thickness, contacts, grasps/release and retained source appearance. Cloth's existing Lab/Blender routes remain available. |
| Volumetric elastic bodies | `solids.py`, `deformable_geometry.py`, `deformable_runtime.py` | Closed control boundary, tetrahedra, native elastic FEM and detailed visual followers. |
| Contact-dependent rigid mechanisms | `rigid_contact.py`, `rigid_contact_runtime.py`, `contact_asset.py` | Concave source meshes/SDFs and explicit test forces; useful for mating threads without prescribing a helical trajectory. |
| Articulation plus flexible surface | `hybrid.py`, `hybrid_runtime.py`, `hybrid_visuals.py` | Shared rigid/cloth construction, two-way attachments and source appearance. |
| Particle materials / liquids | `fluid.py`, `fluid_environment.py`, `material_transition.py` | MPM volumes and evolving surfaces; visual material transitions are distinct from heat-transfer identification. |
| Prefractured rigid volumes | `fracture.py`, `fracture_math.py` | Presegmented geometry and impact-triggered release; not real-time crack propagation. |

Sheet extraction reuses shell geometry, contacts and explicit external grasps;
it is not a separate asset type or a permanent scripted motion in the delivered
asset. Likewise, a threaded assembly is a rigid contact mechanism; a demonstration
controller does not replace its passive geometry and editable parameters.

`native_controls.py` contains shared test controls. The passive loaders do not
silently install demonstration grasps or forces. `scene_native_asset.py` and
`scene_native_runtime.py` preserve these source representations for optional
shared-world assembly. Scene-level behavior still needs independent validation;
standalone acceptance does not prove identical dynamics after conversion.

## Keep reusable code and task evidence separate

- `src/` contains reusable implementation; `scripts/` exposes process/adapter
  entry points; `skills/` explains applicability, assumptions and learned failure
  modes.
- `examples/` holds portable configurations, not somebody's installed paths or
  credentials. Runtime versions and installation dependencies belong in
  [ENVIRONMENTS.md](ENVIRONMENTS.md) and [INSTALLATION.md](INSTALLATION.md).
- Runs retain generated source, input data, numerical states, logs and review
  history in task directories. These and presentation assets remain local and
  ignored by Git. Case-specific diagnosis scripts belong with that evidence.
- Agent autonomy uses general source/task access and native analysis. Add a
  callable tool for a reusable capability or runtime boundary, not merely because
  a new example exposed another symptom. Learned investigation criteria belong
  in the relevant skill.

Measurement context and retained inputs are directly readable. Hash validation
checks measurement integrity; image inspection, skill search, video indexing and
computation/execution tools provide additional task capabilities.
