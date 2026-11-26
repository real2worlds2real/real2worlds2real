# Calibrated scene reconstruction overview

Construct revisable digital twins from photographs or videos, then generate scenario variants, robot demonstrations, and policy evaluations. Physical estimations are replicated in Isaac Sim 5.1 by agents using video planning models and parameters.

The operation objects are typically represented by Blender models created by agents. These models are refined through iterative processing of multi-angle contours, local differences, openings, and other details. Template fitting and SAM 3D object fitting are alternative methods that can be used when necessary. SAM 3D can also be utilized for initializing and setting background objects. If no valid model or evidence is available, the process will stop and requires additional data to be provided.

The process does not read or compare the actual dimensions of the task object and support surface. The scale is derived from a valid calibration board or robot joint/component calibration. The evaluation is conducted after the asset is frozen, and any errors are not fed back into the current modeling iteration.

```text
Original image/video + legal calibration
  Camera, metric world, support surface.
  → Agent Blender performs object iteration → Review of OBJ files and S0–S6 → Stabilization of twin assets
  → Code Room / Completing Mesh Scene Variants → Grabbing and Trajectory → Recording → LeRobot
  Task Environment + Strategy Client → Evaluation/Post-Data Collection

Video + Reviewed Reconstruction + Optional Robot Trajectory
  physics/process: Model planning → Native simulation and parameter iteration → Full-process inspection and independent review
The original physics/replay, physics/video, and physics/soft files continue to be used as dedicated tools.
```

See [Operation Flow](physics/process_workflow.md) for the steps of input, iteration, and delivery of the video physical reproduction.

## module

| module | Responsibilities |
|---|---|
| `core/` | Cross-domain storage, paths, mathematics, video, data contracts, and model input/output adaptation |
| `scene/` | Capture/Calibration, Support Surface, Object, Background, Digital Twin Level, Scene Variant, Review |
| `assets/` | Asset library, kits, materials, meshes, and collision geometry descriptions |
| `robots/` | Specifications, URDF, Pure Computational IK, Task Coordinates, and Positions |
| `grasping/` | BODex, grasping synthesis, trajectory and placement planning |
| `tasks/` | Rigid body, LeHome, rope environment, and task success judgment/assessment adaptation |
| `datagen/` | Recording, demonstration redirection, LeRobot export, and video synthesis |
| `policy/` | Training, services, clients, and general rollouts |
| `physics/` | Model planning for video physical processes, joint calibration and review, as well as tools dedicated to replay/video/software |
| `sim/` | Isaac/PhysX and Newton backend |
| `agent/` | JSON/CLI/MCP tools, runtime storage, and level orchestration |
| `rl/` | Training interface reserved on the task environment, no new implementations |

Apps are combined across various domains after being launched in the backend. Domains collaborate through prepared data and injected interfaces. The core component must have at least two active domains, and it cannot include domain strategies; `core/perception` is only responsible for model loading and input/output adaptation.

For details on the reconstruction of modules, entry points, and dependency boundaries, see [Reconstruction Instructions](scene/reconstruction.md). For updates to new mechanisms, settings, tools, and skills, refer to [Capability Evolution Process](agent/evolution.md): save reusable methods and lessons learned from failures, which can be retrieved for future reconstruction tasks after review and validation of applicability.

## Main Entrance

```sh
# Reconstruction: First, read the current agent workflow and execute and review according to the level.
pipeline/twin.sh --help
pipeline/twin.sh <session> --from <stage> --to <stage>
python -m calibrated_scene_reconstruction.scene.twin.gates sessions/<session> status

# Full Process Tools, States, and MCP
python -m calibrated_scene_reconstruction.agent.tool schemas
python -m calibrated_scene_reconstruction.agent.tool status --request request.json --output outputs/status
python -m calibrated_scene_reconstruction.agent.mcp_server

# Demonstrations, datasets, and videos
tools/run_isaac.sh apps/datagen/episode.py sessions/<session> --robot <robot> --episodes 2
python -m calibrated_scene_reconstruction.datagen.export.lerobot outputs/<dataset> sessions/<session>
python tools/datagen/generate.py --help

# Physics calibration
python -m calibrated_scene_reconstruction.physics.replay.observe --help
python -m calibrated_scene_reconstruction.physics.video.perceive --help
tools/physics/video_physics.sh --help
```

`configs/<session>/` stores configurations, with the agent source code located in its `agent/` subdirectory. `sessions/<session>/` stores observations, model iteration snapshots, evidence, additional logs, and delivery packages. `outputs/` stores datasets, training, and auxiliary outputs. Existing delivery and review records are retained; new experiments use new sessions.

S6 delivery requirements: All applicable previous levels and the current OBJ round must be completed. Numerical checks are separate from manual/agent visual review, and task inspection results are provided by `qa/task_check.json`. Independent reconstruction uses only original footage, legal calibration, and current process documents, without reusing old notes, fits, models, or camera caches.

## Environment and Installation

The environment is isolated by dependency stack and installed as needed. The directory names indicate the corresponding entry points, not that the current machine already has this environment.

| environment/entry | Purpose |
|---|---|
| `envs/recon` | Scenario, calibration, CPU tools and testing |
| `envs/splat` | Gaussian splat training and export |
| `envs/grasp` | Grasp synthesis/solver |
| `envs/policy` | Project Strategy Tool |
| `envs/dp_official`, `envs/openpi` | Upstream Policy Training and Services |
| `envs/da3`, `envs/difix` | Optional Visual Route |
| `tools/run_isaac.sh` | Isaac Launcher specified by local environment configuration |
| `apps/tasks/lehome_run.py` | LeHome Challenge/Isaac Lab environment |
| `tools/runtime/run_newton.sh` | Newton container environment |

To install the project in the Python environment, use the following command: `python -m pip install -e .`. For dependencies related to CUDA/models, refer to the description of fixed dependencies in the `setup/` file. For optional tool dependencies, use the existing `objects` extra package: `python -m pip install -e '.[objects]'`. For multi-task processing, use the memory-constrained launcher. GPU tasks will be allocated based on the environment in which they are executed, and executed serially.

When the dedicated environment is not available, the code can be organized first; complete training, reasoning, and simulation verification require the corresponding environment to be provided. The old modules and training commands still have compatible entries and will not be automatically deleted.

## Document

- [Architecture and Dependencies](architecture.md)
- [Agent Workflow and Skills Index](agent/workflow.md)
- [Twin Process and Levels](scene/twin_pipeline.md)
- [Episodes and LeRobot](datagen/episodes.md), [Demonstration Redirection](datagen/demonstrations.md)
- [Pick and place](policy/pick_place.md), [Task Environment](tasks/README.md)
- [Physical Calibration](physics/README.md), [Robots and Grasping](robots/grasping.md)
- [Complete Compatibility List and Synchronization Steps](compatibility.md)
- [Third-party code and licenses](../THIRD_PARTY.md)

Earlier instructions are not part of this distribution and are not used as input for independent reconstructions.
