# Physics calibration

The new overall entry is the physical reproduction of the video: first refer to [the operation process](process_workflow.md), and for interface details, see [the process protocol](process.md). The agent determines objects, materials, interactions, models, and parameters to be estimated based on the video and optional robot trajectories, and jointly fits the entire process through Isaac Sim 5.1. If the error is unexplained, revise the model or request more observations; even after numerical validation, independent review of native evidence is still required. This module can be called by real2sim. Existing replay/video/soft capabilities are retained for use as specialized tools.

The `physics/replay/` directory handles processing video recordings and mapping robot joints accordingly. The `physics/video/` directory is responsible for identifying segments of the video that represent free movements of the robot. Shared camera projections, mask statistics, and other related functions are located in `physics/common/observations.py`. JSON file handling functions are located in `common/schema.py`. The noise reduction strategies and truncation methods used for both routes differ, so they are kept separate in their respective directories.

Model loading and input/output adaptation are located in `core/perception/`, which does not include object selection, alignment scoring, or admission strategies. The columns of DROID and pose reading are located in `core/adapters/droid.py`, with different domains maintaining their own output formats. Triangular meshes and pose measurement are located in assets, while Isaac's stage construction is located in sim.

The default operating object path is the reviewed agent Blender model. Missing valid model evidence will result in a stop; templates or SAM 3D need to be explicitly selected. Accepted meshes are fixed in the physics calibration; changing the shape requires new modeling and review cycles.

The scene preparation layer reads physical parameters and transmits the prepared data to the backend. The backend does not import the physical identification module. The old `real2sim`, `vidphys` modules, and apps/tools entries are retained for forwarding.

Entry points: `apps/physics/replay.py`, `apps/physics/video.py`, `apps/physics/soft.py`, and `tools/physics/video_physics.sh`. The original physical judgment thresholds in the parameters and files remain unchanged.
