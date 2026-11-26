# Video Physical Reproduction Process

Instructions for operating on the new video. The goal is to replicate the physical processes in the video within Isaac Sim 5.1. real2sim is responsible for scene and object reconstruction; the physics sub-module is responsible for estimating the models and parameters that can explain motion. It is not a fixed parameter fitter assigned by task name, nor is it a service that automatically completes after uploading any video: the agent still needs to understand the input, build the model, review the evidence, and decide the next step.

For detailed restrictions on interfaces, file protocols, and tools, see [process.md](process.md).

## 1. Confirm Input and Use Case

Each new process establishes independent input directories and plans, without copying the focal length, initial state, forces, or material fitting values of old videos. Reviewed object assets can be reused, but the placement and internal components must be rechecked.

| Input | Use and Limitations |
| --- | --- |
| Main Video | The process to be replicated, initial state, interactions, and timeline |
| Reconstructed scene/asset | Geometry, material partitioning, rigid-flex structures, collision and legal scale; proceed with the real2sim modeling cycle when missing |
| camera | Internal parameters, distortion, pose at each moment, and metric registration; a moving camera needs to estimate from a static environment, excluding moving objects and hands |
| Optional Robot/Hand Trajectory | Named joints, coordinate systems, time alignment, and execution errors; defaulting to instructions does not mean real action |
| Optional close-up, photos, operation demonstrations | Supplement the structure, materials, mechanisms, and visible states; clarify which are structural evidence and which are observations from this movement |
| material or performative a priori | Source, scope, and uncertainty; cannot impersonate measurement results |

When there is no robot trajectory, the agent can declare a bounded external action model, and constrain the action period, direction, and position from the visible contact phase. Unknown external forces may compensate for mass and friction, and quantities that cannot be uniquely identified must be reported. The observed pose of the target object should not be directly used as a simulation driver.

The dimensions of real objects, CAD measurement ground truths, and previous evaluation conclusions are only used for external comparisons and do not enter scale, physical search, or admission processes. When videos are insufficient to determine scale, camera, or key contacts, missing inputs are recorded; they cannot be obscured by physical parameters. Multiple videos can serve as named inputs, but the general harness currently uses a single observation timeline: actions captured by different cameras cannot be directly combined into the same movement. Shared parameters, joint targets across multiple shots, and individual drives must be explicitly implemented and verified by the model.

## 2. Understand first, then model

For each component of the robot, specify the following: identity, geometric structure, material, stiffness-softness, independence, joint/contact/connection relationships, initial state, and source of evidence. Whether the task can be completed is just one of the checks; the semantic and physical structure of the object itself must also be valid.

The default modeling method is using Agent Blender; other modeling tools can be used if necessary. The final dynamics simulation is still performed in Isaac Sim 5.1. It is not allowed to use convex hulls that fill all gaps instead of petals or internal containers. Rotating joints cannot replace mating surfaces that require physical contact, and independent objects should not be fixed together for stability purposes. Soft bodies allow reasonable geometric errors based on observed uncertainties, but semantic, material responses, and task capabilities must be preserved.

Model review checks for `structure / materials / mechanisms / native_model`, binding all geometry and source code. The mesh distances, static rendering, or analytical contact during the preparation phase are merely suggestions and cannot replace the stability and contact verification of native simulation. When reusing scenes, verify initial states, occlusions, supports, penetrations, and inheritance speeds individually; do not rely solely on matching the overall center point.

## 3. Establishment of observations from the original video

- The camera is checked independently first. Camera drift, exposure issues, refraction, occlusion, and background adhesion can all create false physical errors.
- Points, contours, regions, axes, angles, deformations, and events are each represented by appropriate observations; complete regions and color pixel subsets cannot be used together.
- The thin edge does not equal the center line; measure using a paired boundary source image, retaining uncertainty. Points blocked are represented as NaN and cannot be filled in with simulation.
- Check the static segments, contact segments, and responses after contact to avoid prolonged static frames masking minor errors. Set up separate channels for parts and stages if necessary.
- First, determine the fit/hold-out index and tolerance. Review the identity, camera, occlusion, and clock. Then, bind the source file hash.

These channels and held-out frames are included in the verification plan. Subsequent failures cannot be resolved by deleting problematic frames, increasing tolerances, or removing enabled checks. If there are actual annotation errors, correct the original image, save the new observations and reviews, re-verify, without modifying the old records. The held-out frames for the same video can only be verified for this repetition; whether the parameters can be migrated requires verification for independent videos.

## 4. Plan parameters and run

Source code and plans are located in `configs/<name>/agent/physics/`, while running records are stored in `<input directory>/physics_process/round_NNNN/`. The input directory can be `real/<name>` or the directory provided by the caller.

```text
configs/<name>/agent/physics/
  model.py                 # create(world, stage, parameters, inputs)
  model_review.json        # Model review, associated with geometry and source code
  observation_review.json  # Newly extracted source image for observation review
  plan.json                # Input, model, parameters, channels, resources, search budget
<Input Directory>/
  observations.npz         # t + named channel; file name can be specified by the plan
  physics_process/
    round_0001/            # Planed snapshots, hashes, native candidates, status, and results
    round_0002/            # Start a new round of revisions
```

Parameter names are determined by the agent based on the model, without restricting quantities like mass/friction. Each quantity specifies initial values, units, range, applicable parts, basis, and either `fit` or `prior_only`; the model must actually use and bind all declared quantities. For example, material stiffness, damping, density, drive error, joint friction, external forces, or initial state—only those included in model implementation, native checks, and observations are added. The framework allows these models, but it does not guarantee that all types of joint identification have been verified.

After setting up the reconstruction environment in the root directory of the repository, the same set of agent interfaces can be called via Python.

```python
from calibrated_scene_reconstruction.physics.process import workflow
from calibrated_scene_reconstruction.physics.process.contract import read

recording = "real/my_process"  # Switch to this input directory
plan = read("configs/my_process/agent/physics/plan.json")  # First, establish and review this plan.
workflow.propose(recording, plan)
print(workflow.run(recording))
print(workflow.status(recording))
```

It can also be called via CLI/MCP's `physics_propose / physics_run / physics_status`; `PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.tool schemas` can query the planning schema. The agent creates models and revisions, while the harness manages clocks, free physical steps, parameter searches, readbacks, scoring, and immutable evidence. The code and inputs remain unchanged during execution.

First run the default candidates, then perform a combined search. By default, `search.objective=feasibility` prioritizes handling the maximum fit level that exceeds the limit; `training_loss` retains the overall fit error objective. Neither of these will read the held-out errors to select candidates or stop prematurely. Hard conditions such as contact penetration and invalid physical states can still reject candidates, and all held-out channels will be checked separately in the final step.

Native execution is allowed through the shared GPU launcher; `deferred` scenarios with insufficient memory can retry the original round. If an experiment is interrupted after some steps have been performed, the history is retained and a new round is created. Partial results cannot be masqueraded as complete executions. Processes that are not being closed in other sessions are not affected. The GPU budget, step size, or pre-staging duration are not fixed for this example.

## 5. Revised according to the reason for failure

| phenomenon | Next step |
| --- | --- |
| Source image has obvious misalignment, follows the background, or is merged with overexposed areas. | Review the original image, fix camera/observation and review, and verify again |
| Inconsistent starting position, orientation, or resting state | Revised bounded initial state assumption and contact, retaining complete static segment inspection |
| Continued shape orientation errors in the same section | Return to modeling loop; synchronize updates for display, collisions, and corresponding mass properties, re-review |
| The error in the force application stage is large, and the initial state and geometry have been established. | Check model function, material mechanism, fixed ratio and parameter omissions; conduct well-founded univariate comparison, and then perform joint fitting |
| Parameter changes have little impact on observations, or multiple parameters compensate for each other. | The report is indistinguishable, retain prior knowledge or request additional observations, do not call the range a confidence interval |
| Pass the fit, fail the held-out. | Still failed in this round; check the model or provide additional independent evidence. Do not select parameters based on held-out errors. |
| Repetitive native execution fluctuations near the acceptance boundary | Record fluctuations and inspect contact, initialization, and discrete errors. Do not claim reliability based on a single low error. |
| Input/source code changes, evidence stale | Retain old records, create new plans and review; do not rewrite old hashes |

Return the problem to the corresponding loop using `physics_request_geometry` and `physics_request_observations`. The local suggestion tool `tools/physics/propose_process_step.py` can generate new experiments from complete sources, but only using fitted data. Linear prediction, historical re-scoring, and analytical contact cannot be automatically transformed into conclusions based on evidence.

The generic improvements should be placed in the `src/calibrated_scene_reconstruction/physics/process/` directory, or in the general tools folder. The ROI, part identities, geometric assumptions, material properties, driving forces, and channel thresholds for the videos should remain in the `configs/<name>/agent/physics/` directory. When adopting this new approach, it’s important to keep the old interfaces and native representations intact. Additionally, it’s necessary to run regression tests for the processes. No new condition branches for task names should be added.

## 6. rendering and independent acceptance

Candidates who have passed the current round and are awaiting review can be rendered using `--candidate`. The default delivery rendering only accepts the current round that has passed independently. Rounds that failed but were executed completely are rendered as `--diagnostic`, and the video remains marked as failed; this does not result in approval.

```bash
PYTHONPATH=src tools/run_shared_isaac.sh apps/physics/render_process.py real/my_process \
  --candidate --output outputs/my_process/candidate_render

# Explicitly viewing failed rounds of history, retaining the current active round and review status
PYTHONPATH=src tools/run_shared_isaac.sh apps/physics/render_process.py real/my_process \
  --diagnostic --round 1 --output outputs/my_process/round_1_diagnostic
```

The above is in command format. Before use, replace the directory, input the name, and determine the number of iterations; the output directory must be a new one. The renderer will replay the native rigid body/PBD state and corresponding camera data. No physics calculations are performed again, and no real-world trajectory is used for animation purposes. Time errors can be checked by holding down the button for half a native step. It’s not possible to move the clock to hide any inconsistencies. Currently, the video comparison tool requires camera data, trace information, ROI data, and contour data for each frame of the original video. Sparse or variable frame-rate observations cannot be directly applied here. Any visual replay of other soft bodies or complete robot joint systems requires appropriate adaptation and verification.

The frame-by-frame source mask, proposed and reviewed by `segment_process_video.py`, can generate panoramas, identical crop sizes, and contour comparisons:

```bash
PYTHONPATH=src envs/recon/bin/python tools/physics/compare_process_video.py \
  --render outputs/my_process/candidate_render \
  --source-masks outputs/my_process/source_masks \
  --regions real/my_process/regions.npz --video data/my_process/input.mp4 \
  --mask-review configs/my_process/agent/physics/mask_review.json \
  --output outputs/my_process/comparison
```

In `regions.npz`, each frame’s `features` give the pixel center in the first two columns and the crop scale in column 4. `--crop-offsets LEFT TOP RIGHT BOTTOM` adjusts the field of view at that scale. Apply the same crop to real and simulated frames. The contour review uses `masks.json` to identify reliable and complete frames for each part. Report full IoU only when the source contour is complete. Use `--line-review` to overlay reviewed visible segments from the source image and omit inferred hidden contours.

If the native model stores information about external forces, you can use the `--force-trace <optimal candidate's NPZ>` option. The `commands` field in the file contains `[t, Fx, Fy, Fz, px, py, pz]`, where these values represent coordinates in the world frame, measured in seconds/Newton/meter. The tool verifies the hash value of the optimal candidate, calculates the actual forces based on the recorded timestamps, and determines the points and directions of application using the rendering camera. The output includes arrows with their magnitudes, world coordinates, and force curves. Save the result as `simulation_force.mp4`. The `--force-scale` option only adjusts the display scale of the arrows (in m/N), without affecting the simulation results. Arrows are not displayed when there are no external forces applied. These values are estimated values only; they cannot be considered true measurements from sensors or actual contact forces. Currently, the interface does not automatically visualize joint moments or couples.

Delivery requires numerical, held-out, and native physical checks to be passed. Then, individual reviews are conducted on `model_semantics / physical_plausibility / whole_process / parameter_evidence`. Actual rendering also depends on initial state, motion, occlusion, structure, and material responses; only central point error cannot be relied upon. `physics_review` is linked to results and evidence within the reviewed rounds; only after passing can it become `physics_accepted` or `physics_attach`. Additional results are integrated into scenarios S5/S6. The frozen assets are not modified, nor are any old exports automatically injected.

## 7. Delivery artifacts

Retain models/plans, geometric and source hashes, parameter evidence, optimal native initial state, states, predictions, checks, and reviews. Visual deliveries are kept simple: `real_vs_simulation.mp4`, `silhouette_overlay.mp4`, `simulation_replay.mp4`, and `real_input.mp4`. Indicate whether they pass, missing evidence, and priors, with complete paths provided. The comparison tools include both normal-speed and half-speed displays of identical frames, without shape interpolation or new physics synthesis.
