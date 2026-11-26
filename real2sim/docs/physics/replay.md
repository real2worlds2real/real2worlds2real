Replay identification uses video and joint trajectories; video identification uses free-motion evidence. They share observations and data contracts while retaining their own identification strategies.

# Real-to-sim physics validation and calibration

A real robot recording (joint trajectory + fixed-camera video of the robot moving an object) is replayed in a twin of
its scene. The adapter converts observations, SAM 3 tracks the object, and Isaac replays joint motion in the calibrated scene. Compare projected tracks and event timing before fitting the physical parameters.

## Data flow

Task-object reconstruction now defaults to the agent Blender loop. The tabletop
entry first writes `model_initialization.json` from the observed support and
silhouettes, then stops if no current reviewed model exists. Author
`configs/<name>/agent/objects/object/model.py` and run:

```bash
python -m calibrated_scene_reconstruction.agent.physics_models fit real/<name> --mode replay
python -m calibrated_scene_reconstruction.agent.physics_models review real/<name> --round 1 --record review.json
python -m calibrated_scene_reconstruction.real2sim.tabletop real/<name>
```

Inspect the round evidence and previews before writing an `OBJ.object` review
with the exact round and source hash, `checks`, `review`, `pass` and resolved
`findings`. This uses the same image-evidence thresholds as scene reconstruction,
without measured task-object sizes. Tracked centroids initialize motion in the
recording camera; they do not independently resolve unobserved rotations.
The simulation entry also verifies the accepted source, observations, camera,
mesh and applied pose before starting Isaac. With a reconstructed session it
requires that session's current Blender object reviews instead.

Reviewed Blender geometry stays fixed during physics calibration. Geometry
nuisance parameters `twin.depth_scale` and `twin.support_dz` require a new model
iteration and review; the replay entry rejects these search parameters or
nonidentity values before starting Isaac. Mesh observations use the mesh
silhouette path, preserving visible concavities rather than filling their hull.
Geometry scans apply to the explicit template fallback.

`--object-route template` on the tabletop or replay entry explicitly restores the
primitive fallback, including for legacy numerical controls. Existing primitive
fitting functions remain available as low-level initialization/control APIs.

```
real/<name>/{meta.json, trajectory.csv|npz, video.mp4, camera.json}      the recording (format below)
  ├─ python -m calibrated_scene_reconstruction.real2sim.droid <episode.parquet> <camera.mp4> real/<name> --object "<prompt>"   (DROID adapter)
  └─ python -m calibrated_scene_reconstruction.real2sim.observe real/<name> [--stride 1]            GPU: SAM 3 (loaded once, frames batched)
       → real/<name>/track.npz                                            centroid, bbox, area, masks per frame
  └─ twin: meta "session" (a pipeline twin of the scene), or
     python -m calibrated_scene_reconstruction.real2sim.tabletop real/<name>                        GPU: SAM 3 support + MoGe-2, first frame
       → real/<name>/{twin.json, twin_qa.png}                             support box + primitive object
  └─ tools/run_isaac.sh apps/real2sim_replay.py real/<name>               replay with default / given physics
       → real/<name>/replay/{sim_track.npz, loss.json}
  └─ ... --scan p1,p2 --batch 12 [--params fit.json]                      loss along each parameter (identifiability)
       → real/<name>/replay/{scan.json, scan_<p>_<k>.npz}
  └─ ... --calibrate --batch 18 --repeats 3 --active ... [--params start]   CMA-ES / CEM, N scene copies at once
       → configs/<scene>/physics.json (session) or real/<name>/replay/physics.json (tabletop twin),
         replay/{history.json, sim_track_{default,calibrated}.npz, loss_{default,calibrated}.json}
  └─ python -m calibrated_scene_reconstruction.real2sim.overlay real/<name> --tracks sim_track_default.npz sim_track_calibrated.npz [--scan]
       → replay/{overlay.png, track_plot.png, scan.png}
configs/<scene>/physics.json → sim.scene.build_scene (object, support); sim.physics.apply_robot (gripper, drives)
```

With a session, the twin was reconstructed from a frame of the same scene, so the object, the support and
`camera.json` share its world frame. A dataset recording has no session: its world frame is the robot base (DROID's
extrinsics are given in it) and `real2sim.tabletop` builds a minimal twin in that frame.

## Recording format (`real/<name>/`, git-ignored)

| file | content |
|---|---|
| `meta.json` | `robot` (configs/robots spec), `joints` (DOF names = column order of q, matching the spec's USD), `object` (SAM 3 prompt), `base` [x, y, yaw_deg] of the robot in the world frame; `session` (`sessions/<scene>`) and `object_name` (twin object, default the task target), or `twin` {`support` prompt, `shape` cylinder / box, `mass`?} and optional `base_z` (default 0) without a session; optional `video_offset_s`, `ee_link` (the link whose pose `ee_pose` records: kinematic check) |
| `trajectory.csv` | header `t,<joints...>[,gripper][,ee_x,ee_y,ee_z,ee_qw,ee_qx,ee_qy,ee_qz]`; t in seconds, any start; gripper command 0 open .. 1 closed |
| `trajectory.npz` | alternative: `t` (T,), `q` (T, J), optional `gripper` (T,), `ee_pose` (T, 7) |
| `video.mp4` | fixed camera; its timestamps are the wall clock |
| `camera.json` | `K` 3x3, `width`, `height`, `T_world_cam` 4x4 camera-to-world in OpenCV axes (x right, y down, z forward), optional `dist`, `K_source` |
| `twin.json` | tabletop twin (`real2sim.tabletop`): `support` {z, box, tilt_deg}, `object` {shape, dims, pose [x, y, z, yaw_deg], mass, first_frame_iou} |
| `events.json` | optional hand labels in video seconds (`lift`, `slip`, `drop`, `settle`); not used by the loss yet |

`real2sim.data.load` checks columns, shapes, finite and strictly increasing timestamps and a proper camera rotation.
`align` resamples q linearly and the gripper by zero-order hold to the 120 Hz physics rate. The trajectory-video offset
is `meta.video_offset_s` or estimated (`estimate_offset`: normalised cross-correlation of the video's frame-difference
energy with the joint speed; give it a recording that starts or ends at rest).

## Public real-robot data: DROID

The adapter reads a separately obtained DROID episode in LeRobot v2.1 format. Select a recording and an exterior camera in which the robot's interaction is visible. Follow the source terms for the recording you use. [DROID project](https://droid-dataset.github.io/) · [LeRobot dataset card](https://huggingface.co/datasets/cadene/droid_1.0.1)

- Trajectory inputs are the joint positions, gripper position and flange pose. The replay uses `panda_link8` as the end-effector link.
- The adapter derives `T_world_cam` from the recorded exterior-camera extrinsics. Supply calibrated intrinsics for your device and crop using `--fx`, `--cx` and `--cy`. The nominal HD720 values are only starting estimates. Check `droid.overlay_tcp` against the recording.
- Video is decoded with AV1 support and converted to lossless H.264. Verify alignment between the video and trajectory clocks.
- Select the robot model that matches the recording, including the mounted gripper and fingertip geometry.

## Tabletop twin (`real2sim.tabletop`, `sim.tabletop`)

From the first frame: the SAM 3 support mask minus the object; MoGe-2 points with the camera's horizontal FOV from K;
world points through `camera.json`. The support is estimated from the plane inliers and the object's initial silhouette supplies its image-space constraint. Examine `twin_qa.png` against the first frame. A silhouette alone leaves object size and distance ambiguous. Use independent calibration and review any change to the source geometry before fitting physical parameters.

## Parameters (`real2sim.params.PARAMS`; unit box, log-uniform where marked)

| name | bounds | default | note |
|---|---|---|---|
| `object.mass` (log) | 0.02 – 3 kg | 0.3 | weigh it when possible and fix it (`Space(fixed=...)`) |
| `object.com_offset.{x,y,z}` | ±3 cm | 0 | from the mesh bbox centre, object frame; z is the identifiable one (tipping) |
| `object.restitution` | 0 – 0.6 | 0.05 | only seen in drops / bounces |
| `gripper.static_friction` (log), `gripper.dynamic_ratio` | 0.1 – 2, 0.5 – 1 | 1.0, 0.85 | object-pad pair; slip under the grasp |
| `support.static_friction` (log), `support.dynamic_ratio` | 0.05 – 1.5, 0.5 – 1 | 0.5, 0.85 | object-support pair; pushes, slides |
| `drive.stiffness_scale`, `drive.damping_scale` (log) | 0.25 – 4 | 1 | multipliers on the spec's gains; tracking lag of the arm |
| `gravity`, `time_scale` | 9.0 – 10.6, 0.9 – 1.1 | 9.81, 1 | sanity checks, off by default (below) |
| `twin.depth_scale` | 0.7 – 1.3 | 1 | tabletop twin only: scale about the camera (nuisance, not physics) |
| `twin.support_dz` | ±5 cm | 0 | tabletop twin only: support and object lifted together (nuisance) |

Default active set: mass, com z, both friction pairs, restitution. Pair coefficients are exact: the object material has
friction 1 with combine mode `multiply` (restitution: `max`), so the pair gets the pad's / support's own coefficient
(`sim.physics`). physics.json is the nested form (`{"object": {"mass", "com_offset", "restitution"}, "gripper":
{"static_friction", "dynamic_friction"}, "support": {...}, "drive": {...}, "gravity", "time_scale", "twin":
{"depth_scale", "support_dz"}, "calibration": {source, loss, loss_default, loss_copies, object_name, active,
generations}}`); `params.flatten` inverts it.

## Loss (`real2sim.loss.discrepancy`, in the real image, on the video clock)

- track: mean Huber distance (px, delta 10) between the projected sim object centre and the SAM 3 mask centroid;
- iou: 1 - mean IoU of the sim silhouette and the SAM 3 mask (weight 20);
- events: mean timing error (s) of `move` (first sustained displacement over valid frames: an occlusion gap neither
  counts nor breaks it), `settle` (entry into a 1.5 px band around the final resting place, after a move; a band, not a
  speed threshold, because mask centroids jitter by a pixel) and `slip` (object moves relative to the projected TCP
  after closing), 1 s for an event only one side shows (weight 30 px/s);
- arm: mean distance (mm) of the replayed end-effector link from the recorded one (`meta.ee_link`, weight 1 px/mm).
  Inspect discrepancies for unintended contact between the model and the environment.

The tracker segments batches of eight frames. It initializes with the highest-scoring instance and then selects the instance nearest the last known centroid within 8% of the image diagonal. It leaves occluded frames unassigned.

Optimizers (`real2sim.optim`): CMA-ES (rank-one + rank-mu) and CEM on [0, 1]^d with batched ask / tell; out-of-box
samples are evaluated clipped with a quadratic penalty. The replay evaluates one generation at once in `--batch` copies
of the whole scene (the `episode.py --batch` layout), each with its own materials, mass, drives and (tabletop) twin
geometry. Every rollout stops the timeline, authors the USD, then `world.reset()`: authoring while playing was undone
by the reset (stop restores the state saved at play; seen as copies that kept the previous generation's collision
mesh). PhysX's mass of every copy is read back after the reset and checked against the authored value. The fit is
re-evaluated in all copies at the end (`loss_copies`: mean, std): copies differ only by their place in the scene, so the
spread measures how much float round-off changes the contact outcome.



Use the [process workflow](process_workflow.md) to fit and review the response on your own capture.
