# Data from one demonstration (`calibrated_scene_reconstruction.demo`)

A photo of an environment and one robot (or teleoperated) trajectory of a task the pipeline has never scripted there
(put a cup in a basket, stack, pour) → many demonstrations of the same task in the reconstructed twin and its cousins.
Object-centric re-targeting in the spirit of MimicGen (segments at gripper events, each replayed in its reference
object's frame) and DemoGen (only the part near the object is replayed, the free-space part is replanned).

## Data flow

```
real/<demo>/{demo.json, traj.npz, photo}         ─ demo.io.load: base frame → photo camera → twin world
   │  + sessions/<name>/ (twin objects, cousins)
   ▼
demo.segment.segment     gripper events → segments {ref object, object in hand, TCP poses in its frame, skill start}
   │                     + the demo's final object state (the object in hand follows the TCP)
   ▼
demo.retarget.success_spec   last released object relative to its container (pos / tilt tolerances), or a lift
   ▼
demo.functional.hold     per close: the hold measured on the object's mesh (+ the gripper's opening) → functional or
   │                     generic, region, closing axis, in-hand pose, symmetry; candidates (BODex records from
   │                     grasping.synth --demo, or antipodal pairs) ranked by in-hand error
   ▼
per draw (demo.sampler): scene (twin / cousin), object jitter on the support, target augmentation, distractors,
                         lights / HDRI, camera jitter, robot base jitter; one ranked grasp per hold →
                         functional.retarget_segments (approach, object path / pour / insert, retreat)
   ▼
demo.retarget.plan       skill: T_ref_new @ ee_rel; bridges: lifted Bezier clear of furniture boxes, minimum jerk;
                         gripper switches at each event, held `actuate` s
   ▼
sessions/<name>/demo/<demo>/{demo.json, plans/plan_XXXX.{npz,json}}
   ▼  TODO (Isaac Sim): apps/episode.py plays a plan
build the variant's scene → retarget.to_joints (IK, trajgen.TrajectoryPlanner.ik_from) → PhysX batch validation
(--batch) → retarget.check_success on the final object poses → record with meta["task"]
   ▼
calibrated_scene_reconstruction.export.lerobot (unchanged: the task text comes from meta.json)
```

## Formats

`real/<demo>/demo.json`

```jsonc
{"robot": "franka_panda",            // configs/robots/<name>.json
 "side": "right",                     // arm of a bimanual robot
 "task": "put the cup in the basket", // dataset task text (a task config may override it)
 "fps": 30,
 "photo": "photo.png",                // the twin's photo, or one from the same viewpoint
 "T_cam_base": [[...4x4...]],         // robot base in the photo camera (OpenCV axes) ...
 "calibration": "calib.json"}         // ... or correspondences to fit it
```

`traj.npz`: `t` (T,) s; `ee_pose` (T, 7: xyz + quat wxyz, TCP in the base frame) and/or `q` (T, D) (joint-only demos
need FK: the live articulation in Isaac Sim, `io.load(..., fk=)`); `gripper` (T,) 0 open .. 1 closed.
`calib.json`: `p_base` / `p_cam` (N ≥ 3 non-collinear): TCP positions from FK and the same points in the photo camera
(a marker on the gripper seen from the photo's viewpoint, depth from `parsing/geometry.npz`); `io.fit_rigid` (Kabsch)
gives `T_cam_base` and the rms residual (written to `demo.json` of the output). Then
`T_world_base = T_world_cam @ T_cam_base` with `T_world_cam` from `Session.world_frame()`.

Object frames are the simulator's (`sim.scene.add_object`): origin at the bottom centre of the world-aligned twin mesh;
a cousin's `xy` / `yaw` move and turn it about the world z. Task config (optional, `--cfg`):
`configs/demo_tasks/<task>.json`, starting from the [example](../../configs/demo_tasks/example.json).
`demo.sampler.DEFAULT` lists every key and is used when no file is given: jittered objects, augmentation probability,
distractor range, scene weights, light / camera / base ranges, success tolerances, skill radius `near`, bridge speed.

## Functional grasps and in-hand skills (`demo.functional`)

Where and how the demo holds the object matters for a mug by its handle, a tool by its grip, a spray bottle at the
trigger, a pot by its handle, a drawer knob, and for pouring, inserting or handing over in a given orientation.

The demo decides; nothing about the object's parts or where to hold it is written by hand.

- Hold (`functional.hold`, per close segment on a twin object; mesh in the sim object frame, bottom centre at the
  origin): the TCP in the object frame (`ee_rel[-1]`) is measured against the mesh. `classify` uses only measurements:
  the material between the fingers (rays from both fingertips at ±stroke/2 toward the TCP, `held_width`) along the
  closing axis that straddles the least material (`closing_axis`: the dataset's gripper frame convention is not
  needed); the demo gripper's own opening while holding when the dataset has one (`generate.opening_of`:
  (1 − position) × `gripper_max_width`, DROID 85 mm), which decides "thin" independent of the reconstruction; the
  TCP's offset from the object's vertical axis over its radius; the contact's connected component (after merging
  coincident vertices; ignored when no component holds half the area, a triangle soup has no parts); the TCP's place
  along an elongated object's long axis. Functional when any of `thin` (held / object width ≤ 0.35), `off_axis`
  (≥ 0.55), `part` (≤ half the area), `end` (≥ 0.45 of the half length, objects ≥ 2.5× longer than wide)
  (`functional.HOLD`, overridable in `grasp.hold`); generic otherwise. Region: the contact's part, else a ball of
  `radius` around the contact. In-hand pose in a canonical TCP frame (`canon`: z approach, y closing / thumb) with
  the measured closing axis; rotational symmetry about z measured on the mesh (`symmetry_z`: 95th percentile of
  turned samples' distance / radius ≤ 0.06).
- Task config `grasp`: `{"type": "auto" | "functional" | "any", "radius", "inhand_pos_tol", "inhand_rot_tol_deg",
  "symmetry": "auto" | null | "z", "top_k", "hold": {thresholds}}`; `auto` (default) = the measurement decides,
  the other two override it (recorded as `decided_by`).
- Candidates (`generate.holds_of`): BODex records of the generated robot's hand when `grasping.synth --demo` wrote
  them (`bodex_tcp`: base-link pose @ `hand_tcp`, then `canon`), else, for a parallel gripper, antipodal pairs sampled
  on the mesh (`functional.antipodal`: opposite surface along the inward normal within the stroke, parallel normals,
  approach turned about the pair, not from below, palm clear), in the region when the hold is functional. `rank`:
  functional → within the in-hand tolerances, best first (m + 0.05 m/rad; a parallel gripper's finger-swapped twin
  counts); generic → every candidate, demo-like first.
- BODex: `python -m calibrated_scene_reconstruction.grasping.synth sessions/<name> --robot <hand robot> --demo sessions/<name>/demo/<demo>`:
  per hold on the twin object's seated mesh, seeds in `functional.seed_region(region)` when functional
  (tabletop `seed_region_center` / `seed_region_radius`, hull-depth filter off), and on the mesh as reconstructed
  (`solid=False`, cavity open, own `__real` key) when the hold straddled a thin wall: the filled solid has no wall.
- Plans (`generate.main`): the demo's TCP poses are first turned about the approach so that the closing axis is the
  measured one (`frame_map`, recorded as `tcp_frame_map_deg`); per plan one of the `top_k` ranked grasps per hold
  (`pick_grasps`; none in tolerance → the demo's own in-hand pose, `source: "demo"`); `retarget_segments`: the close
  segment's approach keeps its motion relative to the TCP (`T_new @ inv(T_old) @ ee_rel`), held segments keep the
  OBJECT path (`object_path` / `tcp_path`), the segment after the release keeps its motion relative to the release
  pose. `skills` (by held segment index): `transport` (default), `pour` (the object turned about a horizontal axis
  through its spout over the reference's origin; axis, spout and height taken from the demo's own pour when missing:
  the axis its z turned about, the rim vertex lowest at the deepest tilt, the spout's height there), `insert` (along
  the axis into the demo's final pose, orientation fixed; `aligned` checks it). `demo.json` records every hold
  (measurements, decision, region, symmetry, candidates, the 32 best ranked); each plan json records the chosen grasp
  per hold (candidate, source, region, in-hand error) and the skill per held segment.
- TODO: a region cost inside the BODex optimisation, the contact spheres' positions in the record (computed now by a GPU side script), named parts
  on fused meshes (a part segmenter), articulated knobs / handles (joint value as success), hand-over poses, the
  generated robot ≠ demo robot (the canonical TCP frame compares them; IK of the other robot is not checked).

## Public datasets

The DROID adapter reads a separately obtained LeRobot v2.1 episode. Refer to the upstream data terms before using a recording. [DROID project](https://droid-dataset.github.io/) · [LeRobot dataset card](https://huggingface.co/datasets/cadene/droid_1.0.1)

```sh
python -m calibrated_scene_reconstruction.demo.adapters.droid "$EPISODE_PARQUET" real/my-demo --video "$EXTERIOR_VIDEO" --check
```

The adapter reads flange poses, gripper position, exterior-camera calibration and the task instruction. Use `--camera` to choose a view and `--tcp-offset` to specify the distance from the flange to the tool centre along its local z axis. Verify that the projected TCP follows the gripper in your recording. The CPU check uses the robot base as its world frame and infers objects at gripper events. Running the generated plan in physics requires a reviewed twin.
