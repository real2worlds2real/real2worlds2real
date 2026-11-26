# Rigid mesh contact request

All distances are metres; vectors use world coordinates unless stated otherwise. Inline data avoids external mesh/path ambiguity.

Top-level fields:

- `bodies`: 2–32 entries with unique `id`, `vertices` (N×3), integer `triangles` (M×3), optional `static` (default false), and optional `pose` `[x,y,z,qx,qy,qz,qw]` (identity default). Use consistently outward, closed, nondegenerate positive-volume meshes. Total triangle budget is 500,000. Geometry checking is not a self-intersection certificate.
- `fps` (default 24), `frames` (240), `substeps` (20), `iterations` (20). The retained physical span is `(frames-1)/fps`.
- `gravity`: default `[0,0,-9.81]`.
- `contact`: `friction` (.1), `density_kg_m3` (900), `gap_m` (.0004), `sdf_band_m` (.001, must cover gap), `sdf_resolution` (256, allowed 32–512). Optional `sdf_voxel_size_m` overrides the resolution target, subject to a 1024-voxel per-body extent budget. These are assumptions, not inferred material properties. Use feature-aware resolution; a nominal grid count does not guarantee adequate thread accuracy.
- `controls`: optional, disjoint time intervals per moving body. No controls are installed by the passive loader.
- `thread_checks`: optional pose measurements, described below.

Each control has `body`, `start_s` (default 0), `end_s` (physical span by default):

- `force_n`, `torque_nm`: 3-vectors, zero default. Optional `linear_velocity_m_s` is the target for linear damping, zero by default. The force law is explicit force + selected position feedback + linear_damping × (velocity target − measured velocity), clipped by force_limit_n. A bounded upward velocity test can exercise removal without prescribing the part trajectory. The report retains actual solver mass and COM to help choose disclosed test wrenches.
- `position_target_m`, `position_axes`: world target and three boolean axes; no position axis is driven by default. `position_stiffness` and `linear_damping` are nonnegative. Linear damping applies to all translational axes, including unselected position axes.
- `angular_velocity_rad_s` and nonnegative `angular_gain`: a world angular velocity servo.
- Optional `alignment_stiffness`: aligns the body's local +Z direction to unit `alignment_axis_world` (default world +Z). This is an explicit external fixture. It is not a mating constraint.
- `force_limit_n`, `torque_limit_nm`: nonnegative per-axis limits applied to the entire test wrench, zero default. Set them intentionally; zero means that axis receives no external force/torque. Gravity remains active.

Example control fragment for a rotating, laterally guided body with free axial position (choose gains and limits for the actual object):

```json
{
  "body": "moving_part", "start_s": 0, "end_s": 4,
  "position_target_m": [0, 0, 0], "position_axes": [true, true, false],
  "position_stiffness": 20, "linear_damping": 0.2,
  "angular_velocity_rad_s": [0, 0, 1], "angular_gain": 0.0002,
  "alignment_stiffness": 0.005,
  "force_limit_n": [0.2, 0.2, 0.1], "torque_limit_nm": [0.001, 0.001, 0.001]
}
```

Each optional `thread_checks` entry specifies distinct `moving_body`, `reference_body`, unit `axis_reference` (default +Z in the reference frame), signed `lead_m_per_turn`, positive `tolerance_m`, inclusive `start_frame`/`end_frame`, and optional `minimum_turns` (default .25, absolute net rotation). These compare measured relative poses with the declared lead; they impose no constraint or trajectory. Choose a settled, engaged interval. Sampling must resolve individual rotations. Tolerance is supplied by the experiment, not automatically loosened on failure.

CLI:

```bash
python scripts/simulate_rigid_contact.py request.json --output behavior
blender --background --factory-startup --python scripts/render_rigid_contact.py -- behavior --mode render
python scripts/encode_object_demo.py behavior --name native_motion
```

Use the isolated native Newton requirements, with that environment available to subprocesses. Set `WARP_CACHE_PATH` and the native overlay through operator configuration when required; never embed an operator's paths or credentials in generated source.

The output retains request and motion hashes. Replay rejects changed caches. `load_rigid_contact(request, cache_dir=..., builder=...)` appends passive bodies to a compatible Newton builder and preserves body IDs. Diagnostic controls are applied only by the test script. This interface is not a robot manipulation implementation.

## Optional Blender source binding

A diagnostic request can supply `source_attempt_id` naming a successful saved Blender attempt. In this mode each body specifies `frame_object` (a rigid Empty/object with applied unit scale), `collision_objects` (evaluated closed mesh objects), and `visual_objects` (unique visible source meshes). Inline vertices/triangles and poses are exported from that source; do not fabricate an unrelated collision candidate. Every visible asset mesh must belong to one body. Hidden simulation proxies, `real2sim_ground` objects and harness-owned `Real2Sim_` objects are excluded from that visual inventory. Collision geometry must still pass the ordinary closed-mesh checks.

The tool retains a copy and hash of the exact source, exports body-local collision coordinates and visual bindings, then replays native transforms on the source meshes with their materials. It checks the original source has not changed during the test. This is source-matched diagnostic evidence. By default this remains diagnostic. Setting record_asset_evidence=true explicitly selects this native rigid review for the ordinary final asset gate. It requires render_video=true and a rigid contract. The saved runtime.json removes test controls and checks; contact_asset.json inventories the source, visuals, construction, trajectory and complete movie. Native numerical checks, fresh original-material replay and unchanged hashes are required before acceptance. An ordinary later run_simulation explicitly reselects the default backend, whose own check must then pass. Final physical calibration and robot policy are separate concerns.

Numerical reports also retain initial/final body poses and final collision-mesh AABB separation. A positive AABB gap is sufficient evidence of final geometric separation; zero gap is inconclusive because bounding boxes may overlap around concave, nonpenetrating surfaces. This check is not a penetration-depth measurement.

Optional `separation_checks` contains distinct body0/body1 identifiers and positive minimum_gap_m; each requires the final world collision-mesh AABB gap to meet that threshold. It is a sufficient, deliberately conservative separation check, not a penetration-depth test.

### Optional contact-loss stop for external rotation

A control may supply `stop_rotation_after_contact_loss: {"other_body":"fixed_part", "dwell_seconds":0.05}`. The test controller first waits until that exact shape pair has reported contact. After contact stays absent for the dwell, it latches off angular velocity/feedforward drive while retaining angular damping, alignment and independent linear forces. Initial absence of contact alone never fires it. The numerical report retains whether the pair contacted, whether the stop fired and its time. This is a disclosed test-controller event, not an imposed screw constraint or a claim of continuous collision certification.

When presenting a mating mechanism, distinguish rotation while engaged from free rotation after release. Continuing a fixed-duration spin after release can misleadingly suggest a larger required turn count. Choose event-aware or justified bounded controls and inspect the actual transition; do not infer engagement from total video rotation alone.
