# Optional final operation movie

`render_motion_video(attempt_id, assumptions)` renders a candidate movie for
`list_video_sources` / `inspect_video`. It leaves acceptance to the agent.
Choose this after the candidate is ready when a continuous operation helps
review contact, articulation, coupling or deformation. No fixed viewing schedule
or per-draft movie is required.

Hybrid candidates replay their retained Newton cache. Rigid and articulated
PhysX candidates use the existing physical operation recorder and replay its
actual body poses through the original editable Blender appearance. This is
physics-driven motion, not scripted Blender joint animation.

Newton cloth candidates replay the passed bundle's `simulation_motion.npz`;
they do not need `motion_plan.json`. Choose the cloth lift/hold protocol using
`run_simulation` parameter overrides when necessary. The movie preserves that
actual trajectory, source appearance and original output cadence.

For a PhysX candidate, write `motion_plan.json` in your agent workspace:

```json
{
  "attempt_id": "attempt-003",
  "plan": {
    "duration": 6,
    "fps": 24,
    "root_height": 0,
    "warmup_seconds": 2,
    "stiffness": 100,
    "damping": 10,
    "controls": [
      {"joint": "EXACT_SIMULATOR_JOINT_NAME", "keyframes": [[0, "lower"], [1, "lower"], [3, "upper"], [4, "upper"], [6, "lower"]]}
    ]
  }
}
```

Choose the meaningful operation and sequence yourself. Use the exact target
joint names returned in simulation diagnostics, which can differ from Blender
names. Joint commands use radians or `"lower"` / `"upper"` from the target limits.
Times increase from zero. Unlisted joints are held at their initial coordinates.
Do not independently command mechanically coupled coordinates inconsistently.
Rigid drop/contact tests use `controls: []` and an appropriate positive
`root_height`. An optional `camera_direction: [x,y,z]` selects the review side.
The recorder uses 24, 30 or 60 FPS and a duration between zero and 30 seconds.
These are temporary test drives; the passive delivered asset stays unchanged.

The tool checks source/bundle integrity, numerical motion and complete video
encoding. Those checks do not prove visual or functional quality. Inspect the
actual resulting movie when you request it, fix demonstrated defects and rerun.
Encoding all frames does not mean the agent inspected all of them. Review dense
intervals or exact frames/crops when short transient defects are a concern.
The same rendered movie is available for delivery after acceptance.
