# Pick-and-place data and training

Start from a reconstructed scene and a task-specific goal on a support surface. Sample collision-free grasps and plan the reach, lift, transfer, release and retreat. Check the final placement after settlement. Record robot state, commanded actions, camera frames and task outcome at matching timestamps.

Use `tools/collect_place.py` for collection with randomized layouts and `calibrated_scene_reconstruction.export.lerobot` for episode export. For DAgger, `tools/fail_states.py` converts a rollout trace into restart states for an expert recovery.

The policy adapters in `tools/policy/` convert the data for Diffusion Policy or OpenPI, and serve trained policies. Configure the dataset, observation channels and action conventions to match the intended deployment. The [station workflow](../../../policy_development/docs/station_harness.md) manages training, development rollouts, freezing and handover.
