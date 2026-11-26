# Physics from one RGB video

Use [observed-motion calibration](process_workflow.md) to design a model, choose parameters and compare a simulated response to the capture. For a recording with an identifiable free-motion phase, the motion-specific estimators also support sliding, rolling, bouncing, inclines, pendulums, rotating supports and flexible materials.

Run from the Real2Sim directory. Place the recording in a directory with its video and a source-derived `meta.json`. Inspect the metadata contract and select applicable stages.

```sh
envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.meta real/my_capture
tools/physics/video_physics.sh real/my_capture
```

Register the camera and metric frame, isolate the observed object and track it on the source clock. Estimate the parameters supported by the observed motion. Retain uncertainty and physically meaningful bounds when scale, loading or contact is ambiguous. The moving object responds to external controls through the selected physical model. Compare simulation readback with independent observations and review the interaction before accepting the asset.
