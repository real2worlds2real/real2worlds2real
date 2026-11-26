"""Real-to-sim physics validation and calibration (docs/real2sim_physics.md). Pure numpy except observe.track_video and
tabletop.perceive (SAM 3 / MoGe-2 on the GPU) and the Isaac side (sim.physics, sim.tabletop, apps/real2sim_replay.py).

data      real/<name>/ recording: loader, validation, resampling to the sim rate, trajectory-video offset
droid     a DROID episode (LeRobot port) -> real/<name>/, in the robot base frame
observe   per-frame object track from the video (mask centroid / bbox / depth centroid) and camera projection
tabletop  minimal twin from the first frame (support plane + primitive object) and its depth-scale family
params    the physical parameter vector: names, bounds, normalisation, physics.json
loss      sim vs video discrepancy: 2D track error, silhouette IoU, event timing
optim     bounded CMA-ES and cross-entropy method with batched ask / tell
overlay   sim silhouettes and robot links over the real frames, track and scan plots
"""
