"""Many demonstrations of an unseen task from one demonstration, object-centric (MimicGen / DemoGen style).

io          demo format (real/<demo>/), poses, photo camera <-> robot base calibration, demo -> twin world frame
segment     gripper events -> object-centric segments (reference object, TCP poses relative to it), final state
retarget    segments onto new object poses, collision-aware bridges, IK to joints, task success from the demo
functional  functional grasp regions (BODex seed constraints), in-hand pose ranking, pour / insert as object paths
sampler     augmentation axes (configs/demo_tasks/<task>.json) -> scene variants
generate    driver: python -m calibrated_scene_reconstruction.demo.generate sessions/<name> --demo real/<demo> --n 200
adapters    public datasets (DROID) -> the demo format
"""
