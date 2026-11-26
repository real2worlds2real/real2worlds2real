"""Apply calibrated or randomized physical parameters to an authored scene."""


def apply_scene_physics(stage, ts, variant):
    cal = ts.inputs.physics  # calibrated against a real recording (calibrated_scene_reconstruction.real2sim); robot: apply_robot
    target = cal.get("calibration", {}).get("object_name", ts.s.scene_graph["task"]["target"])
    if cal and target in ts.objects:
        from .physics import apply_scene

        apply_scene(stage, cal, str(ts.objects[target].GetPath()))
        print(f"physics.json: calibrated {target} and its support", flush=True)
    dr = (variant.get("dr") or {}).get("physics")
    t_dr = ts.s.scene_graph["task"]["target"]
    if dr and "friction" in dr and t_dr in ts.objects and not cal:
        # a domain-randomized configuration (calibrated_scene_reconstruction.dr; its mass came with the object spec): the target's own
        # friction, combined with the hand's and the support's as PhysX's default (average) does -- the calibration's
        # pair coefficients (multiply, apply_scene) took the hand's bare friction and no Sharpa grasp held the cup
        from .physics import bind_colliders, physics_material

        f = float(dr["friction"])
        bind_colliders(stage, [str(ts.objects[t_dr].GetPath())],
                       physics_material(stage, "/World/Looks/dr_object", f, 0.9 * f))
        print(f"physics: randomized {t_dr} friction {f:.2f}", flush=True)
