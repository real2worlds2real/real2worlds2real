"""Native thread fixture using the reconstruction contact runner."""


def run(world, stage, out, part, fixed, object_spec, material):
    import numpy as np
    import trimesh
    from isaacsim.core.prims import SingleRigidPrim
    from calibrated_scene_reconstruction.assets.thread_geometry import shell
    from calibrated_scene_reconstruction.sim.mechanism_stage import add_links
    from calibrated_scene_reconstruction.sim.usd import set_pose
    thread = {'pitch': .004, 'depth': .0012, 'z_min': .005, 'z_max': .029, 'flank_fraction': .35}
    V, F = shell(outer_radius=.030, inner_radius=.026, height=.03, floor=.003,
                 segments=128, thread=thread)
    cap_thread = {**thread, 'z_min': .003, 'z_max': .017, 'phase': .5}
    C, CF = shell(outer_radius=.035, inner_radius=.0302, height=.018, floor=.003,
                  segments=128, thread=cap_thread, thread_side='inner')
    C = C*np.array([1, -1, -1])+[0, 0, .018]
    kit = {'parts': [part('cup', trimesh.Trimesh(V, F, process=False)),
                     part('cap', trimesh.Trimesh(C, CF, process=False))], 'joints': []}
    settings = {'sdf': {'resolution': 768, 'contact_offset': .00003, 'rest_offset': 0.}}
    add_links(stage, '/World/Objects/Test', kit, settings, object_spec, material)
    set_pose(stage.GetPrimAtPath('/World/Objects/Test/cap'), [0, 0, .028], [1, 0, 0, 0])
    fixed('/World/Objects/Test/cup')
    world.reset()
    from calibrated_scene_reconstruction.core.session import Session
    from calibrated_scene_reconstruction.tasks.rigid.task_checks import rot, quat
    from calibrated_scene_reconstruction.tasks.mechanisms.thread_contact import run as thread_check
    cup, cap = SingleRigidPrim('/World/Objects/Test/cup'), SingleRigidPrim('/World/Objects/Test/cap')
    cup.initialize()
    cap.initialize()
    # Use the same general check runner that reconstructable task assets use.
    class Context:
        s = Session(out)
        def body(self, name):
            return {'cup': cup, 'cap': cap}[name]
        def pose(self, name):
            p, q = self.body(name).get_world_pose()
            return np.asarray(p), rot(q)
        def place(self, name, p, R):
            body = self.body(name)
            body.set_world_pose(np.asarray(p), quat(R))
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
    ctx = Context()
    ctx.world, ctx.stage = world, stage
    stage.RemovePrim('/World/Fixtures/cup')
    checked = thread_check(ctx, 'cup', 'cap', lead_m_per_turn=.004,
                           pitch_m=.004, thread_bands={'base': [.005, .029], 'cap': [.001, .015]},
                           entry_z=.028, seated_z=.015, tolerance_m=.0008,
                           output='evidence', max_duration_s=40.)
    return checked
