"""Reusable native robot replay model; agents may extend or replace this factory.

This reuses the existing tabletop/robot backend, not a task-name dispatch. A
primitive scene remains an explicit legacy choice. New reconstruction geometry
must be reviewed before the process plan is proposed.
"""
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import read, require


def robot_replay(world, stage, parameters, inputs):
    from calibrated_scene_reconstruction.physics.replay.data import load
    from calibrated_scene_reconstruction.physics.common.observations import project
    from calibrated_scene_reconstruction.robots.spec import load_spec
    from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot, hand_links
    from calibrated_scene_reconstruction.sim.physics import apply_scene, apply_robot, hand_prims
    from calibrated_scene_reconstruction.sim.tabletop import build_tabletop
    from calibrated_scene_reconstruction.physics.replay import params as parameter_bank
    from scipy.spatial.transform import Rotation
    from pxr import UsdPhysics

    rec = load(Path(inputs['robot_metadata']).parent)
    trajectory = rec.path/('trajectory.npz' if (rec.path/'trajectory.npz').exists() else 'trajectory.csv')
    for name,path in [('robot_metadata',rec.path/'meta.json'),('camera',rec.path/'camera.json'),
                      ('robot_trajectory',trajectory),('video',rec.video)]:
        require(path is not None and Path(inputs[name]).resolve() == path.resolve(),
                'Replay source must match the declared input: '+name)
    require('video_offset_s' in rec.meta, 'Establish the video/trajectory clock alignment before fitting')
    spec = load_spec(inputs['robot_spec'])
    require(Path(spec['usd']).resolve() == Path(inputs['robot_usd']).resolve(), 'Robot asset differs from declared input')
    twin = read(inputs['scene'])
    if twin['object']['shape'] != 'mesh':
        require(inputs.get('legacy_template_opt_in') and read(inputs['legacy_template_opt_in']).get('explicit') is True,
                'Primitive reconstruction is an explicit compatibility fallback, never the default')
    else:
        from calibrated_scene_reconstruction.core.paths import resolve
        require(Path(inputs['object_mesh']).resolve() == resolve(twin['object']['mesh']).resolve(),
                'The native object mesh must be a declared, reviewed input')
    own = next(str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    target = build_tabletop(stage, twin)
    # The caller owns the single native physics scene already created by World.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()) != own:
            stage.RemovePrim(prim.GetPath())
    side = next(iter(spec['arms']))
    base = rec.meta['base']
    add_robot(stage, spec, base[:2], np.radians(base[2]), support_top=float(rec.meta.get('base_z', 0.)))
    supported = {p.name for p in parameter_bank.PARAMS} - {'time_scale', 'twin.depth_scale', 'twin.support_dz'}
    require(set(parameters) <= supported, 'Replay model does not implement these parameters; select or extend the physical model')
    values = parameter_bank.Space([]).flat([])
    values.update(parameters)
    physics = parameter_bank.nested(values)
    apply_scene(stage, physics, target, supports=('/World/Support',), tag='process')
    apply_robot(stage, physics, hand_prims(stage, hand_links(stage, spec, side)), tag='process')
    robot, Q, V = None, None, None
    source_t = rec.t + rec.meta['video_offset_s']
    dt = world.get_physics_dt()
    sim_t = np.arange(0., max(0., source_t[-1])+dt, dt)
    from calibrated_scene_reconstruction.sim.tabletop import object_mesh
    vertices, _ = object_mesh(twin['object'])
    center = (vertices.min(0)+vertices.max(0))/2

    def initialize():
        nonlocal robot, Q, V
        robot = Robot(world, spec)
        require(set(rec.joints) <= set(robot.idx), 'Unknown trajectory joints; explicit joint mapping required')
        Q = np.repeat(robot.home()[None], len(sim_t), axis=0)
        for j, name in enumerate(rec.joints):
            Q[:, robot.idx[name]] = np.interp(sim_t, source_t, rec.q[:, j])
        if rec.gripper is not None:
            ix = np.clip(np.searchsorted(source_t, sim_t, side='right')-1, 0, len(source_t)-1)
            Q = np.array([robot.with_ee(q, float(g), side) for q, g in zip(Q, rec.gripper[ix])])
        V = np.gradient(Q, dt, axis=0) if len(Q)>1 else np.zeros_like(Q)
        robot.teleport(Q[0])

    def drive(t):
        i = min(round(t/dt), len(Q)-1)
        robot.command(Q[i], V[i])

    def observe(t, state):
        pose = state['object']
        R = Rotation.from_quat([*pose[4:],pose[3]]).as_matrix()
        uv = project(rec.K, rec.T_world_cam, (pose[:3]+R@center)[None])[0][0]
        return {'object_centroid_px': uv, 'object_position_m': state['object'][:3],
                'robot_position_m': state['robot'][:3]}

    def checks(trace):
        poses = np.array([r['bodies']['object'] for r in trace])
        top = twin['support']['z']
        minimum = min(float((vertices@Rotation.from_quat([*p[4:],p[3]]).as_matrix().T+p[:3])[:,2].min()) for p in poses)
        tolerance = inputs['_process_context']['physical_limits']['max_support_penetration_m']
        require(tolerance>0, 'Agent must declare a positive contact tolerance')
        return {'support_contact': {'pass': bool(minimum >= top-tolerance),
                  'reason': f'Source mesh minimum z={minimum:.6g}, support={top:.6g}, declared tolerance={tolerance:.6g} m.'}}

    return {'bodies': {'object': {'kind': 'rigid', 'path': target},
                       'support': {'kind': 'static', 'path': '/World/Support/top'},
                       'robot': {'kind': 'rigid', 'path': '/World/Robot/'+rec.meta.get('ee_link', spec['arms'][side]['tcp_link'])}},
            'driver_bodies': ['robot'], 'after_reset': initialize, 'drive': drive, 'observe': observe, 'checks': checks,
            'settle': lambda: robot.command(Q[0],np.zeros_like(Q[0])),
            'joint_states': lambda: {'robot':robot.q()},
            'readbacks':lambda actors: ({'object.mass':{'value':float(actors['object'].get_mass()),'basis':'native PhysX mass'}}
                                       if 'object.mass' in parameters else {}),
            'consumed_parameters': list(parameters), 'effective_parameters': values}
