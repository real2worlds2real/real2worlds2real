"""Finite, slow external sphere contact diagnostics for independent rigid assets.

Only the disclosed external tool is commanded. Tested body poses, velocities,
mass, friction and source geometry are never prescribed. This cannot identify
real stiffness or approve source appearance; it records the modeled response.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def kinematic_target(view, command):
    """Move only one fixture through its active tensor backend, never teleport."""
    command = np.asarray(command, dtype=np.float32)
    if (command.shape != (3,) or not np.isfinite(command).all()
            or view._physics_view.count != 1):
        raise ValueError('A finite target and dedicated single-fixture native view are required')
    values = np.asarray([[*command, 0., 0., 0., 1.]], dtype=np.float32)
    targets = view._backend_utils.convert(values, device=view._device, dtype='float32')
    indices = view._backend_utils.convert(np.asarray([0], dtype=np.int32),
                                          device=view._device, dtype='int32')
    view._physics_view.set_kinematic_targets(targets, indices)


def validate(plan, names):
    required = {'link', 'start_m', 'direction', 'radius_m', 'travel_m',
                'speed_m_s', 'contact_load_N', 'maximum_load_N',
                'maximum_penetration_m'}
    if (not isinstance(plan, dict) or set(plan)-{'actuator'} != required
            or plan['link'] not in names):
        raise ValueError('Explicit existing target and complete external probe plan required')
    start = np.asarray(plan['start_m'], float); axis = np.asarray(plan['direction'], float)
    if start.shape != (3,) or axis.shape != (3,) or not np.isfinite([start, axis]).all():
        raise ValueError('Finite three dimensional probe pose and direction required')
    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-8):
        raise ValueError('Probe direction must be a unit vector')
    bounds = {'radius_m': (.0005, .025), 'travel_m': (.0001, .02),
              'speed_m_s': (.0001, .02), 'contact_load_N': (.001, 5.),
              'maximum_load_N': (.001, 10.), 'maximum_penetration_m': (.000001, .001)}
    for key, (low, high) in bounds.items():
        value = plan[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError('Finite bounded probe setting required: '+key)
    if plan['contact_load_N'] >= plan['maximum_load_N'] or plan['travel_m']/plan['speed_m_s'] > 5:
        raise ValueError('Contact trigger must precede the load limit, within a bounded duration')
    actuator = plan.get('actuator')
    if actuator is not None:
        limits = {'mass_kg': (.0001, .1), 'stiffness_N_m': (1., 500.),
                  'damping_N_s_m': (.01, 5.), 'force_limit_N': (.01, 2.),
                  'maximum_speed_m_s': (.001, .05)}
        if (not isinstance(actuator, dict) or set(actuator) != {'mode', *limits}
                or actuator['mode'] != 'bounded_force'):
            raise ValueError('Explicit bounded dynamic sphere actuator required')
        for key, (low, high) in limits.items():
            value = actuator[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError('Finite bounded dynamic sphere setting required: '+key)
        if actuator['force_limit_N'] >= plan['maximum_load_N']:
            raise ValueError('Commanded finite force must stay below the actual contact guard')
    return json.loads(json.dumps(plan))


def author(stage, path, radius, start, *, actuator=None):
    """Author a separately identified spherical fixture before reset."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from calibrated_scene_reconstruction.sim.usd import set_pose
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(.4, .7, .8)])
    prim = sphere.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateKinematicEnabledAttr(actuator is None)
    if actuator is not None:
        mass = float(actuator['mass_kg']); api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(mass)
        api.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(.4*mass*radius*radius))
        # Disclosed fixture-only gravity compensation: it remains at its free
        # starting point during the independent passive target assay. Target
        # gravity and every source physical parameter are untouched.
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    shape = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    shape.CreateContactOffsetAttr(.0001); shape.CreateRestOffsetAttr(0.)
    set_pose(prim, start)
    return str(prim.GetPath())


def run(ctx, tool_path, names, plan, output):
    """Advance one tool, stop loading on contact force, withdraw and record all states."""
    from isaacsim.core.prims import RigidPrim, SingleRigidPrim
    from pxr import UsdGeom, UsdPhysics
    from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
    from calibrated_scene_reconstruction.tasks.rigid.contact_pairs import depths
    from calibrated_scene_reconstruction.sim.rigid_wrench import apply as apply_wrench

    plan = validate(plan, names); output = Path(output).resolve()
    if not output.is_relative_to((ctx.s.dir/'qa').resolve()):
        raise ValueError('Probe records must stay in session QA')
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve(); (output/'source.py').write_bytes(source.read_bytes())
    tool_prim = ctx.stage.GetPrimAtPath(tool_path)
    dynamic = plan.get('actuator')
    if bool(UsdPhysics.RigidBodyAPI(tool_prim).GetKinematicEnabledAttr().Get()) != (dynamic is None):
        raise ValueError('Tool runtime mode differs from the declared actuator')
    bodies = [ctx.body(name) for name in names]; paths = [body.prim_path for body in bodies]
    if tool_path in paths or any(UsdPhysics.RigidBodyAPI(ctx.stage.GetPrimAtPath(p)).GetKinematicEnabledAttr().Get() for p in paths):
        raise ValueError('Test bodies must remain distinct dynamic rigid bodies')
    tool = SingleRigidPrim(tool_path, name='external_sphere_probe'); tool.initialize()
    native_tool = tool._rigid_prim_view._physics_view
    if native_tool.count != 1:
        raise ValueError('Kinematic target view must contain only the external tool')
    tool_mass = float(as_numpy(tool._rigid_prim_view.get_masses()).sum())
    if dynamic and not np.isclose(tool_mass, dynamic['mass_kg'], rtol=1e-5):
        raise ValueError('Native fixture mass differs from its finite energy hypothesis')
    support = [str(p.GetPath()) for p in ctx.stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
               and not str(p.GetPath()).startswith('/World/Objects/') and str(p.GetPath()) != tool_path]
    filters = paths+support+[tool_path]
    view = RigidPrim(paths, contact_filter_prim_paths_expr=[filters for _ in paths],
                     max_contact_count=32768, reset_xform_properties=False,
                     name='external_contact_'+output.name)
    view.initialize()
    if list(view.prim_paths) != paths:
        raise ValueError('Native tested-body order changed')
    target = names.index(plan['link']); dt = float(ctx.world.get_physics_dt())
    if not 0 < dt <= 1/120:
        raise ValueError('Resolved bounded timestep required')
    start = np.asarray(plan['start_m']); direction = np.asarray(plan['direction'])
    actual_start = as_numpy(tool.get_world_pose()[0])
    if np.linalg.norm(actual_start-start) > 1e-5:
        raise ValueError('External probe did not initialize at the declared free pose')
    steps = int(np.ceil(plan['travel_m']/plan['speed_m_s']/dt))
    poses = []; tools = []; commanded = []; forces = []; contacts = []; peaks = []; velocities = []; commanded_forces = []
    advance = 0.; returning = False; hold_steps = round(.1/dt); resting = 0; failures = []
    source_shapes = {str(q.GetPath()): np.asarray(UsdGeom.Mesh(q).GetPointsAttr().Get()).copy()
                     for p in paths for q in ctx.stage.Traverse()
                     if q.IsA(UsdGeom.Mesh) and str(q.GetPath()).startswith(p+'/')}
    masses = {name: float(ctx.rigid_mass_properties(name)[0]) for name in names}
    stage_file = output/'initial.usda'; ctx.stage.GetRootLayer().Export(str(stage_file))
    try:
        for step in range(2*steps+hold_steps+4):
            if returning:
                advance = max(0., advance-plan['speed_m_s']*dt)
            else:
                advance = min(plan['travel_m'], advance+plan['speed_m_s']*dt)
            command = start+advance*direction
            # Only this external fixture is commanded, never any tested body.
            # SingleRigidPrim.set_world_pose calls set_transforms (teleport),
            # which does not advance the native kinematic target. Use the
            # dedicated fixture-only target API and native xyzw convention.
            external_force = np.zeros(3)
            if dynamic:
                position = as_numpy(tool.get_world_pose()[0]); velocity = as_numpy(tool.get_linear_velocity())
                external_force = (dynamic['stiffness_N_m']*(command-position)
                                  - dynamic['damping_N_s_m']*velocity)
                external_force *= min(1., dynamic['force_limit_N']/max(np.linalg.norm(external_force), 1e-20))
                apply_wrench(tool._rigid_prim_view, [external_force],
                             positions=[position], is_global=True)
            else:
                kinematic_target(tool._rigid_prim_view, command)
            commanded_forces.append(external_force.copy())
            ctx.world.step(render=False)
            p, q = (as_numpy(value) for value in view.get_world_poses())
            poses.append(np.c_[p, q]); tools.append(as_numpy(tool.get_world_pose()[0])); commanded.append(command.copy())
            velocities.append(as_numpy(tool.get_linear_velocity()))
            data = view.get_contact_force_data(dt=dt)
            shape = (len(paths), len(filters)); counts = as_numpy(data[4]).reshape(shape)
            values = depths(as_numpy(data[3]), counts, as_numpy(data[5]).reshape(shape), paths, filters)
            peak = max(values.values(), default=0.); peaks.append(peak)
            matrix = as_numpy(view.get_contact_force_matrix(dt=dt)).reshape(*shape, 3)
            force = matrix[:, -1]; forces.append(force); contacts.append(counts[:, -1].copy())
            load = float(np.linalg.norm(force, axis=1).sum())
            if peak > plan['maximum_penetration_m']:
                failures.append('Actual native contact exceeded the physical penetration limit'); break
            if load > plan['maximum_load_N']:
                failures.append('Actual tool load exceeded the finite load budget'); break
            if not returning and (np.linalg.norm(force[target]) >= plan['contact_load_N'] or advance >= plan['travel_m']):
                returning = True
            if returning and advance == 0:
                resting += 1
                if resting >= hold_steps: break
        if not returning or resting < hold_steps:
            failures.append('Tool withdrawal and unforced tail were incomplete')
    finally:
        np.savez_compressed(output/'trace.npz', poses=poses, names=names, tool_positions=tools,
                            tool_commands=commanded, tool_velocities_m_s=velocities,
                            commanded_tool_forces_N=commanded_forces,
                            tool_forces_N=forces, tool_contact_counts=contacts,
                            maximum_penetration_m=peaks, times_s=dt*np.arange(1, len(poses)+1))
    forces = np.asarray(forces); contacts = np.asarray(contacts)
    target_peak = float(np.linalg.norm(forces[:, target], axis=1).max())
    contact_steps = int(np.count_nonzero(contacts[:, target]))
    if target_peak < plan['contact_load_N'] or contact_steps < 2:
        failures.append('The actual target did not receive a resolved contact load')
    tracking = float(np.linalg.norm(np.asarray(tools)-commanded, axis=1).max())
    if not dynamic and tracking > 1e-5:
        failures.append('Native tool trajectory differs from the declared command')
    maximum_speed = float(np.linalg.norm(velocities, axis=1).max())
    speed_limit = dynamic['maximum_speed_m_s'] if dynamic else plan['speed_m_s']*1.05+1e-5
    if maximum_speed > speed_limit:
        failures.append('Native tool speed exceeded the declared bounded trajectory')
    shape_unchanged = True
    for path, points in source_shapes.items():
        shape_unchanged &= np.array_equal(points, np.asarray(UsdGeom.Mesh(ctx.stage.GetPrimAtPath(path)).GetPointsAttr().Get()))
    if not shape_unchanged: failures.append('Rigid surface coordinates changed during the probe')
    report = {'pass': not failures, 'task_approved': False, 'plan': plan, 'failures': failures,
              'task_actuation': False, 'test_body_pose_commands': False,
              'native_tool_tracking_error_m': tracking, 'target_peak_load_N': target_peak,
              'native_tool_maximum_speed_m_s': maximum_speed,
              'native_tool_command': ('bounded spring/damping force on dynamic sphere only' if dynamic
                                      else 'set_kinematic_targets; fixture-only tensor view'),
              'maximum_commanded_tool_force_N': float(np.linalg.norm(commanded_forces, axis=1).max()),
              'native_tool_mass_kg': tool_mass if dynamic else None,
              'fixture_only_gravity_compensation': bool(dynamic),
              'target_contact_steps': contact_steps, 'maximum_native_penetration_m': max(peaks),
              'rigid_surface_unchanged': bool(shape_unchanged), 'native_mass_kg': masses,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'scope': 'Native contact and rigid approximation under the recorded finite external load. '
                       'Real material stiffness, appearance and source state remain separate.'}
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
