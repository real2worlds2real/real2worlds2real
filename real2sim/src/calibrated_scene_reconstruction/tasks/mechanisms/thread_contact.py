"""Bounded-wrench thread feasibility on unchanged reconstructed surfaces.

The base clamp and lateral/tilt guide are explicit test fixtures. No axial
joint, axial pose control, velocity reset, or z(angle) relation is applied.
"""
import json

import numpy as np
from pxr import Gf, PhysxSchema, UsdPhysics
from isaacsim.core.prims import RigidPrim

from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy, quat
from calibrated_scene_reconstruction.sim.rigid_wrench import apply as apply_wrench
from calibrated_scene_reconstruction.tasks.mechanisms.thread import motion, engagement_window, seating_stall, torque_limited_stall
from calibrated_scene_reconstruction.tasks.mechanisms.bounded_control import axial_wrench, stable_guide_damping, speed_controller_gains
from calibrated_scene_reconstruction.tasks.mechanisms.contact import contact_separations


def run(ctx, base_name, cap_name, *, lead_m_per_turn, pitch_m, thread_bands, entry_z, seated_z,
        tolerance_m, output=None, speed_rad_s=.8, max_torque_Nm=.03,
        preload_N=.25, pull_N=1., physics_hz=960, max_duration_s=60.,
        base_axis_xy=(0., 0.), cap_axis_xy=(0., 0.), max_unscrew_torque_Nm=None,
        tilt_stiffness_Nm_rad=.5, tilt_damping_Nms_rad=.005, max_tilt_torque_Nm=.02,
        seating_angle_tolerance_rad=.01, penetration_tolerance_m=None,
        gravity_compensation=False, axial_damping_Ns_m=0., max_axial_force_N=2., unscrew_pull_N=None,
        torque_stall_window_s=None, lateral_stiffness_N_m=1000., lateral_damping_Ns_m=2.,
        max_lateral_force_N=2.):
    """Offsets are in the base body frame, derived from this model's geometry.

    Cap geometry must have its opening downward along local -Z. Positive lead
    means negative rotation advances it downward. Nothing here measures or
    supplies true task-object dimensions.
    """
    if (lead_m_per_turn <= 0 or entry_z <= seated_z or tolerance_m <= 0
            or speed_rad_s <= 0 or max_torque_Nm <= 0 or pull_N <= 0):
        raise ValueError('Positive geometric thread/force budgets required')
    full_lo, full_hi = engagement_window(thread_bands['base'], thread_bands['cap'], pitch_m)
    unscrew_torque = max_torque_Nm if max_unscrew_torque_Nm is None else max_unscrew_torque_Nm
    unscrew_pull = pull_N if unscrew_pull_N is None else unscrew_pull_N
    if not np.isfinite(unscrew_pull) or unscrew_pull <= 0:
        raise ValueError('Positive bounded unscrewing lift required')
    if torque_stall_window_s is not None and (not np.isfinite(torque_stall_window_s) or torque_stall_window_s<.5):
        raise ValueError('Torque-stall diagnostic window must be at least0.5s')
    if (not np.isfinite([lateral_stiffness_N_m,lateral_damping_Ns_m,max_lateral_force_N]).all()
            or lateral_stiffness_N_m<=0 or lateral_damping_Ns_m<0 or max_lateral_force_N<=0):
        raise ValueError('Finite bounded lateral test-tool gains required')
    penetration_limit = min(tolerance_m, pitch_m/20) if penetration_tolerance_m is None else penetration_tolerance_m
    if (not np.isfinite([unscrew_torque, tilt_stiffness_Nm_rad, tilt_damping_Nms_rad,
                         max_tilt_torque_Nm, seating_angle_tolerance_rad, penetration_limit]).all()
            or min(unscrew_torque, tilt_stiffness_Nm_rad, max_tilt_torque_Nm,
                   seating_angle_tolerance_rad, penetration_limit)<=0 or tilt_damping_Nms_rad<0):
        raise ValueError('Finite bounded wrench, seat and penetration budgets required')
    output = output or ((getattr(ctx, 'trial_output', None) or 'qa')+'/thread_contact')
    folder = ctx.s/output
    folder.mkdir(parents=True, exist_ok=False)
    profile = getattr(getattr(getattr(ctx, 'ts', None), 'inputs', None), 'simulation', None)
    if profile is not None and physics_hz != profile['physics_hz']:
        raise ValueError('Task timestep must match the delivered simulation profile')
    ctx.world.set_simulation_dt(physics_dt=1/physics_hz, rendering_dt=1/30)
    dt = float(ctx.world.get_physics_dt())
    for prim in ctx.stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxSceneAPI.Apply(prim).CreateTimeStepsPerSecondAttr(physics_hz)
    base, cap = ctx.body(base_name), ctx.body(cap_name)
    bp, R = ctx.pose(base_name)
    axis = R[:, 2]
    base_axis_local, cap_axis_local = np.r_[base_axis_xy, 0.], np.r_[cap_axis_xy, 0.]
    if not np.isfinite(np.r_[base_axis_local, cap_axis_local]).all() or len(base_axis_local) != 3 or len(cap_axis_local) != 3:
        raise ValueError('Finite link-local thread axis offsets required')
    base_axis_point = bp+R@base_axis_local
    base_path, cap_path = base.prim_path, cap.prim_path
    fixture_path = '/World/TaskCheckFixtures/thread_base'
    fixture = UsdPhysics.FixedJoint.Define(ctx.stage, fixture_path)
    fixture.CreateBody1Rel().SetTargets([base_path])
    fixture.CreateLocalPos0Attr(Gf.Vec3f(*map(float, bp)))
    q = quat(R)
    fixture.CreateLocalRot0Attr(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
    fixture.CreateLocalPos1Attr(Gf.Vec3f(0))
    fixture.CreateLocalRot1Attr(Gf.Quatf(1))
    cap.set_sleep_threshold(0.)
    contacts = RigidPrim(cap_path, contact_filter_prim_paths_expr=[base_path], max_contact_count=8192,
                         reset_xform_properties=False, name='thread_contacts')
    contacts.initialize()
    # Creating a native view can restore its authored initial transform. Make
    # all sensor views first, then perform the single disclosed initial setup.
    # Never reset a pose once the contact experiment has begun.
    initial_cap = base_axis_point-R@cap_axis_local+axis*entry_z
    ctx.place(cap_name, initial_cap, R)
    actual_cap, _ = ctx.pose(cap_name)
    if np.linalg.norm(actual_cap-initial_cap) > 1e-5:
        raise ValueError('Native cap setup pose differs from the declared contact entry')
    inertia = as_numpy(cap._rigid_prim_view.get_inertias()).reshape(-1, 3, 3)[0]
    I = float(inertia[2, 2])
    if not np.isfinite(I) or I <= 0:
        raise ValueError('Positive actual cap inertia required')
    requested_tilt_damping = tilt_damping_Nms_rad
    minimum_inertia = float(np.linalg.eigvalsh((inertia+inertia.T)*.5).min())
    tilt_damping_Nms_rad = stable_guide_damping(minimum_inertia, dt,
                                               tilt_stiffness_Nm_rad, requested_tilt_damping)
    if type(gravity_compensation) is not bool:
        raise ValueError('Gravity compensation must be an explicit fixture choice')
    mass = float(as_numpy(cap._rigid_prim_view.get_masses()).reshape(-1)[0])
    requested_lateral_damping = lateral_damping_Ns_m
    # The same discrete mass/spring stability bound as the angular guide,
    # using actual mass and linear stiffness/damping for translation.
    lateral_damping_Ns_m = stable_guide_damping(mass,dt,lateral_stiffness_N_m,lateral_damping_Ns_m)
    if not np.isfinite(mass) or mass <= 0 or axial_damping_Ns_m*dt/mass >= 2:
        raise ValueError('Positive actual mass and stable explicit axial damping budget required')
    compensation_N = 0.
    if gravity_compensation:
        gravity_direction, gravity_magnitude = ctx.world.get_physics_context().get_gravity()
        gravity_direction = np.asarray(gravity_direction, float)
        if not np.isfinite(gravity_direction).all() or np.linalg.norm(gravity_direction) == 0:
            raise ValueError('Actual native gravity direction required')
        compensation_N = -mass*float(gravity_magnitude)*float(gravity_direction@axis)/np.linalg.norm(gravity_direction)
    axial_wrench(0., 0., damping_Ns_m=axial_damping_Ns_m,
                 compensation_N=compensation_N, limit_N=max_axial_force_N)
    kp, ki, speed_bandwidth = speed_controller_gains(I, dt)
    rows, phases, time, integral, peak_penetration = [], {}, 0., 0., 0.
    terminations = {}

    def checkpoint(phase):
        tmp = folder/'motion.tmp.npz'
        np.savez_compressed(tmp, values=np.asarray(rows))
        tmp.replace(folder/'motion.npz')
        (folder/'phases.json').write_text(json.dumps(phases, indent=2)+'\n')
        (folder/'progress.json').write_text(json.dumps({'phase': phase, 'physics_time_s': time,
                                                       'complete': False})+'\n')

    def advance(phase, seconds, speed, axial_force, stop=None, minimum_duration_s=.5, stop_hold_s=0., passive=False):
        nonlocal time, integral, peak_penetration
        start, integral, held = len(rows), 0., 0.
        termination = 'duration_limit'
        torque_limit = unscrew_torque if phase == 'screw_off' else max_torque_Nm
        def sample():
            # Record the actual post-step state, including the terminal step.
            # A stop can occur between the regular 60 Hz samples; retaining
            # the previous sample can falsely report a still-engaged cap.
            cp, cq = (as_numpy(v) for v in cap.get_world_pose())
            rp, rq = (as_numpy(v) for v in base.get_world_pose())
            p, _ = ctx.pose(cap_name)
            contact_force = float(np.linalg.norm(as_numpy(contacts.get_contact_force_matrix(dt=dt))))
            row = [time, *rp, *rq, *cp, *cq, contact_force, peak_penetration,
                   *force, *torque, float((R.T@(p-bp))[2])]
            if not np.isfinite(row).all():
                raise ValueError('Nonfinite native thread state')
            if rows and rows[-1][0] == time:
                rows[-1] = row
            else:
                rows.append(row)
        print('[thread] '+phase, flush=True)
        for step in range(int(np.ceil(seconds/dt))):
            p, current_R = ctx.pose(cap_name)
            cp, cq = (as_numpy(v) for v in cap.get_world_pose())
            rp, rq = (as_numpy(v) for v in base.get_world_pose())
            velocity, angular = as_numpy(cap.get_linear_velocity()), as_numpy(cap.get_angular_velocity())
            error = speed-float(angular@axis)
            integral = float(np.clip(integral+error*dt, -torque_limit/ki, torque_limit/ki))
            torque_axis = float(np.clip(kp*error+ki*integral, -torque_limit, torque_limit))
            relative = R.T@(p-bp)
            cap_axis_point = p+current_R@cap_axis_local
            delta = cap_axis_point-base_axis_point
            lateral = delta-axis*float(delta@axis)
            axis_velocity = velocity+np.cross(angular, current_R@cap_axis_local)
            lateral_velocity = axis_velocity-axis*float(axis_velocity@axis)
            force = -lateral_stiffness_N_m*lateral-lateral_damping_Ns_m*lateral_velocity
            length = np.linalg.norm(force)
            force *= min(1., max_lateral_force_N/max(length, 1e-12))
            force += axis*axial_wrench(axial_force, float(axis_velocity@axis),
                                      damping_Ns_m=axial_damping_Ns_m,
                                      compensation_N=compensation_N, limit_N=max_axial_force_N)
            torque = tilt_stiffness_Nm_rad*np.cross(current_R[:, 2], axis)-tilt_damping_Nms_rad*(angular-axis*float(angular@axis))
            torque *= min(1., max_tilt_torque_Nm/max(float(np.linalg.norm(torque)), 1e-12))
            torque += axis*torque_axis
            if passive:
                # No cap gravity compensation, axial/lateral/tilt guide or
                # angular velocity controller during intrinsic retention.
                force = np.zeros(3)
                torque = np.zeros(3)
            apply_wrench(cap._rigid_prim_view, [force], [torque], is_global=True)
            ctx.world.step(render=False)
            time += dt
            data = contacts.get_contact_force_data(dt=dt)
            distances = contact_separations(as_numpy(data[3]), as_numpy(data[4]), as_numpy(data[5]))
            penetration = float(max(0., -distances.min())) if len(distances) else 0.
            peak_penetration = max(peak_penetration, penetration)
            if step % max(1, physics_hz//60) == 0:
                sample()
                if (speed and torque_stall_window_s is not None and
                        torque_limited_stall(rows[start:],torque_limit_Nm=torque_limit,
                                             axis=axis,window_s=torque_stall_window_s)):
                    termination = 'torque_limited_stall'
                    break
            if step and step % (physics_hz*2) == 0:
                checkpoint(phase)
            if step and step % (physics_hz*5) == 0:
                print(f'[thread] {phase}: t={step*dt:.1f}s, z={relative[2]:.6f}', flush=True)
            if stop is not None and step > physics_hz*minimum_duration_s:
                post_p, _ = ctx.pose(cap_name)
                held = held+dt if stop(R.T@(post_p-bp), time) else 0.
                if held > stop_hold_s:
                    termination = 'stop_condition'
                    break
        sample()
        phases[phase] = [start, len(rows)]
        terminations[phase] = termination
        checkpoint(phase)
        print(f'[thread] {phase}: z={rows[-1][-1]:.6f}, duration={len(rows[start:])/60:.2f}s', flush=True)
        return np.asarray(rows[start:])

    seat_margin = max(tolerance_m*.25, .0001)
    seat_cache = [-1, False]
    def seated_and_stalled(relative, _):
        if len(rows)!=seat_cache[0]:
            seat_cache[:] = [len(rows), seating_stall(rows[-32:], seated_z=seated_z, tolerance_m=tolerance_m,
                                                    angle_tolerance_rad=seating_angle_tolerance_rad)]
        return seat_cache[1]
    screw_on = advance('screw_on', max_duration_s, -speed_rad_s, -preload_N,
                       seated_and_stalled)
    # A finite lead-in is not a continuously engaged thread. Select the full
    # overlap interval from source geometry, never from measured lead errors.
    engaged = screw_on[(screw_on[:, -1] > max(full_lo, seated_z+seat_margin))
                       & (screw_on[:, -1] < full_hi)]
    def measure(samples):
        if len(samples) < 3:
            return {'pass': False, 'why': 'Fewer than three poses in the geometry-defined engagement interval'}
        return motion(samples[:, 1:4], samples[:, 4:8], samples[:, 8:11], samples[:, 11:15],
                      lead_m_per_turn=lead_m_per_turn, tolerance_m=tolerance_m, minimum_turns=.5,
                      base_origin=base_axis_local, moving_origin=cap_axis_local)
    measured_on = measure(engaged)
    seated = seating_stall(screw_on[-32:], seated_z=seated_z, tolerance_m=tolerance_m,
                           angle_tolerance_rad=seating_angle_tolerance_rad)
    retained = advance('passive_retention', 1., 0., 0., passive=True)
    retained_z = retained[:, -1]
    retention = bool(seated and np.max(abs(retained_z-retained_z[0])) <= tolerance_m)
    pulled = advance('pull_without_rotation', 2., 0., pull_N)
    pull_retention = bool(seated and np.max(abs(pulled[:, -1]-pulled[0, -1])) <= tolerance_m)
    # Unscrewing includes lifting clear of the finite lead-out. A downward
    # pressing preload, or an upward force below gravity, cannot release a cap.
    # A pull-strength test and normal unthreading need not use the same axial
    # load. With gravity compensation, a gentle lift clears the finite lead-out
    # while avoiding unnecessary flank load. Both remain bounded wrenches.
    screw_off = advance('screw_off', max_duration_s, speed_rad_s, unscrew_pull,
                        lambda relative, _: relative[2] >= entry_z)
    disengaging = screw_off[(screw_off[:, -1] < full_hi)
                            & (screw_off[:, -1] > max(full_lo, seated_z+seat_margin))]
    measured_off = measure(disengaging)
    released = advance('release', 2., 0., pull_N,
                       lambda relative, _: relative[2] > entry_z+2*lead_m_per_turn, minimum_duration_s=0.)
    release = bool(released[-1, -1] > entry_z+lead_m_per_turn and released[-1, 15] < 1e-5)
    contact_observed = bool(np.count_nonzero(screw_on[:, 15] > 1e-5) >= 15)
    values = np.asarray(rows)
    penetration = float(values[:, 16].max())
    checks = {'screw_on': {**measured_on, 'seated': seated, 'pass': measured_on['pass'] and seated},
              'retention': {'pass': retention, 'drift_m': float(np.ptp(retained_z)), 'passive': True,
                            'applied_cap_wrench': False, 'gravity_compensation': False},
              'pull_without_rotation': {'pass': pull_retention, 'pull_N': pull_N,
                                         'drift_m': float(np.ptp(pulled[:, -1]))},
              'screw_off': measured_off, 'release': {'pass': release, 'final_z': float(released[-1, -1])},
              'no_interpenetration': {'pass': penetration <= penetration_limit, 'max_contact_penetration_m': penetration,
                                      'tolerance_m': penetration_limit, 'sampling': 'Every physics step'}}
    np.savez_compressed(folder/'motion.npz', values=values)
    (folder/'phases.json').write_text(json.dumps(phases, indent=2)+'\n')
    (folder/'trace.json').write_text(json.dumps({
        'base': base_name, 'moving': cap_name, 'physics_hz': physics_hz,
        'lead_m_per_turn': lead_m_per_turn, 'pitch_m': pitch_m, 'thread_bands': thread_bands,
        'columns': {'time_s': 0, 'base_p': [1, 4], 'base_q_wxyz': [4, 8], 'moving_p': [8, 11],
                    'moving_q_wxyz': [11, 15], 'normal_force_N': 15, 'penetration_m': 16,
                    'applied_force_N': [17, 20], 'applied_torque_Nm': [20, 23], 'axial_m': 23},
        'prescribed_axial_motion': False,
        'penetration_sampling': 'Column16 is cumulative maximum separation violation, evaluated every physics step',
        'seating': 'Torque-driven tightening to geometry-derived hard seat and sustained native angular stall',
        'phase_terminations': terminations,
    }, indent=2)+'\n')
    ctx.stage.RemovePrim(fixture_path)
    (folder/'progress.json').write_text(json.dumps({'complete': True, 'physics_time_s': time})+'\n')
    return {'pass': all(c['pass'] for c in checks.values()), 'representation': 'surface_contact',
            'prescribed_motion': False, 'contact_observed': contact_observed, 'checks': checks,
            'fixtures': {'base_clamp': True, 'bounded_lateral_tilt_guide': True, 'axial_guide': False,
                         'max_torque_Nm': max_torque_Nm, 'preload_N': preload_N, 'pull_N': pull_N,
                         'max_unscrew_torque_Nm': unscrew_torque,
                         'unscrew_pull_N': unscrew_pull,
                         'speed_controller_bandwidth_rad_s': speed_bandwidth,
                         'max_lateral_force_N': max_lateral_force_N, 'max_tilt_torque_Nm': max_tilt_torque_Nm,
                         'lateral_stiffness_N_m': lateral_stiffness_N_m,
                         'lateral_damping_Ns_m': lateral_damping_Ns_m,
                         'requested_lateral_damping_Ns_m': requested_lateral_damping,
                         'tilt_stiffness_Nm_rad': tilt_stiffness_Nm_rad,
                         'tilt_damping_Nms_rad': tilt_damping_Nms_rad,
                         'requested_tilt_damping_Nms_rad': requested_tilt_damping,
                         'minimum_inertia_kg_m2': minimum_inertia,
                         'seating_angle_tolerance_rad': seating_angle_tolerance_rad,
                         'gravity_compensation': gravity_compensation,
                         'gravity_compensation_N': compensation_N,
                         'axial_damping_Ns_m': axial_damping_Ns_m,
                         'max_axial_force_N': max_axial_force_N,
                         'torque_stall_window_s': torque_stall_window_s,
                         'passive_retention_has_no_cap_wrench': True,
                         'base_axis_xy': list(base_axis_xy), 'cap_axis_xy': list(cap_axis_xy)},
            'geometry_engagement_interval_m': [full_lo, full_hi],
            'evidence_files': [str((folder/n).relative_to(ctx.s.dir)) for n in ('motion.npz', 'phases.json', 'trace.json')]}
