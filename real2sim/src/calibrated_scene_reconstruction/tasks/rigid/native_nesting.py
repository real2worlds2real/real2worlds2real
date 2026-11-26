"""Object-only native nesting with disclosed finite-wrench test fixtures.

The receiving cup is clamped. During insertion/withdrawal a bounded lateral
and tilt guide represents a test tool; there is no axial joint, commanded
vertical pose, or velocity reset. All guide forces are removed for retention.
"""
import json
import numpy as np


def run(ctx, moving_name, receiver_name, *, moving_height_m, receiver_height_m,
        penetration_tolerance_m=.00025, withdrawal_pull_N=.15, axial_force_limit_N=.8,
        withdrawal_twist_Nm=0., insertion_terminal_speed_m_s=None,
        insertion_duration_s=3., release_settling_s=0., output=None):
    from pxr import Gf, UsdPhysics
    from isaacsim.core.prims import RigidPrim
    from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy, quat
    from calibrated_scene_reconstruction.tasks.mechanisms.contact import contact_separations
    from calibrated_scene_reconstruction.tasks.mechanisms.bounded_control import stable_guide_damping
    from calibrated_scene_reconstruction.sim.rigid_wrench import apply as apply_wrench

    if (not np.isfinite([moving_height_m, receiver_height_m, penetration_tolerance_m]).all()
            or min(moving_height_m, receiver_height_m, penetration_tolerance_m)<=0):
        raise ValueError('Positive candidate geometry and penetration budget required')
    if (not np.isfinite([withdrawal_pull_N, axial_force_limit_N]).all()
            or not 0<withdrawal_pull_N<axial_force_limit_N<=10):
        raise ValueError('Explicit finite bounded withdrawal fixture force required')
    if not np.isfinite(withdrawal_twist_Nm) or not abs(withdrawal_twist_Nm)<=.02:
        raise ValueError('Withdrawal twist must be an explicit bounded finite torque')
    if (insertion_terminal_speed_m_s is not None and
            (not np.isfinite(insertion_terminal_speed_m_s) or not .005<=insertion_terminal_speed_m_s<=.3)):
        raise ValueError('Optional insertion terminal speed must be in0.005..0.3m/s')
    if not np.isfinite(insertion_duration_s) or not 1<=insertion_duration_s<=30:
        raise ValueError('Bounded1..30s insertion fixture duration required')
    if not np.isfinite(release_settling_s) or not 0<=release_settling_s<=5:
        raise ValueError('Optional unforced release settling must be bounded within0..5s')
    hz = ctx.ts.inputs.simulation['physics_hz']
    dt = float(ctx.world.get_physics_dt())
    if not np.isclose(dt, 1/hz):
        raise ValueError('Native timestep must match the experiment profile')
    folder = ctx.s/(output or (ctx.trial_output+'/nesting'))
    folder.mkdir(parents=True, exist_ok=False)
    moving, receiver = ctx.body(moving_name), ctx.body(receiver_name)
    bp, R = ctx.pose(receiver_name)
    axis = R[:, 2]
    fixture = UsdPhysics.FixedJoint.Define(ctx.stage, '/World/TaskCheckFixtures/nesting_base')
    fixture.CreateBody1Rel().SetTargets([receiver.prim_path])
    fixture.CreateLocalPos0Attr(Gf.Vec3f(*map(float, bp)))
    q = quat(R)
    fixture.CreateLocalRot0Attr(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
    fixture.CreateLocalPos1Attr(Gf.Vec3f(0))
    fixture.CreateLocalRot1Attr(Gf.Quatf(1))
    contacts = RigidPrim(moving.prim_path, contact_filter_prim_paths_expr=[receiver.prim_path],
                         max_contact_count=8192, reset_xform_properties=False, name='nesting_contacts')
    contacts.initialize()
    moving.set_sleep_threshold(0.)
    # All sensor views exist before the single disclosed assay initial placement.
    ctx.place(moving_name, bp+axis*(receiver_height_m+.012), R)
    mass = float(as_numpy(moving._rigid_prim_view.get_masses()).ravel()[0])
    inertia = as_numpy(moving._rigid_prim_view.get_inertias()).reshape(-1, 3, 3)[0]
    minimum_inertia = float(np.linalg.eigvalsh((inertia+inertia.T)/2).min())
    if mass<=0 or minimum_inertia<=0:
        raise ValueError('Actual positive mass and source-derived inertia required')
    insertion_drag=(.3 if insertion_terminal_speed_m_s is None
                    else mass*9.81/insertion_terminal_speed_m_s)
    if insertion_drag*dt/mass>=1.:
        raise ValueError('Insertion drag is unresolved at this timestep; refine the numerical profile')
    lateral_k, tilt_k = 30., .02
    lateral_d = stable_guide_damping(mass, dt, lateral_k, .7)
    tilt_d = stable_guide_damping(minimum_inertia, dt, tilt_k, .001)
    rows, phases, time, peak = [], {}, 0., 0.

    def advance(phase, seconds, *, passive=False, lift=False):
        nonlocal time, peak
        start = len(rows)
        for step in range(round(seconds*hz)):
            p, current = ctx.pose(moving_name)
            velocity = as_numpy(moving.get_linear_velocity())
            angular = as_numpy(moving.get_angular_velocity())
            delta = p-bp
            lateral = delta-axis*(delta@axis)
            lateral_v = velocity-axis*(velocity@axis)
            force = -lateral_k*lateral-lateral_d*lateral_v
            force *= min(1., .5/max(float(np.linalg.norm(force)), 1e-12))
            # Finite upward pull only during withdrawal. During insertion only
            # physical gravity and tool dissipation provide axial motion.
            # Explicit passive drag prevents an unintended impact-wedging
            # experiment when the intended operation is gentle stacking. This
            # is a finite tool wrench, not a position/velocity assignment. Its
            # numerical coefficient and duration are recorded with the result.
            axial = (mass*9.81+withdrawal_pull_N if lift else 0.)-(.3 if lift else insertion_drag)*float(velocity@axis)
            force += axis*np.clip(axial, -axial_force_limit_N, axial_force_limit_N)
            torque = tilt_k*np.cross(current[:, 2], axis)-tilt_d*(angular-axis*(angular@axis))
            torque *= min(1., .01/max(float(np.linalg.norm(torque)), 1e-12))
            if lift:
                # A disclosed finite twist can test release of a friction-locked
                # taper. Orientation is observed from the solver, never set.
                torque += axis*withdrawal_twist_Nm
            if passive:
                force = np.zeros(3)
                torque = np.zeros(3)
            apply_wrench(moving._rigid_prim_view, [force], [torque], is_global=True)
            ctx.world.step(render=False)
            time += dt
            data = contacts.get_contact_force_data(dt=dt)
            separation = contact_separations(as_numpy(data[3]), as_numpy(data[4]), as_numpy(data[5]))
            if len(separation):
                peak = max(peak, float(max(0., -separation.min())))
            p, current = ctx.pose(moving_name)
            height = float((p-bp)@axis)
            complete = lift and height>receiver_height_m+.015
            if phase == 'release_settle' or step % max(1, hz//60) == 0 or step == round(seconds*hz)-1 or complete:
                row = [time, *p, *quat(current), height,
                       float(np.linalg.norm(as_numpy(contacts.get_contact_force_matrix(dt=dt)))),
                       peak, *force, *torque]
                if not np.isfinite(row).all():
                    raise ValueError('Nonfinite native nesting state')
                rows.append(row)
            if complete:
                break
        phases[phase] = [start, len(rows)]
        np.savez_compressed(folder/'motion.npz', values=np.asarray(rows))
        (folder/'phases.json').write_text(json.dumps(phases, indent=2)+'\n')
        return np.asarray(rows[start:])

    inserted = advance('nest', insertion_duration_s)
    # A task release can settle deeper into its support. Observe that transient
    # separately from the unchanged final stability window. No tool force,
    # state reset or unobserved physics step is allowed; containment and the
    # original all-phase penetration bound still cover the entire transient.
    # This option never qualifies drift of the captured initial scene.
    settling = advance('release_settle', release_settling_s, passive=True) if release_settling_s else None
    held = advance('rest_stability', 1., passive=True)
    released = advance('unnest', 3., lift=True)
    overlap = receiver_height_m-float(np.median(held[:, 8]))
    drift = float(np.max(np.linalg.norm(held[:, 1:4]-held[0, 1:4], axis=1)))
    checks = dict(nest=dict(pass_check=overlap>=.2*moving_height_m, overlap_m=overlap),
                  contact=dict(pass_check=bool(np.max(inserted[:, 9])>1e-4), peak_force_N=float(np.max(inserted[:, 9]))),
                  rest_stability=dict(pass_check=drift<=.001, drift_m=drift, guide_disabled=True),
                  unnest=dict(pass_check=bool(released[-1, 8]>receiver_height_m+.005), final_height_m=float(released[-1, 8])),
                  no_interpenetration=dict(pass_check=peak<=penetration_tolerance_m,
                                           peak_m=peak, tolerance_m=penetration_tolerance_m))
    if settling is not None:
        from calibrated_scene_reconstruction.tasks.rigid.nesting import release_containment
        checks['release_containment'] = release_containment(
            settling[:, 1:4], settling[:, 8], moving_height_m, receiver_height_m)
        checks['rest_stability']['after_unforced_settling_s'] = release_settling_s
    return dict(pass_native=all(c['pass_check'] for c in checks.values()), checks=checks,
                accepted=False, native_states_read=True, prescribed_motion=False,
                fixture=dict(receiver_clamped=True, lateral_force_limit_N=.5, axial_force_limit_N=axial_force_limit_N,
                             withdrawal_pull_N=withdrawal_pull_N,
                             withdrawal_twist_Nm=withdrawal_twist_Nm,
                             insertion_terminal_speed_m_s=insertion_terminal_speed_m_s,
                             insertion_drag_Ns_m=insertion_drag, insertion_duration_s=insertion_duration_s,
                             release_settling_s=release_settling_s,
                             tilt_torque_limit_Nm=.01, force_free_retention_s=1.),
                columns=dict(time_s=0, moving_position=[1, 4], moving_quaternion_wxyz=[4, 8],
                             height_in_receiver=8, contact_force_N=9, peak_penetration_m=10,
                             applied_force_N=[11, 14], applied_torque_Nm=[14, 17]),
                phases=phases, motion=str(folder/'motion.npz'), scope=__doc__)
