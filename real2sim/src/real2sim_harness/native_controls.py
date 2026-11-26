"""Shared native review controls; installed only by explicit test protocols."""
import warp as wp


@wp.kernel
def retain_contact_capacity_maxima(
    rigid: wp.array[wp.int32],
    soft: wp.array[wp.int32],
    maxima: wp.array[wp.int32],
):
    maxima[0] = wp.max(maxima[0], rigid[0])
    maxima[1] = wp.max(maxima[1], soft[0])


@wp.kernel
def track_contact_loss(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    first: int,
    second: int,
    dt: float,
    dwell: float,
    time: float,
    state: wp.array[wp.int32],
    timer: wp.array[float],
    event: wp.array[float],
):
    active = int(0)  # noqa: UP018 - Warp requires a mutable loop-local value.
    for i in range(wp.min(count[0], shape0.shape[0])):
        a, b = shape0[i], shape1[i]
        if (a == first and b == second) or (a == second and b == first):
            active = 1
    if active:
        state[0] = 1
        timer[0] = 0.0
    elif state[0] > 0 and state[1] == 0:
        timer[0] = timer[0] + dt
        if timer[0] >= dwell:
            state[1] = 1
            event[0] = time



@wp.kernel
def apply_wrench(
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    f: wp.array[wp.spatial_vector],
    idx: int,
    rotation_state: wp.array[wp.int32],
    target: wp.vec3,
    mask: wp.vec3,
    velocity_target: wp.vec3,
    force: wp.vec3,
    speed: wp.vec3,
    torque: wp.vec3,
    kp: float,
    damping: float,
    gain: float,
    align: float,
    axis: wp.vec3,
    flimit: wp.vec3,
    tlimit: wp.vec3,
):
    pos = wp.transform_get_translation(q[idx])
    vel = wp.spatial_top(qd[idx])
    omega = wp.spatial_bottom(qd[idx])
    current_axis = wp.quat_rotate(wp.transform_get_rotation(q[idx]), wp.vec3(0.0, 0.0, 1.0))
    ff = force + wp.cw_mul(mask, kp * (target - pos)) + damping * (velocity_target - vel)
    if rotation_state[1] > 0:
        tt = -gain * omega + align * wp.cross(current_axis, axis)
    else:
        tt = torque + gain * (speed - omega) + align * wp.cross(current_axis, axis)
    for j in range(3):
        ff[j] = wp.clamp(ff[j], -flimit[j], flimit[j])
        tt[j] = wp.clamp(tt[j], -tlimit[j], tlimit[j])
    f[idx] = wp.spatial_vector(ff, tt)



@wp.kernel
def move_probe(
    body: int,
    target: wp.vec3,
    dt: float,
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
):
    velocity = (target - wp.transform_get_translation(q0[body])) / dt
    pose = wp.transform(target, wp.transform_get_rotation(q0[body]))
    q0[body] = pose
    q1[body] = pose
    v0[body] = wp.spatial_vector(velocity, wp.vec3(0.0))
    v1[body] = wp.spatial_vector(velocity, wp.vec3(0.0))



@wp.kernel
def move_grasp(
    ids: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    offset: wp.vec3,
    dt: float,
    released: bool,
    flags: wp.array[wp.int32],
    q0: wp.array[wp.vec3],
    q1: wp.array[wp.vec3],
    v0: wp.array[wp.vec3],
    v1: wp.array[wp.vec3],
):
    i = ids[wp.tid()]
    if released:
        flags[i] = flags[i] | 1
    else:
        flags[i] = flags[i] & ~1
        target = rest[i] + offset
        velocity = (target - q0[i]) / dt
        q0[i] = target
        q1[i] = target
        v0[i] = velocity
        v1[i] = velocity
