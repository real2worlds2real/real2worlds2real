"""Optional forces evaluated by Newton inside solver substeps."""

import warp as wp


@wp.kernel
def still_air_drag(
    velocity: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    rate: float,
    force: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    if inverse_mass[index] > 0.0:
        force[index] = force[index] - velocity[index] * (rate / inverse_mass[index])


@wp.kernel
def rigid_still_air_drag(
    velocity: wp.array(dtype=wp.spatial_vector),
    mass: wp.array(dtype=float),
    rate: float,
    force: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    force[index] = force[index] + wp.spatial_vector(
        wp.vec3(0.0), -rate * mass[index] * wp.spatial_bottom(velocity[index])
    )


@wp.kernel
def particle_air_drag_rates(
    velocity: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    rates: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    if inverse_mass[index] > 0.0:
        force[index] = force[index] - velocity[index] * (rates[index] / inverse_mass[index])


@wp.kernel
def body_air_drag_rates(
    velocity: wp.array(dtype=wp.spatial_vector),
    mass: wp.array(dtype=float),
    rates: wp.array(dtype=float),
    force: wp.array(dtype=wp.spatial_vector),
):
    index = wp.tid()
    force[index] = force[index] + wp.spatial_vector(
        wp.vec3(0.0), -rates[index] * mass[index] * wp.spatial_bottom(velocity[index])
    )
