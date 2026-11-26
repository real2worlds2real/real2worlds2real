"""Closed finite helical surfaces for agent-authored mating rigid parts.

No joint or motion is created. Parameters are geometric hypotheses, to be
estimated from images and checked for contact feasibility in the simulator.
"""
import numpy as np


def helical_profile(theta, z, *, pitch, depth, starts=1, handedness=1, phase=0.,
                    z_min, z_max, flank_fraction=.30):
    """Trapezoidal radial ridge with finite, smoothly tapered thread ends."""
    if (not np.isfinite([pitch, depth, phase, z_min, z_max, flank_fraction]).all()
            or pitch <= 0 or depth <= 0 or z_max <= z_min or type(starts) is not int
            or starts < 1 or handedness not in (-1, 1) or not 0 < flank_fraction < .5):
        raise ValueError('Invalid finite thread geometry')
    t = (np.asarray(z) / pitch - handedness * starts * np.asarray(theta) / (2*np.pi) + phase) % 1.
    distance = np.abs(t - .5)
    ridge = np.clip((.5-distance)/flank_fraction, 0, 1)
    # Both the lead-in and the runout disappear into the underlying wall.
    taper = np.clip(np.minimum(np.asarray(z)-z_min, z_max-np.asarray(z)) / (pitch*.5), 0, 1)
    return depth * ridge * taper


def shell(*, outer_radius, inner_radius, height, floor, segments=192, thread=None,
          thread_side='outer', z_step=None, outer_profile=None, inner_profile=None):
    """An open-top vessel with a closed bottom; invert it for a screw cap.

    Internal mating surfaces are grooves (the same helix radial field plus
    clearance), rather than an overlapping duplicate external ridge.
    """
    if not thread:
        from calibrated_scene_reconstruction.assets.shell_geometry import shell as plain_shell
        return plain_shell(outer_radius=outer_radius, inner_radius=inner_radius, height=height,
                           floor=floor, segments=segments, surface_side=thread_side, z_step=z_step,
                           outer_profile=outer_profile, inner_profile=inner_profile)
    if (not np.isfinite([outer_radius, inner_radius, height, floor]).all()
            or not 0 < inner_radius < outer_radius or not 0 < floor < height
            or type(segments) is not int or segments < 32 or thread_side not in {'outer', 'inner'}):
        raise ValueError('Invalid shell dimensions or thread side')
    if thread:
        helical_profile(0., 0., **thread)
        if thread['z_min'] < 0 or thread['z_max'] > height:
            raise ValueError('Finite thread range must lie inside the shell height')
        if thread_side == 'inner' and inner_radius + thread['depth'] >= outer_radius:
            raise ValueError('Female thread cuts through the outer wall')
        step = min(thread['pitch']/16, z_step or thread['pitch']/16)
    else:
        step = z_step or height/24
    theta = np.arange(segments)*2*np.pi/segments
    profiles = {'outer': outer_profile, 'inner': inner_profile}
    for profile in profiles.values():
        if profile is not None:
            values = np.asarray(profile, float)
            if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 2
                    or not np.isfinite(values).all() or (np.diff(values[:, 0]) <= 0).any()
                    or (values[:, 1] <= 0).any() or values[0, 0] > 0 or values[-1, 0] < height):
                raise ValueError('Radius profiles require increasing z/radius pairs covering the shell')
    rings = []
    # Outer surface ascending, mouth annulus, inner surface descending, floor.
    for radius, lower, side in ((outer_radius, 0., 'outer'), (inner_radius, floor, 'inner')):
        if thread:
            fine = (np.arange(max(lower, thread['z_min']-thread['pitch']),
                              min(height, thread['z_max']+thread['pitch']), step)
                    if side == thread_side else [])
            knots = np.asarray(profiles[side])[:, 0] if profiles[side] is not None else []
            levels = np.unique(np.clip(np.r_[np.linspace(lower, height, 25), fine, knots], lower, height))
            # Grid samples and profile knots can differ by floating roundoff.
            # Such duplicate rings form zero-area triangles after Blender or
            # the metric validator welds vertices. Merge only nanometre-scale
            # duplicates, retaining the exact mouth and floor boundaries.
            levels = levels[np.r_[True, np.diff(levels) > 1e-8]]
            levels[0], levels[-1] = lower, height
        else:
            levels = np.linspace(lower, height, max(2, int(np.ceil((height-lower)/step))+1))
        if side == 'inner':
            levels = levels[::-1]
        for z in levels:
            profile = profiles[side]
            radius_at_z = radius if profile is None else np.interp(z, *np.asarray(profile).T)
            r = np.full(segments, radius_at_z, dtype=float)
            if thread and side == thread_side:
                r += helical_profile(theta, z, **thread)
                if side == 'inner':
                    # A female lead-in/runout must clear the male crests. A
                    # ridge tapered to the *root* radius instead closes the
                    # entrance into an unthreaded interference ring.
                    taper = np.clip(min(z-thread['z_min'], thread['z_max']-z)/(thread['pitch']*.5), 0, 1)
                    r += thread['depth']*(1-taper)
            rings.append(np.c_[r*np.cos(theta), r*np.sin(theta), np.full(segments, z)])
    if len(rings)*segments > 1_000_000:
        raise ValueError('Thread tessellation budget exceeded')
    vertices = np.concatenate(rings)
    triangles = []
    for i in range(len(rings)-1):
        a = i*segments + np.arange(segments)
        b = i*segments + (np.arange(segments)+1) % segments
        triangles.extend(np.c_[a, b, b+segments].tolist())
        triangles.extend(np.c_[a, b+segments, a+segments].tolist())
    bottom, inside = len(vertices), len(vertices)+1
    vertices = np.vstack([vertices, [0, 0, 0], [0, 0, floor]])
    for j in range(segments):
        k = (j+1) % segments
        triangles.append([bottom, k, j])
        a = (len(rings)-1)*segments
        triangles.append([inside, a+j, a+k])
    return vertices, np.asarray(triangles, np.int32)
