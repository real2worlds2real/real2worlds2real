"""Material coordinates and an initially presented fold; no physics constraints."""
import numpy as np


def soft_strip_rest(grip_location='center', grip_height=.0175):
    if grip_location == 'edge':
        lengths = np.r_[[.002, .008, .0055, .0065], np.full(12, .068 / 12)]
        boundary = np.r_[0., lengths.cumsum()]
        knots = np.array([[0., 0., .0235 - s] if s <= .022 + 1e-9 else
                          [0., -(s - .022), .0015] for s in boundary])
        knots[:, 2] -= grip_height
        thickness = np.r_[np.full(4, .003), np.full(12, .0007)]
        grip_panel = 1
    elif grip_location == 'center':
        # A folded 14 x 5 cm wipe has two 6.6 cm free tails around an 8 mm
        # middle grip patch. Three short segments resolve the upper U bend.
        lengths = np.r_[np.full(7, .008), [.004, .003, .003], .008, np.full(11, .006)]
        grip_panel = 10
        thickness = np.full(len(lengths), .0007)
        thickness[grip_panel] = .003
        floor = .0015 - grip_height
        backward = [np.array([0., 0., .004])]
        for length, angle in zip([.003, .003, .004], np.deg2rad([30., 90., 150.])):
            backward.append(backward[-1] + length * np.array([0., -np.sin(angle), np.cos(angle)]))
        for length in reversed(lengths[:7]):
            down = np.clip(backward[-1][2] - floor, 0., length)
            backward.append(backward[-1] + [0., -np.sqrt(max(0., length**2 - down**2)), -down])
        knots = list(reversed(backward)) + [np.array([0., 0., -.004])]
        for length in lengths[grip_panel + 1:]:
            down = np.clip(knots[-1][2] - floor, 0., length)
            knots.append(knots[-1] + [0., np.sqrt(max(0., length**2 - down**2)), -down])
        knots = np.asarray(knots)
    else:
        raise ValueError('Grip location must be center or edge')
    boundary = np.r_[0., lengths.cumsum()]
    z_axes = (knots[:-1] - knots[1:]) / lengths[:, None]
    x_axes = np.tile([1., 0., 0.], (len(lengths), 1))
    rotations = np.stack([x_axes, np.cross(z_axes, x_axes), z_axes], axis=-1)
    material_grip_s = boundary[grip_panel] + lengths[grip_panel] / 2
    return dict(lengths=lengths, thickness=thickness, knots=knots, rotations=rotations,
                grip_panel=grip_panel, boundary=boundary, material_grip_s=material_grip_s)
