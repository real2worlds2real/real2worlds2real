"""Re-frame view-dependent radiance along with exported Gaussian geometry."""
import numpy as np


def rotate_ply_radiance(vertices, rotation, chunk=50000):
    """Rotate channel-major PLY SH using the renderer's own tested SH basis.

    3DGRUT exports coefficients in the trained model's frame. Baking a rotation
    into positions/orientations also requires f_new(d)=f_old(R.T@d). DC, density,
    covariance sizes and every geometry field remain unchanged.
    """
    rotation = np.asarray(rotation, float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)
            or type(chunk) is not int or chunk < 1):
        raise ValueError('Proper rigid rotation and positive chunk size required')
    names = sorted([name for name in vertices.dtype.names if name.startswith('f_rest_')],
                   key=lambda name: int(name.rsplit('_', 1)[1]))
    output = vertices.copy()
    if not names or np.allclose(rotation, np.eye(3), atol=1e-12):
        return output
    if names != [f'f_rest_{i}' for i in range(len(names))] or len(names) % 3:
        raise ValueError('PLY radiance fields must contain contiguous RGB SH bands')
    coefficients = len(names)//3
    degree = int(round(np.sqrt(coefficients+1)-1))
    if degree not in (1, 2, 3) or (degree+1)**2-1 != coefficients:
        raise ValueError('Expected complete SH bands up to degree3')
    import torch
    from threedgrut.export.sh_rotation import rotate_specular

    R = torch.as_tensor(rotation, dtype=torch.float64)
    for start in range(0, len(vertices), chunk):
        stop = min(start+chunk, len(vertices))
        raw = np.column_stack([vertices[name][start:stop] for name in names])
        features = raw.reshape(-1, 3, coefficients).transpose(0, 2, 1).reshape(-1, coefficients*3)
        transformed = rotate_specular(torch.from_numpy(features.copy()), R, degree).numpy()
        packed = transformed.reshape(-1, coefficients, 3).transpose(0, 2, 1).reshape(-1, len(names))
        for i, name in enumerate(names):
            output[name][start:stop] = packed[:, i]
    return output
