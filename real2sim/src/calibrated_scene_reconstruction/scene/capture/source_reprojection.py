"""Read actual sparse-track projection residuals without changing calibration.

These are diagnostic residuals, not camera-pose uncertainty estimates or a
reason to waive object source gates. Empty local regions remain unobserved.
The caller supplies the reconstruction that actually produced its cameras.
"""
import numpy as np


def _summary(residuals):
    residuals = np.asarray(residuals, float).reshape(-1, 2)
    if not len(residuals):
        return {'samples': 0, 'observed': False, 'rms_px': None,
                'median_px': None, 'p95_px': None, 'mean_vector_px': None}
    norms = np.linalg.norm(residuals, axis=1)
    return {'samples': len(residuals), 'observed': True,
            'rms_px': float(np.sqrt(np.mean(norms**2))),
            'median_px': float(np.median(norms)),
            'p95_px': float(np.quantile(norms, .95)),
            'mean_vector_px': residuals.mean(0).tolist()}


def measure(reconstruction, regions):
    """Per-image residuals and optional source-image pixel ROI residuals.

    ``regions`` maps exact registered image names to ``None`` or [x,y,w,h].
    Each matched 3D track is projected with that image's actual camera model,
    including distortion if the supplied reconstruction has it. A missing
    image or invalid region is an error, never an implicit zero-error view.
    """
    images = {image.name: image for image in reconstruction.images.values()}
    if not regions or set(regions)-set(images):
        raise ValueError('Explicit registered source image names are required')
    result = []
    for name, region in regions.items():
        box = None if region is None else np.asarray(region, float)
        if box is not None and (box.shape != (4,) or not np.isfinite(box).all()
                                or (box[2:] <= 0).any()):
            raise ValueError('Source ROI requires finite x,y and positive width,height')
        image = images[name]
        camera = reconstruction.cameras[image.camera_id]
        transform = np.asarray(image.cam_from_world().matrix(), float)
        if transform.shape != (3, 4) or not np.isfinite(transform).all():
            raise ValueError('Actual finite camera transform required')
        residuals, local, rejected = [], [], 0
        for point in image.points2D:
            if not point.has_point3D():
                continue
            xyz = np.asarray(reconstruction.points3D[point.point3D_id].xyz, float)
            observed = np.asarray(point.xy, float)
            if xyz.shape != (3,) or observed.shape != (2,) or not np.isfinite(np.r_[xyz, observed]).all():
                raise ValueError('Finite source feature and matched 3D track required')
            xyz_cam = transform[:, :3] @ xyz + transform[:, 3]
            if xyz_cam[2] <= 0:
                rejected += 1
                continue
            projected = np.asarray(camera.img_from_cam(xyz_cam), float)
            if projected.shape != (2,) or not np.isfinite(projected).all():
                raise ValueError('Actual camera model failed to project a source track')
            residual = projected-observed
            residuals.append(residual)
            if box is not None and ((observed >= box[:2]) & (observed < box[:2]+box[2:])).all():
                local.append(residual)
        result.append({'frame': name, 'camera_id': int(image.camera_id),
                       'image_size': [int(camera.width), int(camera.height)],
                       'camera_model': str(camera.model),
                       'all_tracks': _summary(residuals),
                       'region': None if box is None else box.tolist(),
                       'region_tracks': None if box is None else _summary(local),
                       'behind_camera_tracks': rejected})
    return {'diagnostic_only': True, 'calibration_changed': False, 'frames': result,
            'scope': 'Actual fitted sparse tracks; these residuals do not bound pose uncertainty or accept object geometry.'}
