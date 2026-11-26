"""Condition a shared camera rig on explicit model-to-image correspondences.

Only one positive similarity is optimized. Relative camera poses are preserved;
the result cannot improve an incorrect relative rig or certify assumed geometry.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .artifacts import atomic_json, sha256_file
from .posed_observations import camera_spec, file_path, write_report


def align(request, base, output):
    if request.get('schema_version') != 1:
        raise ValueError('Use schema_version=1')
    frame = request.get('frame', {})
    prior = request.get('prior_frame', {})
    if (not isinstance(frame.get('source'), str) or not frame['source'].strip()
            or type(frame.get('metric_verified')) is not bool
            or not isinstance(prior.get('source'), str) or not prior['source'].strip()):
        raise ValueError('Describe model frame/metric_verified and separate prior_frame provenance')
    rows = request.get('views')
    if not isinstance(rows, list) or not 2 <= len(rows) <= 64:
        raise ValueError('Shared-rig alignment needs 2..64 observed views')
    cameras, models, pixels, hashes, photos, initial_cameras = [], [], [], {}, [], []
    total_pixels = total_points = 0
    for row in rows:
        if not isinstance(row, dict) or 'prior_world_to_camera' not in row:
            raise ValueError('Every view needs an explicit shared-prior pose')
        # prior_world_to_camera deliberately has a different field name from a
        # camera already registered to the supplied model frame.
        camera = camera_spec({**row, 'world_to_camera': row['prior_world_to_camera']})
        total_pixels += camera['width'] * camera['height']
        if total_pixels > 32_000_000:
            raise ValueError('Rig image budget exceeds 32 million pixels')
        if not np.allclose(camera['K'][[0, 1], [1, 0]], 0):
            raise ValueError('Use zero-skew pinhole intrinsics')
        xyz = np.asarray(row.get('model_points'), dtype=float)
        uv = np.asarray(row.get('image_points'), dtype=float)
        if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or not 4 <= len(xyz) <= 2000
                or uv.shape != (len(xyz), 2) or not np.isfinite(xyz).all()
                or not np.isfinite(uv).all()
                or np.linalg.matrix_rank(xyz - xyz.mean(0)) < 2
                or (uv < 0).any() or (uv >= [camera['width'], camera['height']]).any()):
            raise ValueError('Each view needs 4..2000 noncollinear paired model points and displayed pixels')
        total_points += len(xyz)
        if total_points > 20000:
            raise ValueError('Rig correspondence budget exceeds 20000 points')
        # PnP supplies an initializer only. Its independently fitted rotations
        # never replace the shared rig in the optimization or returned cameras.
        ok, rv, tv = cv2.solvePnP(xyz, uv, camera['K'], None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise ValueError('Could not initialize rig alignment from correspondences')
        rotation = cv2.Rodrigues(rv)[0]
        if not np.isfinite(tv).all() or ((xyz @ rotation.T + tv[:, 0])[:, 2] <= 0).any():
            raise ValueError('Initializer has nonfinite values or points behind a camera')
        e = np.eye(4)
        e[:3, :3], e[:3, 3] = rotation, tv[:, 0]
        initial_cameras.append(e)
        image = None
        if 'image' in row:
            path = file_path(base, row['image'])
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (camera['height'], camera['width']):
                raise ValueError('Image dimensions must match the declared pixel/intrinsic domain')
            hashes[str(path)] = sha256_file(path)
        cameras.append(camera)
        models.append(xyz)
        pixels.append(uv)
        photos.append(image)
    centers = np.array([c['center'] for c in cameras])
    model_centers = np.array([-e[:3, :3].T @ e[:3, 3] for e in initial_cameras])
    scales = []
    for i in range(len(cameras)):
        for j in range(i):
            a = np.linalg.norm(centers[i] - centers[j])
            b = np.linalg.norm(model_centers[i] - model_centers[j])
            if a > 1e-8 and b > 1e-8:
                scales.append(a / b)
    if not scales:
        raise ValueError('A translating shared rig is required to align scale; pure rotation is insufficient')
    scale = float(np.median(scales))
    p0 = cameras[0]['pose']
    e0 = initial_cameras[0]
    rotation = p0[:3, :3].T @ e0[:3, :3]
    translation = p0[:3, :3].T @ (scale * e0[:3, 3] - p0[:3, 3])
    initial = np.r_[cv2.Rodrigues(rotation)[0][:, 0], translation, np.log(scale)]
    sigma = request.get('robust_scale_px', 2.)
    budget = request.get('max_evaluations', 300)
    gate = request.get('max_reprojection_rms_px')
    if (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not np.isfinite(sigma) or sigma <= 0
            or type(budget) is not int or not 10 <= budget <= 2000
            or (gate is not None and (isinstance(gate, bool)
                or not isinstance(gate, (int, float)) or not np.isfinite(gate) or gate <= 0))):
        raise ValueError('Use positive finite pixel tolerances and 10..2000 max_evaluations')

    def project(p):
        r = cv2.Rodrigues(p[:3])[0]
        s = np.exp(p[6])
        predictions, depths = [], []
        for c, xyz in zip(cameras, models, strict=True):
            prior_xyz = s * (xyz @ r.T) + p[3:6]
            camera_xyz = prior_xyz @ c['pose'][:3, :3].T + c['pose'][:3, 3]
            z = camera_xyz[:, 2]
            safe_z = np.where(np.abs(z) < 1e-9, np.where(z < 0, -1e-9, 1e-9), z)
            homogeneous = camera_xyz @ c['K'].T
            predictions.append(homogeneous[:, :2] / safe_z[:, None])
            depths.append(z)
        return predictions, depths

    initial_predictions, initial_depths = project(initial)
    depth_scale = max(float(np.median(np.concatenate(initial_depths))), 1e-6)

    def residual(p):
        predicted, depths = project(p)
        errors = [(a - b).ravel() for a, b in zip(predicted, pixels, strict=True)]
        # Do not permit a mirrored/behind-camera image match to count as a fit.
        errors.extend(np.minimum(z - 1e-8, 0) / depth_scale * 1000 for z in depths)
        return np.concatenate(errors)

    lower = np.r_[[-np.inf] * 6, initial[6] - np.log(10000)]
    upper = np.r_[[np.inf] * 6, initial[6] + np.log(10000)]
    result = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1',
                           f_scale=sigma, max_nfev=budget, x_scale='jac')
    p = result.x
    r, s = cv2.Rodrigues(p[:3])[0], float(np.exp(p[6]))
    predicted, depths = project(p)
    all_errors = []
    records, per_view, previews = [], [], []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'rig_request.json', request)
    hashes[str(output / 'rig_request.json')] = sha256_file(output / 'rig_request.json')
    for i, (c, observed, uv, z) in enumerate(zip(cameras, pixels, predicted, depths, strict=True)):
        errors = np.linalg.norm(uv - observed, axis=1)
        all_errors.extend(errors.tolist())
        e = np.eye(4)
        e[:3, :3] = c['pose'][:3, :3] @ r
        e[:3, 3] = (c['pose'][:3, :3] @ p[3:6] + c['pose'][:3, 3]) / s
        records.append({'source': 'Model-conditioned shared rig from ' + c['source'],
                        'world_to_camera': e.tolist(), 'intrinsics': c['K'].tolist(),
                        'image_size': [c['width'], c['height']], 'view_index': i})
        per_view.append({'view_index': i, 'point_count': len(observed),
                         'reprojection_rms_px': float(np.sqrt(np.mean(errors ** 2))),
                         'max_reprojection_px': float(errors.max()),
                         'per_point_reprojection_px': errors.tolist(),
                         'points_in_front': bool((z > 0).all())})
        if photos[i] is not None:
            image = photos[i].copy()
            for q in observed:
                cv2.circle(image, tuple(np.rint(q).astype(int)), 3, (30, 40, 240), 1)
            for q in uv:
                if np.isfinite(q).all() and (q >= 0).all() and (q < [c['width'], c['height']]).all():
                    cv2.drawMarker(image, tuple(np.rint(q).astype(int)), (60, 220, 70),
                                   cv2.MARKER_CROSS, 7, 1)
            name = f'rig-projection-{i:03d}.png'
            cv2.imwrite(str(output / name), image)
            previews.append(name)
    rms = float(np.sqrt(np.mean(np.square(all_errors))))
    front = all(row['points_in_front'] for row in per_view)
    similarity = np.eye(4)
    similarity[:3, :3], similarity[:3, 3] = s * r, p[3:6]
    atomic_json(output / 'candidate_cameras.json', {'frame': frame, 'views': records})
    return write_report(output, {
        'schema_version': 1, 'ok': True, 'operation': 'camera_rig_alignment',
        'frame': frame, 'prior_frame': prior, 'input_hashes': hashes,
        'world_to_prior_similarity': similarity.tolist(), 'scale': s,
        'cameras': records, 'per_view': per_view, 'reprojection_rms_px': rms,
        'initial_reprojection_rms_px': float(np.sqrt(np.mean(np.concatenate([
            np.linalg.norm(a - b, axis=1) ** 2
            for a, b in zip(initial_predictions, pixels, strict=True)])))),
        'max_reprojection_rms_px': gate,
        'quality_target_passed': bool(front and all(v['reprojection_rms_px'] <= gate for v in per_view)) if gate else None,
        'all_points_in_front': front, 'optimizer_converged': bool(result.success),
        'optimizer_message': str(result.message), 'evaluations': result.nfev,
        'relative_rig_geometry_preserved': True,
        'independent_metric_evidence_added': False, 'existing_cameras_modified': False,
        'preview_images': previews,
        'limits': [
            'One model-conditioned positive similarity; relative prior cameras are not refined.',
            'Assumed model dimensions and annotated pixels remain assumptions/observations, not independent calibration.',
            'Reprojection is evaluated on the fitted correspondences, not heldout validation.',
            'Inspect all views and current-source geometry before choosing these candidate cameras. No asset or scene is accepted.',
        ],
    })
