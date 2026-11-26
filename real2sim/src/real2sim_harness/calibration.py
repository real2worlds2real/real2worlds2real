"""ChArUco detection, calibrated camera evidence and explicit frame registration."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .artifacts import atomic_json, sha256_file
from .camera import solve_pnp


def board_spec(value):
    from .measurement import positive
    keys = {'type', 'squares_x', 'squares_y', 'square_length_m', 'marker_length_m', 'dictionary'}
    if not isinstance(value, dict) or set(value) - keys or value.get('type') != 'charuco':
        raise ValueError('Provide a known ChArUco board specification')
    for key in ('squares_x', 'squares_y'):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or not 3 <= value[key] <= 40:
            raise ValueError('Board square counts must be integers from 3 to 40')
    square = positive(value.get('square_length_m'), 'square_length_m')
    marker = positive(value.get('marker_length_m'), 'marker_length_m')
    dictionary = value.get('dictionary')
    if (marker >= square or not isinstance(dictionary, str) or not dictionary.startswith('DICT_')
            or not hasattr(cv2.aruco, dictionary) or not isinstance(getattr(cv2.aruco, dictionary), int)):
        raise ValueError('Use a supported ArUco dictionary and marker_length_m < square_length_m')
    return {**value, 'square_length_m': square, 'marker_length_m': marker}


def camera_spec(value):
    if not isinstance(value, dict) or set(value) - {'image_size', 'intrinsics', 'source'}:
        raise ValueError('Camera fields are image_size, intrinsics and optional source')
    size = value.get('image_size', [])
    if len(size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in size):
        raise ValueError('Camera image_size must be positive integer [width,height]')
    intr = value.get('intrinsics', {})
    if set(intr) - {'fx', 'fy', 'cx', 'cy', 'distortion'}:
        raise ValueError('Unknown intrinsic camera fields')
    numbers = np.asarray([intr.get(key, np.nan) for key in ('fx', 'fy', 'cx', 'cy')], dtype=float)
    coefficients = np.asarray(intr.get('distortion', []), dtype=float)
    if not np.isfinite(numbers).all() or np.any(numbers[:2] <= 0) or coefficients.ndim != 1 or len(coefficients) not in {0, 4, 5, 8, 12, 14} or not np.isfinite(coefficients).all():
        raise ValueError('Invalid camera intrinsics or OpenCV distortion coefficients')
    return {'image_size': size, 'intrinsics': {**dict(zip(('fx', 'fy', 'cx', 'cy'), numbers.tolist(), strict=True)),
                                             'distortion': coefficients.tolist()},
            'source': str(value.get('source', 'user_supplied_calibration'))}


def make_board(spec):
    spec = board_spec(spec)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec['dictionary']))
    return cv2.aruco.CharucoBoard((spec['squares_x'], spec['squares_y']),
                                spec['square_length_m'], spec['marker_length_m'], dictionary)


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'))


def intrinsic_matrix(camera):
    intr = camera_spec(camera)['intrinsics']
    return np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], dtype=float)


def detect_board(path, spec, camera=None, *, upsample=1):
    board = make_board(spec)
    rgb = read_rgb(path)
    height, width = rgb.shape[:2]
    if upsample not in (1, 2):
        raise ValueError('Board detector upsample must be 1 or 2')
    parameters = cv2.aruco.CharucoParameters()
    if camera is not None:
        camera = camera_spec(camera)
        if camera['image_size'] != [width, height]:
            raise ValueError('Image resolution differs from calibration; explicit calibrated rescaling is required')
        detection_matrix = intrinsic_matrix(camera)
        detection_matrix[:2, :2] *= upsample
        detection_matrix[:2, 2] = (detection_matrix[:2, 2] + .5) * upsample - .5
        parameters.cameraMatrix = detection_matrix
        parameters.distCoeffs = np.asarray(camera['intrinsics']['distortion'], dtype=float)
    corners, identifiers, markers, _ = cv2.aruco.CharucoDetector(board, parameters).detectBoard(
        cv2.cvtColor(cv2.resize(rgb, None, fx=upsample, fy=upsample), cv2.COLOR_RGB2GRAY))
    if identifiers is None or len(identifiers) < 6:
        raise ValueError('Need at least six detected ChArUco corners for this measurement')
    corners = (corners + .5) / upsample - .5
    points, pixels = board.matchImagePoints(corners, identifiers)
    points, pixels = points.reshape(-1, 3).astype(float), pixels.reshape(-1, 2).astype(float)
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError('Detected ChArUco points are collinear')
    if cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))) / (width * height) < 0.005:
        raise ValueError('Board covers too little of the image for this measurement')
    return {'image_size': [width, height], 'points_m': points.tolist(), 'pixels': pixels.tolist(),
            'corner_ids': identifiers.reshape(-1).tolist(), 'detected_markers': len(markers)}


def calibrate_camera(paths, spec, *, min_views=5, max_reprojection_px=1.0,
                     model='full', subset_trials=0, detector_upsample=1):
    from .measurement import positive
    max_reprojection_px = positive(max_reprojection_px, 'max_reprojection_px')
    if not 3 <= min_views <= 100:
        raise ValueError('min_views must be 3–100')
    if model not in {'full', 'constrained'} or type(subset_trials) is not int or not 0 <= subset_trials <= 32:
        raise ValueError('Choose full/constrained camera model and 0..32 subset trials')
    observations, rejected, hashes = [], [], set()
    for index, path in enumerate(paths):
        digest = sha256_file(Path(path))
        if digest in hashes:
            rejected.append({'index': index, 'reason': 'duplicate image'})
            continue
        hashes.add(digest)
        try:
            row = detect_board(path, spec, upsample=detector_upsample)
        except ValueError as exc:
            rejected.append({'index': index, 'reason': str(exc)})
            continue
        if observations and row['image_size'] != observations[0]['image_size']:
            raise ValueError('Calibration images must have the same displayed resolution and camera optics')
        observations.append({**row, 'index': index, 'sha256': digest})
    if len(observations) < min_views:
        raise ValueError(f'Need {min_views} usable distinct calibration views; found {len(observations)}')
    objects = [np.asarray(row['points_m'], dtype=np.float32).reshape(-1, 1, 3) for row in observations]
    pixels = [np.asarray(row['pixels'], dtype=np.float32).reshape(-1, 1, 2) for row in observations]
    size = tuple(observations[0]['image_size'])
    flags = 0
    initial = None
    if model == 'constrained':
        width, height = size
        initial = np.array([[1.2 * max(size), 0, width / 2], [0, 1.2 * max(size), height / 2], [0, 0, 1.]])
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, matrix, distortion, rotations, translations, intrinsic_std, _, per_view = cv2.calibrateCameraExtended(
        objects, pixels, size, initial, None, flags=flags)
    subset_focals = []
    rng = np.random.default_rng(0)
    for _ in range(subset_trials):
        indices = sorted(rng.choice(len(objects), max(3, (len(objects) + 1) // 2), replace=False))
        _, k, *_ = cv2.calibrateCamera([objects[i] for i in indices], [pixels[i] for i in indices],
                                     size, initial.copy() if initial is not None else None, None, flags=flags)
        subset_focals.append([float(k[0, 0]), float(k[1, 1])])
    finite = all(np.isfinite(value).all() for value in (matrix, distortion, intrinsic_std, per_view))
    normals = np.asarray([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    angles = np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1)))
    angular_span = float(angles.max())
    max_view = float(np.max(per_view))
    ok = bool(finite and rms <= max_reprojection_px and max_view <= max_reprojection_px and angular_span >= 8)
    failures = []
    if not finite:
        failures.append('nonfinite_calibration')
    if rms > max_reprojection_px or max_view > max_reprojection_px:
        failures.append('excessive_reprojection_error')
    if angular_span < 8:
        failures.append('limited_viewpoint_diversity')
    camera = camera_spec({'image_size': observations[0]['image_size'],
        'intrinsics': {'fx': float(matrix[0, 0]), 'fy': float(matrix[1, 1]),
                       'cx': float(matrix[0, 2]), 'cy': float(matrix[1, 2]),
                       'distortion': distortion.reshape(-1).tolist()}, 'source': 'opencv_charuco_calibration'})
    return {'schema_version': 1, 'ok': ok, 'failures': failures, 'camera': camera,
            'camera_model': model, 'subset_focals_px': subset_focals,
            'subset_scope': 'Random-half sensitivity diagnostic; not an uncertainty confidence interval',
            'rms_reprojection_px': float(rms), 'max_view_rms_px': max_view,
            'per_view_rms_px': per_view.reshape(-1).tolist(), 'normal_angular_span_deg': angular_span,
            'intrinsic_standard_deviations': intrinsic_std.reshape(-1).tolist(),
            'max_allowed_reprojection_px': max_reprojection_px,
            'used_views': [{'index': row['index'], 'sha256': row['sha256'], 'corners': len(row['pixels'])} for row in observations],
            'rejected_views': rejected, 'scope': 'camera calibration; not a metric object-accuracy certificate'}


def calibrate_store(store, *, model='full', subset_trials=0, detector_upsample=1):
    from .measurement import inputs
    spec = inputs(store)
    if not spec or not spec.get('calibration_images'):
        raise ValueError('Supply calibration_images and a known board in measurement inputs')
    result = calibrate_camera([store.root / path for path in spec['calibration_images']], spec['board'],
                             model=model, subset_trials=subset_trials, detector_upsample=detector_upsample)
    result['measurement_input_sha256'] = json.loads((store.root / 'run.json').read_text())['measurement_input']['sha256']
    atomic_json(store.root / 'measurements/camera_calibration.json', result)
    return result


def comparison_reference(store, reference_index, solution):
    run = json.loads((store.root / 'run.json').read_text())
    reference = store.root / run['references'][reference_index]['path']
    distortion = solution['intrinsics']['distortion']
    if distortion and np.any(distortion):
        camera = {'image_size': solution['image_size'], 'intrinsics': solution['intrinsics']}
        rgb = cv2.undistort(read_rgb(reference), intrinsic_matrix(camera), np.asarray(distortion),
                            None, intrinsic_matrix(camera))
        target = store.root / f'measurements/undistorted-reference-{reference_index:03d}.png'
        target.parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(target)
        solution['comparison_reference'] = {'path': str(target.relative_to(store.root)), 'sha256': sha256_file(target)}
        solution['comparison_domain'] = 'undistorted_pixels_with_original_intrinsics'
    else:
        solution['comparison_domain'] = 'original_pixels_zero_distortion'
    return solution


def measure_board(store, *, reference_index, asset_to_board=None):
    from .measurement import inputs, rigid_transform
    spec = inputs(store)
    if not spec or 'board' not in spec:
        raise ValueError('Supply known board dimensions; do not infer its physical size from appearance')
    camera = spec.get('camera')
    if camera is None:
        path = store.root / 'measurements/camera_calibration.json'
        result = json.loads(path.read_text()) if path.is_file() else {}
        digest = json.loads((store.root / 'run.json').read_text())['measurement_input']['sha256']
        if not result.get('ok') or result.get('measurement_input_sha256') != digest:
            raise ValueError('Provide calibrated camera intrinsics or call calibrate_measurement_camera first')
        camera = result['camera']
    run = json.loads((store.root / 'run.json').read_text())
    if not isinstance(reference_index, int) or not 0 <= reference_index < len(run['references']):
        raise ValueError('reference_index is out of range')
    reference = store.root / run['references'][reference_index]['path']
    if sha256_file(reference) != run['references'][reference_index]['sha256']:
        raise ValueError('Reference image changed')
    detection = detect_board(reference, spec['board'], camera)
    points = np.asarray(detection['points_m'])
    if asset_to_board is not None:
        transform = np.linalg.inv(rigid_transform(asset_to_board))
        points = points @ transform[:3, :3].T + transform[:3, 3]
    intr = camera['intrinsics']
    solution = solve_pnp(points.tolist(), detection['pixels'], width=camera['image_size'][0],
                         height=camera['image_size'][1], **intr)
    if solution['reprojection']['max_px'] > 2:
        raise ValueError('Board pose rejected: maximum reprojection error exceeds 2 pixels')
    solution.update(source='charuco_metric_board', reference_index=reference_index,
                    reference_sha256=run['references'][reference_index]['sha256'], board=spec['board'],
                    detection=detection, asset_frame_registered=asset_to_board is not None,
                    asset_to_board=asset_to_board,
                    limits=['Board pose alone does not determine the depth of an arbitrary object.',
                            'A board-to-asset registration must be independently justified.'])
    rgb = read_rgb(reference).copy()
    cv2.aruco.drawDetectedCornersCharuco(rgb,
        np.asarray(detection['pixels'], dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(detection['corner_ids'], dtype=np.int32).reshape(-1, 1))
    annotation = store.root / f'measurements/board_reference_{reference_index:03d}.png'
    annotation.parent.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(annotation)
    solution['annotation'] = str(annotation.relative_to(store.root))
    if asset_to_board is not None:
        comparison_reference(store, reference_index, solution)
        atomic_json(store.root / f'camera_solutions/reference_{reference_index:03d}.json', solution)
    atomic_json(store.root / f'measurements/board_reference_{reference_index:03d}.json', solution)
    return solution
