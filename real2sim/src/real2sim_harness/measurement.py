"""Optional metric evidence shared by scale and geometric measurement tasks."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return float(value)


def rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError('Provide a rigid 4x4 transform; scale, shear and reflections are not allowed')
    return matrix


def validate_spec(value):
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('Measurement inputs require schema_version=1')
    allowed = {'schema_version', 'dimensions', 'board', 'camera', 'calibration_images', 'surface'}
    if set(value) - allowed:
        raise ValueError('Unknown measurement input fields')
    result = {'schema_version': 1, 'dimensions': []}
    for row in value.get('dimensions', []):
        if set(row) - {'part', 'axis', 'value_m', 'tolerance_m', 'source'}:
            raise ValueError('Unknown dimension fields')
        if row.get('axis') not in {'X', 'Y', 'Z'} or not isinstance(row.get('source'), str) or not row['source'].strip():
            raise ValueError('Dimensions need an X/Y/Z axis and an explicit measurement source')
        part = row.get('part')
        if part is not None and (not isinstance(part, str) or not part.strip()):
            raise ValueError('part must be a semantic part/object name or null for the whole asset')
        result['dimensions'].append({**row, 'part': part,
            'value_m': positive(row.get('value_m'), 'value_m'),
            'tolerance_m': positive(row.get('tolerance_m'), 'tolerance_m')})
    if value.get('board') is not None:
        from .calibration import board_spec
        result['board'] = board_spec(value['board'])
    if value.get('camera') is not None:
        from .calibration import camera_spec
        result['camera'] = camera_spec(value['camera'])
    if value.get('calibration_images'):
        if 'board' not in result or not all(isinstance(p, str) for p in value['calibration_images']):
            raise ValueError('Calibration images require a board specification and image paths')
        result['calibration_images'] = value['calibration_images']
    if value.get('surface') is not None:
        surface = value['surface']
        allowed = {'path', 'source', 'measured_to_asset', 'tolerance_m', 'max_points'}
        if set(surface) - allowed or not isinstance(surface.get('path'), str) or not surface.get('source'):
            raise ValueError('Surface inputs need a point-cloud path and measurement provenance')
        transform = rigid_transform(surface.get('measured_to_asset'))
        count = surface.get('max_points', 100_000)
        if not isinstance(count, int) or not 100 <= count <= 1_000_000:
            raise ValueError('surface.max_points must be 100–1000000')
        result['surface'] = {**surface, 'measured_to_asset': transform.tolist(), 'max_points': count,
                             'tolerance_m': positive(surface.get('tolerance_m'), 'surface.tolerance_m')}
    if not any(result.get(key) for key in ('dimensions', 'board', 'camera', 'surface')):
        raise ValueError('Provide dimensions, a known board, a calibrated camera or metric surface observations')
    return result


def load_metric_points(path):
    path = Path(path)
    if path.suffix == '.npy':
        points = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            points = data['points_m'].copy()
    else:
        raise ValueError('Use .npy Nx3 points in metres, or .npz with points_m')
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not 3 <= len(points) <= 2_000_000 or not np.isfinite(points).all():
        raise ValueError('Metric points must be a finite Nx3 array with 3–2000000 points')
    return points


def retain_inputs(store, configuration):
    configuration = Path(configuration).resolve()
    spec = validate_spec(json.loads(configuration.read_text()))
    folder = store.root / 'measurements'
    folder.mkdir(exist_ok=True)
    files = {}

    def retain(name, source):
        path = (configuration.parent / source).resolve()
        if not path.is_file():
            raise FileNotFoundError('Missing measurement input file')
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[str(target.relative_to(store.root))] = sha256_file(target)
        return str(target.relative_to(store.root))

    spec['calibration_images'] = [retain(f'calibration/{i:03d}{Path(name).suffix.lower()}', name)
                                  for i, name in enumerate(spec.get('calibration_images', []))]
    if 'surface' in spec:
        points = load_metric_points(configuration.parent / spec['surface']['path'])
        destination = folder / 'surface.npz'
        np.savez_compressed(destination, points_m=points)
        spec['surface']['path'] = str(destination.relative_to(store.root))
        files[spec['surface']['path']] = sha256_file(destination)
    spec['retained_files'] = files
    atomic_json(folder / 'inputs.json', spec)
    run_path = store.root / 'run.json'
    run = json.loads(run_path.read_text())
    run['measurement_input'] = {'path': 'measurements/inputs.json', 'sha256': sha256_file(folder / 'inputs.json')}
    atomic_json(run_path, run)
    return inspect_inputs(store)


def declare_dimensions(store, *, dimensions_json):
    """Retain dimensions stated in the original user request before modeling."""
    import tempfile
    run = json.loads((store.root / 'run.json').read_text())
    if run.get('measurement_input') or any(store.attempts_dir.glob('attempt-*')):
        raise ValueError('Declare dimensions once, before building; existing evidence cannot be replaced')
    dimensions = validate_spec({'schema_version': 1, 'dimensions': json.loads(dimensions_json)})['dimensions']
    unit_scale = {'mm': 0.001, '毫米': 0.001, 'cm': 0.01, '厘米': 0.01, 'm': 1, '米': 1}
    for row in dimensions:
        quote = row['source']
        if quote not in run['objective']:
            raise ValueError('source must quote the original user request exactly')
        readings = re.findall(r'(?<![\d.])(\d+(?:\.\d+)?)\s*(mm|cm|m|毫米|厘米|米)(?![A-Za-z])', quote)
        if not any(math.isclose(float(number) * unit_scale[unit], row['value_m'], rel_tol=1e-8, abs_tol=1e-10)
                   for number, unit in readings):
            raise ValueError('The quoted request must state this dimension with a supported unit; do not promote an estimate')
    with tempfile.TemporaryDirectory(prefix='metric-input-') as directory:
        path = Path(directory) / 'inputs.json'
        atomic_json(path, {'schema_version': 1, 'dimensions': dimensions})
        return retain_inputs(store, path)


def inputs(store):
    run = json.loads((store.root / 'run.json').read_text())
    record = run.get('measurement_input')
    if record is None:
        return None
    path = (store.root / record['path']).resolve()
    if not path.is_relative_to(store.root.resolve()) or sha256_file(path) != record['sha256']:
        raise ValueError('Measurement inputs changed; start a new task with the revised evidence')
    spec = json.loads(path.read_text())
    for name, digest in spec['retained_files'].items():
        retained = (store.root / name).resolve()
        if not retained.is_relative_to(store.root.resolve()) or not retained.is_file() or sha256_file(retained) != digest:
            raise ValueError('Retained measurement evidence is missing or changed')
    return spec


def inspect_inputs(store):
    spec = inputs(store)
    if spec is None:
        return {'available': False, 'evidence_level': 'visual', 'metric_accuracy_claimed': False}
    level = 'geometry_observed' if spec.get('surface') else 'scale_constrained' if spec.get('dimensions') or spec.get('board') else 'camera_calibrated'
    return {'available': True, 'evidence_level': level, 'inputs': spec,
            'metric_accuracy_claimed': False,
            'limits': ['A board anchors its own frame; object depth/frame must still be established.',
                       'Camera reprojection error is not a bound on object surface error.',
                       'Surface checks cover supplied observations, not hidden or unobserved surfaces.',
                       'Cousins do not automatically inherit twin measurement constraints.']}


def review_asset(executor, store, *, attempt_id):
    spec = inputs(store)
    if spec is None or not (spec.get('dimensions') or spec.get('surface')):
        raise ValueError('No retained dimension or surface constraints to review')
    if not isinstance(attempt_id, str) or not re.fullmatch(r'attempt-[0-9]{3}', attempt_id):
        raise ValueError('Use a saved attempt ID')
    attempt = store.attempts_dir / attempt_id
    source = attempt / 'scene.blend'
    source_digest = sha256_file(source)
    number = len(list(attempt.glob('measurement-[0-9][0-9][0-9]'))) + 1
    output = attempt / f'measurement-{number:03d}'
    output.mkdir()
    helper = Path(__file__).with_name('blender_measurement_harness.py')
    with (output / 'execution.log').open('w') as log:
        completed = subprocess.run([str(executor.blender), '--background', str(source),
            '--python-exit-code', '1', '--python', str(helper), '--', str(store.root), str(output)],
            stdout=log, stderr=subprocess.STDOUT, timeout=executor.timeout_seconds, check=False)
    path = output / 'measurement.json'
    if completed.returncode or not path.is_file():
        raise RuntimeError('Metric geometry review failed; inspect its execution log')
    result = json.loads(path.read_text())
    if sha256_file(source) != source_digest:
        raise ValueError('Source changed during measurement review')
    result.update(source_scene_sha256=source_digest,
                  measurement_input_sha256=json.loads((store.root / 'run.json').read_text())['measurement_input']['sha256'])
    if 'surface' in result:
        panel = output / 'surface_errors.png'
        residual_panel(output / 'surface_residuals.npz', panel, result['surface'])
        result['annotation'] = str(panel.relative_to(store.root))
        result['retained_artifacts'] = {str(path.relative_to(attempt)): sha256_file(path)
                                       for path in (panel, output / 'surface_residuals.npz')}
    atomic_json(path, result)
    report_path = attempt / 'report.json'
    report = json.loads(report_path.read_text())
    report['measurement'] = {**result, 'report_path': str(path.relative_to(attempt)), 'report_sha256': sha256_file(path)}
    atomic_json(report_path, report)
    store.append_event({'type': 'measurement_review', 'attempt_id': attempt_id, 'ok': result['ok']})
    return result


def residual_panel(cache, destination, metrics):
    from PIL import Image, ImageDraw
    with np.load(cache, allow_pickle=False) as data:
        observed, closest, errors = data['observed_points_m'], data['closest_points_m'], data['distance_m']
    image = Image.new('RGB', (1260, 470), '#fafaf6')
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), 'OBSERVED REGION: metric samples and nearest evaluated mesh points', fill='#172019')
    selection = np.linspace(0, len(observed)-1, min(5000, len(observed))).astype(int)
    for panel_index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        x0 = 20 + 415 * panel_index
        draw.text((x0, 40), f'{"XYZ"[a]} / {"XYZ"[b]} (metres)', fill='#172019')
        both = np.concatenate((observed[:, [a, b]], closest[:, [a, b]]))
        center = (both.max(axis=0) + both.min(axis=0)) / 2
        scale = 330 / max(float(np.ptp(both, axis=0).max()), 1e-8)
        def pixel(point, x0=x0, a=a, b=b, center=center, scale=scale):
            return (x0 + 185 + (point[a]-center[0])*scale, 230 - (point[b]-center[1])*scale)
        for index in selection:
            x, y = pixel(closest[index])
            draw.ellipse((x-1, y-1, x+1, y+1), fill='#aeb4ab')
        for index in selection:
            x, y = pixel(observed[index])
            fraction = min(1, errors[index] / metrics['tolerance_m'])
            color = (int(45+190*fraction), int(150-110*fraction), 55)
            draw.ellipse((x-1.4, y-1.4, x+1.4, y+1.4), fill=color)
    draw.text((20, 405), f'Point-to-triangle p95: {metrics["p95_m"]*1000:.3f} mm | max: {metrics["max_m"]*1000:.3f} mm | tolerance: {metrics["tolerance_m"]*1000:.3f} mm', fill='#172019')
    draw.text((20, 433), 'Gray: nearest mesh | green to red: sample error / tolerance. Partial observations do not establish completeness.', fill='#172019')
    image.save(destination)


def require_current_review(store, attempt, report):
    spec = inputs(store)
    if not spec or not (spec.get('dimensions') or spec.get('surface')):
        return
    evidence = report.get('measurement', {})
    record = json.loads((store.root / 'run.json').read_text())['measurement_input']
    if not evidence.get('ok'):
        raise ValueError('Measurement constraints require a successful review_measurements before acceptance')
    if evidence.get('source_scene_sha256') != sha256_file(attempt / 'scene.blend') or evidence.get('measurement_input_sha256') != record['sha256']:
        raise ValueError('Measurement evidence is stale; review the current asset again')
    path = (attempt / evidence.get('report_path', '')).resolve()
    if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != evidence.get('report_sha256'):
        raise ValueError('Measurement report is missing or changed')
    for name, digest in evidence.get('retained_artifacts', {}).items():
        retained = (attempt / name).resolve()
        if not retained.is_relative_to(attempt.resolve()) or not retained.is_file() or sha256_file(retained) != digest:
            raise ValueError('Measurement residual evidence is missing or changed')
