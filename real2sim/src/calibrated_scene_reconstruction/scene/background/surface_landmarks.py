"""Calibrated planar proposals from fixed landmarks and an unbounded edge.

An occlusion endpoint is never a rectangle corner. Upright is an explicit
manufactured-surface hypothesis, not a measured pose or supplied dimension.
The visible extent, support, material and unseen geometry need separate checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from calibrated_scene_reconstruction.scene.background.structure_fit import triangulate
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def solve(cameras, frames, pixels, segments, held_out, *, hypothesis, error_px=6.):
    if hypothesis != 'upright_edge_horizontal':
        raise ValueError('Explicit supported surface-orientation hypothesis required')
    if not 0 < error_px <= 6 or len(held_out) < 2:
        raise ValueError('At least two held-out frames and at most six-pixel uncertainty required')
    names = list(map(str, cameras['names']))
    pixels = np.asarray(pixels, float)
    segments = np.asarray(segments, float)
    if (pixels.ndim != 3 or pixels.shape[0] != len(frames) or pixels.shape[1] < 4
            or pixels.shape[2] != 2 or segments.shape != (len(frames), 2, 2)):
        raise ValueError('Four matched physical points and one visible edge per fitting view required')
    observations = []; seen = set()
    for frame, points, edge in [*zip(frames, pixels, segments),
                                *[(r['frame'], r['pixels'], r['segment']) for r in held_out]]:
        points, edge = np.asarray(points, float), np.asarray(edge, float)
        if (frame in seen or frame not in names or points.shape != pixels.shape[1:]
                or edge.shape != (2, 2) or not np.isfinite(points).all() or not np.isfinite(edge).all()):
            raise ValueError('Distinct registered source frames with finite corresponding features required')
        seen.add(frame); i = names.index(frame); w, h = cameras['size'][i]
        combined = np.vstack([points, edge])
        if (np.any(combined < 0) or np.any(combined[:, 0] >= w) or np.any(combined[:, 1] >= h)
                or np.linalg.norm(edge[1] - edge[0]) < 20):
            raise ValueError('Visible raster points and a useful straight edge required')
        observations.append((frame, points, edge))
    points, initial = triangulate(cameras, frames, pixels, max_error_px=error_px)
    origin = points[0]
    direction = points[1:].mean(0) - origin
    if np.linalg.norm(direction[:2]) < .01:
        raise ValueError('Landmarks must span the surface horizontally')
    angle = np.arctan2(direction[1], direction[0])
    axis = np.array([np.cos(angle), np.sin(angle), 0.])
    coordinates = np.c_[(points[1:] - origin) @ axis, origin[2] - points[1:, 2]]
    if np.linalg.svd(points - points.mean(0), compute_uv=False)[1] < .005:
        raise ValueError('Noncollinear physical landmarks required')
    start = np.r_[origin, angle, coordinates.ravel()]

    def decode(v):
        axis = np.array([np.cos(v[3]), np.sin(v[3]), 0.])
        down = np.array([0., 0., -1.])
        axes = np.stack([axis, np.cross(down, axis), down], axis=1)
        world = np.vstack([v[:3], v[:3] + v[4:].reshape(-1, 2) @ axes[:, [0, 2]].T])
        return axes, world

    def project(vertices, frame):
        i = names.index(frame); pc = (vertices - cameras['centre'][i]) @ cameras['R_world_cam'][i]
        if np.any(pc[:, 2] <= .05):
            raise ValueError('Physical feature behind a source camera')
        uv = pc @ cameras['K'][i].T
        return uv[:, :2] / uv[:, 2:]

    def measure(v, row):
        frame, observed, edge = row; axes, world = decode(v)
        uv = project(world, frame)
        endpoints = project(np.array([v[:3], v[:3] + axes[:, 0]]), frame)
        line = np.cross(*np.c_[endpoints, np.ones(2)])
        if np.linalg.norm(line[:2]) < 1e-8:
            raise ValueError('Surface edge projects to a point')
        line /= np.linalg.norm(line[:2])
        return uv, np.c_[edge, np.ones(2)] @ line

    def residual(v):
        try:
            return np.concatenate([np.r_[(measure(v, row)[0] - row[1]).ravel(), measure(v, row)[1]]
                                   for row in observations[:len(frames)]])
        except ValueError:
            return np.full(len(frames) * (pixels.shape[1] * 2 + 2), 1e6)

    fit = least_squares(residual, start, max_nfev=1000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    axes, world = decode(fit.x); checks = []
    for j, row in enumerate(observations):
        uv, edge_error = measure(fit.x, row)
        errors = np.linalg.norm(uv - row[1], axis=1)
        checks.append({'frame': row[0], 'held_out': j >= len(frames), 'point_error_px': errors.tolist(),
                       'line_error_px': edge_error.tolist(), 'projected': uv.tolist(),
                       'pass': bool(errors.max() <= error_px and np.abs(edge_error).max() <= error_px)})
    result = {'origin': fit.x[:3].tolist(), 'rotation': axes.tolist(), 'points_world': world.tolist(),
              'triangulation': initial, 'checks': checks, 'hypothesis': hypothesis,
              'error_limit_px': error_px, 'source_fit_pass': all(r['pass'] for r in checks),
              'approved': False, 'scope': __doc__.strip()}
    if not result['source_fit_pass']:
        failure = ValueError('Surface landmarks or edge exceed source uncertainty')
        failure.proposal = result
        raise failure
    return result


def run(session, config, output):
    config, output = Path(config).resolve(), Path(output).resolve()
    if (output.exists() or not output.is_relative_to(session.dir.resolve())
            or (session/'qa/independent_freeze.json').exists()):
        raise ValueError('New unfrozen session-local surface proposal required')
    q = json.loads(config.read_text())
    entities = {e['id'] for e in session.config('environment.json')['parameters']['entities']}
    graph = session.scene_graph
    if (q.get('entity') not in entities or q['entity'] in {graph['task']['support'], *graph.get('dynamic_objects', {})}
            or not q.get('feature_review', '').strip()):
        raise ValueError('Reviewed fixed physical features on an explicit background entity required')
    camera = session/'parsing/posed_frames/cameras.npz'
    frames = {*q['frames'], *[r['frame'] for r in q['held_out']]}
    if q.get('camera_sha256') != digest(camera) or set(q.get('source_sha256', {})) != frames:
        raise ValueError('Bind all source images and calibrated cameras')
    from calibrated_scene_reconstruction.scene.background import structure_fit
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = [output.with_name(output.stem + '_solver.py'), output.with_name(output.stem + '_triangulation.py')]
    if any(p.exists() for p in snapshots):
        raise ValueError('New immutable implementation snapshots required')
    for source, target in zip([Path(__file__), Path(structure_fit.__file__)], snapshots):
        target.write_bytes(source.read_bytes())
    inputs = {str(p): digest(p) for p in [config, camera.resolve(), *snapshots]}
    for frame in frames:
        source = (session/'splat/data/images'/frame).resolve()
        if Path(frame).name != frame or digest(source) != q['source_sha256'][frame]:
            raise ValueError('Registered source image changed')
        inputs[str(source)] = digest(source)
    failure = None
    try:
        result = solve(np.load(camera), q['frames'], q['pixels'], q['segments'], q['held_out'],
                       hypothesis=q['hypothesis'], error_px=q.get('error_px', 6.))
    except ValueError as error:
        if not hasattr(error, 'proposal'):
            raise
        failure = error; result = error.proposal
    result.update(entity=q['entity'], part=q.get('part'), feature_review=q['feature_review'],
                  input_sha256=inputs, protected_task_state=protected_state(session))
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Source plane evidence changed')
    output.write_text(json.dumps(result, indent=2) + '\n')
    if failure:
        raise failure
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session'); parser.add_argument('config'); parser.add_argument('--output', required=True)
    a = parser.parse_args(); result = run(Session(a.session), a.config, a.output)
    print('Unapproved source surface:', result['entity'])


if __name__ == '__main__':
    main()
