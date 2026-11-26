"""Source-bound review of visible structure that silhouette scores cannot prove.

The agent judges geometry-only and task-pose views. This validates provenance
and completeness of that review, not automatic recognition of an object class.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration(session, obj):
    graph=session.scene_graph
    spec=graph.get('dynamic_objects',{}).get(obj) or graph.get('furniture',{}).get(obj,{})
    fit=spec.get('fit',{})
    features = fit.get('geometry_features', [])
    plan = {}
    if hasattr(session, 'config'):
        plan = session.config('agent/modeling_plan.json', {})
    elif hasattr(session, 'config_dir'):
        source = Path(session.config_dir) / 'agent/modeling_plan.json'
        if source.is_file():
            plan = json.loads(source.read_text())
    planned = plan.get('physical_instances', {}).get(obj, {}).get('geometry_features')
    if hasattr(session, 'config_dir') and (Path(session.config_dir)/'agent/object_understanding.json').is_file():
        from .object_understanding import declaration as understanding
        parts = understanding(session, obj)['parts']
        understood = [{'id': p['id'], 'description': p['geometry_feature']} for p in parts]
        if planned is not None and planned != understood:
            raise ValueError('Visible geometry contract differs from object understanding')
        planned = understood
    elif obj in graph.get('furniture',{}):
        raise ValueError('Reconstructed furniture needs semantic and structural understanding')
    if planned is not None:
        if 'geometry_features' in fit and features != planned:
            raise ValueError('Visible geometry contract differs from the high-level modeling plan')
        features = planned
    if (not isinstance(features, list) or any(not isinstance(x, dict) or
            set(x) != {'id', 'description'} or not x['id'] or not x['description'] for x in features)
            or len({x['id'] for x in features}) != len(features)):
        raise ValueError('Geometry features require distinct ids and source descriptions')
    return features, fit.get('geometry_review')


def review_path(session, obj):
    features, name = declaration(session, obj)
    if not features:
        return None
    if not isinstance(name, str):
        raise ValueError('Visible geometry features need a structural review')
    path = (session.dir / name).resolve()
    if not path.is_relative_to((session.dir / 'qa').resolve()):
        raise ValueError('Structural review must stay in session QA')
    return path


def current(session, obj, round_dir):
    features, _ = declaration(session, obj)
    path = review_path(session, obj)
    if path is None:
        return None
    review = json.loads(path.read_text())
    if (review.get('version') != 1 or review.get('object') != obj
            or review.get('features') != features or review.get('pass') is not True):
        raise ValueError('Current passing structural review required')
    round_dir = Path(round_dir)
    for name in ('model.py', 'mesh.npz'):
        if review.get('candidate_sha256', {}).get(name) != digest(round_dir / name):
            raise ValueError('Structural review belongs to a different candidate')
    rendered = review.get('geometry_render', {})
    render_path = (session.dir / rendered.get('path', '')).resolve()
    if (not render_path.is_relative_to((session.dir / 'qa').resolve()) or not render_path.is_file()
            or digest(render_path) != rendered.get('sha256')):
        raise ValueError('Hash-bound neutral geometry render report required')
    render = json.loads(render_path.read_text())
    if (render.get('material_policy') != 'uniform_diffuse_no_texture'
            or render.get('mesh_sha256') != digest(round_dir / 'mesh.npz')):
        raise ValueError('Geometry review must render this candidate without textures')
    views = review.get('views', {})
    kinds = {row.get('kind') for row in views.values()}
    if not {'source', 'geometry_only', 'novel_geometry', 'task_pose_geometry'} <= kinds:
        raise ValueError('Source, neutral geometry, novel and task-pose geometry views required')
    for view_id, row in views.items():
        file = (session.dir / row['path']).resolve()
        if (not file.is_relative_to(session.dir.resolve()) or not file.is_file()
                or digest(file) != row.get('sha256')):
            raise ValueError('Structural review evidence changed or escaped its session')
        if row['kind'] != 'source':
            actual = render.get('views', {}).get(view_id, {})
            if (actual.get('sha256') != row['sha256']
                    or (render_path.parent / actual.get('file', '')).resolve() != file):
                raise ValueError('Claimed neutral view is absent from the actual geometry render')
    neutral = {name for name, row in views.items() if row['kind'] != 'source'}
    view_checks = review.get('view_checks', {})
    if set(view_checks) != neutral or any(row.get('pass') is not True or not row.get('reason')
                                         for row in view_checks.values()):
        raise ValueError('Every neutral and task-pose view needs its own passing structural judgment')
    checks = review.get('checks', {})
    if set(checks) != {x['id'] for x in features}:
        raise ValueError('Every declared visible structure needs its own review')
    for check in checks.values():
        evidence = check.get('views', [])
        if (check.get('pass') is not True or not check.get('reason')
                or check.get('basis') not in ('observed', 'inferred')
                or not evidence or set(evidence) - views.keys()
                or not any(views[name]['kind'] in ('geometry_only', 'novel_geometry') for name in evidence)):
            raise ValueError('Visible structure cannot pass from texture or silhouette alone')
    return digest(path)
