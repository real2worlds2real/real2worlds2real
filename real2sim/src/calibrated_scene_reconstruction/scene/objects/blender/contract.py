"""Read a model's literal parameter contract without importing agent code into the host process."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from calibrated_scene_reconstruction.assets.metric_mesh import geometry as geometry  # noqa: F401 -- public contract API
from calibrated_scene_reconstruction.assets.metric_mesh import DEFAULT_SDF

SDF = dict(DEFAULT_SDF)


def contract(path):
    tree = ast.parse(Path(path).read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in ("PARAMS", "FIXED", "LINKS", "JOINTS"):
                    values[target.id] = ast.literal_eval(node.value)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" and len(n.args.args) == 1 for n in tree.body):
        raise ValueError('model.py must define build(params)')
    params, fixed = values.get('PARAMS'), values.get('FIXED', {})
    if not isinstance(params, dict) or not params or not isinstance(fixed, dict):
        raise ValueError('PARAMS must be a nonempty literal dictionary; FIXED a literal dictionary')
    if params.keys() & fixed.keys():
        raise ValueError('PARAMS and FIXED must be disjoint')
    for name, p in params.items():
        if not isinstance(p, dict) or set(p) != {'init', 'lo', 'hi'}:
            raise ValueError(f'{name}: expected init, lo, hi')
        v = np.array([p['lo'], p['init'], p['hi']], dtype=float)
        if not np.isfinite(v).all() or not v[0] <= v[1] <= v[2] or v[0] == v[2]:
            raise ValueError(f'{name}: finite ordered bounds containing init required')
    result = {'params': params, 'fixed': fixed, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if any(isinstance(n,ast.FunctionDef) and n.name=='SURFACE_BINDINGS' for n in tree.body):
        result['surface_bindings_dynamic']=True
    if 'LINKS' in values:
        from calibrated_scene_reconstruction.assets.mechanisms import links, joints
        result['links'] = links(values['LINKS'])
        result['joints'] = joints(values.get('JOINTS', []), result['links'])
        result['joints_dynamic'] = any(isinstance(n, ast.FunctionDef) and n.name == 'JOINTS'
                                       and len(n.args.args) == 1 for n in tree.body)
    elif 'JOINTS' in values:
        raise ValueError('JOINTS requires explicit LINKS')
    dependencies = {}
    root = Path(__file__).resolve().parents[4]
    for node in ast.walk(tree):
        modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                   [a.name for a in node.names] if isinstance(node, ast.Import) else [])
        for module in modules:
            if module.startswith('calibrated_scene_reconstruction.'):
                source = root.joinpath(*module.split('.')).with_suffix('.py')
                if not source.exists():
                    source = root.joinpath(*module.split('.'), '__init__.py')
                if source.exists():
                    dependencies[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['dependencies'] = dependencies
    return result


def parameters(spec, overrides=None):
    p = {k: v['init'] for k, v in spec['params'].items()}
    for k, v in (overrides or {}).items():
        if k not in p:
            raise ValueError(f'unknown fitted parameter {k}')
        b = spec['params'][k]
        if not np.isfinite(v) or not b['lo'] <= v <= b['hi']:
            raise ValueError(f'{k}: outside declared bounds')
        p[k] = float(v)
    return {**spec['fixed'], **p}




def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)
