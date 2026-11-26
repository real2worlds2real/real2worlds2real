"""Repair explicitly reviewed dangling internal refs on empty visual frames.

This copies a complete local asset directory into a new output. It never edits
the input. Only a reviewed empty Xform's single missing internal reference may
be removed; all composed attributes, relationships, geometry and physics must
remain identical. A missing geometry file is not eligible for this repair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(item):
    from pxr import Sdf
    if item is None or isinstance(item, (str, bool, int, float)):
        return item
    if isinstance(item, Sdf.AssetPath):
        return {'asset_path': item.path}
    if isinstance(item, Sdf.Path):
        return {'prim_path': str(item)}
    if isinstance(item, dict):
        return {str(k): value(v) for k, v in sorted(item.items())}
    if hasattr(item, 'GetReal') and hasattr(item, 'GetImaginary'):
        return [float(item.GetReal()), value(item.GetImaginary())]
    array = np.asarray(item)
    if array.dtype.kind in 'biufc':
        return {'dtype': str(array.dtype), 'shape': list(array.shape),
                'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(item, (list, tuple)) or array.ndim > 0:
        return [value(v) for v in item]
    return str(item)


def composed(stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    rows = {}
    counts = dict(prims=0, geometry=0, colliders=0, rigid_bodies=0, joints=0)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        attrs = {}
        for attr in prim.GetAttributes():
            attrs[attr.GetName()] = {
                'type': str(attr.GetTypeName()), 'default': value(attr.Get()),
                'samples': {str(t): value(attr.Get(t)) for t in attr.GetTimeSamples()},
                'connections': [str(p) for p in attr.GetConnections()]}
        row = {'type': prim.GetTypeName(), 'active': prim.IsActive(),
               'schemas': list(prim.GetAppliedSchemas()), 'attributes': attrs,
               'relationships': {r.GetName(): [str(p) for p in r.GetTargets()]
                                 for r in prim.GetRelationships()}}
        rows[str(prim.GetPath())] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        counts['prims'] += 1
        counts['geometry'] += int(prim.IsA(UsdGeom.Gprim))
        counts['colliders'] += int(prim.HasAPI(UsdPhysics.CollisionAPI))
        counts['rigid_bodies'] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        counts['joints'] += int(prim.IsA(UsdPhysics.Joint))
    return {'counts': counts, 'prim_state_sha256': rows,
            'sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def run(asset, reviewed, output, *, max_bytes=100_000_000):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from calibrated_scene_reconstruction.sim.asset_composition import audit
    asset, output = Path(asset).resolve(), Path(output).resolve()
    root = asset.parent
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Use a new nonoverlapping output directory')
    if not isinstance(reviewed, list) or not reviewed:
        raise ValueError('Explicit independently reviewed missing-reference entries required')
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in root.rglob('*')) or any(not p.resolve().is_relative_to(root) for p in files):
        raise ValueError('Asset closure must be local and contain no links')
    if sum(p.stat().st_size for p in files) > max_bytes:
        raise ValueError('Asset copy exceeds the explicit byte budget')
    original = Usd.Stage.Open(str(asset))
    before, before_errors = composed(original), audit(original)
    if before_errors['pass']:
        raise ValueError('Asset has no broken composition to repair')
    selected = []
    seen = set()
    for row in reviewed:
        if not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('Each empty-frame interpretation needs an explicit review reason')
        relative = Path(row['layer'])
        layer_path = (root/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not layer_path.is_relative_to(root):
            raise ValueError('Reviewed layer must belong to the copied asset')
        if sha(layer_path) != row['layer_sha256']:
            raise ValueError('Reviewed source layer changed')
        key = (str(relative), row['prim'])
        if key in seen:
            raise ValueError('Duplicate repair entry')
        seen.add(key)
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        spec = layer.GetPrimAtPath(row['prim'])
        if (spec is None or spec.typeName != 'Xform' or spec.path.name != 'visuals'
                or spec.properties or spec.nameChildren):
            raise ValueError('Only an empty visual Xform is eligible')
        refs = list(spec.referenceList.GetAppliedItems())
        if len(refs) != 1 or refs[0].assetPath or str(refs[0].primPath) != row['target']:
            raise ValueError('Only the exact single reviewed internal reference may be removed')
        layer_stage = Usd.Stage.Open(layer)
        if layer_stage.GetPrimAtPath(refs[0].primPath):
            raise ValueError('A real reference target must never be removed')
        parent = original.GetPrimAtPath(spec.path.GetParentPath())
        if not parent or not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('The empty frame must belong to an existing rigid body')
        if any(p.IsA(UsdGeom.Gprim) or p.HasAPI(UsdPhysics.CollisionAPI)
               for p in Usd.PrimRange(parent, Usd.TraverseInstanceProxies())):
            raise ValueError('A parent with actual geometry/colliders is not an empty frame')
        selected.append(row)
    output.mkdir(parents=True, exist_ok=False)
    record = {'version': 1, 'pass': False, 'source_asset': str(asset),
              'source_sha256': {str(p.relative_to(root)): sha(p) for p in files},
              'reviewed_repairs': selected, 'before': before, 'before_composition': before_errors,
              'scope': 'Explicit empty internal visual references only; no geometry/body/joint/material changes or task approval.'}
    (output/'request.json').write_text(json.dumps(record, indent=2)+'\n')
    shutil.copytree(root, output/'original', copy_function=shutil.copy2)
    shutil.copytree(root, output/'candidate', copy_function=shutil.copy2)
    for row in selected:
        layer = Sdf.Layer.FindOrOpen(str(output/'candidate'/row['layer']))
        layer.GetPrimAtPath(row['prim']).referenceList.ClearEdits()
        layer.Save()
    candidate = output/'candidate'/asset.name
    stage = Usd.Stage.Open(str(candidate))
    after, errors = composed(stage), audit(stage)
    record.update(candidate_asset=str(candidate), after=after, after_composition=errors,
                  candidate_sha256={str(p.relative_to(output/'candidate')): sha(p)
                                    for p in sorted((output/'candidate').rglob('*')) if p.is_file()})
    record['pass'] = before == after and errors['pass']
    (output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    if not record['pass']:
        raise ValueError('Repair changed composed state or retained composition errors; candidate not usable')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset'); parser.add_argument('reviewed'); parser.add_argument('output')
    parser.add_argument('--max-bytes', type=int, default=100_000_000)
    args=parser.parse_args()
    result=run(args.asset,json.loads(Path(args.reviewed).read_text()),args.output,max_bytes=args.max_bytes)
    print(json.dumps({k:result[k] for k in ('pass','candidate_asset','scope')},indent=2))


if __name__ == '__main__':
    main()
