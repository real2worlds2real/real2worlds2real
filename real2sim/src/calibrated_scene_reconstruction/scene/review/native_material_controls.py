"""Separate native geometry, projected photos and GS in appearance diagnostics.

Close only the selected background owners' GS cutouts, preserving complete
native geometry. Compare photographed and original native materials, each with
GS enabled/disabled. These are counterfactual controls, never delivery assets.
"""
import argparse
import json
from pathlib import Path

from calibrated_scene_reconstruction.scene.review.environment import digest


def ungapped(stage, material):
    from pxr import UsdShade
    seen = set()
    while material:
        path = str(material.GetPath())
        if path in seen:
            raise ValueError('Cyclic native gap material references')
        seen.add(path)
        original = material.GetPrim().GetCustomDataByKey('r2s2r_gap_source_material')
        if not original:
            return material
        material = UsdShade.Material(stage.GetPrimAtPath(original))
    raise ValueError('Native gap material lost its original surface')


def restore(stage, owners, uniform=False):
    from pxr import UsdGeom, UsdShade
    if (not isinstance(owners, list) or not 1 <= len(owners) <= 8
            or len(set(owners)) != len(owners)
            or any(not isinstance(n, str) or not n or '/' in n for n in owners)):
        raise ValueError('Name up to eight distinct background owners')
    counts = dict.fromkeys(owners, 0); replaced = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        parts = str(prim.GetPath()).split('/')
        if len(parts) < 5 or parts[1:3] != ['World', 'Environment'] or parts[3] not in counts: continue
        counts[parts[3]] += 1
        base = ungapped(stage, UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(base)
        for subset in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Mesh(prim), UsdGeom.Tokens.face, 'materialBind'):
            target = subset.GetPrim()
            material = ungapped(stage, UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()[0])
            # Strip only the generated capture atlas. Preserve intrinsic native
            # face materials such as metal trim or a fabric inlay.
            if uniform and '_CapturedAppearance' in str(material.GetPath()).split('/'):
                material = base; replaced += 1
            UsdShade.MaterialBindingAPI.Apply(target).Bind(material)
    if not all(counts.values()):
        raise ValueError('Selected background owner absent from actual rendered geometry')
    return {'meshes':counts, 'photo_subsets_disabled':replaced}


def prepare(composite_path, owners, sources, output):
    from pxr import Usd
    from calibrated_scene_reconstruction.scene.review import ablation, frozen_scope
    from calibrated_scene_reconstruction.scene.review.frozen_scope_cache import same_geometry
    composite_path, out = Path(composite_path).resolve(), Path(output).resolve()
    if out.exists(): raise ValueError('Use a new material control directory')
    review = json.loads(composite_path.read_text())
    if review.get('diagnostic') or not review.get('environment_context'):
        raise ValueError('An actual composed hybrid source/free-view review is required')
    if not sources or len(set(sources)) != len(sources) or not 2 <= len(sources) <= 16:
        raise ValueError('Choose two to sixteen distinct captured source names')
    native_path = Path(review['environment']['asset']).parent/'manifest.json'
    native = json.loads(native_path.read_text())
    matches = [Path(p) for p, sha in native['input_sha256'].items()
               if sha == review['environment_context']['evidence_sha256']]
    if (len(matches) != 1 or native['asset'] != review['environment']['asset']
            or digest(native['asset']) != review['environment']['asset_sha256']):
        raise ValueError('Actual immutable native build and one field evidence record required')
    context_path = matches[0]
    if digest(context_path) != review['environment_context']['evidence_sha256']:
        raise ValueError('Actual rendered field evidence changed')
    context = json.loads(context_path.read_text())
    if not set(owners) <= set(context['appearance_ownership']):
        raise ValueError('Controls are restricted to declared background owners')
    cameras = json.loads(Path(review['camera_manifest']).read_text())
    cameras['anchors'] = [a for a in cameras['anchors'] if Path(a['source']).name in sources]
    if len(cameras['anchors']) != len(sources): raise ValueError('Requested source camera absent from actual review')
    # The native manifest records its then-current configuration as history.
    # Diagnose the immutable rendered asset, not today's mutable config file.
    # Actual layers/textures are bound by the render, field evidence by context.
    inputs = {**context['input_sha256'], **review['input_sha256']}
    for p in (composite_path, native_path, context_path, Path(__file__).resolve(), Path(ablation.__file__).resolve()):
        inputs[str(p)] = digest(p)
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Material control inputs changed')
    original = Usd.Stage.Open(review['usd']); task = context['protected_task_state']
    args = (task['task']['support'], task['objects'])
    protected = frozen_scope.meshes(original, *args)
    out.mkdir(parents=True); (out/'cameras.json').write_text(json.dumps(cameras, indent=2)+'\n')
    variants = {}
    for name, plain in [('photographed', False), ('native', True)]:
        folder = out/name; folder.mkdir()
        stage = ablation.copy_stage(review['usd'], folder/'scene.usda',
            'Native material control: selected cutouts closed; '+name+'; not a delivery candidate')
        variants[name] = restore(stage, owners, plain)
        if not same_geometry(protected, frozen_scope.meshes(stage, *args)):
            raise ValueError('Material controls changed frozen task geometry')
        stage.GetRootLayer().Save()
        inputs[str(folder/'scene.usda')] = digest(folder/'scene.usda')
    inputs[str(out/'cameras.json')] = digest(out/'cameras.json')
    result = {'approved':False, 'appearance_pass':False, 'composite':str(composite_path),
        'owners':owners, 'variants':variants, 'native_cutouts_closed':True,
        'field_rows_unchanged':True, 'frozen_geometry_unchanged':True,
        'input_sha256':inputs, 'scope':__doc__.strip()}
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Material controls changed during preparation')
    (out/'controls.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('composite')
    p.add_argument('--owners', nargs='+', required=True); p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    prepare(a.composite, a.owners, a.sources, a.output)
    print('Unapproved material controls prepared; render each scene with GS on and --context-disabled')


if __name__ == '__main__': main()
