"""The task_zone Blender branch. Agent review is separate from numerical fitting and export."""
import argparse

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.blender import export, fit, texture


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('session')
    p.add_argument('--max-evaluations', type=int, default=300)
    p.add_argument('--objects', nargs='*')
    p.add_argument('--texture', action='store_true')
    p.add_argument('--object-route', choices=('agent_blender', 'template'), default='agent_blender')
    a = p.parse_args()
    s = Session(a.session)
    from calibrated_scene_reconstruction.scene.objects.routes import ensure
    ensure(s, a.object_route, apply=True)
    for name, obj in s.scene_graph.get('dynamic_objects', {}).items():
        req = obj.get('fit') or {}
        if req.get('kit') and req.get('model'):
            raise ValueError(f'{name}: choose fit.kit OR fit.model')
        if not req.get('model') or (a.objects and name not in a.objects):
            continue
        if a.texture:
            texture.run(s, name)
            continue
        if req.get('accepted'):
            checked=export.verify_round(s, name, s / 'objects' / name / f"round_{obj['fix']['round']}", exported=True)
            if not checked.get('effective_ok',checked.get('ok',False)):
                raise ValueError(f'{name}: source acceptance is no longer valid')
            from calibrated_scene_reconstruction.scene.objects.blender.contract import contract
            if obj.get('fix', {}).get('model_sha256') != contract(fit.model_path(s, name))['sha256']:
                raise ValueError(f'{name}: accepted model changed; run a new round')
            continue
        result, out = fit.run(s, name, a.max_evaluations)
        print(f'[objects] {name}: IoU {result["metrics"]["iou_median"]:.4f}, '
              f'local {result["metrics"]["local_difference"]:.4f}, default_pass={result["ok"]}, '
              f'effective_pass={result.get("effective_ok",result["ok"])}', flush=True)
        if not result.get('effective_ok',result['ok']):
            raise SystemExit(f'{name}: fit thresholds failed; inspect {out}')
        export.run(s, name, out)


if __name__ == '__main__':
    main()
