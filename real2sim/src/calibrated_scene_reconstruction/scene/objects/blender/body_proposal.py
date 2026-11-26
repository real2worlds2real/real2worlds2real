"""Initialize rigid geometry before an attached sheet has a valid draped state.

This produces a proposal in QA, never an object round, exported asset or gate
approval. The full Blender loop still checks every source part and the complete
authored material/contact state before it can accept the object.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import evidence, fit, proposal_stage
from .blender_run import Blender
from .contract import contract, write_json
from .data_inputs import snapshot
from .store import digest


def excluded_parts(spec, names):
    """A rigid proposal cannot hide another rigid mechanism or omit its body."""
    links = spec.get('links', {})
    if (not names or not set(names) <= links.keys()
            or any(links[name]['kind'] == 'rigid' for name in names)
            or not any(link['kind'] == 'rigid' for link in links.values())):
        raise ValueError('Exclude named flexible links only; retain all rigid bodies and mechanisms')


def unapproved(result, obj):
    return {**result, 'object': obj, 'ok': False, 'proposal_only': True,
            'source_approved': False, 'task_approved': False,
            'full_initial_state_checked': False,
            'scope': 'Rigid source initialization only. The loose material is omitted '
                     'from this proposal objective, not from the asset or its final '
                     'source, material, contact, mechanism and native task checks.'}


def run(session, obj):
    from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration, validate_build
    from calibrated_scene_reconstruction.scene.objects.mechanisms import validate_model
    declaration(session, obj)
    model = fit.model_path(session, obj)
    spec = contract(model)
    validate_model(session, obj, spec)
    validate_build(session, obj, {'parts': [dict(name=n, kind=v['kind'])
        for n, v in spec.get('links', {}).items()], 'joints': spec.get('joints', [])}, preflight=True)
    settings = proposal_stage.settings(session, obj)
    if settings is None:
        raise ValueError('Declare an explicit initial_rigid_stage before requesting a body proposal')
    excluded_parts(spec, settings['exclude_links'])
    full = evidence.load(session, obj, parts=True)
    observed = proposal_stage.observations(session, obj, full, settings['exclude_links'])
    root = session/'qa/body_proposal'/obj
    root.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    out = root/f'round_{number}'
    out.mkdir()
    shutil.copyfile(model, out/'model.py')
    assets = snapshot(session, obj, model, out)
    sources = [model, *assets, *[out/relative for relative in assets.values()],
               Path(__file__), Path(fit.__file__), Path(evidence.__file__), Path(proposal_stage.__file__)]
    sources += list(session.config_dir.glob('*.json'))+list((session.config_dir/'agent').glob('*.json'))
    sources += [session/name for name in ['parsing/posed_frames/cameras.npz',
                'splat/align.json', f'splat/object_fit/{obj}.npz',
                f'splat/object_fit/{obj}.components.acquired.npz']]
    for item in full[0].get('_source_parts', []):
        z = item['observations'][0]
        sources += [session.dir/name for name in z.get('_object_domain_inputs', {})]
    # Identity labels are inputs too, even though they are not model geometry.
    from .part_evidence import requirements
    for item in requirements(session, obj):
        if item.get('part_review'):
            directory = session.dir/item['part_review']
            sources += [directory/name for name in ['manifest.json', 'review.json', 'labels.npz']]
    inputs = {str(path.resolve()): digest(path) for path in sources if path.is_file()}
    from .triangle_support import request as triangle_request, inputs as triangle_inputs
    triangle_modules=[]
    if triangle_request(session,obj) is not None:
        inputs.update(triangle_inputs(session,obj))
        for relative in ['scene/objects/blender/triangle_support.py','assets/support_placement.py',
                         'assets/rigid_initial_state.py','assets/surface_distance.py']:
            module=Path(__file__).resolve().parents[3]/relative
            inputs[str(module)]=digest(module);triangle_modules.append(module)
    proposal_fixed = session.scene_graph['dynamic_objects'][obj]['fit'].get('body_proposal_fixed')
    write_json(out/'input.json', {'inputs': inputs, 'settings': settings,
                                'proposal_fixed': proposal_fixed,
                                'full_observation_frames': full[1].tolist(),
                                'proposal_observation_frames': np.asarray(observed[1]).tolist()})
    code = out/'engine'
    code.mkdir()
    for module in [Path(__file__), Path(fit.__file__), Path(evidence.__file__), Path(proposal_stage.__file__), *triangle_modules]:
        shutil.copyfile(module, code/module.name)
    try:
        # Optional numeric proposal settings can reduce construction work on
        # excluded flexible parts. This raw initializer never installs an
        # asset; the full object loop uses its independent full-model settings.
        with Blender(out/'model.py', session/'objects'/obj/'cache',
                     source_files=[*assets, *[out/r for r in assets.values()]],
                     proposal_fixed=proposal_fixed) as builder:
            mesh_path, first = builder.evaluate()
            initial = fit.initial_pose(session, obj, first['geometry']['extent_m'])
            with np.load(mesh_path/'mesh.npz') as mesh:
                initial = fit.ground_pose(mesh['V'], first.get('parts'), initial, faces=mesh['F'])
                initial = fit.seed_body_centre(session, obj, mesh['V'], first.get('parts'), initial)
                initial, orientation = fit.select_orientation(session, obj, mesh['V'], mesh['F'],
                                            initial, observed, model_parts=first.get('parts'))
            write_json(out/'initial_pose.json', initial)
            write_json(out/'yaw_hypotheses.json', orientation)
            from .projection import optional
            request = session.scene_graph['dynamic_objects'][obj]['fit']
            result = fit.optimize(builder, builder.spec, observed, initial, settings['budget'], optional(),
                                  parameter_names=settings['parameters'], anchor_body=True,
                                  pose_parameters=request.get('pose_parameters'),
                                  checkpoint=out/'progress.json', method=request.get('search_method', 'Nelder-Mead'))
            with np.load(result['mesh']) as mesh:
                result['cpu_source_metrics'], _ = evidence.score(mesh['V'], mesh['F'], result['pose'],
                    observed, model_parts=result['build'].get('parts'))
            result = unapproved(result, obj)
        changed = [path for path, sha in inputs.items() if digest(path) != sha]
        if changed:
            raise ValueError('Rigid proposal inputs changed: '+', '.join(changed))
        write_json(out/'result.json', result)
        return result, out
    except Exception as error:
        write_json(out/'failure.json', unapproved({'error': str(error)}, obj))
        raise


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('object')
    args = parser.parse_args()
    _, out = run(Session(args.session), args.object)
    print('Unapproved rigid initialization:', out)


if __name__ == '__main__':
    main()
