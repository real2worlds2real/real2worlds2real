"""Select task-object reconstruction explicitly; agent-authored Blender models are the default."""
import argparse
import copy
import json

from calibrated_scene_reconstruction.core.session_code import source_path
from calibrated_scene_reconstruction.scene.session import Session

ROUTES = ('agent_blender', 'template')


def task_objects(graph):
    dynamic = graph.get('dynamic_objects', {})
    task = graph.get('task') or {}
    names = {task[k] for k in ('target', 'goal') if isinstance(task.get(k), str)}
    names.update(n for n, spec in dynamic.items() if spec.get('fit'))
    return sorted((names & dynamic.keys()) or dynamic.keys())


def routed_graph(graph, route='agent_blender'):
    if route not in ROUTES:
        raise ValueError(f'Unknown object route: {route}')
    result = copy.deepcopy(graph)
    task = result.get('task') or {}
    dynamic = result.setdefault('dynamic_objects', {})
    static = result.get('static_objects', {})
    for key in ('target', 'goal'):
        name = task.get(key)
        if isinstance(name, str) and name in static and name not in dynamic:
            dynamic[name] = static.pop(name)
    requests = []
    for name in task_objects(result):
        obj = result['dynamic_objects'][name]
        fit = dict(obj.get('fit') or {})
        if fit.get('kit') and fit.get('model'):
            raise ValueError(f'{name}: fit.kit and fit.model are mutually exclusive')
        selected = 'template' if route == 'template' else fit.get('route', route)
        if selected not in ROUTES:
            raise ValueError(f'{name}: unknown fit.route {selected!r}')
        if selected == 'template':
            fallback = fit.get('template_fallback') or {}
            kit = fit.get('kit') or fallback.get('kit')
            if not kit:
                raise ValueError(f'{name}: explicit template fallback requires a kit')
            if fit.get('model'):
                fit['agent_model'] = fit.pop('model')
                fit.pop('accepted', None)
                obj.pop('fix', None)
            fit.update(route='template', kit=kit)
            if 'params' in fallback:
                fit['params'] = fallback['params']
        else:
            model = fit.get('model') or fit.get('agent_model') or f'agent/objects/{name}/model.py'
            if fit.get('kit'):
                fit['template_fallback'] = {k: fit[k] for k in ('kit', 'params') if k in fit}
                fit.pop('kit', None)
                fit.pop('params', None)
                fit.pop('accepted', None)
                obj.pop('fix', None)
            fit['model'] = model
        fit.setdefault('prompt', obj.get('_label', name))
        obj['fit'] = fit
        requests.append({'object': name, 'route': selected, 'model': fit.get('model')})
    return result, requests


def ensure(session, route='agent_blender', *, apply=False, require_reviewed=False):
    original = session.config('scene_graph.json')
    graph, requests = routed_graph(original, route)
    if apply and graph != original:
        if (session / 'export').exists():
            raise ValueError('Object route would change a delivered session; create a new session first')
        path = session.config_dir / 'scene_graph.json'
        path.write_text(json.dumps(graph, indent=1, ensure_ascii=False) + '\n')
    missing = []
    for req in requests:
        if req['route'] == 'agent_blender':
            req['ready'] = source_path(session.config_dir, req['model']).is_file()
            if not req['ready']:
                missing.append(req['object'])
    if apply:
        from calibrated_scene_reconstruction.scene.capture.supplementary import evidence as supplemental_evidence
        session.write_json('qa/model_requests.json', {
            'default_route': route, 'objects': requests,
            'mechanism_declaration': str(source_path(session.config_dir, 'mechanisms.json')),
            'physical_priors': str(source_path(session.config_dir, 'physics_priors.json')),
            'object_understanding': str(session.config_dir/'agent/object_understanding.json'),
            'intrinsic_object_requirement': 'Before fitting, inventory all semantic parts, source-visible geometry, material alternatives, expected response, connections and chosen physical links in agent/object_understanding.json. Include task-secondary parts. OBJ and delivery require current per-part semantic/geometry/representation judgments and independent measured Isaac Sim5.1 passive, contact and material response evidence. Task success cannot substitute for these checks.',
            'tool_selection_requirement': 'Blender remains the default modeling tool. When unsuitable for a part, the agent may use an appropriate external geometry/CAD tool or native physical representation, recording the reason and snapshotting its generated data through the model asset contract. Final simulation and task feasibility always run in Isaac Sim5.1 with the actual delivered geometry and physical representation; another simulator or a rendered animation cannot approve delivery.',
            'physical_observation_requirement': 'For new tasks set physical_observation_contract:1 in task.json and observation_contract:1 with source-backed passive-shape obligations in mechanisms.json. Validate actual native source-state agreement before task actuation; retain material uncertainty.',
            'exposure_requirement': 'Inspect source color metadata and HDR exposure alternates before deciding part connectivity, geometry or material response. Resolve source-bound exposure reviews; refresh affected primary images and labels. Bright merged regions are not evidence of connected parts.',
            'supplemental_evidence': supplemental_evidence(session, require_review=False),
            'next': 'Agent declares parts/mechanisms, source-observed physical behavior and physical priors (also rigid-only), writes model.py, fits metric silhouettes and functional features, validates unforced source-state agreement, tests operation and prior uncertainty, reviews evidence, revises and repeats; templates require explicit selection.',
        })
    if missing:
        raise ValueError('Agent Blender models required for ' + ', '.join(missing) +
                         '; see qa/model_requests.json and docs/twin_agent.md. No template fallback was run.')
    if require_reviewed:
        from calibrated_scene_reconstruction.scene.objects.blender.export import verify_round
        from calibrated_scene_reconstruction.scene.twin.reviews import object_state
        for req in requests:
            if req['route'] == 'agent_blender':
                state = object_state(session, req['object'], graph['dynamic_objects'][req['object']], verify_round)
                if not state['passed']:
                    raise ValueError(f"OBJ.{req['object']}: {state['reason']}; complete the Blender iteration before delivery")
    return graph


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('session')
    parser.add_argument('--object-route', choices=ROUTES, default='agent_blender')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--require-reviewed', action='store_true')
    args = parser.parse_args()
    try:
        ensure(Session(args.session), args.object_route, apply=args.apply, require_reviewed=args.require_reviewed)
    except ValueError as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
