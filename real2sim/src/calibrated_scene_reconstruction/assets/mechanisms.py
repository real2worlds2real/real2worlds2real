"""Physical mechanism declarations shared by asset authoring and simulation.

Declarations describe the intended representation, not a claim that it passed a
task. Unobserved dimensions/materials remain assumptions, never measurements.
"""
import hashlib
import json
import re

import numpy as np
import re


KINDS = {'rigid', 'deformable', 'cloth', 'rope'}
TYPES = {'revolute', 'prismatic', 'fixed', 'contact', 'deformation'}
ROLES = {'primary', 'related', 'secondary'}
DECLARATION_FIELDS = {'version', 'objects', 'mechanisms', 'open_questions', 'material_state_contract',
                      'observation_contract', 'observations', 'parameter_constraints'}


def backend(spec):
    kind=spec['kind']
    allowed={'cloth':{'particle','surface_fem'},'deformable':{'legacy','volume_fem'},'rope':{'legacy','volume_fem'}}
    if 'backend' in spec and spec['backend'] not in allowed.get(kind,set()):
        raise ValueError('Physical backend must match its cloth or volume kind')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def links(value):
    """Validate Blender mesh ownership; each moving part has its own body."""
    if not isinstance(value, dict) or not value:
        raise ValueError('LINKS must be a nonempty dictionary')
    owned = set()
    for name, spec in value.items():
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Link names must be valid USD identifiers')
        if not isinstance(spec, dict) or spec.get('kind') not in KINDS:
            raise ValueError(f'{name}: an explicit physical kind is required')
        backend(spec)
        meshes = spec.get('meshes')
        if not isinstance(meshes, list) or not meshes or any(not isinstance(m, str) or not m for m in meshes):
            raise ValueError(f'{name}: nonempty mesh names required')
        if len(set(meshes)) != len(meshes) or owned.intersection(meshes):
            raise ValueError('A mesh cannot belong to multiple physical links')
        owned.update(meshes)
    return value


def validate(value):
    """Session-wide links may span objects, e.g. two independent mating bodies."""
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Mechanism contract version must be 1')
    objects = value.get('objects')
    if not isinstance(objects, dict) or not objects:
        raise ValueError('Declare every task object and its physical links')
    if 'material_state_contract' in value and (type(value['material_state_contract']) is not int
                                               or value['material_state_contract'] not in {1,2}):
        raise ValueError('Material state contract version must be1 or2')
    refs, kinds = set(), {}
    for obj, spec in objects.items():
        body_links = spec.get('links', {})
        if not isinstance(body_links, dict) or not body_links:
            raise ValueError(f'{obj}: links required (also for a single rigid body)')
        for name, body in body_links.items():
            if body.get('kind') not in KINDS:
                raise ValueError(f'{obj}/{name}: physical kind required')
            backend(body)
            refs.add(f'{obj}/{name}')
            kinds[f'{obj}/{name}'] = body['kind']
            effects=body.get('required_contact_effects',[])
            if (not isinstance(effects,list)
                    or any(not isinstance(e,str) or e not in {'rigid_friction','self_friction','surface_pair_friction'} for e in effects)
                    or len(set(effects))!=len(effects)
                    or effects and body['kind']!='cloth'):
                raise ValueError('Declared cloth contact effects require supported names and independent native controls')
            support=body.get('initial_support')
            if support is not None:
                if (body['kind']!='cloth' or not isinstance(support,dict)
                        or set(support)-{'link','relation','up','source','surface_region','material_contact','connected_folds'}
                        or not {'link','relation','up','source','surface_region'}<=set(support)
                        or support['surface_region'] not in {'all_upward','lowest_upward_patch','highest_upward_patch'}
                        or support['relation']!='above_unthreaded' or not isinstance(support['link'],str) or support['link']==name
                        or body_links.get(support['link'],{}).get('kind')!='rigid'
                        or not isinstance(support['source'],str) or not support['source'].strip()):
                    raise ValueError('Initial cloth support requires a real rigid link, explicit relation and evidence source')
                direction=np.asarray(support['up'],float)
                if direction.shape!=(3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction),1.):
                    raise ValueError('Initial support requires a unit up direction in model coordinates')
                if 'material_contact' in support:
                    from .support_route import validate_contact
                    validate_contact(support['material_contact'])
                if 'connected_folds'in support:
                    from .material_support import validate_fold_support
                    if 'material_contact'not in support:raise ValueError('Connected-fold support requires direct-contact evidence')
                    validate_fold_support(support['connected_folds'])
        connections=spec.get('surface_connections',[])
        if not isinstance(connections,list):raise ValueError('Surface connections must be a list')
        connected=set()
        for bond in connections:
            if (not isinstance(bond,dict) or set(bond)!={'surface','rigid','basis'}
                    or not isinstance(bond['basis'],str) or not bond['basis'].strip()
                    or body_links.get(bond['surface'],{}).get('backend','particle') not in {'surface_fem','particle'}
                    or body_links.get(bond['surface'],{}).get('kind')!='cloth'
                    or body_links.get(bond['rigid'],{}).get('kind')!='rigid'
                    or bond['surface'] in connected):raise ValueError('Declare unique real surface-to-rigid root connections')
            connected.add(bond['surface'])
    questions = value.get('open_questions', [])
    if not isinstance(questions, list):
        raise ValueError('Mechanism open_questions must be a list')
    question_ids = set()
    for question in questions:
        name = question.get('id') if isinstance(question, dict) else None
        if (not isinstance(name, str) or not name.strip() or name in question_ids
                or question.get('role') not in ROLES
                or question.get('status') not in {'open', 'resolved'}
                or any(not isinstance(question.get(key), str) or not question[key].strip()
                       for key in ('question', 'impact', 'source'))):
            raise ValueError('Open questions need unique ids, task roles, status, source and modeling impact')
        question_ids.add(name)
        if question['status'] == 'resolved':
            resolution = question.get('resolution', {})
            if (not isinstance(resolution, dict) or any(not isinstance(resolution.get(key), str)
                    or not resolution[key].strip() for key in ('answer', 'source'))):
                raise ValueError('A resolved mechanism question requires an answer and its evidence source')
    mechanisms = value.get('mechanisms')
    if not isinstance(mechanisms, list):
        raise ValueError('mechanisms must be a list; use [] for rigid-only tasks')
    ids = set()
    for m in mechanisms:
        name = m.get('id')
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError('Mechanism ids must be unique nonempty strings')
        ids.add(name)
        kind = m.get('type')
        if kind not in TYPES or m.get('role') not in ROLES:
            raise ValueError(f'{name}: physical type and task role required')
        bodies = m.get('links', [])
        if (not isinstance(bodies, list) or not bodies or len(set(bodies)) != len(bodies)
                or any(b not in refs for b in bodies)):
            raise ValueError(f'{name}: unknown or repeated link')
        if kind != 'deformation' and len(bodies) != 2:
            raise ValueError(f'{name}: two distinct links required')
        representation = m.get('representation')
        expected = 'surface_contact' if kind == 'contact' else ('native_deformable' if kind == 'deformation' else 'native_joint')
        if representation != expected:
            raise ValueError(f'{name}: {kind} requires {expected}, not pose playback or a substitute joint')
        evidence = m.get('evidence')
        if not isinstance(evidence, dict) or evidence.get('status') not in {'observed', 'inferred', 'unknown'} or not evidence.get('source'):
            raise ValueError(f'{name}: observed/inferred/unknown evidence and source required')
        checks = m.get('checks')
        if not isinstance(checks, list) or not checks or any(not isinstance(c, str) or not c for c in checks):
            raise ValueError(f'{name}: named validation checks required')
        if kind in {'revolute', 'prismatic'}:
            axis, limits = np.asarray(m.get('axis', []), float), np.asarray(m.get('limits', []), float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-6:
                raise ValueError(f'{name}: normalized joint axis required')
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError(f'{name}: finite ordered limits required (radians or metres)')
            required = {'geometry', 'relative_motion', 'limits', 'attachment'}
            if not required.issubset(checks):
                raise ValueError(f'{name}: real joint validation needs {sorted(required)}')
            if 'initial_state' in m:
                from .joint_state import interval
                interval(m,limits)
        approximate=m.get('response_accuracy','quantitative')=='task_and_semantics'
        if (m.get('response_accuracy','quantitative') not in {'quantitative','task_and_semantics'}
                or type(m.get('requires_deformation',False)) is not bool):
            raise ValueError('Explicit supported response accuracy and boolean deformation requirement needed')
        if kind == 'deformation':
            if not any(kinds[b] in {'cloth','deformable','rope'} for b in bodies):
                raise ValueError('A deformation mechanism needs an actual flexible link')
            required={'geometry','task_interaction'}
            if not approximate or m['role']=='primary' or m.get('requires_deformation',False):required.add('solver_deformation')
            else:required|={'native_flexible_parts','physical_plausibility','contact_preserved'}
            if not required.issubset(checks):
                raise ValueError(f'{name}: deformation requires {sorted(required)} checks')
        if kind == 'deformation' and value.get('material_state_contract') and any(kinds[b] == 'cloth' for b in bodies):
            needed = {'rest_metric','no_initial_interpenetration'}
            needed |= {'physical_plausibility','contact_preserved'} if approximate else {'material_area','strain_bounded'}
            if value['material_state_contract'] == 2:
                needed.add('no_initial_self_intersection')
                needed.add('self_contact_recorded' if approximate and m['role']!='primary' else 'no_self_intersection')
            if not needed.issubset(checks):
                raise ValueError(f'{name}: cloth material-state contract requires {sorted(needed)}')
        if m.get('subtype'):
            from calibrated_scene_reconstruction.core.capabilities import invoke
            invoke('mechanism', m['subtype'], m)
    digest(value)  # Reject NaN/Infinity anywhere, including assumptions.
    if 'observations' in value and 'observation_contract' not in value:
        raise ValueError('Observations require physical observation contract1')
    if 'observation_contract' in value:
        if type(value['observation_contract']) is not int or value['observation_contract'] != 1:
            raise ValueError('Physical observation contract must be1')
        observations = value.get('observations')
        if not isinstance(observations, list) or not observations:
            raise ValueError('Declare independently observed physical behavior before task checks')
        observation_ids = set()
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError('Observed behavior must be an object')
            name = observation.get('id')
            mechanism = next((m for m in mechanisms if m['id'] == observation.get('mechanism')), None)
            files = observation.get('evidence_files')
            if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in observation_ids or mechanism is None
                    or observation.get('kind') != 'passive_shape'
                    or observation.get('link') not in mechanism['links']
                    or observation.get('check') not in mechanism['checks']
                    or not observation.get('source') or not observation.get('basis')
                    or not isinstance(files, dict) or not files
                    or any(not isinstance(path, str) or not path or not isinstance(sha, str)
                           or re.fullmatch(r'[0-9a-f]{64}', sha) is None for path, sha in files.items())):
                raise ValueError('Observed behavior needs unique identity, mechanism/link/check and source evidence')
            observation_ids.add(name)
            if 'link_group' in observation:
                group=observation['link_group']
                if (not isinstance(group,list) or not group or len(set(group))!=len(group)
                        or observation['link'] not in group
                        or any(ref not in kinds or ref.split('/')[0]!=observation['link'].split('/')[0] for ref in group)):
                    raise ValueError('Observed semantic group must contain actual same-assembly links and its anchor')
            for key in ('part_prompt','exterior_to','part_review'):
                if key in observation and (not isinstance(observation[key],str) or not observation[key].strip()):
                    raise ValueError('Independent part observations need an explicit semantic prompt')
            if observation.get('exterior_to') and not observation.get('part_prompt'):
                raise ValueError('Exterior comparison requires an independently labelled target part')
            if observation.get('part_review') and not observation.get('part_prompt'):
                raise ValueError('Source part review requires an independently acquired semantic prompt')
            if 'transmitting_links' in observation or 'transmission_basis' in observation:
                refs=observation.get('transmitting_links')
                if (not isinstance(refs,list) or not refs
                        or any(not isinstance(ref,str) or ref not in kinds or ref==observation['link']
                               or ref.split('/')[0]!=observation['link'].split('/')[0] for ref in refs)
                        or len(set(refs))!=len(refs)
                        or not isinstance(observation.get('transmission_basis'),str)
                        or not observation['transmission_basis'].strip() or not observation.get('part_prompt')):
                    raise ValueError('Optical transmission needs source evidence/basis and distinct same-assembly physical links')
            duration, tolerance = observation.get('duration_s'), observation.get('max_projection_motion_px')
            if (type(duration) not in (int, float) or type(tolerance) not in (int, float)
                    or not np.isfinite([duration, tolerance]).all() or not .25 <= duration <= 10
                    or not 0 < tolerance <= 20):
                raise ValueError('Declare a bounded passive observation and source-based pixel-motion tolerance')
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke, require
    for key in value.keys() - DECLARATION_FIELDS:
        require('mechanism_fields', key)
    for name in entries('mechanism_validators'):
        invoke('mechanism_validators', name, value)
    return value


def joints(value, bodies):
    if not isinstance(value, list):
        raise ValueError('JOINTS must be a list')
    pairs = set()
    for joint in value:
        a, b, kind = joint.get('parent'), joint.get('child'), joint.get('type')
        if kind not in {'fixed', 'revolute', 'prismatic'} or a == b or a not in bodies or b not in bodies:
            raise ValueError('JOINTS require supported real joint types and distinct declared links')
        if bodies[a]['kind'] != 'rigid' or bodies[b]['kind'] != 'rigid':
            raise ValueError('Flexible attachments require native attachment support, never a rigid joint substitute')
        pair = frozenset((a, b))
        if pair in pairs:
            raise ValueError('Only one joint may connect a pair of links')
        pairs.add(pair)
        for k in ('anchor0', 'anchor1'):
            point = np.asarray(joint.get(k, []), float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Each joint needs finite link-local anchors')
        if kind != 'fixed':
            axis, limits = np.asarray(joint.get('axis', []), float), np.asarray(joint.get('limits', []), float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-6:
                raise ValueError('Normalized joint axis required')
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Finite ordered joint limits required')
            initial=joint.get('initial_position',0.)
            if (type(initial) not in (int,float) or not np.isfinite(initial)
                    or not limits[0]<=initial<=limits[1]):
                raise ValueError('Initial joint position must be finite and inside its actual limits')
            if 'frame0' in joint and 'initial_position' not in joint:
                raise ValueError('Observed-state joint frames require an explicit initial position')
        joint_frames(joint)
    return value


def quaternion_matrix(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or not np.isclose(q@q,1.,atol=1e-6):
        raise ValueError('Joint frame requires a normalized finite wxyz quaternion')
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def joint_frames(joint):
    """Link-local frames preserve a captured nonzero articulation state.

    Geometry may be authored in its observed state in a common asset frame.
    Distinct joint frames encode its real coordinate without a target drive,
    animation, pose overwrite, or an incorrect change to the closed limit.
    """
    axis=np.asarray(joint.get('axis',[1.,0.,0.]),float)
    if ('frame0' in joint)!=('frame1' in joint):
        raise ValueError('Declare both link-local joint frames')
    if 'frame0' in joint:
        frames=[np.asarray(joint['frame'+str(k)],float) for k in [0,1]]
    else:
        q=np.r_[1+axis[0],0.,-axis[2],axis[1]]
        q=q/np.linalg.norm(q) if np.linalg.norm(q)>1e-8 else np.array([0.,0.,0.,1.])
        frames=[q,q.copy()]
    for frame in frames:
        rotation=quaternion_matrix(frame)
        if joint['type']!='fixed' and not np.allclose(rotation[:,0],axis,atol=1e-6):
            raise ValueError('Joint frame X axis differs from declared physical joint axis')
    return frames


def revolute_position(parent_rotation,child_rotation,frame0,frame1):
    relative=quaternion_matrix(frame0).T@np.asarray(parent_rotation).T@np.asarray(child_rotation)@quaternion_matrix(frame1)
    return float(np.arctan2(relative[2,1]-relative[1,2],relative[1,1]+relative[2,2]))


def joint_components(bodies, value):
    """Connected native joint trees, excluding independent rigid/flexible bodies.

    Sharing an authored asset frame does not attach a loose part to its joints.
    Reduced-coordinate articulations cannot contain a closed joint cycle.
    """
    joints(value, bodies)
    neighbors = {name: set() for name in bodies}
    parents = {name: name for name in bodies}

    def find(name):
        while parents[name] != name:
            name = parents[name]
        return name

    children = set()
    for joint in value:
        a, b = joint['parent'], joint['child']
        ra, rb = find(a), find(b)
        if ra == rb:
            raise ValueError('Native reduced-coordinate articulations require joint trees, not cycles')
        parents[rb] = ra
        neighbors[a].add(b)
        neighbors[b].add(a)
        children.add(b)
    components, seen = [], set()
    for name in bodies:
        if not neighbors[name] or name in seen:
            continue
        pending, members = [name], set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(neighbors[current] - members)
        ordered = [body for body in bodies if body in members]
        root = next((body for body in ordered if body not in children), ordered[0])
        components.append({'links': ordered, 'root': root})
        seen.update(members)
    return components


def check_results(contract, results, *, relaxed_observations=(), relaxed_questions=()):
    """Every task-relevant mechanism and named check must pass independently."""
    validate(contract)
    results = results if isinstance(results, dict) else {}
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    failures = []
    for name in entries('mechanism_results'):
        failures.extend(invoke('mechanism_results', name, contract, results))
    relaxed_observations = set(relaxed_observations)
    declared_observations = {o['id']: o for o in contract.get('observations', [])}
    if not relaxed_observations <= set(declared_observations):
        raise ValueError('Only explicitly declared physical observations can be conditionally relaxed')
    relaxed_questions=set(relaxed_questions)
    if not relaxed_questions<={q['id'] for q in contract.get('open_questions',[])}:
        raise ValueError('Only explicitly declared modeling questions can be conditionally relaxed')
    for question in contract.get('open_questions', []):
        if question['role'] != 'secondary' and question['status'] != 'resolved' and question['id'] not in relaxed_questions:
            failures.append(question['id'] + ': task-relevant mechanism question unresolved')
    for observation in contract.get('observations', []):
        if observation['id'] in relaxed_observations:
            continue  # The caller retains the raw failed result and its input acceptance proof.
        observed = results.get(observation['mechanism'], {}).get('checks', {})
        result = observed.get('observation:'+observation['id'], observed.get(observation['check'], {}))
        if (result.get('pass') is not True or result.get('observation_id') != observation['id']
                or result.get('native_states_read') is not True or result.get('external_actuation') is not False
                or result.get('source_evidence_sha256') != observation['evidence_files']):
            failures.append(observation['id']+': physical behavior does not match source observations')
    for m in contract['mechanisms']:
        if m['role'] == 'secondary':
            continue
        result = results.get(m['id'], {})
        checks = result.get('checks', {})
        waived = {o['check'] for key,o in declared_observations.items()
                  if key in relaxed_observations and o['mechanism'] == m['id']}
        if ((result.get('pass') is not True and not waived) or result.get('representation') != m['representation']
                or any(checks.get(c, {}).get('pass') is not True for c in m['checks'] if c not in waived)):
            failures.append(m['id'])
        if m['type'] == 'contact' and (result.get('prescribed_motion') is not False
                                       or result.get('contact_observed') is not True):
            failures.append(m['id'] + ': contact evidence missing or motion prescribed')
        if m['type'] == 'deformation' and result.get('solver_deformation_observed') is not True:
            qualitative=(m.get('response_accuracy')=='task_and_semantics' and m['role']!='primary'
                         and not m.get('requires_deformation',False)
                         and result.get('native_flexible_parts') is True
                         and result.get('physical_plausibility_pass') is True
                         and checks.get('task_interaction',{}).get('pass') is True)
            if not qualitative:failures.append(m['id'] + ': physical deformation not demonstrated')
    return {'pass': not failures, 'failures': failures}


def parameter_constraints(contract, parameters, *, require_all=True):
    """Check shared functional geometry across separately fitted objects.

    An operand is a parameter reference, or a list of references whose product
    expresses a metric dimension (for example unit*radius_ratio). No arbitrary
    expression evaluation or unbound reference dimension is allowed.
    """
    for constraint in contract.get('parameter_constraints', []):
        refs = constraint.get('parameters', [])
        factors = [[r] if isinstance(r,str) else r for r in refs]
        if (not refs or any(not isinstance(group,list) or not group or
                any(not isinstance(r,str) or r.count('/')!=1 or not all(r.split('/'))
                    for r in group) for group in factors)):
            raise ValueError('Parameter constraints require object/parameter references')
        values = []
        missing=False
        for group in factors:
            value=1.
            for ref in group:
                obj, key = ref.split('/')
                if obj not in parameters:
                    if require_all:
                        raise ValueError('Missing fitted object for shared mechanism: '+obj)
                    missing=True;break
                if key not in parameters[obj]:
                    raise ValueError('Missing shared mechanism parameter: '+ref)
                v=parameters[obj][key]
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):
                    raise ValueError('Shared mechanism parameters must be finite scalars')
                value*=v
            if missing:break
            values.append(value)
        if len(values) != len(refs):
            continue
        if not np.isfinite(values).all():
            raise ValueError('Shared mechanism parameters must be finite')
        kind = constraint.get('type')
        if kind == 'equal':
            tolerance = constraint.get('atol', 1e-8)
            if not np.isfinite(tolerance) or tolerance < 0 or max(values)-min(values) > tolerance:
                raise ValueError('Shared mechanism parameters disagree: '+', '.join('*'.join(g) for g in factors))
        elif kind == 'difference' and len(values) == 2:
            lo, hi = constraint['bounds']
            if not np.isfinite([lo, hi]).all() or lo > hi or not lo <= values[0]-values[1] <= hi:
                raise ValueError('Mating clearance outside declared hypothesis: '+', '.join('*'.join(g) for g in factors))
        else:
            raise ValueError('Supported mechanism constraints: equal or bounded difference')
    return {'pass': True}
