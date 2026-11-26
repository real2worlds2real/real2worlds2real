"""Task-independent process plans and observation-based acceptance."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import uuid

import numpy as np


KINDS = {'video', 'robot_trajectory', 'camera', 'observations', 'geometry', 'model_source',
         'model_review', 'material_prior', 'auxiliary'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())




def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)




def valid_observations(values, metric='rmse'):
    values=np.asarray(values,float)
    if metric=='unordered_points':
        require(values.ndim==3 and 1<=values.shape[1]<=256 and 1<=values.shape[2]<=3,
                'Unordered observations require bounded timestamp/point/coordinate arrays')
        finite=np.isfinite(values).all(-1);missing=np.isnan(values).all(-1)
        require(np.all(finite|missing),'A missing point must omit all its coordinates')
        return finite.any(-1)
    return np.isfinite(values.reshape(len(values),-1)).all(1)


def channel_residual(channel, observed, predicted, indices):
    """Residuals for source-visible data only; unordered parts retain no false IDs."""
    observed,predicted=np.asarray(observed,float),np.asarray(predicted,float)
    metric=channel.get('metric','rmse');valid=valid_observations(observed[indices],metric)
    require(valid.sum()>=2,'Insufficient valid comparison timestamps')
    if metric=='unordered_points':
        from scipy.optimize import linear_sum_assignment
        rows=[]
        for i in np.asarray(indices)[valid]:
            target=observed[i,np.isfinite(observed[i]).all(-1)]
            cost=np.sum((target[:,None,:]-predicted[i,None,:,:])**2,axis=-1)
            a,b=linear_sum_assignment(cost)
            rows.append((predicted[i,b]-target[a]).ravel())
        return np.concatenate(rows)
    error=(predicted[indices]-observed[indices]).reshape(len(indices),-1)[valid]
    if metric=='angular':error=np.arctan2(np.sin(error),np.cos(error))
    return error.ravel()


def resolve_inputs(plan, root):
    inputs = plan.get('inputs', {})
    require(isinstance(inputs, dict) and inputs, 'Named input files are required')
    result = {}
    for name, item in inputs.items():
        require(isinstance(name,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*',name),
                'Input identifiers must be simple names, not paths')
        require(isinstance(item, dict) and item.get('kind') in KINDS, 'Unknown input kind: ' + name)
        path = Path(item['path'])
        path = (path if path.is_absolute() else Path(root) / path).resolve()
        forbidden = {'evals', 'attic', 'ground_truth', 'ground-truth'}
        require(not forbidden.intersection(p.lower() for p in path.parts)
                and path.stem.lower() not in {'gt', 'references', 'ground_truth'},
                'Development ground truth cannot be a calibration input')
        require(path.is_file(), 'Missing declared input: ' + str(path))
        result[name] = {'path': str(path), 'kind': item['kind'], 'sha256': digest(path)}
    return result


def validate(plan, root):
    require(plan.get('version') == 1 and bool(plan.get('objective')), 'Process plan version 1 and objective required')
    inputs = resolve_inputs(plan, root)
    require(any(i['kind'] == 'video' for i in inputs.values()), 'Original video evidence required')
    observation = plan.get('observations')
    require(observation in inputs and inputs[observation]['kind'] == 'observations',
            'observations must name a declared observation NPZ')
    observation_review=plan.get('observation_review')
    if observation_review is not None:
        require(observation_review in inputs and inputs[observation_review]['kind']=='auxiliary',
                'Observation review must name a declared auxiliary input')
        reviewed=read(inputs[observation_review]['path'])
        sources={k:v['sha256'] for k,v in inputs.items() if v['kind'] in {'video','observations','camera'}}
        require(reviewed.get('pass') is True and reviewed.get('reason') and reviewed.get('inputs')==sources,
                'Observation review must bind current videos, observations and cameras')
        checks=reviewed.get('checks',{})
        require({'identity','camera','occlusion','timebase'}<=set(checks)
                and all(v.get('pass') is True and v.get('reason') for v in checks.values()),
                'Review observation identity, camera, occlusion and timebase before physical fitting')
    model = plan.get('model', {})
    require(isinstance(model.get('required_checks'), list) and model['required_checks']
            and all(isinstance(c, str) and c for c in model['required_checks']),
            'Agent must plan explicit physical plausibility checks')
    require(all(number(v) and v > 0 for v in model.get('physical_limits', {}).values()),
            'Physical upper limits must be declared finite positive values')
    bodies = model.get('bodies', [])
    require(isinstance(bodies, list) and bodies, 'Agent must declare the physical bodies')
    ids = [b['id'] for b in bodies]
    require(len(set(ids)) == len(ids), 'Duplicate physical body IDs')
    motion=model.get('initial_motion')
    if motion is not None:
        require(isinstance(motion,dict) and motion.get('mode')=='stationary' and motion.get('review') in inputs
                and inputs[motion['review']]['kind']=='auxiliary','Stationary kinetic initialization needs an explicit source review')
        receipt=read(inputs[motion['review']]['path'])
        sources={k:v['sha256'] for k,v in inputs.items() if v['kind'] in {'video','camera','observations'}}
        require(receipt.get('pass') is True and receipt.get('reason') and receipt.get('inputs')==sources,
                'Initial motion review must bind current source video, cameras and observations')
        require('initial_kinetic_state' in model['required_checks'],
                'Stationary kinetic initialization must remain an explicit native audit')
        require(type(motion.get('condition_after_settling',False)) is bool,
                'Source-time-zero conditioning must be an explicit boolean')
        if motion.get('condition_after_settling',False):
            require('source_time_zero' in model['required_checks'],
                    'Source-time-zero conditioning requires its native audit')
    for body in bodies:
        require(body.get('representation') and body.get('material') and body.get('reason'),
                'Every body needs a representation, material hypothesis and reason')
        require(body.get('sources') and all(k in inputs for k in body['sources']),
                'Every body must identify its declared source inputs')
        require(body.get('motion') in {'static', 'dynamic', 'prescribed'}, 'Declare static body, dynamic response or prescribed driver')
        if body['motion'] == 'prescribed':
            driver = body.get('driver_input')
            require(driver in inputs and inputs[driver]['kind'] in {'robot_trajectory', 'auxiliary'},
                    'Prescribed drivers require declared trajectory evidence')
    require(any(b['motion'] == 'dynamic' for b in bodies), 'At least one freely simulated response body required')
    initial_review=model.get('initial_state_review')
    if 'initial_state' in inputs or initial_review is not None:
        require(initial_review in inputs and inputs[initial_review]['kind']=='auxiliary',
                'Reused initial states require a source-grounded per-part initial-state review')
        receipt=read(inputs[initial_review]['path'])
        evidence=model.get('initial_evidence',[])
        require(all(k in inputs for k in evidence),'Initial-state evidence must name declared inputs')
        bound={k:v['sha256'] for k,v in inputs.items()
               if v['kind'] in {'video','camera','geometry','model_source','observations'} or k in evidence}
        require(receipt.get('pass') is True and receipt.get('reason') and receipt.get('inputs')==bound,
                'Initial-state review must bind current source observations, cameras and geometry/model inputs')
        dynamic={b['id'] for b in bodies if b['motion']=='dynamic'}
        covered=[]
        for part in receipt.get('parts',[]):
            require(part.get('bodies') and set(part['bodies'])<=dynamic and part.get('reason'),
                    'Initial-state review must identify physical parts and their source basis')
            checks=part.get('checks',{})
            require({'pose','structure','visibility','contact'}<=set(checks)
                    and all(c.get('pass') is True and c.get('reason') for c in checks.values()),
                    'Review each initial part pose, structure, visibility and contact; centroids alone do not qualify a scene')
            covered.extend(part['bodies'])
        require(set(covered)==dynamic and len(covered)==len(set(covered)),
                'Initial-state review must cover every dynamic part exactly once, with grouped flexible parts allowed')
        require('initial_state_source' in model['required_checks'],
                'Native initial-state source agreement must remain an explicit execution check')
    interactions = model.get('interactions', [])
    for interaction in interactions:
        require(interaction.get('mechanism') and interaction.get('reason')
                and set(interaction.get('bodies', [])) <= set(ids) and interaction.get('bodies'),
                'Interactions must name known bodies and explain the physical mechanism')
    params = plan.get('parameters', {})
    require(isinstance(params, dict), 'parameters must be a named mapping')
    for name, p in params.items():
        bounds = p.get('bounds', [])
        require(len(bounds) == 2 and all(number(v) for v in bounds) and bounds[0] < bounds[1]
                and number(p.get('initial')) and bounds[0] <= p['initial'] <= bounds[1],
                'Finite ordered prior bounds containing the initial value required: ' + name)
        require(p.get('unit') and p.get('reason') and p.get('source') in
                {'material_prior', 'observed_motion', 'assumption'}, 'Parameter units and prior basis required: ' + name)
        require(p.get('identification') in {'fit', 'prior_only'}, 'Specify fit or prior_only: ' + name)
        require(p.get('targets') and set(p['targets']) <= set(ids), 'Parameter must target known bodies: ' + name)
        require(not p.get('log') or bounds[0] > 0, 'Log parameter must have positive bounds')
        require(name not in {'twin.depth_scale', 'twin.support_dz'},
                'Geometry changes require a new modeling and review round')
    sim = plan.get('simulation', {})
    require(sim.get('engine') == 'isaacsim' and str(sim.get('version')) == '5.1',
            'Final process reproduction requires Isaac Sim 5.1')
    require(sim.get('purpose','validation') in {'screening','validation'},'Explicit supported simulation purpose required')
    require(sim.get('tensor_backend','numpy') in {'numpy','torch'} and sim.get('device','cpu') in {'cpu','cuda:0'},
            'Explicit supported native tensor backend/device required')
    require(sim.get('device','cpu')=='cpu' or sim.get('tensor_backend')=='torch','CUDA tensor readback requires torch')
    require(sim.get('runtime','isaac_world') in {'isaac_world','physics_only'},'Unsupported Isaac runtime')
    if sim.get('runtime')=='physics_only':
        require(sim.get('tensor_backend','numpy')=='numpy' and sim.get('device','cpu')=='cpu',
                'Physics-only runtime exposes CPU readback while preserving native GPU dynamics')
    require(number(sim.get('dt')) and 1 / 2000 <= sim['dt'] <= 1 / 30
            and number(sim.get('duration')) and sim['dt'] <= sim['duration'] <= 300,
            'Bounded native simulation dt and duration required')
    if sim.get('purpose')=='screening' or 'validation_dt' in sim:
        require(number(sim.get('validation_dt')) and 1/2000<=sim['validation_dt']<=sim['dt'],
                'Screening needs an explicit finer validation timestep')
    require(number(sim.get('settle_seconds',0.)) and 0 <= sim.get('settle_seconds',0.) <= 5.,
            'Initial settling must be explicit and bounded')
    runner = sim.get('runner')
    resources=sim.get('resources')
    if resources is not None:
        require(isinstance(resources,dict) and all(type(resources.get(k)) is int and 1024<=resources[k]<=131072
                for k in ('expected_vram_mib','shared_headroom_mib'))
                and resources['shared_headroom_mib']>=4096,'Explicit resource estimates and at least 4096 MiB shared headroom required')
        basis=resources.get('basis_input')
        require(basis in inputs and inputs[basis]['kind']=='auxiliary','Bind a measured runtime resource profile before changing admission estimates')
    require(runner in inputs and inputs[runner]['kind'] == 'model_source', 'Native model runner source required')
    require(model.get('review') in inputs and inputs[model['review']]['kind'] == 'model_review',
            'A source-bound model review is required')
    review = read(inputs[model['review']]['path'])
    expected = {k: v['sha256'] for k, v in inputs.items() if v['kind'] in {'model_source', 'geometry'}}
    require(review.get('pass') is True and review.get('inputs') == expected and review.get('reason'),
            'Model review must pass and bind every current geometry and model source')
    checks = review.get('checks', {})
    require({'structure', 'materials', 'mechanisms', 'native_model'} <= set(checks)
            and all(v.get('pass') is True and v.get('reason') for v in checks.values()),
            'Review the model structure, materials, mechanisms and native representation explicitly')
    budget = plan.get('search', {})
    require(type(budget.get('population')) is int and 2 <= budget['population'] <= 128
            and type(budget.get('generations')) is int and 1 <= budget['generations'] <= 200,
            'Bounded population and generation budget required')
    require(type(budget.get('timeout_seconds')) is int and 1 <= budget['timeout_seconds'] <= 86400,
            'Explicit native execution timeout required')
    require(budget['population'] * budget['generations'] <= 2048, 'Use at most 2048 candidates per round')
    require(budget.get('sensitivity_points', 3) in {0, 3, 5}, 'Sensitivity sampling must be 0, 3 or 5 points')
    require(number(budget.get('initial_sigma',.25)) and 0<budget.get('initial_sigma',.25)<=.5,
            'Initial search sigma must be in (0, .5] of the parameter box')
    scales=budget.get('parameter_scales',{})
    require(isinstance(scales,dict) and set(scales)<={k for k,p in params.items() if p['identification']=='fit'}
            and all(number(v) and 0<v<=10 for v in scales.values()),
            'Search scales must be positive finite multipliers for active parameters')
    require(type(budget.get('stop_when_fit_passes',False)) is bool,'Fit-only stopping must be a boolean')
    require(budget.get('objective','feasibility') in {'feasibility','training_loss'},
            'Search objective must use fit-only feasibility or training loss')
    if 'proposals' in budget:
        proposed=budget['proposals'];active={k for k,p in params.items() if p['identification']=='fit'}
        require(active and isinstance(proposed,list) and 1<=len(proposed)<=budget['population']*budget['generations']
                and budget.get('proposals_reason'),'Bounded explicit proposals need a source/model rationale and active parameters')
        for row in proposed:
            require(isinstance(row,dict) and set(row)==active and all(number(v) and params[k]['bounds'][0]<=v<=params[k]['bounds'][1]
                    for k,v in row.items()),'Each proposal must bind exactly the active parameters within their declared bounds')
    require(1 + budget['population']*budget['generations'] + budget.get('sensitivity_points',3)*
            sum(p['identification']=='fit' for p in params.values()) <= 2048, 'Total trial budget exceeds 2048')
    require(type(sim.get('max_trace_values', 20_000_000)) is int
            and 1 <= sim.get('max_trace_values', 20_000_000) <= 100_000_000, 'Bounded native trace budget required')
    channels = plan.get('channels', [])
    require(isinstance(channels, list) and channels and len({c['id'] for c in channels}) == len(channels),
            'Unique observation channels required')
    for c in channels:
        require(c.get('observed') and c.get('predicted') and c.get('unit') and c.get('reason'),
                'Each channel needs observed/predicted keys, units and a reason')
        require(number(c.get('tolerance')) and c['tolerance'] > 0 and number(c.get('weight')) and c['weight'] > 0,
                'Positive channel tolerance and weight required')
        require(c.get('metric','rmse') in {'rmse','angular','unordered_points'},'Unknown observation metric')
    observation_checks=model.get('observation_checks',{})
    require(isinstance(observation_checks,dict),'Observation-check bindings must be a mapping')
    channel_ids={c['id'] for c in channels}
    for name,bound in observation_checks.items():
        require(name in model['required_checks'] and isinstance(bound,list) and bound and set(bound)<=channel_ids,
                'Bind each source-observation check to its explicit scored channels')
    if initial_review is not None:
        require('initial_state_source' in observation_checks,
                'Initial source agreement must be scored as observation channels, not used as a held-out physical candidate filter')
    if model.get('observation_definitions'):
        from .region_observations import validate_bindings
        require(plan.get('observation_review') in inputs,'Region observations require a bound source review')
        validate_bindings(plan,read(inputs[plan['observation_review']]['path']))
    with np.load(inputs[observation]['path'], allow_pickle=False) as data:
        t = np.asarray(data['t'], float)
        require(t.ndim == 1 and len(t) >= 4 and np.isfinite(t).all() and np.all(np.diff(t) > 0)
                and t[0] >= 0 and t[-1] <= sim['duration'], 'Observation timestamps outside native horizon')
        split = plan.get('comparison', {})
        train, held = split.get('fit_indices', []), split.get('validation_indices', [])
        require(len(train) >= 2 and len(held) >= 2 and all(type(i) is int and 0 <= i < len(t) for i in train + held)
                and len(set(train)) == len(train) and len(set(held)) == len(held)
                and not set(train).intersection(held), 'Disjoint fit and held-out observation indices required')
        for c in channels:
            a = np.asarray(data[c['observed']], float)
            require(len(a) == len(t), 'Channel timestamps do not match: ' + c['id'])
            for indices in (train, held):
                require(valid_observations(a[indices],c.get('metric','rmse')).sum() >= 2,
                        'At least two valid samples per channel and split required: ' + c['id'])
    return inputs


def no_relaxation(previous, plan):
    """An unsuccessful model cannot be accepted by dropping or relaxing a measurement."""
    require(plan['objective'] == previous['objective'], 'A new objective requires a new process recording')
    target=previous['simulation'].get('validation_dt')
    if target is not None:
        require(plan['simulation'].get('validation_dt') is not None and plan['simulation']['validation_dt']<=target,
                'Cannot drop or coarsen the declared validation timestep')
        require(plan['simulation'].get('purpose','validation')=='screening' or plan['simulation']['dt']<=target,
                'Final validation must run at the declared finer timestep')
    require(not previous.get('observation_review') or plan.get('observation_review'),
            'Cannot drop an existing observation review')
    require(not previous.get('model',{}).get('initial_state_review') or plan.get('model',{}).get('initial_state_review'),
            'Cannot drop an existing initial-state review')
    require(not previous.get('model',{}).get('initial_motion') or plan.get('model',{}).get('initial_motion'),
            'Cannot drop a reviewed initial kinetic-state policy')
    if previous.get('model',{}).get('initial_motion',{}).get('condition_after_settling',False):
        require(plan.get('model',{}).get('initial_motion',{}).get('condition_after_settling',False),
                'Cannot drop source-time-zero conditioning after a failed round')
    require(set(previous.get('model',{}).get('observation_definitions',{}))<=
            set(plan.get('model',{}).get('observation_definitions',{})),
            'Cannot drop reviewed region observation definitions')
    for name,bound in previous.get('model',{}).get('observation_checks',{}).items():
        require(set(bound)<=set(plan.get('model',{}).get('observation_checks',{}).get(name,[])),
                'Cannot remove scored source-check bindings: '+name)
    current = {c['id']: c for c in plan['channels']}
    for old in previous['channels']:
        new = current.get(old['id'])
        require(new and new['unit'] == old['unit'] and new['tolerance'] <= old['tolerance'],
                'Cannot remove or relax an existing acceptance channel: ' + old['id'])
        require(new.get('metric','rmse')==old.get('metric','rmse'),'Cannot weaken an existing correspondence metric: '+old['id'])
    for split in ('fit_indices','validation_indices'):
        require(set(previous['comparison'][split]) <= set(plan['comparison'][split]),
                'Do not remove or move existing fit/held-out samples; new samples may be added')
    require(set(previous['model']['required_checks']) <= set(plan['model']['required_checks']),
            'Cannot drop a physical plausibility check after a failure')
    for key, old in previous['model'].get('physical_limits', {}).items():
        new = plan['model'].get('physical_limits', {}).get(key)
        require(new is not None and new <= old, 'Cannot relax a physical upper limit: '+key)


def compare(plan, observed, predicted):
    """Score arbitrary measured features, never prescribed simulated object poses."""
    ot, pt = np.asarray(observed['t'], float), np.asarray(predicted['t'], float)
    require(pt.ndim == 1 and len(pt) >= 2 and np.isfinite(pt).all() and np.all(np.diff(pt) > 0)
            and pt[0] <= ot[0] + 1e-8 and pt[-1] >= ot[-1] - 1e-8,
            'Simulation must cover the whole observation horizon; no extrapolation')
    result = {'fit_loss': 0., 'channels': {}, 'pass': True}
    for c in plan['channels']:
        require(c['predicted'] in predicted, 'Unexplained observation channel: ' + c['id'])
        a, b = np.asarray(observed[c['observed']], float), np.asarray(predicted[c['predicted']], float)
        require(a.shape[1:] == b.shape[1:] and len(b) == len(pt) and np.isfinite(b).all(),
                'Invalid native prediction shape or values: ' + c['id'])
        flat = b.reshape(len(pt), -1)
        if c.get('metric')=='angular':flat=np.unwrap(flat,axis=0)
        interp = np.stack([np.interp(ot, pt, v) for v in flat.T], axis=1).reshape(a.shape)
        row = {}
        for split, key in [('fit', 'fit_indices'), ('validation', 'validation_indices')]:
            ids = plan['comparison'][key]
            error=channel_residual(c,a,interp,ids)
            row[split + '_rmse'] = float(np.sqrt(np.mean(error ** 2)))
        row['pass'] = max(row['fit_rmse'], row['validation_rmse']) <= c['tolerance']
        row['unit'], row['tolerance'] = c['unit'], c['tolerance']
        result['channels'][c['id']] = row
        result['fit_loss'] += c['weight'] * (row['fit_rmse'] / c['tolerance']) ** 2
        result['pass'] &= row['pass']
    return result

from calibrated_scene_reconstruction.core.records import write, require
