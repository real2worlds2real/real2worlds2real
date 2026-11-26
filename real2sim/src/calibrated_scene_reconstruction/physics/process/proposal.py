"""A bounded local proposal from completed native trials, never acceptance evidence."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import lsq_linear,minimize

from .contract import digest,read,require,channel_residual
from .search import hard_checks_pass


def perturbations(plan,steps,*,repeat_baseline=False):
    """One independent, bounded physical perturbation per selected quantity.

    Steps are expressed in declared physical units. No observations (including
    held-out frames) are read. Responses still require native execution.
    """
    active={k:p for k,p in plan['parameters'].items() if p['identification']=='fit'}
    require(steps and set(steps)==set(active),'Perturb every selected active quantity explicitly')
    baseline={k:p['initial'] for k,p in active.items()};rows=[baseline.copy()] if repeat_baseline else []
    for name,step in steps.items():
        require(np.isfinite(step) and step!=0,'Nonzero finite physical perturbations required')
        value=baseline[name]+step;lo,hi=active[name]['bounds']
        require(lo<=value<=hi,'Perturbation leaves declared parameter bounds: '+name)
        rows.append({**baseline,name:float(value)})
    return rows


def linear_step(round_directory,*,trust_radius=.15,parameters=None,history_only=False,
                physical_steps=None,objective='training_loss'):
    """Fit a local native response Jacobian using fit frames only.

    Geometry, model and initial conditions are shared within the source-bound
    round. Unmeasured directions and ill-conditioned trial designs are rejected.
    The suggested candidate must be executed and independently checked again.
    """
    root=Path(round_directory);plan=read(root/'plan.json');prefix=None
    require(objective in {'training_loss','feasibility'},'Explicit supported local proposal objective required')
    if history_only:
        from .workflow import intact
        require((root/'started.json').is_file() and intact(root),
                'Unsealed history proposals require an unchanged, started worker round')
        raw=(root/'history.jsonl').read_bytes();end=raw.rfind(b'\n')+1
        require(end>0,'At least one complete worker history record required')
        prefix=raw[:end];result={'history':[json.loads(line) for line in prefix.splitlines()]}
    else:
        result=read(root/'result.json')
        require(result.get('engine',{}).get('name')=='isaacsim' and str(result['engine']['version']).startswith('5.1'),
                'Completed native Isaac Sim 5.1 trials required')
    active=[k for k,p in plan['parameters'].items() if p['identification']=='fit']
    names=active if parameters is None else list(parameters)
    require(names and len(names)==len(set(names)) and set(names)<=set(active),
            'Local proposal parameters must be a nonempty active subset')
    require(names and 0<trust_radius<=.5,'Active parameters and a bounded local trust radius required')
    observations=np.load(root/'observations.npz');indices=plan['comparison']['fit_indices']
    rows=[];sections=[]
    def residual(predicted):
        parts=[];offset=0;current=[]
        for c in plan['channels']:
            target=observations[c['observed']][indices]
            response=predicted[c['predicted']][indices]
            require(response.shape==target.shape,'Native/observed feature shape differs')
            # Equal channel weighting, as in compare(); no held-out sample here.
            error=channel_residual(c,observations[c['observed']],predicted[c['predicted']],indices)/c['tolerance']
            parts.append(error*np.sqrt(c['weight']/len(error)))
            current.append((c['id'],slice(offset,offset+len(error)),c['weight']));offset+=len(error)
        if not sections:sections.extend(current)
        return np.concatenate(parts)
    def ratios(r):return np.array([np.linalg.norm(r[section])/np.sqrt(weight) for _,section,weight in sections])
    for row in result['history']:
        if row.get('error') or not row.get('artifacts') or not row.get('physical_checks'):continue
        if not hard_checks_pass(plan,row['physical_checks']):continue
        artifacts=row['artifacts']
        paths=[p for p in artifacts if p.endswith('predictions.npz')]
        require(len(paths)==1,'Each native trial must bind one prediction artifact')
        path=(root/paths[0]).resolve();require(path.is_relative_to(root.resolve()) and digest(path)==artifacts[paths[0]],
            'Native prediction artifact changed or escaped its round')
        predicted=np.load(path)
        require(np.array_equal(predicted['t'],observations['t']),'Native trial timebase changed')
        rows.append((row,residual(predicted)))
    require(len(rows)>=len(names)+1,'Need at least one independent native perturbation per active quantity')
    fixed=set(plan['parameters'])-set(names)
    require(all(np.isclose(row['parameters'][k],rows[0][0]['parameters'][k],rtol=0,atol=1e-12)
                for row,_ in rows for k in fixed),
            'Unmodeled parameter directions changed; use all varied quantities in the local proposal')
    if objective=='feasibility':
        from .fit_priority import score
        def priority(item):
            values=ratios(item[1]);return score({'fit_loss':float(item[1]@item[1]),
                'channels':{key:{'fit_rmse':float(value),'tolerance':1.} for (key,_,_),value in zip(sections,values)}})
        base=min(rows,key=priority)
    else:base=min(rows,key=lambda r:np.sum(r[1]**2))
    row0,r0=base
    bounds=np.array([plan['parameters'][k]['bounds'] for k in names],float)
    widths=np.diff(bounds).ravel();x0=np.array([row0['parameters'][k] for k in names])
    steps=widths*trust_radius
    if physical_steps is not None:
        require(set(physical_steps)==set(names),'Declare one physical trust step for every proposed parameter')
        steps=np.array([physical_steps[k] for k in names],float)
        require(np.isfinite(steps).all() and (steps>0).all() and (steps<=widths*.5).all(),
                'Physical trust steps must be finite, positive and bounded by half the prior range')
    D=[];R=[]
    for row,r in rows:
        if row['index']==row0['index']:continue
        D.append((np.array([row['parameters'][k] for k in names])-x0)/steps);R.append(r-r0)
    D,R=np.asarray(D),np.asarray(R)
    weights=np.ones(len(D)) if physical_steps is None else np.exp(-np.minimum(np.sum(D*D,axis=1)/8,20))
    weighted=D*np.sqrt(weights[:,None]);singular=np.linalg.svd(weighted,compute_uv=False)
    require(len(singular)==len(names) and singular[-1]>1e-6 and singular[0]/singular[-1]<1000,
            'Native perturbations do not resolve independent parameter directions')
    J=np.linalg.lstsq(weighted,R*np.sqrt(weights[:,None]),rcond=None)[0].T
    model_errors=D@J.T-R
    model_rmse={key:float(np.sqrt(np.mean(np.sum(model_errors[:,section]**2,axis=1)/weight)))
                for key,section,weight in sections}
    lower=np.maximum(-1,(bounds[:,0]-x0)/steps);upper=np.minimum(1,(bounds[:,1]-x0)/steps)
    step=lsq_linear(J,-r0,bounds=(lower,upper)).x
    if objective=='feasibility':
        initial=np.r_[step,ratios(r0+J@step).max()]
        solved=minimize(lambda u:float(u[-1]+.01*np.sum((r0+J@u[:-1])**2)),initial,
            method='SLSQP',bounds=[*zip(lower,upper),(0.,1e4)],
            constraints=[{'type':'ineq','fun':lambda u:u[-1]-ratios(r0+J@u[:-1])}],
            options={'maxiter':150,'ftol':1e-10})
        require(solved.success,'Local feasibility proposal did not converge');step=solved.x[:-1]
    proposed=x0+step*steps
    files=[root/'plan.json',root/'observations.npz',Path(__file__)]
    if not history_only:files.append(root/'result.json')
    history_binding=None
    if history_only:
        require((root/'history.jsonl').read_bytes()[:len(prefix)]==prefix,'Completed worker history prefix changed')
        history_binding={'path':str((root/'history.jsonl').resolve()),'bytes':len(prefix),
                         'sha256':hashlib.sha256(prefix).hexdigest()}
    return {'parameters':{**row0['parameters'],**dict(zip(names,proposed.tolist()))},
        'base_trial':row0['index'],'source_trials':[r['index'] for r,_ in rows],
        'trust_radius_fraction':None if physical_steps is not None else trust_radius,
        'physical_trust_steps':dict(zip(names,steps.tolist())),
        'unit_step':dict(zip(names,(step*steps/widths).tolist())),
        'estimated_fit_loss':float(np.sum((r0+J@step)**2)),
        'base_fit_loss':float(np.sum(r0**2)),
        'objective':objective,'estimated_largest_normalized_fit_error':float(ratios(r0+J@step).max()),
        'surrogate_training_error_in_tolerances':model_rmse,
        'design_condition':float(singular[0]/singular[-1]),
        'conditional_fixed_parameters':sorted(fixed),
        'source_history_prefix':history_binding,'source_round_sealed':not history_only,
        'scope':'Local proposal from declared recorded fit-frame responses only; not a simulation, native execution certificate, unique identification, held-out validation or approval. Unsealed worker history can propose a new trial but cannot approve its source round.',
        'accepted':False,'inputs':{str(p.resolve()):digest(p) for p in files}}
