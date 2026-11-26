"""Joint search over agent-selected parameters; validation never selects a candidate."""
import numpy as np

from calibrated_scene_reconstruction.physics.replay.optim import CMAES, penalised
from calibrated_scene_reconstruction.physics.replay.params import Param, Space
from .contract import compare, require
from .fit_priority import score as fit_priority_score


def hard_checks_pass(plan, checks):
    observed=plan.get('model',{}).get('observation_checks',{})
    return bool(checks) and all(c.get('pass') is True and c.get('reason') for name,c in checks.items() if name not in observed)


def fit(plan, observations, evaluate, progress=None):
    """evaluate(values, index) returns native predictions and physical checks.

    The native worker supplies evaluate. CPU fixtures can exercise optimization
    without pretending their predictions are Isaac simulation evidence.
    """
    bank = [Param(name, *p['bounds'], p['initial'], p.get('log', False))
            for name, p in plan['parameters'].items()]
    active = [name for name, p in plan['parameters'].items() if p['identification'] == 'fit']
    space = Space(active, params=bank)
    history, candidates = [], []

    def remember(row):
        history.append(row)
        if progress is not None:
            progress(row)

    def objective(U):
        scores = []
        for u in U:
            index = len(history)
            values = space.flat(u)
            try:
                predictions, checks, artifacts = evaluate(values, index)
                metrics = compare(plan, observations, predictions)
                for name,channels in plan.get('model',{}).get('observation_checks',{}).items():
                    expected=all(metrics['channels'][key]['pass'] for key in channels)
                    require(name in checks and checks[name].get('pass') is expected and checks[name].get('reason'),
                            'Source-observation check differs from its declared scored channels: '+name)
                good = hard_checks_pass(plan,checks)
                value = metrics['fit_loss'] if good else 1e12
                optimization_value=fit_priority_score(metrics,plan['search'].get('objective','feasibility')) if good else 1e12
                row = {'index': index, 'parameters': values, 'fit_loss': value,
                       'search_score':optimization_value,
                       'metrics': metrics, 'physical_checks': checks, 'artifacts': artifacts}
                candidates.append(row)
            except Exception as exc:
                row = {'index': index, 'parameters': values, 'fit_loss': 1e12, 'error': str(exc)}
                value = 1e12
                optimization_value=1e12
            remember(row)
            scores.append(optimization_value)
        return np.asarray(scores)

    # Always preserve the prior/default replay for a genuine before/after.
    objective([space.default_unit()])
    baseline=history[0]
    def fit_satisfied(row):
        return (row.get('fit_loss',1e12)<1e12 and
                all(c['fit_rmse']<=c['tolerance'] for c in row['metrics']['channels'].values()))
    # This optional stopping decision reads training error and native validity
    # only. Held-out success or failure never decides whether to search.
    stop_on_fit=plan['search'].get('stop_when_fit_passes',False)
    if active and not (stop_on_fit and fit_satisfied(baseline)):
        if 'proposals' in plan['search']:
            for row in plan['search']['proposals']:
                objective([space.to_unit([row[k] for k in active])])
                if stop_on_fit and fit_satisfied(history[-1]):break
        else:
            opt = CMAES(space.default_unit(), popsize=plan['search']['population'],
                        seed=plan['search'].get('seed', 0),sigma=plan['search'].get('initial_sigma',.25))
            scales=plan['search'].get('parameter_scales',{})
            opt.C=np.diag([scales.get(k,1.)**2 for k in space.names()])
            for _ in range(plan['search']['generations']):
                values=opt.ask();opt.tell(values,penalised(objective,values))
                if (stop_on_fit and any(fit_satisfied(c) for c in candidates)) or opt.done():break
    if not candidates or min(c['fit_loss'] for c in candidates) >= 1e12:
        return {'pass': False, 'history': history, 'findings': [
            {'id': 'native_execution', 'kind': 'model_or_execution',
             'reason': 'No physically valid native candidate; inspect per-trial failures.',
             'errors': sorted({h['error'] for h in history if 'error' in h})}]}
    # Selection uses the fit split and physical validity only, never held-out error.
    feasible=[c for c in candidates if c['fit_loss']<1e12 and
              all(v['fit_rmse']<=v['tolerance'] for v in c['metrics']['channels'].values())]
    if feasible:best=min(feasible,key=lambda c:c['fit_loss'])
    else:best=min(candidates,key=lambda c:c['search_score'])
    sensitivities = {}
    points = plan['search'].get('sensitivity_points', 3)
    for name in active if points else []:
        p = plan['parameters'][name]
        grid = np.geomspace(*p['bounds'], points) if p.get('log') else np.linspace(*p['bounds'], points)
        rows = []
        for value in grid:
            values = {**best['parameters'], name: float(value)}
            try:
                predicted, checks, artifacts = evaluate(values, len(history))
                loss = compare(plan, observations, predicted)['fit_loss']
                physical = hard_checks_pass(plan,checks)
                row = {'value': float(value), 'fit_loss': loss, 'physical_pass': physical}
                remember({'index': len(history), 'diagnostic': 'sensitivity', 'parameters': values,
                          'artifacts': artifacts, **row})
            except Exception as exc:
                row = {'value': float(value), 'error': str(exc)}
                remember({'index': len(history), 'diagnostic': 'sensitivity', 'parameters': values, **row})
            rows.append(row)
        losses = [r['fit_loss'] for r in rows if r.get('physical_pass')]
        flat = len(losses) == points and np.ptp(losses) <= 1e-6*max(1., min(losses))
        sensitivities[name] = {'samples': rows, 'flat': flat,
            'scope': 'One parameter at a time, others fixed; sensitivity does not establish joint identifiability.'}
    findings = [{'id': name, 'kind': 'unexplained_observation',
                 'reason': 'Fit or held-out error exceeds the declared tolerance.', 'metrics': metrics}
                for name, metrics in best['metrics']['channels'].items() if not metrics['pass']]
    varied={name:np.ptp([r['parameters'][name] for r in candidates if hard_checks_pass(plan,r['physical_checks'])])
            >1e-12*max(1.,abs(best['parameters'][name])) for name in active}
    return {'pass': not findings and all(c.get('pass') is True for c in best['physical_checks'].values()), 'best': best, 'before': history[0], 'history': history,
            'search_objective':plan['search'].get('objective','feasibility'),
            'findings': findings, 'parameter_evidence': {
                name: {'value': best['parameters'][name], 'bounds': p['bounds'], 'unit': p['unit'],
                       'status': ('unconstrained_on_scan' if sensitivities.get(name, {}).get('flat') else
                                  'fitted_not_uniquely_identified' if varied[name] else 'not_varied_in_search')
                                 if p['identification'] == 'fit' else 'prior_only',
                       'source': p['source'], 'reason': p['reason']}
                for name, p in plan['parameters'].items()},
            'sensitivity': sensitivities,
            'validation_scope': 'Held-out frames of this recording, not independent-video validation.'}
