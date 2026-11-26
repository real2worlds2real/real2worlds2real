"""Qualify a particle-wide rigid-friction ablation from matched native controls.

The operation is not finger-specific. Task use additionally requires measured
absence of every other rigid support/contact throughout the causal interval.
"""
from pathlib import Path
import json
import numpy as np

from .contact_response import _trial,compare,sha,control_parameters


def response(reference,ablated,*,preserve):
    """Compare resolved load responses after the material update, not t0 drift."""
    switch=ablated['rigid_contact_ablation'];time=float(switch['time_s'])
    if (switch.get('method')!='particle_base_and_self_scale'
            or switch.get('native_readback_after_step') is not True
            or switch.get('affects_all_particle_rigid_contacts') is not True
            or not 0<switch['base_friction']<=1e-6
            or not np.isclose(switch['target_particle_pair_friction'],ablated['parameters']['friction'])
            or not np.isclose(switch['base_friction']*switch['authored_particle_friction_scale'],
                              switch['target_particle_pair_friction'])):
        raise ValueError('Current explicit particle-wide ablation and stepped native readback required')
    shifts=[]
    for data in [reference,ablated]:
        rows=data['rows'];matches=[r for r in rows if abs(r['time_s']-time)<1e-8]
        if len(matches)!=1 or rows[-1]['time_s']<=time+.05:
            raise ValueError('Saved pre-ablation state and resolved force interval required')
        shifts.append(float(rows[-1]['relative_x_m']-matches[0]['relative_x_m']))
    before,after=shifts
    tolerance=max(5e-5,.15*abs(before))
    if abs(before)<.0002:
        raise ValueError('Control response too small to resolve preservation or removal')
    if not preserve:
        initial=next(r for r in reference['rows'] if abs(r['time_s']-time)<1e-8)
        free=reference['rows'][-1]['free_traction_displacement_m']-initial['free_traction_displacement_m']
        if free<.001 or abs(before/free-1)>.15:
            raise ValueError('Zero-rigid-friction reference does not recover free traction')
    return {'pass':bool(abs(after-before)<=tolerance),'reference_displacement_m':before,
            'ablated_displacement_m':after,'absolute_tolerance_m':tolerance,
            'self_response_preserved':preserve,'switch_time_s':time}


def run(paths):
    required={f'{support}_{mode}' for support in ['rigid','self'] for mode in ['zero','nominal','ablated']}
    if set(paths)!=required:raise ValueError('Six matched native contact controls required')
    # Qualification stores absolute trial paths for subsequent task-side
    # verification. Normalize before computing nested checks too, so relative
    # CLI input and replay of the recorded proof produce the same report.
    paths={name:str(Path(path).resolve()) for name,path in paths.items()}
    data={};checks={};files={}
    for name,path in paths.items():
        data[name],checks[name]=_trial(path)
        files[str(Path(path).resolve())]=sha(path)
    from . import contact_response
    result={'version':1,'pass':False,'task_approved':False,'trial_checks':checks,'evidence_sha256':files,
            'trials':{name:str(Path(path).resolve()) for name,path in paths.items()},
            'producer_sha256':{str(Path(p).resolve()):sha(p) for p in [__file__,contact_response.__file__]},
            'scope':'Particle-wide rigid-friction ablation only; task use must independently exclude all non-hand rigid contact throughout the causal interval'}
    if not all(c['pass'] for c in checks.values()):return result
    effects={support:compare(paths[support+'_zero'],paths[support+'_nominal']) for support in ['rigid','self']}
    result['contact_capabilities']=effects
    if not all(c['pass'] for c in effects.values()):return result
    failures=[];responses={}
    for support in ['rigid','self']:
        try:
            original=data[support+'_nominal'];changed=data[support+'_ablated']
            if (changed['parameters'].get('rigid_ablation') is not True
                    or changed['parameters']['support']!=support
                    or changed.get('native_spring_readback') is not True):
                raise ValueError('Ablation must use the supported native spring/contact profile')
            for key in ['backend','physics_hz','native_contact_settings','spring_settings','spring_model',
                        'spring_damping_kg_s','material_edge_range_m','areal_density_kg_m2',
                        'physical_thickness_m','native_mass_kg','node_count','load_force_N','input_sha256']:
                if changed.get(key)!=original.get(key):raise ValueError('Ablation changes native profile: '+key)
            params=lambda d:control_parameters(d,{'output','rigid_ablation'})
            if params(changed)!=params(original):raise ValueError('Ablation changes another control parameter')
            switch=changed['rigid_contact_ablation'];time=float(switch['time_s'])
            if not np.allclose(changed['native_friction_initial'],original['parameters']['friction'],rtol=0,atol=1e-7):
                raise ValueError('Ablation already changed the initial native coefficient')
            if not np.allclose(changed['native_friction_final'],switch['base_friction'],atol=1e-10,rtol=0):
                raise ValueError('Missing actual near-zero particle base-friction readback')
            with np.load(Path(paths[support+'_nominal']).with_name('states.npz')) as a,\
                 np.load(Path(paths[support+'_ablated']).with_name('states.npz')) as b:
                for key in ['rest','faces','masses']:
                    if not np.array_equal(a[key],b[key]):raise ValueError('Ablation changes native material: '+key)
                ia=next(i for i,r in enumerate(original['rows']) if abs(r['time_s']-time)<1e-8)
                ib=next(i for i,r in enumerate(changed['rows']) if abs(r['time_s']-time)<1e-8)
                if a['points'].shape[1:]!=b['points'].shape[1:] or not np.allclose(a['points'][ia],b['points'][ib],rtol=0,atol=5e-5):
                    raise ValueError('Native states already differ before the ablation')
            reference=data['rigid_zero'] if support=='rigid' else original
            responses[support]=response(reference,changed,preserve=support=='self')
        except (KeyError,ValueError,TypeError,StopIteration,IndexError,OSError) as error:
            failures.append(support+': '+str(error))
    result.update(failures=failures,responses=responses)
    result['pass']=not failures and len(responses)==2 and all(r['pass'] for r in responses.values())
    return result


def verify(path):
    """Recompute a saved qualification before changing a task's native material."""
    recorded=json.loads(Path(path).read_text())
    if recorded.get('version')!=1 or recorded.get('pass') is not True:
        raise ValueError('Passing current particle friction-ablation qualification required')
    current=run(recorded.get('trials',{}))
    if not current['pass'] or current!=recorded:
        raise ValueError('Particle friction qualification, native evidence or producer changed')
    return current
