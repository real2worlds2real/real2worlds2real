"""Matched diagnostic shell release; never production or task approval."""
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.deformable_release import modal_decay


def compare(zero,material):
    paths=[Path(zero),Path(material)];data=[json.loads(p.read_text()) for p in paths]
    failures=[];decays=[];profiles=[]
    required=['apps/scene/check_particle_flexure.py','src/calibrated_scene_reconstruction/sim/discrete_bending.py',
        'src/calibrated_scene_reconstruction/sim/discrete_membrane.py','src/calibrated_scene_reconstruction/sim/discrete_dissipation.py',
        'src/calibrated_scene_reconstruction/assets/cloth_projection.py','src/calibrated_scene_reconstruction/assets/deformable_release.py']
    root=Path(__file__).resolve().parents[4]
    for label,path,d in zip(['zero','material'],paths,data):
        backend=d.get('backend',{})
        if (d.get('control_complete') is not True or d.get('synthetic_control') is not True
                or d.get('native_physics_errors',{}).get('pass') is not True
                or d.get('external_forces') is not False or d.get('gravity_enabled') is not False
                or d.get('fixed_nodes')!=0 or d.get('positions_written_after_reset') is not False
                or d.get('native_springs_disabled') is not True or d.get('native_spring_and_mass_readback') is not True
                or d.get('nonlinear_all_steps_converged') is not True
                or backend.get('name')!='isaacsim' or not str(backend.get('version','')).startswith('5.1.')
                or backend.get('representation')!='particle' or backend.get('solver')!='PGS'):
            failures.append(label+': incomplete unforced native shell control')
        hashes=d.get('input_sha256',{})
        for relative in required:
            p=root/relative
            if hashes.get(str(p))!=hashlib.sha256(p.read_bytes()).hexdigest():
                failures.append(label+': implementation changed: '+relative)
        for name,digest in hashes.items():
            p=Path(name)
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
                failures.append(label+': control dependency changed: '+name)
        snapshot=path.with_name('source.py')
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=hashes.get(str(root/required[0])):
            failures.append(label+': missing matched executed source')
        protocol=d.get('parameters',{});profiles.append({k:v for k,v in protocol.items()
            if k not in {'output','bending_viscosity_time_s','membrane_viscosity_time_s'}})
        try:
            rows=d['rows'];t=np.asarray([r['time_s'] for r in rows]);q=np.asarray([r['modal_amplitude_m'] for r in rows])
            if not all(r['material']['pass'] for r in rows):raise ValueError('Invalid material trace')
            if t[-1]<max(.8,3.2/d['expected_free_strip_frequency_hz'])-1/240:raise ValueError('Incomplete release')
            decays.append(modal_decay(t,q,d['expected_free_strip_frequency_hz']))
        except (KeyError,ValueError,TypeError,IndexError) as error:
            failures.append(label+': '+str(error));decays.append(None)
    for key in ['backend','settings','physics_hz','position_iterations','native_mass_kg','expected_free_strip_frequency_hz']:
        if data[0].get(key)!=data[1].get(key):failures.append('Unmatched release: '+key)
    if profiles[0]!=profiles[1]:failures.append('Unmatched release numerical protocol')
    for key in ['bending_viscosity_time_s','membrane_viscosity_time_s']:
        if data[0].get('parameters',{}).get(key)!=0:failures.append('Zero control has nonzero or unspecified viscosity')
    nominal=[data[1].get('parameters',{}).get(k) for k in ['bending_viscosity_time_s','membrane_viscosity_time_s']]
    if any(v is None or not np.isfinite(v) or v<0 for v in nominal) or not any(nominal):
        failures.append('Material control needs an explicit positive viscosity hypothesis')
    try:
        with np.load(paths[0].with_name('states.npz')) as a,np.load(paths[1].with_name('states.npz')) as b:
            if any(not np.array_equal(a[k],b[k]) for k in ['rest','faces','masses']):raise ValueError('Unmatched native material')
            if not np.array_equal(a['points'][0],b['points'][0]):raise ValueError('Unmatched initial excitation')
        ta=np.array([r['time_s'] for r in data[0]['rows']]);tb=np.array([r['time_s'] for r in data[1]['rows']])
        if not np.array_equal(ta,tb):raise ValueError('Unmatched release sampling')
    except (KeyError,ValueError,OSError) as error:failures.append(str(error))
    relative=None
    if all(decays):
        baseline=decays[0]['final_to_initial_proxy_ratio']
        if baseline<=1e-8:failures.append('Numerical damping already erased the baseline mode')
        else:
            relative=decays[1]['final_to_initial_proxy_ratio']/baseline
            if relative>1.1:failures.append('Material release grows relative to zero viscosity')
    return {'pass':not failures,'failures':failures,'relative_retained_modal_proxy':relative,
        'resolved_dissipation':not failures and relative is not None and relative<.9,
        'zero':str(paths[0]),'material':str(paths[1]),'release_decay':decays,
        'diagnostic_only':True,'production_backend_registered':False,'material_identified':False,'task_approved':False,
        'scope':'Matched native dissipation coverage only. Source behaviour, contact, timestep convergence and export/reload remain required.'}


def convergence(coarse,fine):
    """Report the numerical change in the material effect, separately from zero loss."""
    failures=[]
    pairs=[compare(*coarse),compare(*fine)]
    records=[json.loads(Path(pair[0]).read_text()) for pair in [coarse,fine]]
    materials=[json.loads(Path(pair[1]).read_text()) for pair in [coarse,fine]]
    if not all(p['pass'] and p['resolved_dissipation'] for p in pairs):failures.append('Both release pairs must resolve dissipation')
    a,b=records
    if b.get('physics_hz')!=2*a.get('physics_hz',0):failures.append('Refined profile must halve the timestep')
    omitted={'physics_hz','output'}
    if any({k:v for k,v in x.get('parameters',{}).items() if k not in omitted}
           !={k:v for k,v in y.get('parameters',{}).items() if k not in omitted}
           for x,y in [(a,b),tuple(materials)]):failures.append('Timestep study changes other protocol settings')
    if a.get('settings')!=b.get('settings'):failures.append('Timestep study changes material properties')
    ratios=[p['relative_retained_modal_proxy'] for p in pairs]
    change=None if any(x is None or x<=0 for x in ratios) else abs(ratios[1]/ratios[0]-1)
    if change is None or change>.15:failures.append('Dissipation effect has not converged within the 15% diagnostic screen')
    return {'pass':not failures,'failures':failures,'pairs':pairs,'relative_effect_change':change,
        'diagnostic_only':True,'production_backend_registered':False,'material_identified':False,'task_approved':False}
