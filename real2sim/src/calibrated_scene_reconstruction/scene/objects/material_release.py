"""Matched native free-release coverage, separate from source identification."""
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.deformable_release import modal_decay
from calibrated_scene_reconstruction.assets.flexure_reference import frequency
from .material_response import sha


def compare(path,zero_path,parameters,*,physics_hz,iterations,minimum_material_edge_m):
    failures=[];decays={};traces={};protocols={}
    expected=frequency(.08,parameters['flexural_rigidity'],parameters['areal_density'])
    root=Path(__file__).resolve().parents[4]
    rates=['elasticity_damping','bend_damping','linear_damping']
    for label,source,mode in [('material',path,'material'),('zero',zero_path,'disabled')]:
        data=json.loads(Path(source).read_text());backend=data.get('backend',{});protocol=data.get('parameters',{})
        settings=data.get('settings',{});protocols[label]=protocol
        if (data.get('control_complete') is not True or data.get('synthetic_control') is not True
                or data.get('external_forces') is not False or data.get('gravity_enabled') is not False
                or data.get('fixed_or_kinematic_nodes')!=0 or data.get('positions_written_after_reset') is not False
                or data.get('native_physics_errors',{}).get('pass') is not True
                or data.get('damping_mode')!=mode or protocol.get('damping_mode')!=mode):
            failures.append(label+': missing complete unforced native release')
        if backend.get('name')!='isaacsim' or not str(backend.get('version','')).startswith('5.1.') or backend.get('solver')!='PGS':
            failures.append(label+': release backend differs from Isaac5.1 PGS')
        spacing=protocol.get('spacing_m')
        if (protocol.get('physics_hz')!=physics_hz or protocol.get('position_iterations')!=iterations
                or not isinstance(spacing,(int,float)) or not np.isfinite(spacing) or spacing<=0
                or minimum_material_edge_m<.9*spacing):
            failures.append(label+': release numerical profile or sampling differs from runtime')
        for name in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density',*rates]:
            value=settings.get(name)
            target=parameters.get(name,parameters.get('elasticity_damping') if name=='bend_damping' else None)
            if label=='zero' and name in rates:target=0.
            if value is None or target is None or not np.isfinite([value,target]).all() or not np.isclose(value,target,rtol=1e-5,atol=1e-12):
                failures.append(label+': release physical parameter differs: '+name)
        for relative in ['apps/scene/check_surface_flexure.py','src/calibrated_scene_reconstruction/assets/flexure_reference.py',
                         'src/calibrated_scene_reconstruction/assets/deformable_release.py','src/calibrated_scene_reconstruction/sim/surface_cloth.py']:
            source_file=root/relative
            if data.get('input_sha256',{}).get(str(source_file.resolve()))!=sha(source_file):
                failures.append(label+': release implementation changed: '+relative)
        snapshot=Path(source).with_name('source.py')
        if not snapshot.is_file() or sha(snapshot)!=sha(root/'apps/scene/check_surface_flexure.py'):
            failures.append(label+': executed release snapshot changed')
        try:
            rows=data['rows'];times=np.asarray([r['time_s'] for r in rows]);values=np.asarray([r['modal_amplitude_m'] for r in rows])
            if times[-1]<max(.8,3.2/expected)-1/240 or not all(r['material']['pass'] for r in rows):
                raise ValueError('Incomplete or materially invalid release trace')
            decays[label]=modal_decay(times,values,expected);traces[label]=(times,values)
        except (KeyError,TypeError,ValueError,IndexError) as error:
            failures.append(label+': '+str(error))
    relative=None
    if len(traces)==2:
        t,q=traces['material'];z,p=traces['zero']
        if (t.shape!=z.shape or not np.allclose(t,z,rtol=0,atol=1e-8)
                or not np.isclose(q[0],p[0],rtol=.01,atol=1e-9)
                or protocols['material'].get('spacing_m')!=protocols['zero'].get('spacing_m')):
            failures.append('Release controls do not have matched excitation, sampling and duration')
        baseline=decays['zero']['final_to_initial_proxy_ratio']
        if baseline<=1e-8:
            failures.append('Undamped release already lost the resolvable mode; dissipation is confounded')
        else:
            relative=decays['material']['final_to_initial_proxy_ratio']/baseline
            # A gross energy-growth screen, not an inferred decay target.
            # Small effects stay reported as unresolved and cannot identify a
            # material or replace independent source-behaviour acceptance.
            if relative>1.1:failures.append('Material release grows relative to its matched zero-damping control')
    return {'pass':not failures,'failures':failures,'material_control':str(path),'zero_control':str(zero_path),
            'release_decay':decays,'relative_retained_modal_proxy':relative,
            'material_identified':False,'source_behavior_approved':False,
            'scope':'Matched native release coverage and gross growth screen only. Numerical damping is reported separately; a weak difference is not evidence of effective material damping or realistic source recoil.'}
