"""Sequential native hand/cloth assay harness for input acceptance.

Every attempt owns its output, retains failed reports, and fails on missing
native result files even if Kit returns zero. No task or delivery approval.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

from calibrated_scene_reconstruction.core.paths import ROOT


from calibrated_scene_reconstruction.core.environment import setting

def invoke(app, arguments, output):
    output = Path(output)
    # Respect an existing outer lease; otherwise acquire one for each native
    # assay when the caller selected shared-GPU admission. Other deployments
    # retain their configured host/container launcher.
    shared=setting('SHARED_GPU_RESERVE_MIB') is not None
    active=setting('SHARED_GPU_ACTIVE')=='1'
    runner='run_shared_isaac.sh' if shared and not active else 'run_isaac.sh'
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.run([str(ROOT / 'tools' / runner), 'apps/scene/' + app,
                                  *arguments, '--output', str(output)], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT)
    report = output / 'result.json'
    if process.returncode or not report.is_file():
        error = output / 'error.json'
        return {'complete': False, 'pass': False, 'physical_pass': False,
                'failure': json.loads(error.read_text()) if error.is_file()
                           else f'Native process={process.returncode}, result={report.exists()}'}
    return json.loads(report.read_text())


def run(session, obj, candidate, hand_plan, output, *, criterion, case='nominal',
        fixture_proof=None, register=False, inside_finger='either', minimum_normal_alignment=None,
        closure_descent_m=None, require_contact_free_approach=False, pinch_settle_seconds=.4,
        track_pinch_material=False, particle_friction_proof=None, pinch_axis=None, minimum_pinch_axis_alignment=.94, pinch_material_axis=None,
        closure_mode='open_descent',open_descent_seconds=.5,closure_contact_links=None,closure_contact_basis=None,
        pinch_material_frame=None,timing=None,material_contact_mode='same_component'):
    from calibrated_scene_reconstruction.assets.acceptance import declaration
    root = session.dir.resolve(); output = Path(output).resolve()
    candidate, hand_plan = Path(candidate).resolve(), Path(hand_plan).resolve()
    if material_contact_mode not in {'single_patch','same_component'}:
        raise ValueError('Explicit material contact association required')
    if not 0<minimum_pinch_axis_alignment<=1:raise ValueError('Pinch-axis cosine must be within(0,1]')
    if minimum_normal_alignment is not None and not 0<=minimum_normal_alignment<=1:raise ValueError('Normal alignment cosine must be within0..1')
    if closure_descent_m is not None and not 0<=closure_descent_m<=.03:
        raise ValueError('Closure descent must be within0..30mm')
    if closure_mode not in {'open_descent','coupled'} or not .04<=open_descent_seconds<=5:
        raise ValueError('Explicit closure mode and bounded open-descent duration required')
    timing={} if timing is None else timing
    if not isinstance(timing,dict) or set(timing)-{'opening','opening_hold','approach','pinch'}:
        raise ValueError('Explicit supported motion timing fields required')
    for name,value in timing.items():
        if type(value) not in (int,float) or not (0 if name=='opening_hold' else .25)<=value<=5:
            raise ValueError('Finite bounded fixture motion timing required')
    closure_contact_links=[] if closure_contact_links is None else closure_contact_links
    if (not isinstance(closure_contact_links,list) or any(not isinstance(n,str) or not n for n in closure_contact_links)
            or len(set(closure_contact_links))!=len(closure_contact_links)
            or bool(closure_contact_links)!=bool(isinstance(closure_contact_basis,str) and closure_contact_basis.strip())):
        raise ValueError('Expected closure contacts require unique link names and a source/task basis')
    if (type(require_contact_free_approach) is not bool or type(track_pinch_material) is not bool
            or not 0<=pinch_settle_seconds<=2.):
        raise ValueError('Explicit approach policy and bounded pinch settling required')
    if (case not in {'nominal', 'low', 'high'} or inside_finger not in {'either','thumb','index'}
            or not output.is_relative_to(root / 'qa')
            or not candidate.is_relative_to(root / 'qa')
            or not hand_plan.is_relative_to(session.config_dir.resolve())
            or obj not in session.scene_graph['dynamic_objects']):
        raise ValueError('Current-session candidate, QA output, config hand plan and object required')
    requests = declaration(session.config('task.json', {}))['criteria']
    request = next((row for row in requests if row['id'] == criterion), None)
    if request is None or request.get('validation') != 'native_physics':
        raise ValueError('Explicit native input acceptance criterion required')
    required_mode=request.get('interpretation',{}).get('closure_mode')
    if ((required_mode is not None and required_mode!=closure_mode) or
            'open_descent_before_closure' in request.get('checks',[]) and closure_mode!='open_descent'):
        raise ValueError('Closure mode must preserve the input-required open-descent sequence')
    required_axis=request.get('interpretation',{}).get('pinch_material_axis')
    required_contact=request.get('interpretation',{}).get('material_contact_mode')
    if required_contact is not None and required_contact!=material_contact_mode:
        raise ValueError('Material contact association must preserve the input-required scope')
    required_frame=request.get('interpretation',{}).get('pinch_material_frame')
    if pinch_material_frame is None:pinch_material_frame=required_frame or 'spatial'
    if pinch_material_frame not in {'spatial','opening_plane'} or required_frame is not None and required_frame!=pinch_material_frame:
        raise ValueError('Material direction frame must match the input interpretation')
    if required_axis is not None:
        if pinch_axis is not None or pinch_material_axis is not None and list(pinch_material_axis)!=required_axis:
            raise ValueError('Grasp must use the material direction specified by the current input')
        pinch_material_axis=required_axis
    if minimum_normal_alignment is None:minimum_normal_alignment=0. if pinch_axis is not None or pinch_material_axis is not None else .7
    if pinch_axis is not None and pinch_material_axis is not None:raise ValueError('Choose one source-axis convention')
    if pinch_material_axis is not None:
        import numpy as np
        axis=np.asarray(pinch_material_axis,float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:raise ValueError('Resolved material axis required')
    if pinch_axis is not None:
        from calibrated_scene_reconstruction.assets.hand_cloth import pinch_axis_alignment
        if not pinch_axis_alignment([[0,0,0],pinch_axis],pinch_axis,minimum=minimum_pinch_axis_alignment)['resolved']:
            raise ValueError('Source pinch axis has no resolved projection')
    if 'source_pinch_axis' in request.get('checks',[]) and pinch_axis is None and pinch_material_axis is None:
        raise ValueError('Input requires a source-reviewed longitudinal pinch axis')
    from calibrated_scene_reconstruction.apps.reconstruction.hand_acceptance import CHECKS
    if not set(request.get('checks',[])) <= CHECKS:
        raise ValueError('This hand/cloth checker does not implement every requested check')
    # Reject undeclared or malformed native inputs before reserving the GPU.
    candidate_data=json.loads(candidate.read_text());hand_data=json.loads(hand_plan.read_text())
    if (candidate_data.get('preparation_only') or candidate_data.get('requires_additional_preparation')
            or candidate_data.get('requires_material_restore')):
        raise ValueError('Preparation-only geometry cannot represent the requested initial task state')
    if candidate_data.get('requires_experimental_particle_shell_runtime'):
        raise ValueError('Experimental shell candidate needs qualified task/export runtime; native FEM cannot silently substitute for it')
    if fixture_proof is not None:
        fixture_proof = Path(fixture_proof).resolve()
        if not fixture_proof.is_relative_to(root / 'qa') or not fixture_proof.is_file():
            raise ValueError('Current-session free-hand proof required')
    if particle_friction_proof is not None:
        from calibrated_scene_reconstruction.scene.objects.particle_friction_ablation import verify
        particle_friction_proof=Path(particle_friction_proof).resolve()
        if not particle_friction_proof.is_relative_to(root/'qa'):
            raise ValueError('Current-session particle friction proof required')
        verify(particle_friction_proof)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / 'source.py')
    (output / 'request.json').write_text(json.dumps({'object': obj, 'candidate': str(candidate),
        'hand_plan': str(hand_plan), 'criterion': criterion, 'case': case,
        'inside_finger':inside_finger,'pinch_axis':pinch_axis,'pinch_material_axis':pinch_material_axis,'minimum_pinch_axis_alignment':minimum_pinch_axis_alignment,
        'pinch_material_frame':pinch_material_frame,
        'timing':timing,'material_contact_mode':material_contact_mode,
        'minimum_normal_alignment':minimum_normal_alignment,
        'closure_descent_m':closure_descent_m,
        'closure_mode':closure_mode,'open_descent_seconds':open_descent_seconds,
        'expected_closure_contacts':{'links':closure_contact_links,'basis':closure_contact_basis},
        'require_contact_free_approach':require_contact_free_approach,
        'pinch_settle_seconds':pinch_settle_seconds,
        'track_pinch_material':track_pinch_material,
        'particle_friction_proof':None if particle_friction_proof is None else str(particle_friction_proof),
        'fixture_proof': None if fixture_proof is None else str(fixture_proof)}, indent=2) + '\n')
    def failed(stage, native):
        result={'ok': False, 'pass': False, 'task_approved': False, 'failed_stage': stage,
                'native_result': native, 'evidence_images': [], 'directory': str(output)}
        if stage=='positive' and all((output/'positive'/name).is_file() for name in ['result.json','states.npz','protocol.json']):
            try:
                from calibrated_scene_reconstruction.scene.objects.cloth_availability import review
                result['failure_review']=review(session,output/'positive',output/'failure_review')
                result['evidence_images']=[result['failure_review']['image']]
            except (KeyError,ValueError,OSError,ImportError,RuntimeError) as error:
                result['failure_review_error']=str(error)
        return result
    if fixture_proof is None:
        profile=session.config('simulation.json',{})
        hz=profile.get('physics_hz')
        if type(hz) is not int or not 240<=hz<=7680 or profile.get('solver')!='PGS':
            raise ValueError('Free-hand preflight requires the actual supported task simulation profile')
        result = invoke('check_hand_fixture.py', ['--plan', str(hand_plan),
            '--physics-hz',str(hz),'--finger-effort',str(hand_data['finger_effort_limit_Nm'])], output / 'free_hand')
        if result.get('pass') is not True:
            return failed('free_hand', result)
        fixture_proof = output / 'free_hand/result.json'
    common = [str(root), '--candidate', str(candidate), '--hand-plan', str(hand_plan),
              '--fixture-proof', str(fixture_proof), '--assembly', obj, '--case', case,
              '--inside-finger',inside_finger,'--minimum-normal-alignment',str(minimum_normal_alignment),
              '--closure-mode',closure_mode,'--open-descent-seconds',str(open_descent_seconds),
              '--pinch-material-frame',pinch_material_frame,
              '--material-contact-mode',material_contact_mode,
              '--pinch-settle-seconds',str(pinch_settle_seconds)]
    if pinch_material_axis is not None:common += ['--pinch-material-axis',*map(str,pinch_material_axis),'--minimum-pinch-axis-alignment',str(minimum_pinch_axis_alignment)]
    if pinch_axis is not None:common += ['--pinch-axis',*map(str,pinch_axis),'--minimum-pinch-axis-alignment',str(minimum_pinch_axis_alignment)]
    if closure_descent_m is not None:common += ['--closure-descent-m',str(closure_descent_m)]
    for name,value in sorted(timing.items()):common+=['--'+name.replace('_','-')+'-seconds',str(value)]
    if closure_contact_links:
        for name in closure_contact_links:common+=['--closure-contact-link',name]
        common+=['--closure-contact-basis',closure_contact_basis]
    if require_contact_free_approach:common.append('--require-contact-free-approach')
    if track_pinch_material:common.append('--track-pinch-material')
    if particle_friction_proof is not None:common+=['--particle-friction-proof',str(particle_friction_proof)]
    positive = invoke('check_hand_cloth.py', common, output / 'positive')
    if (positive.get('complete') is not True or positive.get('physical_pass') is not True
            or (positive.get('lift') or {}).get('pass') is not True):
        return failed('positive', positive)
    motion=output/'positive/motion.json'
    if not motion.is_file():
        return failed('matched_motion',{'failure':'Passing positive trial did not save its executed motion'})
    negative = invoke('check_hand_cloth.py', [*common, '--zero-friction-lift','--replay-motion',str(motion)], output / 'zero_friction')
    if negative.get('complete') is not True or negative.get('physical_pass') is not True:
        return failed('zero_friction', negative)
    from calibrated_scene_reconstruction.apps.reconstruction.hand_acceptance import run as verify_pair
    result = verify_pair(session, output / 'positive', output / 'zero_friction',
                         output / 'criterion', criterion=criterion, register=register)
    return {**result, 'ok': result['pass'], 'directory': str(output), 'evidence_images': []}
