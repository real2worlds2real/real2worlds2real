"""Bind requested task-video phases to recorded native states, not preparation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def check(manifest,trace,quality):
    request=manifest.get('task_motion')
    if request is None:return None
    required={'result','states','protocol','indices','required_phases','allow_failed_attempt'}
    optional={'initial_phase','maximum_initial_hinge_angle_rad','open_hinge_interval_rad'}
    if (not isinstance(request,dict) or not required<=request.keys()
            or request.keys()-required-optional or type(request['allow_failed_attempt']) is not bool):
        raise ValueError('Explicit bound native task-motion request required')
    for key in ['result','states','protocol']:
        path=Path(request[key]);expected=manifest['sha256'].get(str(path))
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Native task-motion input is unbound or changed: '+key)
    result=json.loads(Path(request['result']).read_text())
    protocol=json.loads(Path(request['protocol']).read_text());rows=result['rows']
    indices=np.asarray(request['indices'])
    if (indices.ndim!=1 or not len(indices) or indices.dtype.kind not in 'iu'
            or indices[0]!=0 or (np.diff(indices)<0).any() or indices[-1]>=len(rows)
            or len(quality)!=len(indices)):
        raise ValueError('Task video must start at recorded capture state and follow actual ordered samples')
    phases=[rows[i]['phase'] for i in indices]
    if phases[0]!=request.get('initial_phase','initial'):
        raise ValueError('Task video must begin at capture state, not a preparation phase')
    required_phases=request['required_phases']
    if (not isinstance(required_phases,list) or not required_phases
            or any(not isinstance(p,str) or not p for p in required_phases)):
        raise ValueError('Explicit requested native phases required')
    missing=set(required_phases)-set(phases)
    if missing:raise ValueError('Requested task video lacks recorded phases: '+', '.join(sorted(missing)))
    if len(set(required_phases))!=len(required_phases) or any(
            a>=b for a,b in zip([phases.index(p) for p in required_phases],
                                [phases.index(p) for p in required_phases][1:])):
        raise ValueError('Recorded task phases do not follow the requested order')
    if [r['phase'] for r in quality]!=phases:
        raise ValueError('Video phase labels differ from the actual native samples')
    for key in ['initial_articulation_state','initial_material_distribution','initial_support_route']:
        if key in protocol and (not isinstance(protocol[key],dict) or protocol[key].get('pass') is not True):
            raise ValueError('Requested captured initial state was not established: '+key)
    if protocol.get('cloth_forces') or protocol.get('cloth_attachments'):
        raise ValueError('Temporary cloth forces or attachments cannot represent a hand grasp')
    if 'maximum_initial_hinge_angle_rad' in request:
        limit=request['maximum_initial_hinge_angle_rad']
        if not np.isfinite(limit) or limit<0 or abs(rows[0]['lid_angle_rad'])>limit:
            raise ValueError('Native initial hinge does not match the requested captured state')
    if 'open_hinge_interval_rad' in request:
        interval=np.asarray(request['open_hinge_interval_rad'],float)
        if interval.shape!=(2,) or not np.isfinite(interval).all() or interval[0]>=interval[1] or 'pinch' not in phases:
            raise ValueError('Explicit open hinge interval and recorded pinch required')
        pinch=phases.index('pinch')
        checked=[rows[indices[max(0,pinch-1)]]]+[rows[i] for i in indices if rows[i]['phase']=='pinch']
        if any(not interval[0]<=r['lid_angle_rad']<=interval[1] for r in checked):
            raise ValueError('Native open hinge was not established and retained for the pinch')
    with np.load(request['states']) as source:
        if (len(source['times'])!=len(rows) or not np.array_equal(trace['times'],source['times'][indices])
                or not np.array_equal(trace['points'],source['points'][indices])):
            raise ValueError('Rendered cloth/time must equal the actual native states without interpolation')
        for name,fields in manifest.get('links',{}).items():
            if 'points' in fields and fields['points']!='points':
                source_field=fields.get('source_points')
                source_indices=np.asarray(fields.get('source_indices',[]))
                if (source_field not in source or source_indices.ndim!=1 or not len(source_indices)
                        or source_indices.dtype.kind not in 'iu' or source_indices.min()<0
                        or source_indices.max()>=source[source_field].shape[1]
                        or len(set(source_indices.tolist()))!=len(source_indices)
                        or not np.array_equal(trace[fields['points']],source[source_field][indices][:,source_indices])):
                    raise ValueError('Rendered flexible skin differs from actual native nodes: '+name)
            if 'position' not in fields:continue
            saved=source[name+'_poses'][indices]
            if (not np.array_equal(trace[fields['position']],saved[:,:3]) or
                    not np.array_equal(trace[fields['rotation']],saved[:,3:].reshape(-1,3,3))):
                raise ValueError('Rendered rigid motion differs from actual native poses: '+name)
        for context in manifest.get('rigid_context',[]):
            body=context.get('native_body')
            if body is None:raise ValueError('Requested tool motion needs an explicit native body binding')
            saved=source['hand_links'][indices,result['hand_body_names'].index(body)]
            q=saved[:,3:].astype(float);q/=np.linalg.norm(q,axis=1,keepdims=True)
            x,y,z,w=q.T
            rotations=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)
            if (not np.array_equal(trace[context['position']],saved[:,:3]) or
                    not np.allclose(trace[context['rotation']],rotations,rtol=0.,atol=1e-12)):
                raise ValueError('Rendered contact tool motion differs from actual native poses: '+body)
    passed=result.get('complete') is True and result.get('physical_pass') is True and (result.get('lift') or {}).get('pass') is True
    if not request['allow_failed_attempt'] and not passed:
        raise ValueError('Requested passing motion is only a failed or incomplete attempt')
    return {'requested_phases_recorded':True,'phases':list(dict.fromkeys(phases)),
            'starts_at_native_capture':True,'native_geometry_unchanged':True,
            'source_complete':result.get('complete') is True,'source_physical_pass':result.get('physical_pass') is True,
            'source_failure':result.get('failure'),'task_approved':result.get('task_approved') is True,
            'scope':'Requested action sequence is present in original native samples. A failed diagnostic remains failed; rendering grants no acceptance.'}
