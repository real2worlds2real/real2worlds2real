"""Unforced rigid motion measurements, separate from flexible contact validity.

A valid cloth mesh or a holder-assisted manipulation cannot certify a stable
initial assembly. Bounds come from the declared component response tests.
"""
import numpy as np


def rigid_drift(vertices, poses):
    """Maximum source-surface displacement, including body rotation."""
    vertices, poses = np.asarray(vertices, float), np.asarray(poses, float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or poses.ndim != 2 or poses.shape[1:] != (12,) or len(poses) < 2
            or not np.isfinite(vertices).all() or not np.isfinite(poses).all()):
        raise ValueError('Source vertices and at least two finite native rigid poses required')
    rotations = poses[:, 3:].reshape(-1, 3, 3)
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)):
        raise ValueError('Native rigid poses must contain proper rotations')
    initial = vertices @ rotations[0].T + poses[0, :3]
    values = [float(np.linalg.norm(vertices @ r.T + p[:3] - initial, axis=1).max())
              for p, r in zip(poses, rotations)]
    return {'maximum_surface_displacement_m':max(values),
            'surface_displacement_per_sample_m':values,
            'worst_sample':int(np.argmax(values)),
            'scope':'Sampled native poses of the unchanged rigid source surface; not a stiffness identification.'}


def surface_drift(points, frame_poses):
    """Actual material-node motion in a declared rigid reference frame."""
    points=np.asarray(points,float);frames=np.asarray(frame_poses,float)
    if (points.ndim!=3 or points.shape[2]!=3 or not points.shape[1]
            or frames.shape!=(len(points),12) or len(points)<2 or not np.isfinite(points).all()):
        raise ValueError('Matching finite material surface and reference-frame samples required')
    # Reuse proper-rotation and finite-pose validation; no best-fit alignment
    # that could conceal free-body drift or loss of support.
    rigid_drift([[0.,0,0]],frames)
    rotations=frames[:,3:].reshape(-1,3,3)
    local=np.einsum('tni,tij->tnj',points-frames[:,:3,None].transpose(0,2,1),rotations)
    values=np.linalg.norm(local-local[0],axis=2).max(axis=1)
    return {'maximum_surface_displacement_m':float(values.max()),
            'surface_displacement_per_sample_m':values.tolist(),
            'scope':'Actual indexed material nodes in an explicit rigid frame; not a material-law or support certificate.'}


def evaluate(understanding, parts, poses, times, *, flexible_states=None, observations=None):
    """Resolve only explicitly declared rigid passive-drift bounds.

    Missing/unsupported passive measurements stay unresolved; no task pass is
    substituted. The caller must independently establish an unforced interval.
    """
    times = np.asarray(times, float)
    if (times.ndim != 1 or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0)):
        raise ValueError('Ordered native passive sample times required')
    parts = {p['name']:p for p in parts}
    checks = {}
    extent = None
    def normalized(peak):
        nonlocal extent
        if extent is None:
            from .passive_extent import initial_extent
            extent = initial_extent(parts, poses, flexible_states, len(times))
        return peak/extent['value_m']
    for part in understanding.get('parts', []):
        links = part['links']
        rigid=all(name in parts and parts[name]['kind']=='rigid' for name in links)
        for test in part.get('physics_tests', []):
            if test['purpose'] != 'passive_state':
                continue
            metrics = test['metrics']
            if not rigid:
                if not all((m['name'] in {'surface_drift_m','normalized_unforced_drift'}
                            and m['unit']==('m' if m['name']=='surface_drift_m' else 'object_extent')
                            and m.get('reference_link') in poses
                            and (m['name']=='surface_drift_m' or (m.get('reference_link') in parts
                                 and parts[m['reference_link']]['kind']=='rigid')))
                           or (m['name'] not in {'surface_drift_m','normalized_unforced_drift'}
                               and all(m['name'] in (observations or {}).get(name,{}) for name in links)) for m in metrics):
                    # Other flexible tests have their own producers. Do not
                    # silently interpret them as rigid motion or a pass.
                    checks[test['id']]={'pass':False,'why':'Unsupported or unframed flexible passive metric'}
                    continue
                measured={};passed=True
                for metric in metrics:
                    if metric['name'] not in {'surface_drift_m','normalized_unforced_drift'}:
                        samples=[]
                        for name in links:
                            value=observations[name][metric['name']];values=np.asarray(value['values'],float)
                            if value['unit']!=metric['unit'] or values.shape!=times.shape or not np.isfinite(values).all():
                                raise ValueError('Matching native metric units and every passive sample required')
                            samples.extend(values)
                        passed &= all(metric['bounds'][0]<=v<=metric['bounds'][1] for v in samples)
                        measured[metric['name']]={'minimum':float(min(samples)),'maximum':float(max(samples)),
                                                  'bounds':metric['bounds'],'unit':metric['unit']}
                        continue
                    values={}
                    for name in links:
                        if name not in (flexible_states or {}):raise ValueError('Declared flexible passive state needs actual material samples')
                        values[name]=surface_drift(flexible_states[name],poses[metric['reference_link']])
                    peak=max(v['maximum_surface_displacement_m']for v in values.values())
                    value=normalized(peak) if metric['name']=='normalized_unforced_drift' else peak
                    passed &= metric['bounds'][0]<=value<=metric['bounds'][1]
                    measured[metric['name']]={'value_m':peak,'bounds':metric['bounds'],'reference_link':metric['reference_link'],'links':values}
                    if metric['name']=='normalized_unforced_drift':
                        measured[metric['name']].update(value_ratio=value,initial_object_extent=extent)
                checks[test['id']]={'pass':bool(passed),'part':part['id'],'metrics':measured}
                continue
            if not all((m['name']=='rigid_drift_m' and m['unit']=='m')
                       or (m['name']=='normalized_unforced_drift' and m['unit']=='object_extent'
                           and m.get('reference_frame')=='world' and 'reference_link' not in m)
                       for m in metrics):
                checks[test['id']] = {'pass':False, 'why':'Unsupported declared passive metric'}
                continue
            measured = {}
            for name in links:
                if name not in poses or len(poses[name])!=len(times):
                    raise ValueError('Every declared rigid link needs matching native samples')
                measured[name] = rigid_drift(parts[name]['V'], poses[name])
            peak = max(row['maximum_surface_displacement_m'] for row in measured.values())
            metric_values={m['name']:(normalized(peak) if m['name']=='normalized_unforced_drift' else peak) for m in metrics}
            checks[test['id']] = {'pass':bool(all(m['bounds'][0]<=metric_values[m['name']]<=m['bounds'][1] for m in metrics)),
                'part':part['id'],'value_m':peak,'bounds':[m['bounds'] for m in metrics],
                'links':measured}
            if 'normalized_unforced_drift' in metric_values:
                checks[test['id']].update(value_ratio=metric_values['normalized_unforced_drift'],
                                         initial_object_extent=extent,reference_frame='world')
    return {'pass':bool(checks) and all(row['pass'] for row in checks.values()),
            'resolved':bool(checks), 'duration_s':float(times[-1]-times[0]), 'checks':checks,
            'scope':'Declared rigid motion and indexed flexible motion in explicit rigid frames only; material response, support, contact and full task remain separate.'}
