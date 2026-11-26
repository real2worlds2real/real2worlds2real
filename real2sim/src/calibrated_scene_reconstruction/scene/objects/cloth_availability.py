"""Review loss of a rim grasp region using saved native states, without approval.

This distinguishes physical validity from task affordance. It does not identify
friction, stiffness or natural curvature from a failed manipulation.
"""
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.cloth_container_state import check as distribution
from calibrated_scene_reconstruction.assets.hand_cloth import rim_target


def measurements(points, times, rest, faces, opening, outer_radius):
    points, times = np.asarray(points), np.asarray(times)
    if (points.ndim != 3 or points.shape[2] != 3 or len(points) != len(times)
            or not len(times) or not np.isfinite(points).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Finite matching sequential native states and times required')
    rows=[]
    for time, state in zip(times, points):
        data=distribution(state,rest,faces,opening,{})['measurements']
        try:
            selected=rim_target(state,rest,faces,opening,outer_radius=outer_radius)
            available=True; reason=None
        except ValueError as error:
            selected=None; available=False; reason=str(error)
        rows.append({'time_s':float(time),'inside_material_fraction':data['inside_fraction'],
                     'rim_interior_patch_available':available,'reason':reason,
                     'selected_material_node':None if selected is None else selected['material_node']})
    return {'diagnostic_only':True,'task_approved':False,'rows':rows,
            'first_unavailable_time_s':next((r['time_s'] for r in rows if not r['rim_interior_patch_available']),None),
            'scope':'Measured material distribution and grasp-region availability only; not material identification or a task verdict.'}


def pinch_tracking(points, rows, material_node):
    """Track the same material point against actual pad references.

    A reference point is not the full curved contact surface. These distances
    diagnose targeting error, not opposing contact normals or normal force.
    """
    points=np.asarray(points,float)
    if (points.ndim!=3 or points.shape[2]!=3 or len(points)!=len(rows)
            or type(material_node) is not int or not 0<=material_node<points.shape[1]
            or not np.isfinite(points).all()):
        raise ValueError('Matching finite native states and a selected material node required')
    measured=[]
    for state,row in zip(points,rows):
        if row['phase'] not in ['approach','pinch','lift','hold','release']:
            continue
        pads=np.asarray(row['finger_pads'],float)
        if pads.shape!=(2,3) or not np.isfinite(pads).all():
            raise ValueError('Two finite actual pad references required')
        point=state[material_node];offset=point-pads.mean(0);axis=pads[1]-pads[0]
        gap=float(np.linalg.norm(axis));along=float(offset@axis/gap) if gap>1e-12 else None
        lateral=float(np.linalg.norm(offset-axis*(along/gap))) if along is not None else None
        measured.append({'time_s':row['time_s'],'phase':row['phase'],
                         'pad_reference_gap_m':gap,
                         'material_to_pad_midpoint_m':float(np.linalg.norm(offset)),
                         'material_lateral_offset_m':lateral,
                         'material_offset_along_pinch_axis_m':along,
                         'material_world_point':point.tolist(),
                         'near_surface_samples':row.get('hand_contact',{}).get('near_surface_samples',{})})
    return {'diagnostic_only':True,'task_approved':False,'material_node':material_node,'rows':measured,
            'scope':'Actual selected material and pad-reference motion only. Near-surface sample counts '
                    'are proximity, not opposing contact or compressive load. Curved pad shoulders, '
                    'multiple cloth layers and dynamic drift require contact review; friction causality '
                    'still requires the matched native lift/control test.'}


def review(session, trial, output):
    root=session.dir.resolve();trial=Path(trial).resolve();output=Path(output).resolve()
    if not all(p.is_relative_to(root/'qa') for p in [trial,output]):
        raise ValueError('Current-session native trace and new QA output required')
    result=json.loads((trial/'result.json').read_text())
    protocol=json.loads((trial/'protocol.json').read_text())
    candidate=json.loads((trial/'candidate_input.json').read_text())
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    parts={p['name']:p for p in asset({'mesh':str(Path(candidate['path'])/'mesh.npz')})['parts']}
    sheet=parts[protocol['cloth_link']];body=parts[protocol['body_link']]
    radius=float(np.ptp(body['V'],axis=0)[:2].mean()/2)
    with np.load(trial/'states.npz') as trace:
        points=trace['points'];times=trace['times']
        if len(points)!=len(result['rows']) or not np.allclose(times,[r['time_s'] for r in result['rows']]):
            raise ValueError('Saved native samples and quality rows disagree')
        # Native states are world coordinates; the target is body-local.
        local=np.stack([(p-np.asarray(r['body_pose'])[:3])@np.asarray(r['body_pose'])[3:].reshape(3,3)
                        for p,r in zip(points,result['rows'])])
    report=measurements(local,times,sheet['cloth_rest']['points'],sheet['F'],candidate['report']['opening'],radius)
    report['native_failure']=result.get('failure')
    report['sampled_physical_validity']=all(r['physical_pass'] for r in result['rows'])
    selection=result.get('postopening_selection')
    if selection is not None:
        report['pinch_tracking']=pinch_tracking(points,result['rows'],selection['material_node'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot(times,[r['inside_material_fraction'] for r in report['rows']],label='Material inside')
    axes[0].step(times,[int(r['rim_interior_patch_available']) for r in report['rows']],where='post',label='Rim grasp region available')
    axes[0].set(xlabel='Native time (s)',ylabel='Fraction / availability',ylim=(-.05,1.05));axes[0].legend(fontsize=8)
    opening=candidate['report']['opening'];centre=np.asarray(opening['xy'])
    for index in np.unique(np.linspace(0,len(points)-1,4).astype(int)):
        p=local[index];radial=np.linalg.norm(p[:,:2]-centre,axis=1)
        axes[1].scatter(radial*1000,p[:,2]*1000,s=.15,label=f'{times[index]:.2f}s')
    axes[1].plot([0,radius*1000],[opening['rim_z']*1000]*2,'k--',lw=1,label='Rim height')
    axes[1].axvline(radius*1000,color='k',ls=':',lw=1,label='Outer radius')
    axes[1].set(xlabel='Distance from opening axis (mm)',ylabel='Height (mm)',aspect='equal');axes[1].legend(fontsize=8,markerscale=6)
    figure.suptitle('Native cloth diagnostic; task not approved',fontsize=11)
    image=output/'availability.png';figure.savefig(image,dpi=150);plt.close(figure)
    return {'report':str((output/'result.json').relative_to(root)),'image':str(image.relative_to(root))}
