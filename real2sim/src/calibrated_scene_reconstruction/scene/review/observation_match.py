"""Read-only source-state and passive-shape checks before task actuation."""
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.objects.blender import evidence
from calibrated_scene_reconstruction.core.geometry.visibility import VisibleParts


def material_motion_profile(rest,states,*,body_poses=None,bins=6):
    """Localize transient motion in material coordinates, without naming a cause.

    Optional measured rigid poses remove the assembly's motion only. Every
    flexible node retains its independent motion; this is not a new reference
    for the source or material validity gates.
    """
    rest,states=np.asarray(rest,float),np.asarray(states,float)
    if (rest.ndim!=2 or rest.shape[1]!=3 or states.ndim!=3 or states.shape[1:]!=rest.shape
            or not len(states) or not np.isfinite(rest).all() or not np.isfinite(states).all()
            or type(bins) is not int or not 2<=bins<=16):
        raise ValueError('Matching finite material coordinates and actual sampled node states required')
    local=states.copy()
    if body_poses is not None:
        poses=np.asarray(body_poses,float)
        if poses.shape!=(len(states),12) or not np.isfinite(poses).all():
            raise ValueError('One actual rigid position/rotation per material sample required')
        rotations=poses[:,3:].reshape(-1,3,3)
        if not np.allclose(rotations@rotations.transpose(0,2,1),np.eye(3),atol=1e-5):
            raise ValueError('Material motion anchor must be a rigid measured frame')
        local=np.einsum('sni,sij->snj',states-poses[:,None,:3],rotations)
    delta=local-local[0];distance=np.linalg.norm(delta,axis=2)
    _,singular,axes=np.linalg.svd(rest-rest.mean(0),full_matrices=False)
    rows=[]
    for index,(singular_value,axis) in enumerate(zip(singular,axes)):
        if singular_value<=max(1e-12,singular[0]*1e-8):continue
        if axis[np.argmax(abs(axis))]<0:axis=-axis
        coordinate=(rest-rest.mean(0))@axis;low,high=coordinate.min(),coordinate.max()
        assignment=np.minimum(bins-1,((coordinate-low)/(high-low)*bins).astype(int))
        regions=[]
        for cell in range(bins):
            selected=assignment==cell
            if not selected.any():continue
            per_sample=np.percentile(distance[:,selected],95,axis=1);worst=int(np.argmax(per_sample))
            regions.append({'fraction':[cell/bins,(cell+1)/bins],'nodes':int(selected.sum()),
                'maximum_sample_p95_motion_m':float(per_sample[worst]),'maximum_motion_m':float(distance[:,selected].max()),
                'worst_sample_index':worst,'median_displacement_at_worst_sample_m':np.median(delta[worst,selected],axis=0).tolist()})
        rows.append({'material_axis':index,'direction_in_material_coordinates':axis.tolist(),'regions':regions})
    return {'frame':'actual_primary_rigid' if body_poses is not None else 'world',
            'sample_count':len(states),'maximum_motion_m':float(distance.max()),'axes':rows,
            'physical_parameters_identified':False,
            'scope':'Material-coordinate motion localization only. Contact roles require source/geometry review; no gate, relaxed reference, attachment or constitutive parameter is inferred.'}


def passive_series(world,duration,read,check,*,interval=.05,on_sample=None):
    """Check the whole unforced interval, including transient shape collapse.

    Retain actual native states; a return to the initial shape at the endpoint
    cannot erase an earlier response failure. No force or position controls.
    """
    dt=float(world.get_physics_dt())
    if not np.isfinite([dt,duration,interval]).all() or min(dt,duration,interval)<=0:
        raise ValueError('Finite positive passive duration, time step and sampling interval required')
    steps=int(np.ceil(duration/dt));stride=max(1,int(np.floor(interval/dt)))
    states=[];records=[]
    def record(step):
        actual=np.asarray(read()).copy();states.append(actual)
        records.append({'time_s':step*dt,'result':check(actual)})
        if on_sample is not None:on_sample(records[-1],actual)
    record(0)
    for step in range(1,steps+1):
        world.step(render=False)
        if step%stride==0 or step==steps:record(step)
    passed=all(row['result']['pass'] for row in records)
    return {'pass':passed,'duration_s':steps*dt,'maximum_sample_interval_s':stride*dt,
            'samples':records,'external_actuation':False,
            'scope':'Sampled native unforced response; not continuous-time certification'},np.asarray(states)


def response_diagnosis(series):
    """Route failures by evidence without asserting an identified modulus."""
    samples=series['samples'];first=samples[0]['result']
    if not first.get('source_fit_pass',False) or not first['pass']:
        category='source_geometry_or_native_initialization'
        next_step='Resolve source part geometry, label/exposure ambiguity and initial contact state before material tuning.'
    elif (series.get('material_and_self_contact_pass') is False or any(
            row['result'].get('actual_material_state',{}).get('pass') is False for row in samples)):
        category='contact_or_material_validity_failure'
        next_step='Resolve actual material strain, self-contact and numerical validity before interpreting visual softness.'
    elif not series['pass']:
        category='unforced_response_mismatch'
        next_step=('Inspect sampled shape, rigid penetration and self-contact, then compare supported physical hypotheses '
                   '(fold memory, support/clamping, thickness, friction and stiffness) under unchanged source/strain/task gates. '
                   'Use matched gravity/contact negative controls only for diagnosis of initialization or stored-strain release. '
                   'Do not assume low stiffness is the cause or make the sheet rigid.')
    else:
        category='source_response_consistent'
        next_step='Continue actual task motion and prior sensitivity checks; passive agreement does not identify properties.'
    return {'category':category,'next_step':next_step,'physical_parameters_identified':False,
            'source_of_target':'Current capture source observations, never user stiffness feedback or a relaxed simulator state'}


class ObservedState:
    """Fit to source images plus motion of initially visible material points.

    The reference is the source-fitted assembly, never a relaxed simulator
    shape. Source images remain the silhouette target. Visible material motion
    is tested separately so a small part cannot hide in assembly-wide averages.
    This constrains a response, not unique mass/friction/material parameters.
    """
    def __init__(self, reference, faces, target_faces, observations, *, max_motion_px, device='cpu',
                 part_observations=None,transmitting_faces=None):
        self.reference = np.asarray(reference, float).copy()
        self.faces = np.asarray(faces).copy()
        self.observations = observations
        self.limit = float(max_motion_px)
        self.target_faces=np.asarray(target_faces,bool)
        self.part_observations=part_observations
        self.device=device
        self.transmitting_faces=transmitting_faces
        self.pose = {'xy': [0., 0.], 'yaw_deg': 0., 'support_z': 0.}
        self.reference_metrics, _ = evidence.score(self.reference, self.faces, self.pose, observations)
        self.reference_part_metrics=None
        if part_observations is not None:
            from calibrated_scene_reconstruction.scene.objects.physical_observations import compare
            self.reference_part_metrics,_=compare(self.reference,self.faces,self.target_faces,
                                                   part_observations,device=device,transmitting_faces=transmitting_faces)
            self.reference_part_metrics['pass']=(self.reference_part_metrics['iou_median']>=.85
                                                  and self.reference_part_metrics['local_difference']<=.02)
        visibility = VisibleParts(reference, faces, target_faces, device=device,transmitting_faces=transmitting_faces)
        z, frames, masks, windows = observations
        self.visible = [visibility.visible_vertices(z['K'][i], z['R_world_cam'][i], z['centre'][i], w, m.shape,
                        domain=m & z['_object_domains'][k] if '_object_domains' in z else m)
                        for k, (i, m, w) in enumerate(zip(frames, masks, windows))]
        if sum(len(ids) >= 3 for ids in self.visible) < 8:
            raise ValueError('Source-state comparison needs at least8 views of the selected physical part')

    def check(self, actual):
        actual = np.asarray(actual, float)
        if actual.shape != self.reference.shape or not np.isfinite(actual).all():
            raise ValueError('Read matching finite actual native state; no remeshing or relaxed reference')
        metrics, _ = evidence.score(actual, self.faces, self.pose, self.observations)
        z, frames, _, windows = self.observations
        motion = []
        for i, window, ids in zip(frames, windows, self.visible):
            if len(ids) < 3:
                continue
            projected = []
            for points in (self.reference[ids], actual[ids]):
                camera = (points-z['centre'][i])@z['R_world_cam'][i]
                if (camera[:, 2] <= .05).any():
                    raise ValueError('Previously observed material moved behind the comparison camera')
                uv = camera@z['K'][i].T
                projected.append((uv[:, :2]/uv[:, 2:]-window[:2])*window[4])
            motion.append(float(np.percentile(np.linalg.norm(projected[1]-projected[0], axis=1), 95)))
        maximum = max(motion)
        part=None
        if self.part_observations is not None:
            from calibrated_scene_reconstruction.scene.objects.physical_observations import compare
            part,_=compare(actual,self.faces,self.target_faces,self.part_observations,device=self.device,
                           transmitting_faces=self.transmitting_faces)
            part['pass']=part['iou_median']>=.85 and part['local_difference']<=.02
        source_pass=bool(self.reference_metrics['ok'] and (self.reference_part_metrics is None
                                                         or self.reference_part_metrics['pass']))
        return {'pass': bool(source_pass and metrics['ok'] and maximum <= self.limit
                             and (part is None or part['pass'])),
                'source_fit_pass': source_pass, 'actual_image_metrics': metrics,
                'reference_independent_part_metrics':self.reference_part_metrics,
                'independent_part_metrics':part,
                'maximum_view_p95_visible_material_motion_px': maximum,
                'view_p95_motion_px': motion, 'limit_px': self.limit,
                'source_reference_changed_by_settling': False,
                'scope': 'Source silhouette consistency and initially visible material hold; not parameter identification'}
