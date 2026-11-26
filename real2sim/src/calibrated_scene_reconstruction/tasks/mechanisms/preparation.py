"""Finite-force cloth-pose preparation; no task acceptance or state targets.

An agent supplies explicitly inferred material grasp areas and rigid-local
controller paths. Temporary fixture forces prepare geometry only; the finished
candidate still needs source, released-native and task checks without fixtures.
"""
import numpy as np
import json

from calibrated_scene_reconstruction.assets.mechanisms import joint_frames,revolute_position
from .force_fixture import ForceHandle,material_patch
from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy


def frames(values,dimensions):
    value=np.asarray(values,float)
    if (value.ndim!=2 or value.shape[1]!=dimensions+1 or len(value)<2
            or not np.isfinite(value).all() or value[0,0]<0 or (np.diff(value[:,0])<=0).any()):
        raise ValueError('Explicit finite increasing fixture-time paths required')
    return value


def interpolate(path,time):return np.array([np.interp(time,path[:,0],path[:,j]) for j in range(1,path.shape[1])])


class MaterialPreparation:
    def __init__(self,ctx,assembly,kit,plan):
        if plan.get('version')!=1 or plan.get('assembly')!=assembly or not plan.get('basis'):
            raise ValueError('Current-assembly preparation plan and inferred construction basis required')
        self.ctx,self.assembly,self.plan=ctx,assembly,plan
        initial_tolerance=plan.get('initial_target_tolerance_m')
        if initial_tolerance is not None and (not np.isfinite(initial_tolerance) or not 0<initial_tolerance<=.01):
            raise ValueError('Finite bounded initial tool-target frame tolerance required')
        self.parts=kit['parts']
        self.name=assembly+'/'+plan['cloth_link'];part=next(p for p in kit['parts'] if p['name']==plan['cloth_link'] and p['kind']=='cloth')
        from .preparation_geometry import material_coordinates
        self.chart_angle=plan.get('material_chart_rotation_rad',0.)
        chart=material_coordinates(part['cloth_rest']['points'],self.chart_angle);points=ctx.nodes(self.name)
        self.part=part
        self.masses=as_numpy(ctx._cloth[self.name].get_particle_masses())[0];self.grips=[]
        rigid={p['name'] for p in kit['parts'] if p['kind']=='rigid'};owned=set()
        for grip in plan.get('grips',[]):
            path=frames(grip['keyframes'],3);active=np.asarray(grip['active_s'],float)
            if active.shape!=(2,) or not np.isfinite(active).all() or not 0<=active[0]<active[1]:
                raise ValueError('Explicit bounded fixture activation interval required')
            ids=material_patch(chart,grip['centre_fraction'],grip['size_m']);share=grip['mass_share']
            if grip['anchor_link'] not in rigid or owned.intersection(ids):
                raise ValueError('Disjoint material patches and a current rigid anchor required')
            owned.update(ids)
            gravity_share=grip.get('gravity_share',share)
            gravity_ramp=grip.get('gravity_ramp_s',.5)
            ramp=grip.get('release_ramp_s',0.)
            if (not np.isfinite([share,gravity_share]).all() or not 0<share<=1
                    or not 0<=gravity_share<=1 or grip['speed_m_s']>.1
                    or not np.isfinite(gravity_ramp) or not .001<=gravity_ramp<=10):
                raise ValueError('Bounded mass share and finite-force controller speed required')
            if not np.isfinite(ramp) or not 0<=ramp<=active[1]-active[0]:
                raise ValueError('Finite release ramp within the temporary grip interval required')
            handle=ForceHandle(points[ids],self.masses[ids],dt=ctx.world.get_physics_dt(),
                               held_mass=float(self.masses.sum()*share),force_limit=grip['force_limit_N'],
                               speed=grip['speed_m_s'],stiffness=grip.get('stiffness_N_m',20.),ramp_seconds=gravity_ramp)
            position,rotation=ctx.pose(self.assembly+'/'+grip['anchor_link'])
            initial_goal=path[0,1:]@rotation.T+position
            initial_error=float(np.linalg.norm(handle.target-initial_goal))
            if initial_tolerance is not None and initial_error>initial_tolerance:
                raise ValueError('Temporary grip starts in a different anchor frame; rebase its path after model centering')
            self.grips.append({'plan':grip,'path':path,'active':active,'ids':ids,'handle':handle,
                               'initial_target_error_m':initial_error,
                               'gravity_share':gravity_share,'release_ramp_s':float(ramp),'peak_applied_force_N':0.,
                               'last_control_time_s':None,'last_applied_force_N':0.})
        if ((not self.grips and not plan.get('hinge')) or sum(g['plan']['mass_share'] for g in self.grips)>1+1e-10
                or sum(g['gravity_share'] for g in self.grips)>1+1e-10):
            raise ValueError('Finite material grips or hinge actuation, with total mass share at most one required')
        self.hinge=plan.get('hinge');self.hinge_path=None;self.hinge_active=None;self.peak_torque=0.
        self.hinge_bias=None;self.release_condition_report=None
        self.post_release_condition_reports=[]
        self.surface_contact_readers={}
        self.hinge_driver=None;self.hinge_control='effort'
        self.hinge_last_command_Nm=0.;self.hinge_last_control_time_s=None
        self.hinge_effort_cleared=not bool(self.hinge)
        self.hinge_release_receipt=None
        self.release_verified=not bool(plan.get('release_conditions'))
        if self.hinge:
            h=self.hinge;self.hinge_path=frames(h['keyframes'],1)
            self.hinge_control=h.get('control','effort')
            if self.hinge_control not in {'effort','native_drive'}:
                raise ValueError('Explicit effort or finite native drive preparation control required')
            if self.hinge_control=='native_drive' and h.get('torque_bias_keyframes') is not None:
                raise ValueError('Native preparation drives do not silently mix external bias torques')
            self.hinge_active=np.asarray(h.get('active_s',self.hinge_path[[0,-1],0]),float)
            if (self.hinge_active.shape!=(2,) or not np.isfinite(self.hinge_active).all()
                    or not 0<=self.hinge_active[0]<self.hinge_active[1]):
                raise ValueError('Explicit bounded hinge fixture activation interval required')
            self.joint=next(j for j in kit['joints'] if j['type']=='revolute' and j['parent']==h['parent'] and j['child']==h['child'])
            self.joint_frames=joint_frames(self.joint)
            if h.get('torque_bias_keyframes') is not None:
                self.hinge_bias=frames(h['torque_bias_keyframes'],1)
            if (not np.isfinite([h['torque_limit_Nm'],h['stiffness_Nm_rad'],h['damping_Nms_rad']]).all()
                    or min(h['torque_limit_Nm'],h['stiffness_Nm_rad'],h['damping_Nms_rad'])<=0):
                raise ValueError('Finite positive fixture hinge gains and torque budget required')
        self.checkpoints=plan.get('checkpoints',[]);self.checkpoint_reports=[]
        self.failed_checkpoint=None
        if not isinstance(self.checkpoints,list):
            raise ValueError('Preparation checkpoints must be an explicit ordered list')
        names=set();previous=-1.
        for item in self.checkpoints:
            if (not isinstance(item,dict) or set(item)!={'name','time_s','conditions','basis'}
                    or not isinstance(item['name'],str) or not item['name'].strip() or item['name'] in names
                    or not isinstance(item['basis'],str) or not item['basis'].strip()
                    or type(item['time_s']) not in (float,int) or not np.isfinite(item['time_s'])
                    or not 0<=item['time_s']<self.release_time or item['time_s']<=previous):
                raise ValueError('Named, ordered pre-release checkpoints with an explicit construction basis required')
            self.validate_conditions(item['conditions'],required=True)
            names.add(item['name']);previous=item['time_s']

    def validate_conditions(self,request,*,required=False):
        allowed={'maximum_hinge_angle_rad','container_state','material_support','interfaces','surface_contacts'}
        if not isinstance(request,dict) or set(request)-allowed or required and not request:
            raise ValueError('Explicit supported preparation conditions required')
        if 'maximum_hinge_angle_rad' in request:
            limit=request['maximum_hinge_angle_rad']
            if not self.hinge or type(limit) not in (float,int) or not np.isfinite(limit) or limit<0:
                raise ValueError('A real hinge and a finite nonnegative angular condition required')
        if 'surface_contacts' in request:
            from .surface_contact_state import validate
            if not isinstance(request['surface_contacts'],list) or not request['surface_contacts']:
                raise ValueError('Explicit nonempty contact-neighbourhood requirements required')
            for item in request['surface_contacts']:validate(item)

    def check_due_checkpoints(self,time):
        """Inspect actual native state before issuing this step's fixture commands.

        A failed checkpoint does not retime the experiment, teleport material,
        or grant a source/task pass. The caller must stop the failed trial.
        Scheduled fixture release still clears persistent native drives.
        """
        if self.failed_checkpoint is not None:return False
        for item in self.checkpoints[len(self.checkpoint_reports):]:
            if time<item['time_s']:break
            result={'name':item['name'],'scheduled_time_s':float(item['time_s']),
                    'observed_time_s':float(time),'basis':item['basis'],
                    **self.check_conditions(item['conditions'])}
            self.checkpoint_reports.append(result)
            if not result['pass']:
                self.failed_checkpoint=item['name']
                return False
        return True

    def step(self,time):
        # Articulation tensor effort commands persist across physics steps.
        # Merely ceasing to write is not a release, unlike a one-step nodal impulse.
        if self.hinge and time>=self.hinge_active[1] and not self.hinge_effort_cleared:
            h=self.hinge
            self.hinge_release_receipt=(self.hinge_driver.release() if self.hinge_driver is not None else
                self.ctx.apply_joint_effort(self.assembly,h['parent'],h['child'],0.,effort_limit=h['torque_limit_Nm']))
            self.hinge_last_command_Nm=0.;self.hinge_last_control_time_s=float(time)
            self.hinge_effort_cleared=True
        checkpoints_valid=self.check_due_checkpoints(time)
        if time>=self.release_time:
            if not self.release_verified:
                self.release_condition_report=self.check_release_conditions()
                # Forces have already been cleared. A failed construction goal
                # is not a native-physics exception and must not hide what the
                # genuinely released object does next. Keep the failed verdict
                # separate from fixture release; it cannot approve a candidate.
                self.release_verified=True
            return None if checkpoints_valid else False
        if not checkpoints_valid:return False
        poses={}
        def pose(name):
            if name not in poses:poses[name]=self.ctx.pose(name)
            return poses[name]
        if self.hinge and self.hinge_active[0]<=time<self.hinge_active[1]:
            h=self.hinge;parent=self.assembly+'/'+h['parent'];child=self.assembly+'/'+h['child']
            if self.hinge_control=='native_drive':
                if self.hinge_driver is None:
                    from .joint_drive_fixture import JointDriveFixture
                    self.hinge_driver=JointDriveFixture(self.ctx,self.assembly,h['parent'],h['child'],
                        stiffness=h['stiffness_Nm_rad'],damping=h['damping_Nms_rad'],effort_limit=h['torque_limit_Nm'])
                self.hinge_driver.target(float(interpolate(self.hinge_path,time)[0]));effort=None
            else:
                Rp=pose(parent)[1];Rc=pose(child)[1]
                theta=revolute_position(Rp,Rc,*self.joint_frames);axis=Rp@np.asarray(self.joint['axis'])
                speed=float((as_numpy(self.ctx.body(child).get_angular_velocity())-
                             as_numpy(self.ctx.body(parent).get_angular_velocity()))@axis)
                bias=0. if self.hinge_bias is None else float(interpolate(self.hinge_bias,time)[0])
                effort=float(np.clip(h['stiffness_Nm_rad']*(interpolate(self.hinge_path,time)[0]-theta)-h['damping_Nms_rad']*speed+bias,
                                     -h['torque_limit_Nm'],h['torque_limit_Nm']))
                self.ctx.apply_joint_effort(self.assembly,h['parent'],h['child'],effort,effort_limit=h['torque_limit_Nm'])
            self.hinge_last_command_Nm=effort;self.hinge_last_control_time_s=float(time)
            self.hinge_effort_cleared=False
            if effort is not None:self.peak_torque=max(self.peak_torque,abs(effort))
        # A hinge-only preparation does not use cloth forces. Avoid copying
        # the whole dense surface and its velocities off the GPU every step.
        if not self.grips:return
        active=[grip for grip in self.grips if grip['active'][0]<=time<grip['active'][1]]
        if not active:return
        patches=None
        if hasattr(self.ctx,'node_patch_state'):
            points,velocity=self.ctx.node_patch_state(self.name,np.concatenate([g['ids'] for g in active]))
            ends=np.r_[0,np.cumsum([len(g['ids']) for g in active])]
            patches=[(points[a:b],velocity[a:b]) for a,b in zip(ends[:-1],ends[1:])]
        else:
            points=self.ctx.nodes(self.name);velocity=as_numpy(self.ctx._cloth[self.name].get_velocities())[0]
        indices=[];forces=[];limits=[]
        for number,grip in enumerate(active):
            p=grip['plan'];ids=grip['ids'];handle=grip['handle']
            position,rotation=pose(self.assembly+'/'+p['anchor_link'])
            goal=interpolate(grip['path'],time)@rotation.T+position
            selected_points,selected_velocity=(patches[number] if patches is not None else (points[ids],velocity[ids]))
            force=handle.forces(selected_points,selected_velocity,goal,load=[0,0,float(self.masses.sum()*grip['gravity_share'])*9.81])
            if grip['release_ramp_s']:
                remaining=np.clip((grip['active'][1]-time)/grip['release_ramp_s'],0.,1.)
                force*=remaining*remaining*(3-2*remaining)
            grip['peak_applied_force_N']=max(grip['peak_applied_force_N'],float(np.linalg.norm(force,axis=1).sum()))
            grip['last_control_time_s']=float(time);grip['last_applied_force_N']=float(np.linalg.norm(force,axis=1).sum())
            indices.append(ids);forces.append(force);limits.append(handle.force_limit*float(handle.weights.max()))
        # Disjoint regions share one native impulse update. Each handle already
        # bounds every nodal force; no intermediate simulation step separated
        # the old per-handle writes. Preserve those forces while avoiding a
        # whole-cloth device roundtrip for each individual grip.
        if indices:
            self.ctx.apply_nodal_forces(self.name,np.concatenate(indices),np.concatenate(forces),force_limit_N=max(limits))

    @property
    def release_time(self):
        return max([g['active'][1] for g in self.grips]+([] if self.hinge_active is None else [self.hinge_active[1]]))

    def released(self,time):return time>=self.release_time and self.release_verified and self.hinge_effort_cleared

    def check_release_conditions(self):
        request=self.plan.get('release_conditions',{})
        return self.check_conditions({} if request is None else request)

    def check_conditions(self,request):
        self.validate_conditions(request)
        checks={}
        if not request:return {'pass':True,'applicable':False,'checks':{}}
        if 'maximum_hinge_angle_rad' in request:
            parent=self.ctx.pose(self.assembly+'/'+self.hinge['parent'])[1]
            child=self.ctx.pose(self.assembly+'/'+self.hinge['child'])[1]
            angle=revolute_position(parent,child,*self.joint_frames)
            checks['hinge']={'pass':abs(angle)<=request['maximum_hinge_angle_rad'],'actual_rad':angle}
        if 'container_state' in request:
            checks['container_state']=self.container_state(request['container_state'])
        if 'material_support' in request:
            from calibrated_scene_reconstruction.assets.support_route import above_unthreaded
            from calibrated_scene_reconstruction.assets.material_support import contact_state
            support=request['material_support']
            if support.get('relation')!='above_unthreaded' or not support.get('material_contact'):
                raise ValueError('Explicit existing material-end support contract required')
            part=next(p for p in self.parts if p['name']==support['link'] and p['kind']=='rigid')
            position,rotation=self.ctx.pose(self.assembly+'/'+support['link'])
            points=(self.ctx.nodes(self.name)-position)@rotation
            routing=above_unthreaded(points,part['V'],part['F'],support['up'],
                surface_region=support['surface_region'])
            contact=contact_state(points,self.part['cloth_rest']['points'],self.part['F'],
                part['V'],part['F'],support,tolerance_m=1e-6)
            checks['material_support']={'pass':bool(routing['pass'] and contact['pass']),
                'routing':routing,'material_end_contact':contact,'loaded_support_force_proven':False}
        if request.get('interfaces'):
            from .interface_contact import check
            checks['interfaces']=check(self.ctx,self.assembly,self.parts,request['interfaces'])
        if request.get('surface_contacts'):
            from .surface_contact_state import Native
            for number,item in enumerate(request['surface_contacts']):
                key=json.dumps(item,sort_keys=True)
                if key not in self.surface_contact_readers:
                    self.surface_contact_readers[key]=Native(self.ctx,self.assembly,self.parts,
                        self.plan['cloth_link'],item)
                checks['surface_contact_'+str(number)]=self.surface_contact_readers[key].check()
        return {'pass':bool(checks) and all(r['pass'] for r in checks.values()),'applicable':True,'checks':checks}

    def observe(self,time,points):
        """Separate path lag, actual patch tracking and finite commanded load."""
        if self.hinge_driver is not None and self.hinge_driver.released:
            self.hinge_release_receipt.update(self.hinge_driver.verify_released())
        if time>=self.release_time and self.plan.get('release_conditions'):
            self.post_release_condition_reports.append({'time_s':float(time),**self.check_release_conditions()})
        points=np.asarray(points,float);poses={};rows=[]
        for grip in self.grips:
            p=grip['plan'];name=self.assembly+'/'+p['anchor_link']
            if name not in poses:poses[name]=self.ctx.pose(name)
            position,rotation=poses[name];goal=interpolate(grip['path'],time)@rotation.T+position
            actual=np.average(points[grip['ids']],axis=0,weights=self.masses[grip['ids']])
            limited=grip['handle'].target
            rows.append({'name':p.get('name'),'material_fraction':p['centre_fraction'],
                'active_interval_contains_sample':bool(grip['active'][0]<=time<grip['active'][1]),
                'actual_world':actual.tolist(),'trajectory_goal_world':goal.tolist(),
                'rate_limited_target_world':limited.tolist(),
                'trajectory_rate_limit_lag_m':float(np.linalg.norm(limited-goal)),
                'actual_tracking_error_m':float(np.linalg.norm(actual-limited)),
                'initial_target_error_m':grip['initial_target_error_m'],
                'last_control_time_s':grip['last_control_time_s'],
                'last_commanded_sum_force_N':grip['last_applied_force_N'],
                'force_limit_N':grip['handle'].force_limit})
        return {'construction_only':True,'native_contact_force_readback':False,'task_approved':False,
                'container_state':(self.container_state(self.plan['container_state'],points=points)
                                   if self.plan.get('container_state') else None),
                'last_hinge_command_Nm':self.hinge_last_command_Nm,
                'last_hinge_control_time_s':self.hinge_last_control_time_s,
                'hinge_effort_cleared':self.hinge_effort_cleared,
                'hinge_release_receipt':self.hinge_release_receipt,
                'material_chart_rotation_rad':float(self.chart_angle),'grips':rows}

    def container_state(self,request,*,points=None):
        from calibrated_scene_reconstruction.assets.cloth_container_state import check
        p,r=self.ctx.pose(self.assembly+'/'+request['anchor_link'])
        opening=self.ctx.sg['dynamic_objects'][self.assembly]['fix']['opening']
        if opening is None:raise ValueError('Container packing constraints require an actual opening')
        points=self.ctx.nodes(self.name) if points is None else np.asarray(points,float)
        return check((points-p)@r,self.part['cloth_rest']['points'],
                     self.part['F'],opening,request['requirements'])

    def report(self):
        construction=None
        if self.plan.get('container_state'):
            construction=self.container_state(self.plan['container_state'])
        return {'preparation_only':True,'task_approved':False,'positions_prescribed':False,'permanent_attachments_created':False,
                'material_chart_rotation_rad':float(self.chart_angle),
                'checkpoints':self.checkpoint_reports,
                'scheduled_checkpoints':len(self.checkpoints),
                'checkpoints_pass':len(self.checkpoint_reports)==len(self.checkpoints) and all(r['pass'] for r in self.checkpoint_reports),
                'failed_checkpoint':self.failed_checkpoint,
                'container_state':construction,
                'release_conditions':self.release_condition_report,
                'final_release_conditions':self.check_release_conditions(),
                'post_release_conditions':self.post_release_condition_reports,
                'release_conditions_met_at_release':(self.release_condition_report or {}).get('pass',not bool(self.plan.get('release_conditions'))),
                'released_conditions_retained':bool((self.release_condition_report or {}).get('pass',not bool(self.plan.get('release_conditions')))
                    and all(r['pass'] for r in self.post_release_condition_reports)),
                'peak_hinge_torque_Nm':self.peak_torque if self.hinge_control=='effort' else None,
                'hinge_control':self.hinge_control,
                'native_drive_readback':self.hinge_driver.initial_readback if self.hinge_driver is not None else None,
                'last_hinge_command_Nm':self.hinge_last_command_Nm,
                'hinge_effort_cleared':self.hinge_effort_cleared,
                'hinge_release_receipt':self.hinge_release_receipt,
                'all_force_release_time_s':float(self.release_time),
                'hinge_active_s':None if self.hinge_active is None else self.hinge_active.tolist(),
                'grips':[{'name':g['plan'].get('name'),'selected_nodes':len(g['ids']),
                          'initial_target_error_m':g['initial_target_error_m'],
                          'held_mass_share':g['plan']['mass_share'],'gravity_compensation_share':g['gravity_share'],
                          'release_ramp_s':g['release_ramp_s'],'peak_applied_sum_force_N':g['peak_applied_force_N'],
                          'controller_report_before_release_scaling':True,
                          'active_s':g['active'].tolist(),**g['handle'].report()} for g in self.grips],
                'scope':'Explicit temporary finite-force construction fixtures only. Source pose, released native state, extraction and wiping remain unapproved.'}
