"""Measure real fixed attachments throughout an object-level task trial."""
import numpy as np

from calibrated_scene_reconstruction.tasks.rigid.task_checks import rot


def errors(p0,R0,p1,R1,anchor0,anchor1,frame0,frame1):
    a=np.asarray(p0)+np.asarray(R0)@anchor0;b=np.asarray(p1)+np.asarray(R1)@anchor1
    relative=(np.asarray(R0)@rot(frame0)).T@(np.asarray(R1)@rot(frame1))
    return float(np.linalg.norm(a-b)),float(np.arccos(np.clip((np.trace(relative)-1)/2,-1.,1.)))


class FixedAttachments:
    def __init__(self,ctx,assembly,declaration,*,position_tolerance,angle_tolerance=.001):
        from pxr import UsdPhysics
        self.ctx=ctx;self.entries={};self.trace=[]
        self.position_tolerance=float(position_tolerance);self.angle_tolerance=float(angle_tolerance)
        for mechanism in declaration['mechanisms']:
            if mechanism['type']!='fixed' or any(not ref.startswith(assembly+'/') for ref in mechanism['links']):continue
            refs=mechanism['links'];paths=[ctx.body(ref).prim_path for ref in refs]
            matching=[]
            for prim in ctx.stage.Traverse():
                if not prim.IsA(UsdPhysics.FixedJoint):continue
                joint=UsdPhysics.FixedJoint(prim)
                targets=[joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()]
                if any(len(target)!=1 for target in targets):continue
                actual=[str(target[0]) for target in targets]
                if set(actual)==set(paths):matching.append((joint,actual))
            entry={'spec':mechanism,'geometry':len(matching)==1,'maximum_anchor_error_m':0.,'maximum_rotation_error_rad':0.}
            if len(matching)==1:
                joint,actual=matching[0];entry['refs']=[refs[paths.index(path)] for path in actual]
                entry['anchors']=[np.asarray(getattr(joint,f'GetLocalPos{k}Attr')().Get()) for k in [0,1]]
                frames=[]
                for k in [0,1]:
                    q=getattr(joint,f'GetLocalRot{k}Attr')().Get();frames.append(np.r_[q.GetReal(),q.GetImaginary()])
                entry['frames']=frames
            self.entries[mechanism['id']]=entry
        self.sample(0.)

    def sample(self,time):
        row={'time_s':float(time),'links':{}}
        for name,entry in self.entries.items():
            if not entry['geometry']:continue
            first,second=(self.ctx.pose(ref) for ref in entry['refs'])
            position,angle=errors(*first,*second,*entry['anchors'],*entry['frames'])
            entry['maximum_anchor_error_m']=max(entry['maximum_anchor_error_m'],position)
            entry['maximum_rotation_error_rad']=max(entry['maximum_rotation_error_rad'],angle)
            row['links'][name]={'anchor_error_m':position,'rotation_error_rad':angle,
                'poses':{ref:{'position':np.asarray(p).tolist(),'rotation':np.asarray(R).tolist()}
                         for ref,(p,R) in zip(entry['refs'],[first,second])}}
        if self.entries:self.trace.append(row)

    def results(self):
        result={}
        for name,entry in self.entries.items():
            attached=(entry['geometry'] and entry['maximum_anchor_error_m']<=self.position_tolerance
                      and entry['maximum_rotation_error_rad']<=self.angle_tolerance)
            checks={key:{'pass':False} for key in entry['spec']['checks']}
            checks['geometry']={'pass':entry['geometry'],'actual_native_fixed_joint':entry['geometry']}
            checks['attachment']={'pass':attached,'maximum_anchor_error_m':entry['maximum_anchor_error_m'],
                'maximum_rotation_error_rad':entry['maximum_rotation_error_rad'],
                'position_tolerance_m':self.position_tolerance,'angle_tolerance_rad':self.angle_tolerance}
            result[name]={'pass':all(c['pass'] for c in checks.values()),'representation':'native_joint','checks':checks,
                          'scope':'Measured actual native fixed-joint frames throughout this task; no pose prescription'}
        return result
