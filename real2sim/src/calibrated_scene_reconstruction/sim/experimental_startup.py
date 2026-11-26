"""Audited environment suspension for unregistered constitutive diagnostics.

Only the native initialization warmup is isolated. Restore every setting before
the first recorded interval; this grants no source, contact or task approval.
No simulated positions or velocities are written here.
"""


class FilteredPairs:
    """Adapt an explicit USD relationship without disabling rigid shapes.

    The installed backend must independently demonstrate filter support;
    successful relationship readback alone is not native contact evidence.
    """
    def __init__(self,relationship):self.relationship=relationship
    def GetPath(self):return self.relationship.GetPath()
    def Get(self):return [str(path) for path in self.relationship.GetTargets()]
    def Set(self,value):return self.relationship.SetTargets(value)


class DeferredStartup:
    def __init__(self,entries):
        self.entries=list(entries);self.saved=[];self.active=False;self.complete=False
        if not self.entries or len({str(attr.GetPath()) for attr,_ in self.entries})!=len(self.entries):
            raise ValueError('Distinct explicit diagnostic startup attributes required')

    def suspend(self):
        if self.active or self.complete:raise ValueError('Diagnostic startup can be suspended only once')
        self.saved=[]
        try:
            for attr,temporary in self.entries:
                value=attr.Get()
                if value is None:raise ValueError('Native startup setting has no explicit effective value')
                self.saved.append((attr,value,temporary))
                if not attr.Set(temporary) or attr.Get()!=temporary:
                    raise ValueError('Native startup suspension failed: '+str(attr.GetPath()))
        except Exception:
            for attr,value,_ in reversed(self.saved):attr.Set(value)
            raise
        self.active=True

    def restore(self):
        if not self.active:raise ValueError('No suspended diagnostic startup to restore')
        changed=[str(attr.GetPath()) for attr,_,temporary in self.saved if attr.Get()!=temporary]
        failures=[]
        for attr,value,_ in reversed(self.saved):
            if not attr.Set(value) or attr.Get()!=value:failures.append(str(attr.GetPath()))
        self.active=False;self.complete=not changed and not failures
        if changed or failures:
            raise ValueError('Diagnostic startup setting changed or failed restoration: '+str(changed+failures))
        return self.report()

    def report(self):
        return {'experimental_only':True,'production_backend_registered':False,'task_approved':False,
                'positions_written':False,'velocities_written':False,'all_settings_restored':self.complete,
                'settings':[{'attribute':str(attr.GetPath()),'before':value,'during_initialization':temporary,
                             'after':attr.Get()} for attr,value,temporary in self.saved],
                'scope':'Environment isolated during native reset only; settings restored before recorded physical interval. Actual contact implementation and dynamics still require native controls.'}


class InitialMaterialPose:
    """One authored pose at t0 after material-space contact cooking, never replay.

    The caller must cook ``cooked`` without forces/contact with other bodies.
    Both arrays use native world coordinates and the same material-node order.
    No geometry, force law, or task approval follows from this initialization.
    """
    def __init__(self,cooked,target,tolerance=2e-6):
        import numpy as np
        self.cooked=np.array(cooked,dtype=float,copy=True)
        self.target=np.array(target,dtype=float,copy=True)
        if (self.cooked.ndim!=2 or self.cooked.shape[1]!=3 or not len(self.cooked)
                or self.target.shape!=self.cooked.shape
                or not np.isfinite(self.cooked).all() or not np.isfinite(self.target).all()
                or not np.isfinite(tolerance) or tolerance<=0):
            raise ValueError('Finite corresponding native material/initial points required')
        self.cooked.setflags(write=False);self.target.setflags(write=False)
        self.tolerance=tolerance;self.attempted=False

    def apply(self,view,*,time_s):
        import hashlib
        import numpy as np
        import torch
        if self.attempted or time_s!=0:
            raise ValueError('Authored initial pose may be applied only once at physical t0')
        self.attempted=True  # A partial write must never be silently retried.
        current=view.get_positions()
        actual=current.detach().cpu().numpy().reshape(-1,3)
        if actual.shape!=self.cooked.shape or not np.allclose(actual,self.cooked,atol=self.tolerance,rtol=0.):
            raise ValueError('Cooked material geometry differs before initial-pose authoring')
        velocities=view.get_velocities()
        if not torch.isfinite(velocities).all() or torch.any(torch.abs(velocities)>1e-6):
            raise ValueError('Material cooking was not quiescent; refusing a velocity reset')
        indices=torch.zeros(1,dtype=torch.int32,device=current.device)
        target=torch.tensor(self.target,dtype=current.dtype,device=current.device).reshape(current.shape)
        view.set_positions(target,indices)
        view.set_velocities(torch.zeros_like(velocities),indices)
        actual=view.get_positions().detach().cpu().numpy().reshape(-1,3)
        error=float(np.abs(actual-self.target).max())
        if not np.isfinite(error) or error>self.tolerance:
            raise ValueError('Authored initial pose readback differs')
        return {'experimental_only':True,'task_approved':False,
                'initial_pose_writes_before_recorded_t0':1,'runtime_position_writes':0,
                'physical_time_s':float(time_s),'node_count':len(self.target),
                'cooked_world_points_sha256':hashlib.sha256(self.cooked.tobytes()).hexdigest(),
                'authored_world_points_sha256':hashlib.sha256(self.target.tobytes()).hexdigest(),
                'initial_pose_readback_maximum_error_m':error,
                'scope':'Initial state authoring only; native contact, material, full physical trace and task checks remain required.'}
