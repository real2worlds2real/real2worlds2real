"""Candidate readback optimization; the original material checks are retained."""
import json
from .flexible_trace import Trace as IndividualTrace


class Trace(IndividualTrace):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,bulk_readback=False,**kwargs)
        from calibrated_scene_reconstruction.sim.particle_batch import Snapshots,SampleView
        self.snapshots=Snapshots(self.views)
        for group in self.snapshots.groups:
            for name in group['names']:self.views[name]=SampleView(self.snapshots,name)

    def record(self,step):
        if self.array is not None and step==self.sample_steps[self.index]:
            self.snapshots.refresh(oracle=step in [0,self.sample_steps[len(self.sample_steps)//2],self.steps])
        return super().record(step)

    def finish(self):
        result=super().finish()
        if result is None:return None
        path=self.output/'flexible_trace.json';full=json.loads(path.read_text())
        result['native_readback']=full['native_readback']=self.snapshots.report()
        path.write_text(json.dumps(full,indent=2)+'\n')
        return result
