"""Read-only particle snapshots with independently checked bulk body ordering.

Grouping changes no solver, source geometry, mass, material, state or sampling
schedule. Unsupported layouts retain their individual readers.
"""
import os
import numpy as np
from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy


class Snapshots:
    def __init__(self,views):
        from isaacsim.core.prims import ClothPrim
        from calibrated_scene_reconstruction.assets.particle_batch_order import check
        self.groups=[];self.current={};self.samples=0;self.oracle_checks=[]
        candidates={name:v for name,v in views.items() if getattr(v,'backend',None)=='particle'}
        for obj in sorted({name.split('/')[0] for name in candidates}):
            names=[name for name in candidates if name.split('/')[0]==obj]
            if len(names)<2:continue
            selected={str(candidates[n].prim.GetPath()):n for n in names}
            parents=[p.rsplit('/',1)[0] for p in selected]
            expression=os.path.commonprefix(parents)+'.*/surface'
            try:
                view=ClothPrim(expression,name='read_only_particle_batch_'+obj,reset_xform_properties=False)
                if set(view.prim_paths)!=set(selected):continue
                view.initialize()
                paths=list(view.prim_paths)
                points=as_numpy(view.get_world_positions())
                single=np.stack([as_numpy(candidates[selected[p]].get_world_positions())[0] for p in paths])
                proof=check(single,points)
            except (ValueError,RuntimeError):
                continue
            self.groups.append({'view':view,'names':[selected[p] for p in paths],
                                'single':[candidates[selected[p]] for p in paths],'registration':proof})

    def refresh(self,*,oracle=False):
        from calibrated_scene_reconstruction.assets.particle_batch_order import check
        for group in self.groups:
            p=as_numpy(group['view'].get_world_positions());v=as_numpy(group['view'].get_velocities())
            if oracle:
                single=np.stack([as_numpy(view.get_world_positions())[0] for view in group['single']])
                proof=check(single,p)
                speed=np.stack([as_numpy(view.get_velocities())[0] for view in group['single']])
                error=float(abs(speed-v).max())
                if not np.allclose(speed,v,rtol=1e-6,atol=1e-7):raise ValueError('Bulk native particle velocity differs from independent readers')
                self.oracle_checks.append({**proof,'sample':self.samples,'maximum_velocity_discrepancy':error})
            for i,name in enumerate(group['names']):self.current[name]=(p[i:i+1],v[i:i+1])
        self.samples+=1

    def report(self):
        return {'groups':[{'names':g['names'],'registration':g['registration']} for g in self.groups],
                'oracle_checks':self.oracle_checks,'physics_changed':False,'sample_schedule_changed':False}


class SampleView:
    def __init__(self,snapshots,name):self.snapshots=snapshots;self.name=name
    def get_world_positions(self):return self.snapshots.current[self.name][0]
    def get_velocities(self):return self.snapshots.current[self.name][1]
