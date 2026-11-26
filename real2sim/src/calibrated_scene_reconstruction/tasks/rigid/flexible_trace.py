"""Native flexible states accompanying a rigid-driven mixed assembly assay."""
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric
from .task_checks import as_numpy


class Trace:
    def __init__(self,ctx,names,steps,output,*,stride=None,bulk_readback=True):
        self.ctx,self.output,self.steps=ctx,Path(output),steps
        objects={name.split('/')[0] for name in names}
        self.names=[name for name in getattr(ctx,'_cloth',{}) if name.split('/')[0] in objects]
        from calibrated_scene_reconstruction.assets.response_policy import mode
        self.accuracy_mode=mode(ctx.task.get('flexible_accuracy'))
        if stride is None:
            stride=max(1,int(.02/ctx.world.get_physics_dt())) if self.accuracy_mode=='task_and_semantics' else 4
        if type(stride) is not int or stride<1:raise ValueError('Positive native trace stride required')
        self.stride=stride
        self.parts={};self.views={};self.metrics={};self.ranges={};offset=0
        from calibrated_scene_reconstruction.assets.metric_mesh import asset
        from calibrated_scene_reconstruction.assets.physical_priors import expand_families
        priors=expand_families(ctx.s.config('agent/physics_priors.json'))['bodies'] if self.names else {}
        kits={}
        for name in self.names:
            obj,link=name.split('/')
            if obj not in kits:kits[obj]=asset(ctx.sg['dynamic_objects'][obj]['fix'])
            kit=kits[obj]
            part=next(p for p in kit['parts'] if p['name']==link)
            backend=part.get('backend','particle')
            if backend not in {'surface_fem','particle'}:raise ValueError('Unsupported native flexible trace backend')
            if backend=='particle' and not getattr(ctx._cloth[name],'index_registration',{}).get('indexed_particle_order_preserved'):
                raise ValueError('Particle trace requires verified native/source indices')
            self.parts[name]=part;self.views[name]=ctx._cloth[name]
            limits=priors[name]['validity_hypothesis']
            self.metrics[name]=MaterialMetric(part['V'],part['F'],part['cloth_rest'],
                **{key:limits[key] for key in ('max_edge_strain','max_area_change')})
            self.ranges[name]=(offset,len(part['V']));offset+=len(part['V'])
        self.sample_steps=list(range(0,steps+1,stride))
        if self.sample_steps[-1]!=steps:self.sample_steps.append(steps)
        self.rows=[];self.index=0;self.maximum_speed=0.
        self.array=None if not self.names else np.lib.format.open_memmap(self.output/'surface_positions.npy',mode='w+',
            dtype='float32',shape=(len(self.sample_steps),offset,3))
        self.snapshots=None
        if bulk_readback and self.names:
            from calibrated_scene_reconstruction.sim.particle_batch import Snapshots,SampleView
            self.snapshots=Snapshots(self.views)
            for group in self.snapshots.groups:
                for name in group['names']:self.views[name]=SampleView(self.snapshots,name)

    def record(self,step):
        if self.array is None or step!=self.sample_steps[self.index]:return
        if self.snapshots is not None:
            self.snapshots.refresh(oracle=step in [0,self.sample_steps[len(self.sample_steps)//2],self.steps])
        row={};passed=True
        for name,view in self.views.items():
            points=as_numpy(view.get_world_positions())[0];velocities=as_numpy(view.get_velocities())[0]
            a,n=self.ranges[name];self.array[self.index,a:a+n]=points
            from calibrated_scene_reconstruction.assets.response_policy import material_verdict
            metric=self.metrics[name].check(points)
            # Complete native nodes and scalar errors remain archived. Dense
            # hotspot lists at every sampled frame add no qualitative proof.
            if self.accuracy_mode=='task_and_semantics' and step:
                metric={k:v for k,v in metric.items() if k!='strain_hotspots'}
            response=material_verdict(metric,self.accuracy_mode)
            passed &= response['pass'];row[name]={**metric,'response_acceptance':response}
            self.maximum_speed=max(self.maximum_speed,float(np.linalg.norm(velocities,axis=1).max()))
        self.rows.append({'physics_step':step,'pass':bool(passed),'material':row});self.index+=1

    def finish(self):
        if self.array is None:return None
        if self.index!=len(self.sample_steps):raise ValueError('Incomplete native flexible trace')
        self.array.flush();path=self.output/'surface_positions.npy'
        result={'pass':all(r['pass'] for r in self.rows),'source':'Actual source-indexed native flexible nodes',
                'backends':{name:part.get('backend','particle') for name,part in self.parts.items()},
                'material_accuracy_mode':self.accuracy_mode,
                'quantitative_material_pass':all(m['pass'] for r in self.rows for m in r['material'].values()),
                'names':self.names,'vertex_ranges':self.ranges,'sample_steps':self.sample_steps,
                'dt_s':float(self.ctx.world.get_physics_dt()),'sample_stride':self.stride,
                'maximum_sampled_node_speed_m_s':self.maximum_speed,
                'positions':path.name,'positions_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
                'native_readback':self.snapshots.report() if self.snapshots is not None else None,
                'rows':self.rows,'scope':'Sampled material validity and exact recorded node states; not continuous CCD, surface penetration, appearance or task approval'}
        (self.output/'flexible_trace.json').write_text(json.dumps(result,indent=2)+'\n')
        return {k:v for k,v in result.items() if k!='rows'}
