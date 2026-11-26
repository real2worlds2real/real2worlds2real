"""Sample actual rigid surfaces throughout an authored joint's full travel.

Other joints remain at their captured coordinates. This rejects a bad hinge
before flexible-material tuning; it does not replace native contact sweeps or
certify every simultaneous configuration of a multi-joint assembly.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def measure(mesh,*,maximum_penetration_m=2e-5,angular_step_rad=.05,linear_step_m=.001,device='cpu'):
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface,triangle_samples
    settings=np.array([maximum_penetration_m,angular_step_rad,linear_step_m],float)
    if not np.isfinite(settings).all() or (settings<=0).any():
        raise ValueError('Finite positive articulation path sampling limits required')
    joints=json.loads(str(mesh.get('joints_json','[]')))
    if not any(j.get('collision_enabled') and j['type']!='fixed' for j in joints):
        return {'pass':True,'joints':[],'scope':'No moving joint requests adjacent contact'}
    links=json.loads(str(mesh['links_json']));parts={}
    for p in links:
        if p['kind']!='rigid':continue
        a,n,b,c=[p[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
        v=mesh['V'][a:a+n];f=mesh['F'][b:b+c]-a
        parts[p['name']]={'V':v,'F':f,'samples':triangle_samples(v,f)}
    suppressed={frozenset([j['parent'],j['child']]) for j in joints if not j.get('collision_enabled',False)}
    reports=[]
    for joint in joints:
        if joint['type']=='fixed' or not joint.get('collision_enabled',False):continue
        downstream={joint['child']}
        while True:
            expanded=downstream|{j['child'] for j in joints if j['parent'] in downstream}
            if expanded==downstream:break
            downstream=expanded
        pairs=[(a,b) for a in downstream if a in parts for b in parts if b not in downstream
               and frozenset([a,b]) not in suppressed]
        surfaces={n:SignedSurface(parts[n]['V'],parts[n]['F'],device=device) for pair in pairs for n in pair}
        initial=float(joint.get('initial_position',0.));low,high=joint['limits']
        step=angular_step_rad if joint['type']=='revolute' else linear_step_m
        count=int(np.ceil((high-low)/step))+1
        if count>4096:raise ValueError('Joint path exceeds sampling budget; select an explicit physical path policy')
        coordinates=np.unique(np.r_[np.linspace(low,high,count),initial]);rows=[]
        axis=np.array(joint['axis']);anchor=np.array(joint['anchor0'])
        for value in coordinates:
            if joint['type']=='revolute':
                r=Rotation.from_rotvec(axis*(value-initial)).as_matrix();t=anchor-r@anchor
            else:r=np.eye(3);t=axis*(value-initial)
            worst=0.;pair_rows=[]
            for a,b in pairs:
                # Query both directions: sparse mesh vertices alone can miss
                # a narrow wall entering a large cap face.
                moved=parts[a]['samples']@r.T+t
                first=surfaces[b].max_penetration(moved)
                moving_local=(parts[b]['samples']-t)@r
                second=surfaces[a].max_penetration(moving_local)
                penetration=max(first,second);worst=max(worst,penetration)
                if penetration>maximum_penetration_m:
                    pair_rows.append({'moving':a,'stationary':b,'penetration_m':penetration})
            rows.append({'coordinate':float(value),'maximum_penetration_m':worst,'failed_pairs':pair_rows})
        reports.append({'parent':joint['parent'],'child':joint['child'],'type':joint['type'],
                        'pass':all(r['maximum_penetration_m']<=maximum_penetration_m for r in rows),
                        'samples':rows,'checked_pairs':[list(p) for p in pairs]})
    return {'pass':all(r['pass'] for r in reports),'joints':reports,
            'maximum_penetration_m':float(maximum_penetration_m),'angular_step_rad':float(angular_step_rad),
            'linear_step_m':float(linear_step_m),
            'scope':'Sampled authored rigid paths with other joints held at capture state; native motion remains required'}


class Checked:
    def __init__(self,builder,output):
        self.builder,self.output=builder,Path(output);self.spec=builder.spec;self.cache={}

    def __getattr__(self,name):return getattr(self.builder,name)

    def evaluate(self,*args,**kwargs):
        path,report=self.builder.evaluate(*args,**kwargs)
        digest=hashlib.sha256((path/'mesh.npz').read_bytes()).hexdigest()
        if digest not in self.cache:
            with np.load(path/'mesh.npz') as mesh:self.cache[digest]=measure(mesh)
            self.output.mkdir(parents=True,exist_ok=True)
            (self.output/(digest+'.json')).write_text(json.dumps(self.cache[digest],indent=2)+'\n')
        result=self.cache[digest]
        if not result['pass']:raise ValueError('Authored articulation path has rigid penetration; see '+str(self.output))
        return path,{**deepcopy(report),'articulation_path':result}
