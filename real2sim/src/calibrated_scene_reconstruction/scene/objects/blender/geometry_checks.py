"""Keep reviewed empty regions as well as solid parts in modeling iterations."""
from pathlib import Path
import hashlib
import json
import numpy as np
from calibrated_scene_reconstruction.assets.passage_geometry import occupancy


def radial_relation(mesh,part,check,origin):
    """Source-declared broad relative direction of a support and attachment."""
    from calibrated_scene_reconstruction.assets.support_route import upward_patch
    up=np.asarray(check['up'],float);interval=np.asarray(check['cosine_interval'],float)
    if (up.shape!=(3,)or not np.isfinite(up).all()or not np.isclose(np.linalg.norm(up),1.,atol=1e-8)
            or interval.shape!=(2,)or not np.isfinite(interval).all()or not -1<=interval[0]<=interval[1]<=1
            or not check.get('origin_link')or not check.get('basis')):
        raise ValueError('Reviewed radial relation needs a body origin, unit up, cosine interval and source basis')
    a,n,b,k=[part[x]for x in ['vertex_start','vertex_count','face_start','triangle_count']]
    v=mesh['V'][a:a+n];f=mesh['F'][b:b+k]-a
    selected=upward_patch(v,f,up,1e-6,check['surface_region']);tri=v[f[selected]]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if area.sum()<=0:raise ValueError('Positive actual support area required')
    centre=np.einsum('n,nd->d',area,tri.mean(1))/area.sum()
    reference=check['reference_joint'];matches=[j for j in json.loads(str(mesh['joints_json']))
        if all(j.get(key)==value for key,value in reference.items())]
    if len(matches)!=1:raise ValueError('One resolved source attachment required for radial relation')
    rays=np.asarray([centre,matches[0]['anchor0']])-origin;rays-=np.outer(rays@up,up)
    lengths=np.linalg.norm(rays,axis=1)
    if (lengths<=1e-6).any():raise ValueError('Support/attachment direction is unresolved at the body centre')
    cosine=float(np.clip((rays[0]@rays[1])/np.prod(lengths),-1.,1.))
    return {'pass':bool(interval[0]<=cosine<=interval[1]),'cosine':cosine,
            'angle_deg':float(np.degrees(np.arccos(cosine))),'cosine_interval':interval.tolist(),
            'support_area_centroid':centre.tolist(),'attachment_anchor':matches[0]['anchor0'],
            'scope':'Actual supplied geometry versus reviewed broad relative direction; no source dimensions, exact pose, material or task approval.'}


def measure(path,plan,root):
    path,root=Path(path),Path(root)
    with np.load(path/'mesh.npz') as mesh:
        parts=json.loads(str(mesh['links_json']));rows=[]
        for check in plan:
            source=root/check['source']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=check['source_sha256']:
                raise ValueError('Source geometry observation is stale')
            p=next(x for x in parts if x['name']==check['link'])
            a,n,b,count=[p[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
            origin=np.zeros(3)
            if check.get('origin_link'):
                anchor=next(x for x in parts if x['name']==check['origin_link'] and x['kind']=='rigid')
                av=mesh['V'][anchor['vertex_start']:anchor['vertex_start']+anchor['vertex_count']]
                origin=(av.min(0)+av.max(0))/2;origin[2]=av[:,2].min()
            kind=check.get('type','segment_occupancy')
            if kind=='radial_relation':
                rows.append({'id':check['id'],'source':str(source),**radial_relation(mesh,p,check,origin)})
                continue
            if kind!='segment_occupancy':raise ValueError('Unknown reviewed source geometry check type')
            measured=occupancy(mesh['V'][a:a+n],mesh['F'][b:b+count]-a,
                               np.asarray(check['starts'])+origin,np.asarray(check['ends'])+origin)
            expected=check['occupied']
            if (len(expected)!=len(measured['occupied']) or any(type(x) is not bool for x in expected)
                    or not check.get('basis')):raise ValueError('Reviewed source labels and basis required for each segment')
            rows.append({'id':check['id'],'pass':measured['occupied']==expected,
                         'expected_occupied':expected,'source':str(source),'probe_origin':origin.tolist(),**measured})
    result={'pass':all(r['pass'] for r in rows),'checks':rows,
            'mesh_sha256':hashlib.sha256((path/'mesh.npz').read_bytes()).hexdigest(),
            'plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()}
    return result



class Checked:
    def __init__(self,builder,plan,output,root):
        self.builder,self.plan,self.output,self.root=builder,plan,Path(output),Path(root)
        self.spec=builder.spec

    def __getattr__(self,name):return getattr(self.builder,name)

    def evaluate(self,*args,**kwargs):
        path,report=self.builder.evaluate(*args,**kwargs)
        result=measure(path,self.plan,self.root)
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/(path.name+'_'+result['plan_sha256'][:12]+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        if not result['pass']:raise ValueError('Source solid/void geometry checks failed: '+json.dumps(result['checks']))
        return path,{**report,'source_geometry_checks':result}
