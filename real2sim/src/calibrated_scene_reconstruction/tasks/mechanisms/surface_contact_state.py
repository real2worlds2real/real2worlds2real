"""Geometric contact-neighbourhood checks; never normal-force evidence."""
import numpy as np


def validate(request):
    if (not isinstance(request,dict) or set(request)!={'partners','centre_fraction','size_m','maximum_gap_m','maximum_penetration_m','minimum_common_area_fraction'}
            or not isinstance(request['partners'],list) or len(request['partners'])<1
            or len(set(request['partners']))!=len(request['partners'])
            or any(not isinstance(name,str) or not name for name in request['partners'])):
        raise ValueError('Explicit distinct contact partners and a material neighbourhood required')
    c=np.asarray(request['centre_fraction'],float);size=np.asarray(request['size_m'],float)
    gap=request['maximum_gap_m'];fraction=request['minimum_common_area_fraction'];penetration=request['maximum_penetration_m']
    if (c.shape!=(2,) or size.shape!=(2,) or not np.isfinite([c,size]).all()
            or (c<0).any() or (c>1).any() or (size<=0).any()
            or not np.isfinite([gap,fraction,penetration]).all()
            or not 0<gap<=.01 or not 0<fraction<=1 or not 0<=penetration<=.01):
        raise ValueError('Bounded positive contact distance, area and material region required')


def measure(points,rest,faces,surfaces,request):
    validate(request)
    p,r,f=np.asarray(points,float),np.asarray(rest,float),np.asarray(faces,int)
    if p.shape!=r.shape or p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite([p,r]).all():
        raise ValueError('Matching actual and immutable material points required')
    centre=r[:,:2].min(0)+np.asarray(request['centre_fraction'])*np.ptp(r[:,:2],axis=0)
    tri=r[f];selected=(abs(tri.mean(1)[:,:2]-centre)<=np.asarray(request['size_m'])/2).all(1)
    if not selected.any():raise ValueError('Contact region contains no material triangles')
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)[selected]/2
    samples=p[f[selected]].mean(1);common=np.ones(len(samples),bool);rows={}
    for name in request['partners']:
        if name not in surfaces:raise ValueError('Actual contact partner missing: '+name)
        distance=np.asarray(surfaces[name](samples),float)
        if distance.shape!=(len(samples),) or not np.isfinite(distance).all():
            raise ValueError('Actual signed contact distances required')
        # Deep penetration is never a contact witness; the ordinary physical
        # solid-validity check remains independently mandatory.
        near=(distance>=-request['maximum_penetration_m'])&(distance<=request['maximum_gap_m'])
        common &= near
        rows[name]={'minimum_distance_m':float(distance.min()),
                    'near_material_area_fraction':float(area[near].sum()/area.sum())}
    fraction=float(area[common].sum()/area.sum())
    return {'pass':fraction>=request['minimum_common_area_fraction'],
            'common_near_material_area_fraction':fraction,'region_area_m2':float(area.sum()),
            'partners':rows,'normal_pressure_proven':False,'task_approved':False,
            'scope':'Same actual material neighbourhood near every declared surface; geometry alone does not prove loaded contact or retention'}


class Native:
    def __init__(self,ctx,assembly,parts,cloth_link,request):
        from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface
        from calibrated_scene_reconstruction.sim.soft_surface import from_view
        validate(request)
        self.ctx,self.assembly,self.cloth_link,self.request=ctx,assembly,cloth_link,request
        self.part=next(p for p in parts if p['name']==cloth_link)
        self.rigid={};self.soft={}
        for name in request['partners']:
            part=next(p for p in parts if p['name']==name)
            if part['kind']=='rigid':self.rigid[name]=SignedSurface(part['V'],part['F'])
            elif part['kind'] in {'deformable','rope'}:
                view=ctx._soft[assembly+'/'+name]
                root=str(ctx.ts.objects[assembly].GetPath())
                self.soft[name]=from_view(view,ctx.stage.GetPrimAtPath(root+'/'+name+'/surface'))
            else:raise ValueError('Contact partner must be an actual closed rigid or flexible solid')

    def check(self):
        surfaces={}
        for name,value in self.rigid.items():
            p,r=self.ctx.pose(self.assembly+'/'+name)
            surfaces[name]=lambda points,value=value,p=p,r=r:value((points-p)@r)
        for name,value in self.soft.items():
            value.refresh();surfaces[name]=value.distance
        return measure(self.ctx.nodes(self.assembly+'/'+self.cloth_link),
            self.part['cloth_rest']['points'],self.part['F'],surfaces,self.request)
