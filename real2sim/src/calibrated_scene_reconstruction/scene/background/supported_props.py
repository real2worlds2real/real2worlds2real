"""Fit background prop envelopes to calibrated source masks on a known support.

The agent chooses the semantic parts and the supporting background surface.
These fits use silhouettes only; they do not import dimensions or certify a
shape, material, physical contact or novel-view appearance.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def observation_scope(row,part=None):
    """A named part fit cannot certify its enclosing object's full extent."""
    if part is None:
        from calibrated_scene_reconstruction.scene.background.observations import geometry_review
        geometry_review(row)
    elif (not isinstance(part,str) or not part.strip() or row.get('geometry_fit_scope')!='surface_part'
          or row.get('geometry_fit_part')!=part or not isinstance(row.get('geometry_fit_review'),str)
          or not row['geometry_fit_review'].strip()):
        raise ValueError('Explicit matching reviewed semantic part required; do not fit a complete prop to a partial mask')


def corners(position, size, yaw):
    local=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [0.,1.]])*size
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return local@R.T+position


def perimeter(poly,count=64):
    poly=np.asarray(poly,float);delta=np.roll(poly,-1,axis=0)-poly;length=np.linalg.norm(delta,axis=1)
    cumulative=np.r_[0.,np.cumsum(length)];distance=np.arange(count)*cumulative[-1]/count
    ids=np.minimum(np.searchsorted(cumulative,distance,side='right')-1,len(poly)-1)
    return poly[ids]+delta[ids]*((distance-cumulative[ids])/np.maximum(length[ids],1e-12))[:,None]


def solve(observations, position, size, yaw, support_z):
    """Use three separated calibrated masks; truncated edges are lower bounds."""
    position=np.asarray(position,float);size=np.asarray(size,float)
    if (len(observations)<3 or position.shape!=(3,) or size.shape!=(3,) or
            not np.isfinite(np.r_[position,size,yaw,support_z]).all() or np.any(size<=0)):
        raise ValueError('Three source views and a finite positive background hypothesis required')
    centres=np.array([r['C'] for r in observations]);directions=position-centres
    unit=directions/np.linalg.norm(directions,axis=1)[:,None]
    if np.linalg.norm(centres[:,None]-centres[None,:],axis=2).max()<.05 or np.degrees(np.arccos(np.clip(unit@unit.T,-1,1))).max()<2:
        raise ValueError('Independent source baselines required')
    data=[]
    for row in observations:
        mask=np.asarray(row['mask'],bool);yy,xx=np.nonzero(mask);h,w=mask.shape
        if len(xx)<64:raise ValueError('Usable source silhouette required')
        target=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
        edge=np.array([target[0]>w*.01,target[1]>h*.01,target[2]<w*.99,target[3]<h*.99])
        hidden=np.asarray(row.get('hidden_edges',np.zeros(4,bool)))
        if hidden.shape!=(4,) or hidden.dtype!=bool:
            raise ValueError('Four reviewed boolean edge-visibility flags required')
        edge &= ~hidden
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:raise ValueError('Usable source boundary required')
        boundary=perimeter(max(contours,key=cv2.contourArea)[:,0])
        outside=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,5).astype(float)
        data.append((row,target,edge,np.array([w,h,w,h]),boundary,outside,hidden))
    def unpack(x):return np.r_[position[:2]+x[:2],support_z],size*np.exp(x[3:]),yaw+x[2]
    def residual(x,prior=True,shape=True):
        p,d,a=unpack(x);vertices=corners(p,d,a);values=[]
        for row,target,edge,norm,boundary,outside,hidden in data:
            pc=(vertices-row['C'])@row['R'];uv=pc@row['K'].T
            if pc[:,2].min()<=.05:values.extend([10.]*(132 if shape else 4));continue
            uv=uv[:,:2]/uv[:,2,None];box=np.r_[uv.min(0),uv.max(0)]
            delta=(box-target)/norm
            # A source-raster boundary only constrains undershoot, not unseen extent.
            delta=np.where(edge,delta,np.maximum(delta*[1,1,-1,-1],0)*[1,1,-1,-1])
            values.extend(delta)
            if shape:
                poly=uv[ConvexHull(uv).vertices];lines=np.roll(poly,-1,axis=0)-poly
                relative=boundary[:,None,:]-poly[None,:,:]
                # A bbox can match while the projected body collapses to a
                # diagonal strip. Require actual source-mask coverage too.
                signed=(lines[None,:,1]*relative[:,:,0]-lines[None,:,0]*relative[:,:,1])/np.maximum(np.linalg.norm(lines,axis=1),1e-12)
                values.extend(np.maximum(0,signed.max(1))/np.linalg.norm(norm[:2]))
                samples=perimeter(poly);valid=(samples[:,0]>=0)&(samples[:,1]>=0)&(samples[:,0]<norm[0])&(samples[:,1]<norm[1])
                distance=map_coordinates(outside,[samples[:,1],samples[:,0]],order=1,mode='nearest',prefilter=False)
                # An occluder is not a physical edge of the supported object.
                # Retain positive visible-mask coverage and known edge bounds;
                # do not shrink hidden geometry into the occluder's silhouette.
                values.extend(np.where(valid & ~hidden.any(),distance,0)/np.linalg.norm(norm[:2]))
        if prior:values.extend(np.r_[x[:2]*.003,x[2]*.001,x[3:]*.003])
        return values
    limit=max(size)*1.5;best=None
    incomplete=any(not edge.all() for _,_,edge,_,_,_,_ in data)
    for offset in [0.,np.pi/2,-np.pi/2,np.pi]:
        x=np.r_[0.,0.,offset,0.,0.,0.]
        fit=least_squares(residual,x,bounds=([-limit,-limit,-2*np.pi,*[-1.2]*3],[limit,limit,2*np.pi,*[1.2]*3]),loss='soft_l1',f_scale=.01,max_nfev=160)
        # Equally plausible visible halves must not win solely by discarding
        # the weak shape prior used during optimization of hidden extents.
        cost=float(np.mean(np.square(residual(fit.x,incomplete))))
        if best is None or cost<best[0]:best=(cost,fit.x)
    p,d,a=unpack(best[1]);checks=[]
    for row,_,_,_,_,_,_ in data:
        pc=(corners(p,d,a)-row['C'])@row['R'];uv=pc@row['K'].T;uv=uv[:,:2]/uv[:,2,None];mask=np.zeros(row['mask'].shape,'uint8');cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype('int32')),1);actual=mask>0;target=np.asarray(row['mask'],bool)
        checks.append(float((actual&target).sum()/max(1,(actual|target).sum())))
    return {'position':p.tolist(),'size':d.tolist(),'yaw_deg':float(np.degrees(a)%360),
        'normalized_bbox_rmse':float(np.sqrt(np.mean(np.square(residual(best[1],False,False))))),
        'bound_hit':bool(np.any(abs(best[1][:2])>=limit-1e-5) or abs(best[1][2])>=2*np.pi-1e-5 or np.any(abs(best[1][3:])>=1.2-1e-5)),
        'source_silhouette_iou':checks,'approved':False,'views':len(data),
            'prior_guided_hidden_extent':incomplete,
            'source_visible_edges':[{'frame':row.get('frame',i),'visible_edges':edge.tolist()}
                                    for i,(row,_,edge,_,_,_,_) in enumerate(data)],
        'scope':'Calibrated silhouette envelope only. Agent part construction, support, source/free-view material and geometry review remain required.'}


def prepare(session, observations, hypotheses, output):
    from calibrated_scene_reconstruction.scene.background import occluded_edges
    out=Path(output).resolve();obs=Path(observations).resolve();hyp=Path(hypotheses).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen background proposal required')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']));specs=json.loads(hyp.read_text())['props']
    records=json.loads(obs.read_text())['records'];cfg=session.config('environment.json');native={e['id']:e for e in cfg['parameters']['entities']}
    out.mkdir(parents=True);implementation=out/'solver.py';implementation.write_bytes(Path(__file__).read_bytes())
    environment_snapshot=out/'environment.json';environment_snapshot.write_bytes((session.config_dir/'environment.json').read_bytes())
    edge_snapshot=out/'edge_visibility.py';edge_snapshot.write_bytes(Path(occluded_edges.__file__).read_bytes())
    inputs={str(p.resolve()):digest(p) for p in [camera,obs,hyp,implementation,environment_snapshot,edge_snapshot]};results=[]
    for spec in specs:
        support=native[spec['on']]
        if support['kind'] not in ('desk','cabinet','cabinet_open','shelf'):raise ValueError('Background horizontal support required')
        rows=[];seen=set()
        for row in records:
            if row['entity']!=spec['entity'] or not row.get('mask') or row['frame']in seen:continue
            observation_scope(row,spec.get('part'))
            for key in ['source','mask']:
                if digest(row[key])!=row[key+'_sha256']:raise ValueError('Source observation changed')
                inputs[str(Path(row[key]).resolve())]=row[key+'_sha256']
            seen.add(row['frame']);i=names.index(row['frame']);mask=cv2.imread(row['mask'],0)
            if mask.shape!=tuple(C['size'][i][::-1]):raise ValueError('Calibrated mask raster required')
            rows.append({'K':C['K'][i],'R':C['R_world_cam'][i],'C':C['centre'][i],'mask':mask>128,
                         'hidden_edges':occluded_edges.hidden(row),'frame':row['frame']})
        result=solve(rows,spec['position'],spec['size'],np.radians(spec['yaw_deg']),support['position'][2]+support['size'][2])
        result.update(entity=spec['entity'],on=spec['on'],source_frames=sorted(seen),
                      geometry_fit_scope=('surface_part' if spec.get('part') else
                          'occluded_object' if any(r.get('geometry_fit_scope')=='occluded_object' for r in records if r['entity']==spec['entity']) else 'whole_object'),part=spec.get('part'))
        if spec.get('part'):result['scope']='Named semantic part envelope only. It does not constrain the full parent prop; reconstruct and physically/source/free-view check its remaining assembly separately.'
        results.append(result)
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Prop evidence changed')
    record={'version':1,'results':results,'input_sha256':inputs,'implementation_snapshot':str(implementation),
            'environment_snapshot':str(environment_snapshot),'protected_task_state':protected_state(session),'pass':False}
    (out/'proposals.json').write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('observations');p.add_argument('hypotheses');p.add_argument('--output',required=True);a=p.parse_args()
    print('Unapproved prop envelopes',len(prepare(Session(a.session),a.observations,a.hypotheses,a.output)['results']))


if __name__=='__main__':main()
