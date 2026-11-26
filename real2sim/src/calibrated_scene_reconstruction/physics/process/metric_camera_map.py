"""Metric static maps anchored by independently reviewed robot camera proposals.

A per-camera sequence may have too little translation to triangulate depth.
Use tracks shared across metric anchor cameras, then refine the small observed
camera motion. This remains a proposal; source camera, clock and physical
response acceptance are separate. Moving target motion is never an input.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

from .contract import digest, require, write


def solve(cameras, tracks, anchors, groups, *, error_px=6., min_angle_deg=2., budget=800):
    names=list(map(str,cameras['names']));lookup={name:i for i,name in enumerate(names)}
    require(len(names)==len(lookup) and set(groups)==set(names), 'Distinct camera frames and complete groups required')
    require(len(anchors)>=2 and set(anchors)<=set(names), 'At least two existing metric anchor cameras required')
    require(0<error_px<=12 and 0<min_angle_deg<=90 and type(budget) is int and 10<=budget<=2000,
            'Bounded source uncertainty, ray angle and optimizer budget required')
    old_R=np.asarray(cameras['R_world_cam'],float);old_C=np.asarray(cameras['centre'],float)
    require(old_R.shape==(len(names),3,3) and old_C.shape==(len(names),3)
            and np.isfinite(old_R).all() and np.isfinite(old_C).all(), 'Finite source camera poses required')
    anchored={};by_group={}
    for name,value in anchors.items():
        K=np.asarray(value['K'],float);R=np.asarray(value['R_world_cam'],float);C=np.asarray(value['centre'],float)
        require(value.get('basis') and K.shape==(3,3) and R.shape==(3,3) and C.shape==(3,)
                and np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(C).all()
                and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2],[0,0,1])
                and np.allclose(R.T@R,np.eye(3),atol=1e-6) and np.isclose(np.linalg.det(R),1.),
                'Metric anchors need explicit evidence and finite proper poses/intrinsics')
        require(groups[name] not in by_group, 'Use one pose anchor per independently calibrated camera group')
        i=lookup[name];anchored[i]=(K,R,C);by_group[groups[name]]=i
    require(set(by_group)==set(groups.values()), 'Every independent camera group needs its own metric anchor')
    pairs=[(a,b)for a in anchored for b in anchored if a<b]
    a,b=max(pairs,key=lambda pair:np.linalg.norm(anchored[pair[0]][2]-anchored[pair[1]][2]))
    native_baseline=np.linalg.norm(anchored[a][2]-anchored[b][2]);map_baseline=np.linalg.norm(old_C[a]-old_C[b])
    require(native_baseline>.01 and map_baseline>1e-8, 'Distinct camera centres must establish a useful baseline')
    scale=native_baseline/map_baseline;R=np.empty_like(old_R);C=np.empty_like(old_C);K=np.zeros_like(old_R)
    for i,name in enumerate(names):
        j=by_group[groups[name]];intrinsic,orientation,centre=anchored[j];A=orientation@old_R[j].T
        R[i]=A@old_R[i];C[i]=centre+scale*A@(old_C[i]-old_C[j]);K[i]=intrinsic
    P=np.array([K[i]@np.c_[R[i].T,-R[i].T@C[i]]for i in range(len(names))])
    points=[];point_ids=[];oc=[];op=[];uv=[];train=[];declined=[]
    for item in tracks:
        rows={lookup[name]:np.asarray(pixel,float) for name,pixel in item['observations'].items() if name in lookup}
        require(all(pixel.shape==(2,) and np.isfinite(pixel).all() for pixel in rows.values()), 'Finite source pixels required')
        ids=sorted(rows.keys()&anchored.keys())
        if len(ids)<2:
            declined.append({'id':item['id'],'reason':'Depth is not observed across independent metric anchor cameras'})
            continue
        A=np.array([row for i in ids for row in [rows[i][0]*P[i,2]-P[i,0],rows[i][1]*P[i,2]-P[i,1]]])
        _,_,vt=np.linalg.svd(A)
        if abs(vt[-1,3])<1e-10:
            declined.append({'id':item['id'],'reason':'Point at infinity'});continue
        point=vt[-1,:3]/vt[-1,3];error=[];depth=[];rays=[]
        for i in ids:
            q=P[i]@np.r_[point,1];depth.append(float(q[2]));error.append(float(np.linalg.norm(q[:2]/q[2]-rows[i])) if q[2] else float('inf'))
            direction=point-C[i];rays.append(direction/max(np.linalg.norm(direction),1e-12))
        angle=max(np.degrees(np.arccos(np.clip(rays[i]@rays[j],-1.,1.))) for i in range(len(ids)) for j in range(i))
        if min(depth)<=0 or max(error)>error_px or angle<min_angle_deg:
            declined.append({'id':item['id'],'reason':'Cross-view geometry failed','error_px':error,'depth':depth,'angle_deg':float(angle)});continue
        j=len(points);points.append(point);point_ids.append(item['id']);eligible=sorted(set(rows)-anchored.keys())
        hold=eligible[j%len(eligible)] if len(rows)>=4 and eligible else None
        for i,pixel in sorted(rows.items()):oc.append(i);op.append(j);uv.append(pixel);train.append(i!=hold)
    require(len(points)>=6, 'Too few cross-camera metric tracks; acquire additional stationary observations')
    points=np.asarray(points);oc=np.asarray(oc);op=np.asarray(op);uv=np.asarray(uv);train=np.asarray(train,bool)
    require(all(np.sum((oc==i)&train)>=6 for i in range(len(names))), 'Each camera pose needs at least six training tracks')
    free=[i for i in range(len(names)) if i not in anchored];index={j:i for i,j in enumerate(free)};nc=len(free)*6
    x0=np.r_[np.c_[Rotation.from_matrix(R[free]).as_rotvec(),C[free]].ravel(),points.ravel()] if free else points.ravel()
    sparsity=lil_matrix((2*sum(train),len(x0)),dtype=int)
    for k,(i,j)in enumerate(zip(oc[train],op[train])):
        if i in index:sparsity[2*k:2*k+2,6*index[i]:6*index[i]+6]=1
        sparsity[2*k:2*k+2,nc+3*j:nc+3*j+3]=1
    def unpack(x):
        rr=R.copy();cc=C.copy()
        if free:
            values=x[:nc].reshape(-1,6);rr[free]=Rotation.from_rotvec(values[:,:3]).as_matrix();cc[free]=values[:,3:]
        return rr,cc,x[nc:].reshape(-1,3)
    def projection(x):
        rr,cc,pp=unpack(x);pc=np.einsum('ni,nij->nj',pp[op]-cc[oc],rr[oc]);q=np.einsum('nij,nj->ni',K[oc],pc)
        return q[:,:2]/np.maximum(q[:,2,None],1e-8),pc[:,2]
    initial,depth=projection(x0)
    require(np.all(depth>0), 'Initial metric map crosses a camera plane; revise anchors instead of fitting a clipped projection')
    fit=least_squares(lambda x:(projection(x)[0][train]-uv[train]).ravel(),x0,jac_sparsity=sparsity.tocsr(),
        x_scale='jac',loss='soft_l1',f_scale=2.,max_nfev=budget,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    rr,cc,pp=unpack(fit.x);pred,depth=projection(fit.x);errors=np.linalg.norm(pred-uv,axis=1)
    def quantile(values):return np.percentile(values,[50,90,95,100]).tolist() if len(values) else None
    report={'accepted':False,'task_approved':False,'points':len(points),'observations':len(uv),
            'training_error_px':quantile(errors[train]),'heldout_error_px':quantile(errors[~train]),
            'behind_camera_observations':int(np.sum(depth<=0)),'optimizer_converged':bool(fit.success),
            'optimizer_message':fit.message,'evaluations':int(fit.nfev),'declined_tracks':declined,
            'camera_rows':[{'name':name,'observations':int(sum(oc==i)),
                'error_px':quantile(errors[oc==i])}for i,name in enumerate(names)],
            'scope':__doc__.strip()}
    return {'names':np.array(names),'K':K,'R_world_cam':rr,'centre':cc}, {
        'world':pp,'point_ids':np.array(point_ids),'observation_camera':oc,'observation_point':op,
        'pixels':uv,'training':train,'error_px':errors},report


def run(source, anchors_file, output, *, model='0', budget=800):
    import pycolmap
    source=Path(source);anchors_file=Path(anchors_file);output=Path(output)
    require(not output.exists(), 'Use a fresh metric camera proposal directory')
    inputs=json.loads((source/'inputs.json').read_text());anchors=json.loads(anchors_file.read_text())['anchors']
    groups={row['image']:row['camera'] for row in inputs['records']}
    reconstruction=pycolmap.Reconstruction(source/'sparse'/model);names=sorted(im.name for im in reconstruction.images.values())
    images={im.name:im for im in reconstruction.images.values()};cameras={'names':names,
        'R_world_cam':np.array([images[name].cam_from_world().rotation.matrix().T for name in names]),
        'centre':np.array([images[name].projection_center() for name in names])}
    tracks=[{'id':int(key),'observations':{reconstruction.images[e.image_id].name:
        reconstruction.images[e.image_id].points2D[e.point2D_idx].xy.tolist() for e in point.track.elements}}
        for key,point in reconstruction.points3D.items()]
    for row in inputs['records']:
        require(digest(source/'images'/row['image'])==row['prepared_sha256'], 'Static source image changed')
    result,points,report=solve(cameras,tracks,anchors,{n:groups[n] for n in names},budget=budget)
    output.mkdir(parents=True);np.savez_compressed(output/'cameras.npz',**result);np.savez_compressed(output/'points.npz',**points)
    report['inputs']={str(p.resolve()):digest(p) for p in [source/'inputs.json',anchors_file,Path(__file__),
        *sorted((source/'sparse'/model).glob('*.bin'))]}
    write(output/'result.json',report)
    return report
