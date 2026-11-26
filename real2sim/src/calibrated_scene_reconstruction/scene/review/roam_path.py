"""One continuous, levelled free-roam camera path through a captured corridor.

This only prepares presentation cameras. Reconstruction inputs, sampling,
calibration, native/GS assets and all scene poses remain unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation,RotationSpline,Slerp

from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session


def smooth_route(centres,rotations,count,sigma=4.,height_offset=.12):
    C=np.asarray(centres,float);R=np.asarray(rotations,float)
    if C.ndim!=2 or C.shape[1]!=3 or len(C)<8 or R.shape!=(len(C),3,3) or count<48 or not np.isfinite(C).all() or not np.isfinite(R).all():
        raise ValueError('At least8 finite registered camera poses and a useful frame count required')
    C=gaussian_filter1d(C,sigma,axis=0,mode='nearest');C[:,2]+=height_offset
    f=R[:,:,2];yaw=gaussian_filter1d(np.unwrap(np.arctan2(f[:,1],f[:,0])),sigma,mode='nearest')
    pitch=np.clip(gaussian_filter1d(np.arctan2(f[:,2],np.linalg.norm(f[:,:2],axis=1)),sigma,mode='nearest'),np.radians(-15),np.radians(5))
    forward=np.c_[np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)]
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right,axis=1)[:,None];down=np.cross(forward,right)
    rotations=Rotation.from_matrix(np.stack([right,down,forward],axis=-1))
    angles=(rotations[:-1].inv()*rotations[1:]).magnitude()
    # Allocate time to translation AND turning, so a pan does not become a cut
    # or a very fast spin at an almost stationary capture position.
    step=np.linalg.norm(np.diff(C,axis=0),axis=1)+.8*angles
    travel=np.r_[0,np.cumsum(np.maximum(step,1e-5))]
    t=np.linspace(0,1,count);ramp=.025;speed=np.minimum(t/ramp,1)*np.minimum((1-t)/ramp,1)
    u=np.cumsum(speed);u=(u-u[0])/(u[-1]-u[0])*travel[-1]
    positions=PchipInterpolator(travel,C,axis=0)(u);orientations=RotationSpline(travel,rotations)(u).as_matrix()
    return positions,orientations


def ceiling_sweeps(rotations):
    """Smooth upward looks while retaining the continuous walking/yaw route."""
    R=np.asarray(rotations);forward=R[:,:,2].copy();t=np.linspace(0,1,len(R))
    pitch=np.arctan2(forward[:,2],np.linalg.norm(forward[:,:2],axis=1))
    lift=np.zeros(len(R))
    for centre,width,degrees in [(.40,.07,50.),(.73,.075,50.)]:
        u=np.clip((t-centre)/width,-1,1)
        lift+=np.radians(degrees)*np.cos(u*np.pi/2)**2
    pitch+=lift;yaw=np.arctan2(forward[:,1],forward[:,0])
    forward=np.c_[np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)]
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right,axis=1)[:,None]
    return np.stack([right,np.cross(forward,right),forward],axis=-1)


def inspect_assemblies(centres,rotations,targets,fps,fov,aspect,obstacles=None):
    """Frame completed assemblies, using bounded collision-checked pullbacks."""
    C=np.array(centres,copy=True);R=np.array(rotations,copy=True);records=[];tan=np.tan(np.radians(fov/2))
    boxes=np.asarray(obstacles,float) if obstacles is not None else np.empty((0,2,3))
    def clear(points):
        return not len(boxes) or not np.any(np.all((points[:,None,:]>=boxes[None,:,0]-.10)&(points[:,None,:]<=boxes[None,:,1]+.10),axis=-1))
    def look_at(position,target):
        f=target-position;length=np.linalg.norm(f)
        if length<1e-8:return None
        f/=length;right=np.cross(f,[0,0,1]);length=np.linalg.norm(right)
        if length<1e-8:return None
        right/=length;return np.stack([right,np.cross(f,right),f],axis=-1)
    for entity,corners in targets.items():
        corners=np.asarray(corners,float);target=corners.mean(0);options=[];window=int(7*fps)
        for i in range(int(len(C)*.12),int(len(C)*.86),3):
            if .31<i/len(C)<.50 or .62<i/len(C)<.82:continue
            direction=C[i]-target;direction[2]=0;length=np.linalg.norm(direction)
            if length<.1:continue
            direction/=length
            for pullback in (0.,.5,1.,1.5,2.):
                position=C[i]+direction*pullback;look=look_at(position,target)
                if look is None:continue
                q=(corners-position)@look;align=float(look[:,2]@R[i,:,2])
                if (q[:,2]>.1).all() and align>.5:
                    extent=np.max(abs(q[:,:2])/q[:,2,None]/[tan,tan/aspect],axis=0)
                    if np.max(extent)<.9:options.append((float(extent.max())-.2*(1-align)-.12*pullback,i,direction*pullback))
        chosen=None
        for _,index,offset in sorted(options,key=lambda row:row[0],reverse=True):
            ids=np.arange(max(0,index-window),min(len(C),index+window+1));weight=np.cos((ids-index)/window*np.pi/2)**2
            positions=C[ids]+weight[:,None]*offset
            if clear(positions):chosen=(index,offset,ids,weight,positions);break
        if chosen is None:raise ValueError('No collision-free corridor can frame complete assembly: '+entity)
        index,offset,ids,weights,positions=chosen;C[ids]=positions
        for i,weight in zip(ids,weights):
            look=look_at(C[i],target)
            if look is not None:R[i]=Slerp([0,1],Rotation.from_matrix(np.stack([R[i],look])))([weight]).as_matrix()[0]
        records.append({'entity':entity,'frame':int(index),'all_bounds_in_frustum':True,'pullback_m':float(np.linalg.norm(offset)),
                        'camera_clearance_m':.10,'scope':'Actual-mesh AABB clearance on the inspection maneuver; frustum framing is not proof of unoccluded visibility or photorealism.'})
    return C,R,records


def build(session,output,seconds=90.,fps=24,end_frame=None,fov=80.,height_offset=.12,width=1920,show_ceiling=False):
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use its environment derivative')
    if not 30<=seconds<=240 or not 12<=fps<=60 or not 60<=fov<=100 or not 0<=height_offset<=.3 or not 640<=width<=3840 or width%32:raise ValueError('Bounded roam duration, FPS, FOV, dimensions and lift required')
    camera=session/'parsing/posed_frames/cameras.npz';z=np.load(camera);names=list(map(str,z['names']))
    groups={name.split('_')[0] for name in names};capture=max(groups,key=lambda key:sum(n.startswith(key+'_') for n in names))
    ids=sorted([i for i,n in enumerate(names) if n.startswith(capture+'_') and (end_frame is None or int(Path(n).stem.split('_')[-1])<=end_frame)],key=lambda i:int(Path(names[i]).stem.split('_')[-1]))
    n=round(seconds*fps);C,R=smooth_route(z['centre'][ids],z['R_world_cam'][ids],n,height_offset=height_offset)
    if show_ceiling:R=ceiling_sweeps(R)
    inspections=[];inspection_inputs={}
    requirements=session.config('environment.json')['parameters'].get('assembly_requirements',{})
    if requirements:
        pointer=session/'qa/environment_physics/twin/current.json';record=Path(json.loads(pointer.read_text())['path']);jobpath=record.parent/'job.json';job=json.loads(jobpath.read_text());geometry=np.load(job['geometry']);targets={}
        for entity in requirements:
            vertices=np.concatenate([geometry[f'v{i}'] for i,row in enumerate(job['parts']) if row['entity']==entity]);lo,hi=vertices.min(0),vertices.max(0)
            targets[entity]=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        if job.get('native_manifest_sha256')!=digest(session/'scene/environment/twin/current.json'):raise ValueError('Inspection cameras require current exported geometry')
        obstacles=np.array([[geometry[f'v{i}'].min(0),geometry[f'v{i}'].max(0)] for i,row in enumerate(job['parts'])])
        C,R,inspections=inspect_assemblies(C,R,targets,fps,fov,16/9,obstacles)
        inspection_inputs={str(p.resolve()):digest(p) for p in [pointer,record,jobpath,Path(job['geometry'])]}
    W,H=width,width*9//16;f=W/(2*np.tan(np.radians(fov/2)));K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
    distance=float(np.linalg.norm(np.diff(C,axis=0),axis=1).sum());angles=(Rotation.from_matrix(R[:-1]).inv()*Rotation.from_matrix(R[1:])).magnitude()
    stats={'translation_m':distance,'rotation_deg':float(np.degrees(angles).sum()),'max_speed_m_s':float(np.linalg.norm(np.diff(C,axis=0),axis=1).max()*fps),
           'max_angular_speed_deg_s':float(np.degrees(angles).max()*fps),'assembly_inspections':inspections,'cuts':0,'fov_deg':fov,'source_capture':capture,'source_camera_count':len(ids),'ceiling_sweeps':show_ceiling}
    if stats['max_speed_m_s']>1. or stats['max_angular_speed_deg_s']>60:raise ValueError('Increase duration: camera motion is too fast')
    cfg=session.config_dir/'environment.json';entities=session.config('environment.json')['parameters']['entities'];seen=np.zeros(len(entities),bool)
    targets=np.array([np.asarray(e['position'])+[0,0,e['size'][2]/2] for e in entities]);tan=np.tan(np.radians(fov/2))
    for c,r in zip(C,R):
        q=(targets-c)@r;seen|=(q[:,2]>.2)&(abs(q[:,0])<q[:,2]*tan)&(abs(q[:,1])<q[:,2]*tan*H/W)
    stats['frustum_coverage']={'entity_centres_inside_view':int(seen.sum()),'total_entities':len(entities),'unseen':[e['id'] for e,visible in zip(entities,seen) if not visible],
                              'scope':'Frustum-only diagnostic; not proof of unoccluded visibility or reconstructed completeness.'}
    record={'purpose':'One uninterrupted free-roam presentation; not video input selection','width':W,'height':H,'fps':fps,'seconds':seconds,'K':K.tolist(),
            'source_size':[W,H],'trajectory':[{'centre':c.tolist(),'R_world_cam':r.tolist()} for c,r in zip(C,R)],'path_stats':stats,
            'input_sha256':{**inspection_inputs,**{str(p.resolve()):digest(p) for p in [camera,cfg,Path(__file__)]}},'scope':'Levelled and smoothed camera corridor with small height lift, arc-length/rotation timing and a wider lens; no cuts or resets.'}
    output=Path(output).resolve()
    if not output.is_relative_to(session.dir.resolve()) or output.exists():raise ValueError('Use a new session-local camera path')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record)+'\n');return stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=90.);p.add_argument('--fps',type=int,default=24);p.add_argument('--end-frame',type=int);p.add_argument('--fov',type=float,default=80.);p.add_argument('--width',type=int,default=1920)
    p.add_argument('--show-ceiling',action='store_true',help='Include smooth upward looks during the uninterrupted room route')
    a=p.parse_args();print(json.dumps(build(Session(a.session),a.output,a.seconds,a.fps,a.end_frame,a.fov,width=a.width,show_ceiling=a.show_ceiling),indent=2))


if __name__=='__main__':main()
