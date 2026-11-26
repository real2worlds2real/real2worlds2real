"""Display recorded external force commands; never infer new dynamics or loads."""
from pathlib import Path

import cv2
import numpy as np

from .contract import digest, read, require


COLOR = (255, 195, 45)  # RGB, distinct from source and simulation contours.


def sample_commands(commands, times, dt):
    """Use actual nearest recorded commands, including the final step boundary."""
    commands=np.asarray(commands,float);times=np.asarray(times,float)
    require(commands.ndim==2 and commands.shape[1]==7 and len(commands)>=2
            and np.isfinite(commands).all() and np.all(np.diff(commands[:,0])>0),
            'Force commands must be finite increasing rows [t, Fx, Fy, Fz, px, py, pz] in world SI units')
    require(times.ndim==1 and np.isfinite(times).all() and np.isfinite(dt) and dt>0,
            'Finite frame times and positive native timestep required')
    right=np.clip(np.searchsorted(commands[:,0],times),0,len(commands)-1)
    left=np.maximum(right-1,0)
    ids=np.where(abs(commands[left,0]-times)<=abs(commands[right,0]-times),left,right)
    # State recording precedes drive(t); the last frame may have no following command.
    require(np.all(abs(commands[ids,0]-times)<=dt+1e-8),
            'Force records do not cover the native frame times; no interpolation or clock shift is allowed')
    return commands[ids].copy()


def project_force(point, force, K, T_world_cam, dist, meters_per_newton):
    """Project a world-space force glyph through the recorded CV camera."""
    require(np.isfinite(meters_per_newton) and meters_per_newton>0,'Positive force display scale required')
    point=np.asarray(point,float);force=np.asarray(force,float)
    K=np.asarray(K,float);T=np.asarray(T_world_cam,float)
    require(point.shape==force.shape==(3,) and K.shape==(3,3) and T.shape==(4,4)
            and np.isfinite(np.r_[point,force,K.ravel(),T.ravel()]).all(),
            'Finite world force, contact point and camera required')
    world=np.array([point,point+meters_per_newton*force])
    camera=(world-T[:3,3])@T[:3,:3]
    if np.any(camera[:,2]<=1e-6):return None
    uv,_=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),K,np.asarray(dist,float))
    return uv.reshape(2,2)


def load_force(metadata, filename, camera_input='process_camera', meters_per_newton=.06):
    """Bind a saved command trace to exactly the rendered best native trial."""
    from .workflow import intact

    root=Path(metadata['round']).resolve();filename=Path(filename).resolve()
    require(intact(root),'Force overlay requires intact native process evidence')
    result=read(root/'result.json');trial=root/'trials'/f"{result['best']['index']:04d}"
    require(filename.parent==trial,'Force trace must belong to the rendered best native trial')
    key=str(filename.relative_to(root))
    require(result['best']['artifacts'].get(key)==digest(filename),
            'Force trace must match the sealed native artifact')
    inputs=read(root/'inputs.json');camera_path=Path(inputs[camera_input]['path']).resolve()
    require(metadata['sha256'].get(str(camera_path))==digest(camera_path),
            'Force overlay camera must be the recorded render camera')
    camera=dict(np.load(camera_path,allow_pickle=False))
    commands=np.load(filename,allow_pickle=False)['commands']
    times=np.array([r['native_t'] for r in metadata['frames']])
    sampled=sample_commands(commands,times,read(root/'plan.json')['simulation']['dt'])
    K=camera['K'].copy();K[0]*=metadata['width']/metadata['source_width'];K[1]*=metadata['height']/metadata['source_height']
    projections=[]
    for i,row in enumerate(sampled):
        frame=metadata['frames'][i]['frame']
        projections.append(project_force(row[4:7],row[1:4],K,camera['T_world_cam'][frame],
                                         camera['dist'],meters_per_newton))
    magnitudes=np.linalg.norm(commands[:,1:4],axis=1);peak=int(np.argmax(magnitudes))
    receipt={'trace':str(filename),'trace_sha256':digest(filename),
        'columns':['t_s','Fx_N','Fy_N','Fz_N','point_x_m','point_y_m','point_z_m'],
        'frame':'world','peak_force_N':float(magnitudes[peak]),'peak_time_s':float(commands[peak,0]),
        'arrow_meters_per_newton':meters_per_newton,
        'camera_input':camera_input,'camera_sha256':digest(camera_path),
        'parameter_evidence':result.get('parameter_evidence',{}),
        'scope':'Estimated external applied force, not measured human force or native contact reaction. Actual saved commands, nearest native step; no force interpolation, refit or new physics.',
        'samples':[{'native_t':float(t),'command_t':float(row[0]),'force_world_N':row[1:4].tolist(),
                    'point_world_m':row[4:7].tolist(),'magnitude_N':float(np.linalg.norm(row[1:4]))}
                   for t,row in zip(times,sampled)]}
    return {'commands':commands,'sampled':sampled,'projections':projections,'receipt':receipt}


def draw_force(image, data, index):
    """Arrow length uses a declared world scale; HUD carries the actual norm."""
    row=data['sampled'][index];magnitude=float(np.linalg.norm(row[1:4]))
    uv=data['projections'][index];height,width=image.shape[:2]
    if uv is not None and magnitude>1e-6:
        a,b=np.rint(uv).astype(int)
        # Clip only the drawn glyph; never move its contact point or change its direction.
        visible,start,end=cv2.clipLine((0,0,width,height),tuple(a),tuple(b))
        if visible:
            cv2.line(image,start,end,(20,20,20),7,cv2.LINE_AA)
            cv2.arrowedLine(image,start,end,COLOR,4,cv2.LINE_AA,tipLength=.22)
            if 0<=a[0]<width and 0<=a[1]<height:cv2.circle(image,tuple(a),5,COLOR,-1,cv2.LINE_AA)
            label=f'{magnitude:.2f} N'
            x=int(np.clip(a[0]+12,5,width-115));y=int(np.clip(a[1]-12,25,height-10))
            cv2.putText(image,label,(x,y),0,.7,(10,10,10),4,cv2.LINE_AA)
            cv2.putText(image,label,(x,y),0,.7,COLOR,2,cv2.LINE_AA)


def draw_force_hud(image, data, index, *, origin=(14,14), size=(470,190)):
    """Magnitude timeline and world components; an overlay, not scene geometry."""
    x,y=origin;w,h=size
    cv2.rectangle(image,(x,y),(x+w,y+h),(18,18,18),-1)
    row=data['sampled'][index];magnitude=float(np.linalg.norm(row[1:4]))
    lines=['Estimated external push (not measured)',
           f'|F| = {magnitude:.2f} N   peak {data["receipt"]["peak_force_N"]:.2f} N',
           f'World: Fx {row[1]:+.2f}  Fy {row[2]:+.2f}  Fz {row[3]:+.2f} N']
    for i,label in enumerate(lines):cv2.putText(image,label,(x+12,y+25+23*i),0,.52,COLOR if i==1 else (238,238,238),1,cv2.LINE_AA)
    commands=data['commands'];m=np.linalg.norm(commands[:,1:4],axis=1)
    t0,t1=commands[0,0],commands[-1,0];peak=max(float(m.max()),1e-8)
    a,b=x+35,y+94;gw,gh=w-52,h-123
    graph=np.column_stack([a+(commands[:,0]-t0)/(t1-t0)*gw,b+gh-m/peak*gh])
    cv2.line(image,(a,b+gh),(a+gw,b+gh),(110,110,110),1)
    cv2.polylines(image,[np.rint(graph).astype(np.int32)],False,COLOR,2,cv2.LINE_AA)
    cursor=int(np.clip(a+(row[0]-t0)/(t1-t0)*gw,a,a+gw))
    cv2.line(image,(cursor,b),(cursor,b+gh),(240,240,240),1,cv2.LINE_AA)
    for label,point in [(f'{t0:.1f}s',(a,b+gh+20)),(f'{t1:.2f}s',(a+gw-50,b+gh+20)),('N',(x+10,b+10))]:
        cv2.putText(image,label,point,0,.4,(200,200,200),1,cv2.LINE_AA)
