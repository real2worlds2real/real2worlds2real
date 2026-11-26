"""Source-labelled sparse landmarks, independent of any candidate geometry.

Use measured COLMAP feature identities in existing reviewed observation views.
This diagnostic neither completes invisible material nor changes source gates.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .blender.store import digest
from .source_motion import full_mask


def supported_tracks(tracks, masks):
    """Require direct foreground support, without testing a candidate surface."""
    result={}
    for name,(xy,identities) in tracks.items():
        mask=cv2.erode(masks[name].astype('uint8'),np.ones((5,5),'uint8'))>0
        pixels=np.rint(xy).astype(int);h,w=mask.shape
        inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        keep=np.zeros(len(xy),bool);keep[inside]=mask[pixels[inside,1],pixels[inside,0]]
        for point,pixel in zip(identities[keep],xy[keep]):
            result.setdefault(int(point),[]).append((name,pixel.copy()))
    return result


def parallax_degrees(point,centres):
    rays=np.asarray(point,float)-np.asarray(centres,float)
    norms=np.linalg.norm(rays,axis=1)
    if len(rays)<2 or np.any(norms<1e-10):return 0.
    rays/=norms[:,None]
    return float(np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1.,1.))))


def localization_covariance(point,cameras,pixel_sigma):
    """Local pixel-noise uncertainty; calibration/systematic errors are extra."""
    if not np.isfinite(pixel_sigma) or pixel_sigma<=0:raise ValueError('Positive pixel uncertainty required')
    jacobians=[]
    for intrinsic,rotation,centre in cameras:
        world_to_cam=np.asarray(rotation,float).T
        q=np.asarray(intrinsic,float)@world_to_cam@(np.asarray(point,float)-centre)
        if q[2]<=0:raise ValueError('Landmark must be in front of each supporting camera')
        projection=np.asarray(intrinsic,float)@world_to_cam
        jacobians.append((projection[:2]*q[2]-q[:2,None]*projection[2])/q[2]**2)
    information=np.concatenate(jacobians).T@np.concatenate(jacobians)/pixel_sigma**2
    values,axes=np.linalg.eigh(information)
    if values[0]<=max(1e-12,values[-1]*1e-10):return None
    return (axes/values)@axes.T


def run(s,obj,prompt,output,*,exclude_prompt=None,min_views=2,min_parallax=1.,max_error=1.5):
    from .physical_observations import observations
    from .blender.part_evidence import requirements
    from calibrated_scene_reconstruction.scene.capture.colmap_points import image_tracks,point_records
    if min_views<2 or not 0<min_parallax<180 or max_error<=0:raise ValueError('Positive multiview quality bounds required')
    output=Path(output).resolve()
    if not output.is_relative_to((s/'qa').resolve()):raise ValueError('Landmarks belong to the current capture QA')
    output.mkdir(parents=True,exist_ok=False)
    z,frames,masks,windows=observations(s,obj,prompt,exterior_to=exclude_prompt)
    raw=s/'splat/object_fit'/f'{obj}.components.acquired.npz'
    acquired=s/'splat/object_fit'/f'{obj}.acquired.npz'
    with np.load(acquired,allow_pickle=True) as data:prompts=str(data['prompt']).split('|')
    with np.load(raw) as data:names=list(map(str,data['names']));labels=data['part_masks']
    if exclude_prompt and exclude_prompt not in prompts:raise ValueError('Unknown optical occluder')
    view_masks={};centres={};camera_indices={};inputs=[raw,acquired,s/'parsing/posed_frames/cameras.npz',
        s/'splat/align.json',s/'splat/object_fit'/f'{obj}.npz']
    inputs.extend(s/name for name in z.get('_object_domain_inputs',{}))
    for item in requirements(s,obj):
        if item['part_prompt']==prompt and item.get('part_review'):
            folder=s/item['part_review'];inputs.extend(p for p in folder.iterdir() if p.is_file())
    for i,mask,window in zip(frames,masks,windows):
        name=str(z['names'][i]);path=s/'splat/data/images'/name
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError('Missing source RGB: '+name)
        region=full_mask(mask,window,image.shape)
        if exclude_prompt:region&=~labels[names.index(name),prompts.index(exclude_prompt)].astype(bool)
        view_masks[name]=region;centres[name]=z['centre'][i];camera_indices[name]=int(i);inputs.append(path)
    sparse=s/'splat/data/sparse/full';inputs.extend([sparse/'images.bin',sparse/'points3D.bin'])
    tracks=image_tracks(sparse/'images.bin',view_masks)
    supported=supported_tracks(tracks,view_masks)
    records=point_records(sparse/'points3D.bin',supported)
    alignment=json.loads((s/'splat/align.json').read_text())['sim3']
    rotation=np.asarray(alignment['R']);translation=np.asarray(alignment['t']);scale=float(alignment['scale'])
    rows=[];world=[];retained={};rejections={'insufficient_direct_views':0,'sparse_reprojection_error':0,'insufficient_parallax':0}
    for identity,views in supported.items():
        point,error,length=records[identity];point=scale*rotation@point+translation
        if len(views)<min_views:rejections['insufficient_direct_views']+=1;continue
        if error>max_error:rejections['sparse_reprojection_error']+=1;continue
        angle=parallax_degrees(point,[centres[name] for name,_ in views])
        if angle<min_parallax:rejections['insufficient_parallax']+=1;continue
        reprojection=[]
        for name,pixel in views:
            i=camera_indices[name];camera=z['R_world_cam'][i].T@(point-z['centre'][i])
            projection=z['K'][i]@camera
            reprojection.append(float(np.linalg.norm(projection[:2]/projection[2]-pixel)) if camera[2]>0 else float('inf'))
        # Independently calibrated-camera consistency is not a candidate-fit filter.
        if max(reprojection)>max_error*3:rejections['sparse_reprojection_error']+=1;continue
        sigma=max(.5,float(np.sqrt(np.mean(np.square(reprojection)))))
        covariance=localization_covariance(point,[(z['K'][camera_indices[name]],z['R_world_cam'][camera_indices[name]],centres[name]) for name,_ in views],sigma)
        rows.append({'point_id':identity,'direct_views':len(views),'track_length':int(length),
            'sparse_error_px':float(error),'parallax_deg':angle,'maximum_calibrated_error_px':max(reprojection),
            'pixel_noise_sigma_px':sigma,'localization_covariance_m2':covariance.tolist() if covariance is not None else None,
            'maximum_pixel_noise_sigma_m':float(np.sqrt(np.linalg.eigvalsh(covariance)[-1])) if covariance is not None else None,
            'observations':[{'name':name,'xy':pixel.tolist()} for name,pixel in views]})
        world.append(point)
        for name,pixel in views:retained.setdefault(name,[]).append(pixel)
    points=np.asarray(world,float).reshape(-1,3)
    np.savez_compressed(output/'landmarks.npz',world=points,point_ids=np.asarray([r['point_id'] for r in rows],np.int64))
    tiles=[]
    for name,region in view_masks.items():
        image=cv2.imread(str(s/'splat/data/images'/name));image[~region]=(image[~region]*.35).astype('uint8')
        for x,y in retained.get(name,[]):cv2.circle(image,(round(float(x)),round(float(y))),3,(20,220,40),-1)
        cv2.putText(image,name+f' ({len(retained.get(name,[]))} landmarks)',(16,30),0,.8,(0,255,255),2)
        tiles.append(cv2.resize(image,(640,360)))
    for i in range(0,len(tiles),8):cv2.imwrite(str(output/f'landmarks_{i//8+1}.jpg'),np.vstack(tiles[i:i+8]))
    (output/'source.py').write_bytes(Path(__file__).read_bytes());inputs.append(output/'source.py')
    result={'source_only':True,'candidate_geometry_used':False,'task_approved':False,'source_frames_removed':False,
        'source_views':len(frames),'retained_landmarks':len(rows),'landmarks':rows,'rejections':rejections,
        'quality_bounds':{'min_direct_views':min_views,'min_parallax_deg':min_parallax,'max_sparse_error_px':max_error,
                          'max_calibrated_error_px':max_error*3},
        'inputs_sha256':{str(p.resolve()):digest(p) for p in inputs},
        'scope':'Sparse direct-material texture evidence only. Transmitted pixels excluded; no candidate residual selects points. Local covariance covers pixel noise only, excluding camera/scale bias and incorrect correspondences. Missing landmarks do not prove missing material; hidden shape and physical parameters remain uncertain.'}
    (output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('object');p.add_argument('prompt')
    p.add_argument('--exclude-prompt');p.add_argument('--output',required=True)
    a=p.parse_args();result=run(Session(a.session),a.object,a.prompt,a.output,exclude_prompt=a.exclude_prompt)
    print(json.dumps({key:result[key] for key in ['source_views','retained_landmarks','rejections']},indent=2))
