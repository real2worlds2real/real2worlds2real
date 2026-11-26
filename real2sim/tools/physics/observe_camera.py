"""CPU sparse camera reconstruction for moving-camera physics recordings.

Run in the installed COLMAP environment. Dynamic image regions are observations
in the per-recording config, never task-specific logic in this tool.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from calibrated_scene_reconstruction.physics.process.contract import digest,write,require


def rectangle_at(regions,frame):
    for region in regions:
        keys=sorted((int(k),box) for k,box in region['keyframes'].items())
        if frame<keys[0][0] or frame>keys[-1][0]:continue
        values=np.array([box for _,box in keys],float)
        yield np.array([np.interp(frame,[k for k,_ in keys],values[:,j]) for j in range(4)]).astype(int)


def prepare(config,case,output,stride=3):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    settings=json.loads(Path(config).read_text());recording=settings['videos'][case]
    video=Path(recording['path']).resolve()
    images=output/'images';images.mkdir();masks=output/'masks';masks.mkdir()
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);timeline=[]
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok,image=cap.read();require(ok,'Incomplete input video')
        if frame%stride:continue
        name=f'frame_{frame:06d}.jpg'
        cv2.imwrite(str(images/name),image)
        mask=np.full(image.shape[:2],255,np.uint8)
        for x1,y1,x2,y2 in rectangle_at(recording.get('moving_regions',[]),frame):
            x1,x2=np.clip([x1,x2],0,image.shape[1]);y1,y2=np.clip([y1,y2],0,image.shape[0])
            mask[y1:y2,x1:x2]=0
        cv2.imwrite(str(masks/(name+'.png')),mask)
        timeline.append({'name':name,'frame':frame,'t':frame/fps})
    cap.release()
    prior=json.loads(Path(settings['camera_prior']).read_text());K=np.array(prior['K'])
    reader=pycolmap.ImageReaderOptions(camera_model='OPENCV',mask_path=masks.resolve(),
        camera_params=','.join(map(str,[K[0,0],K[1,1],K[0,2],K[1,2],0,0,0,0])))
    database=output/'database.db'
    extraction=pycolmap.FeatureExtractionOptions(num_threads=8,max_image_size=1280)
    extraction.sift.max_num_features=4096
    pycolmap.extract_features(database,images,camera_mode=pycolmap.CameraMode.SINGLE,reader_options=reader,
                             extraction_options=extraction,device=pycolmap.Device.cpu)
    matching=pycolmap.FeatureMatchingOptions(num_threads=8,use_gpu=False)
    pairing=pycolmap.SequentialPairingOptions(overlap=8,quadratic_overlap=True,loop_detection=False)
    pycolmap.match_sequential(database,matching_options=matching,pairing_options=pairing,device=pycolmap.Device.cpu)
    options=pycolmap.IncrementalPipelineOptions(num_threads=8,min_model_size=8,max_num_models=3,random_seed=0)
    options.mapper.init_min_tri_angle=2.
    options.mapper.init_min_num_inliers=60
    options.ba_refine_extra_params=False  # Rectified phone-video hypothesis; inspect reprojection before accepting.
    options.ba_use_gpu=False
    options.max_runtime_seconds=900
    sparse=output/'sparse';sparse.mkdir()
    models=pycolmap.incremental_mapping(database,images,sparse,options=options)
    registered={str(i):{'images':m.num_reg_images(),'points':m.num_points3D(),'mean_reprojection_error':m.compute_mean_reprojection_error()} for i,m in models.items()}
    write(output/'camera_preparation.json',{'video':str(video),'case':case,'timeline':timeline,'models':registered,
        'inputs':{str(p.resolve()):digest(p) for p in [video,Path(config),Path(settings['camera_prior']),Path(__file__)]},
        'pycolmap_version':pycolmap.__version__,'device':'cpu','metric':False,
        'scope':'Static masked features only; metric scale and initial object registration still require review.'})
    print(json.dumps(registered),flush=True)


def prepare_joint(config,cases,output,stride=2):
    """Separate camera intrinsics per clip, one static scene, no shared target motion."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    settings=json.loads(Path(config).read_text());images=output/'images';masks=output/'masks'
    timelines={};sources=[Path(config),Path(settings['camera_prior']),Path(__file__)]
    for case in cases:
        recording=settings['videos'][case];video=Path(recording['path']).resolve();sources.append(video)
        (images/case).mkdir(parents=True);(masks/case).mkdir(parents=True)
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));timeline=[]
        for frame in range(n):
            ok,image=cap.read();require(ok,'Incomplete video: '+str(video))
            if frame%stride and frame!=n-1:continue
            name=f'{case}/frame_{frame:06d}.jpg';cv2.imwrite(str(images/name),image)
            mask=np.full(image.shape[:2],255,np.uint8)
            for x1,y1,x2,y2 in rectangle_at(recording.get('moving_regions',[]),frame):
                x1,x2=np.clip([x1,x2],0,image.shape[1]);y1,y2=np.clip([y1,y2],0,image.shape[0]);mask[y1:y2,x1:x2]=0
            cv2.imwrite(str(masks/(name+'.png')),mask);timeline.append({'name':name,'frame':frame,'t':frame/fps})
        cap.release();timelines[case]=timeline
    prior=json.loads(Path(settings['camera_prior']).read_text());K=np.array(prior['K'])
    reader=pycolmap.ImageReaderOptions(camera_model='OPENCV',mask_path=masks.resolve(),
        camera_params=','.join(map(str,[K[0,0],K[1,1],K[0,2],K[1,2],0,0,0,0])))
    database=output/'database.db'
    extraction=pycolmap.FeatureExtractionOptions(num_threads=8,max_image_size=1280);extraction.sift.max_num_features=4096
    pycolmap.extract_features(database,images,camera_mode=pycolmap.CameraMode.PER_FOLDER,reader_options=reader,
        extraction_options=extraction,device=pycolmap.Device.cpu)
    pairs=[]
    for timeline in timelines.values():
        names=[row['name'] for row in timeline]
        pairs.extend((a,b) for i,a in enumerate(names) for b in names[i+1:i+11])
    for i,case in enumerate(cases):
        for other in cases[i+1:]:
            pairs.extend((a['name'],b['name']) for a in timelines[case][::3] for b in timelines[other][::3])
    pair_file=output/'pairs.txt';pair_file.write_text(''.join(a+' '+b+'\n' for a,b in pairs))
    pycolmap.match_image_pairs(database,matching_options=pycolmap.FeatureMatchingOptions(num_threads=8,use_gpu=False),
        pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pair_file),device=pycolmap.Device.cpu)
    options=pycolmap.IncrementalPipelineOptions(num_threads=8,min_model_size=8,max_num_models=3,random_seed=0)
    options.mapper.init_min_tri_angle=2.;options.mapper.init_min_num_inliers=60
    options.ba_refine_extra_params=False;options.ba_use_gpu=False;options.max_runtime_seconds=900
    sparse=output/'sparse';sparse.mkdir()
    models=pycolmap.incremental_mapping(database,images,sparse,options=options)
    report={str(i):{'images':m.num_reg_images(),'points':m.num_points3D(),'mean_reprojection_error':m.compute_mean_reprojection_error(),
        'clips':{case:sum(im.name.startswith(case+'/') for im in m.images.values()) for case in cases}} for i,m in models.items()}
    write(output/'camera_preparation.json',{'models':report,'timelines':timelines,'device':'cpu','metric':False,
        'inputs':{str(p.resolve()):digest(p) for p in sources},'pycolmap_version':pycolmap.__version__,
        'scope':'Shared static scene and separate lenses; target motion and forces are not shared camera evidence.'})
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--case')
    parser.add_argument('--joint-cases',nargs='+')
    parser.add_argument('--output',required=True);parser.add_argument('--stride',type=int,default=3)
    args=parser.parse_args()
    if bool(args.case)==bool(args.joint_cases):parser.error('Choose --case or --joint-cases')
    if args.joint_cases:prepare_joint(args.config,args.joint_cases,args.output,args.stride)
    else:prepare(args.config,args.case,args.output,args.stride)
