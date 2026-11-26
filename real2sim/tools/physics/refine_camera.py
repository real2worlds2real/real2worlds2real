"""Densify a failed video camera run while preserving its valid sparse model."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pycolmap

from observe_camera import rectangle_at
from calibrated_scene_reconstruction.physics.process.contract import digest,write,require


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for key in ['config','case','source','model','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--fixed-focal',action='store_true')
    args=parser.parse_args()
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    images=out/'images';masks=out/'masks';images.mkdir();masks.mkdir()
    cfg=json.loads(Path(args.config).read_text())['videos'][args.case]
    cap=cv2.VideoCapture(cfg['path']);fps=cap.get(cv2.CAP_PROP_FPS);names=[]
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok,image=cap.read();require(ok,'Incomplete input video')
        name=f'frame_{frame:06d}.jpg';names.append(name);cv2.imwrite(str(images/name),image)
        mask=np.full(image.shape[:2],255,np.uint8)
        for x1,y1,x2,y2 in rectangle_at(cfg.get('moving_regions',[]),frame):
            x1,x2=np.clip([x1,x2],0,image.shape[1]);y1,y2=np.clip([y1,y2],0,image.shape[0]);mask[y1:y2,x1:x2]=0
        cv2.imwrite(str(masks/(name+'.png')),mask)
    cap.release()
    database=out/'database.db';shutil.copy2(source/'database.db',database)
    original=pycolmap.Reconstruction(source/'sparse'/args.model)
    camera=next(iter(original.cameras.values()))
    reader=pycolmap.ImageReaderOptions(mask_path=masks.resolve(),existing_camera_id=camera.camera_id)
    extraction=pycolmap.FeatureExtractionOptions(num_threads=8,max_image_size=1280);extraction.sift.max_num_features=4096
    pycolmap.extract_features(database,images,image_names=names,camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,extraction_options=extraction,device=pycolmap.Device.cpu)
    pairs=out/'pairs.txt';pairs.write_text(''.join(a+' '+b+'\n' for i,a in enumerate(names) for b in names[i+1:i+9]))
    pycolmap.match_image_pairs(database,matching_options=pycolmap.FeatureMatchingOptions(num_threads=8,use_gpu=False),
        pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pairs),device=pycolmap.Device.cpu)
    options=pycolmap.IncrementalPipelineOptions(num_threads=8,max_num_models=2,random_seed=0,max_runtime_seconds=900)
    options.ba_refine_focal_length=not args.fixed_focal;options.mapper.abs_pose_refine_focal_length=not args.fixed_focal
    options.ba_refine_extra_params=False;options.mapper.abs_pose_refine_extra_params=False;options.ba_use_gpu=False
    sparse=out/'sparse';sparse.mkdir()
    models=pycolmap.incremental_mapping(database,images,sparse,options=options,input_path=source/'sparse'/args.model)
    report={str(i):{'images':m.num_reg_images(),'points':m.num_points3D(),'mean_reprojection_error':m.compute_mean_reprojection_error()} for i,m in models.items()}
    write(out/'camera_preparation.json',{'models':report,'frames':len(names),'fps':fps,'fixed_focal':args.fixed_focal,
        'prior_camera_params':camera.params.tolist(),'device':'cpu','metric':False,
        'inputs':{str(p.resolve()):digest(p) for p in [Path(args.config),source/'database.db',Path(cfg['path']),Path(__file__)]},
        'scope':'Dense registration after a coverage failure; no target trajectory or physics parameter is a camera target.'})
    print(json.dumps(report),flush=True)
