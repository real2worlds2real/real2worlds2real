"""Export a static sparse reconstruction for metric process-camera registration."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from calibrated_scene_reconstruction.physics.process.vision import model_reference


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--camera-run',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.camera_run);model=root/'sparse'/args.model
    rec=pycolmap.Reconstruction(model)
    settings=json.loads(Path(args.config).read_text())['videos'][args.case]
    images=sorted(rec.images.values(),key=lambda i:i.name)
    source=next(im for im in images if im.name in ['frame_000000.jpg',args.case+'/frame_000000.jpg'])
    points=[];identities=[]
    polygon=np.asarray(settings['table_polygon'],np.int32);x1,y1,x2,y2=settings['initial_object_roi']
    for p in source.points2D:
        if not p.has_point3D():continue
        x,y=p.xy
        if cv2.pointPolygonTest(polygon,(float(x),float(y)),False)<0 or (x1<=x<=x2 and y1<=y<=y2):continue
        point=rec.points3D[p.point3D_id]
        if point.error>2. or point.track.length()<3:continue
        points.append(point.xyz);identities.append(p.point3D_id)
    records=[]
    for im in images:
        if '/' in im.name and not im.name.startswith(args.case+'/'):continue
        pose=im.cam_from_world().matrix();R,t=pose[:,:3],pose[:,3]
        records.append((int(Path(im.name).stem[6:]),-R.T@t,R.T))
    reference=model_reference(root/'database.db',model,[im.name for im in images])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'reference.npz',**reference)
    camera=rec.cameras[source.camera_id]
    np.savez_compressed(out/'reconstruction.npz',frames=[r[0] for r in records],centres=[r[1] for r in records],
        rotations=[r[2] for r in records],K=camera.calibration_matrix(),dist=np.asarray(camera.params[4:]),
        table_xyz=np.asarray(points),table_point_ids=identities)
    print({'registered_frames':len(images),'table_points':len(points),'reference_descriptors':len(reference['xyz'])})
