"""Compare a camera hypothesis using static tracks excluded from bundle adjustment."""
import argparse
from pathlib import Path
import numpy as np
import pycolmap
from calibrated_scene_reconstruction.physics.process.contract import write,digest,require
from calibrated_scene_reconstruction.physics.process.static_points import triangulate


def refine(source,output,*,distortion=False):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reconstruction=pycolmap.Reconstruction(source)
    require(len(reconstruction.cameras)==1,'Use one video camera per intrinsics hypothesis')
    images=sorted(reconstruction.images.values(),key=lambda image:image.name)
    order={image.image_id:i for i,image in enumerate(images)}
    fit=[i for i in range(len(images)) if i%3];validation=[i for i in range(len(images)) if not i%3]
    held=[]
    for identifier,point in sorted(reconstruction.points3D.items()):
        if identifier%5:continue
        pixels=np.full((len(images),2),np.nan)
        for element in point.track.elements:
            if element.image_id in order:pixels[order[element.image_id]]=reconstruction.images[element.image_id].points2D[element.point2D_idx].xy
        if np.isfinite(pixels[fit]).all(1).sum()>=3 and np.isfinite(pixels[validation]).all(1).sum()>=2:
            held.append((identifier,pixels))
    require(len(held)>=20,'At least 20 independent static validation tracks required')
    for identifier,_ in held:reconstruction.delete_point3D(identifier)
    camera=next(iter(reconstruction.cameras.values()));before=camera.params.copy()
    options=pycolmap.BundleAdjustmentOptions();options.refine_extra_params=distortion;options.print_summary=False
    options.ceres.use_gpu=False;options.ceres.loss_function_type=pycolmap.LossFunctionType.SOFT_L1
    options.ceres.solver_options.num_threads=8;options.ceres.solver_options.max_num_iterations=100
    options.ceres.solver_options.max_solver_time_in_seconds=120
    pycolmap.bundle_adjustment(reconstruction,options)
    require(camera.model_name=='OPENCV','This source exporter requires the explicit OpenCV distortion model')
    transforms=[]
    for image in images:
        pose=image.cam_from_world().matrix();T=np.eye(4);T[:3,:3]=pose[:,:3].T;T[:3,3]=-pose[:,:3].T@pose[:,3];transforms.append(T)
    calib={'T_world_cam':np.asarray(transforms),'K':camera.calibration_matrix(),'dist':np.r_[camera.params[4:],0.]}
    scores=[]
    for identifier,pixels in held:
        try:
            score=triangulate(pixels,calib,fit_indices=fit,validation_indices=validation,min_parallax_deg=.5,max_rmse_px=3.)
            scores.append({'point_id':identifier,'fit_rmse_px':score['fit_rmse_px'],'validation_rmse_px':score['validation_rmse_px'],'parallax_deg':score['parallax_deg'],'pass':score['pass']})
        except ValueError as exc:scores.append({'point_id':identifier,'error':str(exc),'pass':False})
    model=output/'sparse/0';model.mkdir(parents=True);reconstruction.write(model)
    finite=[row['validation_rmse_px'] for row in scores if 'validation_rmse_px' in row]
    report={'refine_distortion':distortion,'camera_params_before':before.tolist(),'camera_params_after':camera.params.tolist(),
        'training_reprojection_px':reconstruction.compute_mean_reprojection_error(),'validation_tracks':len(held),
        'validation_scored':len(finite),'validation_median_px':None if not finite else float(np.median(finite)),
        'validation_p90_px':None if not finite else float(np.quantile(finite,.9)),'tracks':scores,
        'source_sha256':{str(p.resolve()):digest(p) for p in [*source.glob('*.bin'),Path(__file__)]},
        'scope':'Static validation tracks are removed entirely from bundle adjustment. Their 3D points use fit views only; validation views score reprojection. This is not metric registration, target-motion fitting or process acceptance.'}
    write(output/'report.json',report);print({k:v for k,v in report.items() if k not in {'tracks','source_sha256','scope'}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--refine-distortion',action='store_true');args=parser.parse_args()
    refine(args.source,args.output,distortion=args.refine_distortion)
