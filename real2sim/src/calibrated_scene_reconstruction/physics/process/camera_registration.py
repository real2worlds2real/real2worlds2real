"""Metric registration using a static sparse scene and an existing object anchor."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contract import require
from .vision import pose


def register(reconstruction,anchor_xyz,anchor_pixels,*,seed=0,plane_orientation='observed',prior_reason=None,normal_source=None):
    """Keep later target motion outside the camera solve.

    The known initial object fixes metric scale; observed static table tracks
    provide plane orientation. No measured physical dimensions are supplied.
    """
    K,D=reconstruction['K'],reconstruction['dist']
    D=np.r_[D,0.] if len(D)==4 else D
    T,indices,errors=pose(anchor_xyz,anchor_pixels,K,D,min_inliers=8)
    anchor_xyz,anchor_pixels=anchor_xyz[indices],anchor_pixels[indices]
    R_source=reconstruction['rotations'][0]
    expected=R_source@T[:3,:3].T[:,2]
    points=reconstruction['table_xyz']
    require(plane_orientation in {'observed','anchor_prior','source_direction'},'Unknown camera-plane hypothesis')
    if plane_orientation!='observed':require(bool(prior_reason),'An explicit source-based upright/support prior is required')
    if plane_orientation=='source_direction':
        normal_source=np.asarray(normal_source,float)
        require(normal_source.shape==(3,) and np.isfinite(normal_source).all() and np.linalg.norm(normal_source)>0,
                'A finite reviewed source gravity direction is required')
    require(len(points)>=20,'Not enough observed table points for metric registration')
    rng=np.random.default_rng(seed)
    threshold=np.median(np.linalg.norm(points-points.mean(0),axis=1))*.02
    best=None
    for _ in range(2500):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        n=np.cross(b-a,c-a);n/=max(np.linalg.norm(n),1e-12)
        if n@expected<0:n=-n
        if n@expected<np.cos(np.deg2rad(20)):continue
        d=a@n;inside=abs(points@n-d)<threshold
        if best is None or inside.sum()>best.sum():best=inside
    require(best is not None and best.sum()>=20,'Source-visible table plane cannot be established')
    center=points[best].mean(0)
    _,_,basis=np.linalg.svd(points[best]-center);n=basis[-1]
    if n@expected<0:n=-n
    d=center@n
    if plane_orientation in {'anchor_prior','source_direction'}:
        n=expected/np.linalg.norm(expected) if plane_orientation=='anchor_prior' else normal_source/np.linalg.norm(normal_source)
        if n@expected<0:n=-n
        projections=points@n;d=float(np.median(projections))
        mad=1.4826*np.median(abs(projections-d))
        best=abs(projections-d)<=max(threshold,2.5*mad)
        require(best.sum()>=20,'Insufficient support tracks under the declared orientation prior')
        d=float(np.median(projections[best]))
    initial_alignment=T[:3,:3]@R_source.T
    correction=Rotation.align_vectors([[0,0,1]],[(initial_alignment@n).tolist()])[0].as_matrix()
    alignment=correction@initial_alignment

    def model(x):
        R=Rotation.from_euler('z',x[0]).as_matrix()@alignment
        Rcw=(R@R_source).T
        C=x[1:]
        pixels=cv2.projectPoints(anchor_xyz,cv2.Rodrigues(Rcw)[0],-Rcw@C,K,D)[0].reshape(-1,2)
        return R,C,pixels

    solved=least_squares(lambda x:(model(x)[2]-anchor_pixels).ravel(),np.r_[0.,T[:3,3]],
        loss='soft_l1',f_scale=1.,bounds=([-np.pi,-5.,-5.,.05],[np.pi,5.,5.,3.]),max_nfev=300)
    R,C,pixels=model(solved.x)
    anchor_rmse=float(np.sqrt(np.mean((pixels-anchor_pixels)**2)))
    require(solved.success and anchor_rmse<=3.,'Metric anchor and observed table plane are inconsistent')
    height=reconstruction['centres'][0]@n-d
    require(height>0,'Static table must lie below the first camera')
    scale=C[2]/height
    translation=C-scale*(R@reconstruction['centres'][0])
    return {'scale':float(scale),'R':R,'translation':translation,'K':K,'dist':D,
        'anchor_rmse_px':anchor_rmse,'anchor_inliers':len(indices),
        'table_inliers':int(best.sum()),'table_samples':len(points),
        'table_rms_m':float(scale*np.sqrt(np.mean((points[best]@n-d)**2))),
        'initial_camera_height_m':float(C[2]),'plane_normal_source':n,'plane_offset_source':float(d),
        'plane_orientation_basis':plane_orientation,'plane_orientation_reason':prior_reason or 'Static scene points',
        'scope':'Initial source geometry plus static video features; no later target poses or measured dimensions used.'}


def transform_reference(reference,registration):
    return {**reference,'xyz':registration['scale']*reference['xyz']@registration['R'].T+registration['translation']}
