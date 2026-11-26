"""Project measured material geometry into observation coordinates."""
import cv2
import numpy as np
from .contract import require

def project_material(vertices,faces,position,rotation,camera,*,opening=(1,1)):
    K,D,T=camera
    Rcw=T[:3,:3].T
    xyz=np.asarray(vertices)@rotation.T+position
    points=cv2.projectPoints(xyz,cv2.Rodrigues(Rcw)[0],-Rcw@T[:3,3],K,D)[0].reshape(-1,2)
    lo=np.floor(points.min(0)-4).astype(int);hi=np.ceil(points.max(0)+4).astype(int)
    require(np.max(hi-lo)<4096,'Projected response left the observable image volume')
    from calibrated_scene_reconstruction.physics.process.mask_raster import triangles
    mask=triangles(points-lo,faces,(max(1,hi[1]-lo[1]),max(1,hi[0]-lo[0])))
    from calibrated_scene_reconstruction.physics.process.region_observations import measure
    return measure(mask,opening=opening,offset=lo)
