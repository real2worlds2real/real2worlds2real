"""Propose initial parallel material interfaces; not inferred adhesion strength."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def propose(points,faces,rest,*,thickness,film_range):
    points,faces,rest=np.asarray(points,float),np.asarray(faces,int),np.asarray(rest,float)
    if (points.shape!=rest.shape or points.ndim!=2 or points.shape[1:]!=(3,)
            or not all(np.isfinite(x).all() for x in [points,rest])
            or not np.isfinite([thickness,film_range]).all() or min(thickness,film_range)<=0):
        raise ValueError('Finite matching material coordinates and explicit physical interface range required')
    tri=points[faces];material=rest[faces]
    edge=np.linalg.norm(material[:,1]-material[:,0],axis=1).max()
    edge=max(edge,np.linalg.norm(material[:,2]-material[:,1],axis=1).max(),np.linalg.norm(material[:,0]-material[:,2],axis=1).max())
    mesh=trimesh.Trimesh(points,faces,process=False)
    count=min(64,len(faces));_,near=cKDTree(tri.mean(1)).query(points,k=count)
    ii=np.repeat(np.arange(len(points)),count);ff=near.reshape(-1)
    far=np.linalg.norm(rest[ii]-material.mean(1)[ff],axis=1)>3*edge
    parallel=abs((mesh.vertex_normals[ii]*mesh.face_normals[ff]).sum(1))>=np.cos(np.radians(20.))
    ii,ff=ii[far&parallel],ff[far&parallel]
    closest=trimesh.triangles.closest_point(tri[ff],points[ii]);delta=points[ii]-closest
    distance=np.linalg.norm(delta,axis=1)
    normal_gap=abs((delta*mesh.face_normals[ff]).sum(1))
    valid=(distance>=.9*thickness)&(distance<=thickness+film_range)&(normal_gap>=.9*distance)
    ii,ff,closest,distance=ii[valid],ff[valid],closest[valid],distance[valid]
    if not len(ii):raise ValueError('No initial nonlocal parallel interfaces inside the declared film range')
    order=np.lexsort((distance,ii));ii,ff,closest=ii[order],ff[order],closest[order]
    _,keep=np.unique(ii,return_index=True);ii,ff,closest=ii[keep],ff[keep],closest[keep]
    bary=trimesh.triangles.points_to_barycentric(tri[ff],closest).clip(0,1)
    bary/=bary.sum(1)[:,None]
    rest_area=np.linalg.norm(np.cross(material[:,1]-material[:,0],material[:,2]-material[:,0]),axis=1)/2
    nodal_area=np.zeros(len(points))
    for column in range(3):np.add.at(nodal_area,faces[:,column],rest_area/3)
    # Symmetric nearest-side proposals are counted half on each side. This
    # quadrature is experimental and must pass area/mesh convergence controls.
    return {'indices':np.c_[ii,faces[ff]],'barycentric':bary,'area':nodal_area[ii]/2,
            'initial_gap_m':np.linalg.norm(points[ii]-closest,axis=1)}
