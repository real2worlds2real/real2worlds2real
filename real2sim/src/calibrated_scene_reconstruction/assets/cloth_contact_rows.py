"""Fresh frictionless contact rows for geometry proposals, never native pins."""
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def normal_cosine_rows(points,faces,pairs,*,minimum_cosine,scale):
    """Linearize a local closing-fold guard without separating its neighbours.

    The scalar residual is scale*(dot(unit_normals)-minimum_cosine).
    This is an offline angular inequality, never a native bending law.
    """
    points=np.asarray(points,float);faces=np.asarray(faces,int);pairs=np.asarray(pairs,int)
    if not np.isfinite([minimum_cosine,scale]).all() or not 0<minimum_cosine<1 or scale<=0:
        raise ValueError('Positive explicit angular guard and metric row scale required')
    triangles=points[faces[pairs]]
    u=triangles[:,:,1]-triangles[:,:,0];v=triangles[:,:,2]-triangles[:,:,0]
    cross=np.cross(u,v);area=np.linalg.norm(cross,axis=2)
    if not np.isfinite(area).all() or (area<=1e-14).any():raise ValueError('Nondegenerate fold guard triangles required')
    normals=cross/area[:,:,None];cosine=(normals[:,0]*normals[:,1]).sum(1)
    w=(normals[:,::-1]-cosine[:,None,None]*normals)/area[:,:,None]
    gb=np.cross(v,w);gc=np.cross(w,u);ga=-gb-gc
    gradient=np.stack([ga,gb,gc],axis=2).reshape(-1,6,3)*scale
    return faces[pairs].reshape(-1,6),gradient,(cosine-minimum_cosine)*scale


def point_triangle(points,faces,material_points,clearance,*,neighbours=48):
    points=np.asarray(points,float);faces=np.asarray(faces);rest=np.asarray(material_points,float)
    if (points.shape!=rest.shape or points.ndim!=2 or points.shape[1:]!=(3,)
            or not np.isfinite(points).all() or not np.isfinite(rest).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min()<0 or faces.max()>=len(points)
            or not np.isfinite(clearance) or clearance<=0 or type(neighbours) is not int or neighbours<1):
        raise ValueError('Finite current/material mesh and positive proposal clearance required')
    tri=points[faces];material=rest[faces]
    edge=max(np.linalg.norm(material[:,i]-material[:,j],axis=1).max() for i,j in [(0,1),(1,2),(2,0)])
    k=min(neighbours,len(tri));_,near=cKDTree(tri.mean(1)).query(points,k=k)
    ii=np.repeat(np.arange(len(points)),k);ff=near.ravel()
    far=np.linalg.norm(rest[ii]-material.mean(1)[ff],axis=1)>3*edge;ii,ff=ii[far],ff[far]
    closest=trimesh.triangles.closest_point(tri[ff],points[ii]);gap=np.linalg.norm(points[ii]-closest,axis=1)
    active=(gap<2*clearance)&(gap>1e-10);ii,ff,closest,gap=ii[active],ff[active],closest[active],gap[active]
    bary=trimesh.triangles.points_to_barycentric(tri[ff],closest)
    normal=(points[ii]-closest)/gap[:,None]
    indices=np.c_[ii,faces[ff]]
    gradient=np.concatenate([normal[:,None],-bary[:,:,None]*normal[:,None]],axis=1)
    return {'indices':indices,'gradient':gradient,'residual':gap-1.1*clearance,'gap_m':gap,
            'scope':'Current closest-feature linearization only. Recompute after every geometry step; positive residual is free clearance, not a zero-displacement constraint.'}
