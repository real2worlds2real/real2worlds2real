"""Uniformly sampled flat cloth from a complete source-derived outline.

This builds material quantity and boundary only, never a folded pose, permanent
paper crease or task result. Physical preparation and native checks are separate.
"""
import cv2
import numpy as np

from .cloth_rest import triangle_edges,validate
from .cloth_topology import measure as topology


def build(outline,*,spacing_m=.001):
    spacing=np.asarray(spacing_m,float)
    if spacing.ndim==0:spacing=np.repeat(spacing,2)
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or (spacing<1e-5).any() or (spacing>.01).any():
        raise ValueError('One or two explicit material spacings in10micrometres..10mm required')
    area=float(outline['area_m2'])
    loops=[np.asarray(c,float)*np.sqrt(area) for c in outline['area_normalized_contours']]
    if area<=0 or not loops or any(c.ndim!=2 or c.shape[1:]!=(2,) or not np.isfinite(c).all() for c in loops):
        raise ValueError('Finite complete unfolded material outline required')
    low=np.concatenate(loops).min(0);high=np.concatenate(loops).max(0)
    counts=np.ceil((high-low)/spacing).astype(int)+1
    if counts.min()<4 or np.prod(counts)>500_000:
        raise ValueError('Outline must resolve at least four material nodes per side within500k-node proposal budget')
    x,y=np.linspace(low[0],high[0],counts[0]),np.linspace(low[1],high[1],counts[1])
    dx,dy=x[1]-x[0],y[1]-y[0]
    # Supersample the source boundary, including holes. Whole quadrilateral
    # cells stay within its sampled footprint; no disconnected offcut is added.
    raster=np.zeros(tuple((4*(counts-1)+1)[::-1]),np.uint8)
    cv2.fillPoly(raster,[np.rint((c-low)/[dx,dy]*4).astype('int32') for c in loops],1)
    inside=raster[::4,::4].astype(bool)
    cells=inside[:-1,:-1]&inside[:-1,1:]&inside[1:,:-1]&inside[1:,1:]&raster[2::4,2::4].astype(bool)
    xx,yy=np.meshgrid(x,y);points=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    grid=np.arange(len(points)).reshape(len(y),len(x));a=grid[:-1,:-1][cells]
    faces=np.stack([np.c_[a,a+1,a+len(x)+1],np.c_[a,a+len(x)+1,a+len(x)]],1).reshape(-1,3)
    keep=np.unique(faces);mapping=np.full(len(points),-1,dtype=int);mapping[keep]=np.arange(len(keep));points=points[keep];faces=mapping[faces]
    topo=topology(len(points),faces)
    if topo['edge_connected_surfaces']!=1 or topo['nonmanifold_edges']:
        raise ValueError('One complete material surface required; refine sampling or review source outline')
    edges=triangle_edges(faces);delta=abs(points[edges[:,1]]-points[edges[:,0]])
    diagonal=(delta[:,0]>.1*dx)&(delta[:,1]>.1*dy)
    all_edges=np.stack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],1).reshape(-1,2)
    unique,inverse,count=np.unique(np.sort(all_edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');offset=np.r_[0,np.cumsum(count)];shared=np.flatnonzero(count==2)
    owners=np.c_[order[offset[shared]]//3,order[offset[shared]+1]//3]
    opposite=faces[owners].sum(2)-unique[shared].sum(1)[:,None]
    bend=np.unique(np.sort(opposite,axis=1),axis=0)
    codes=edges[:,0]*len(points)+edges[:,1];bend=bend[~np.isin(bend[:,0]*len(points)+bend[:,1],codes)]
    rest={'points':points.tolist(),'springs':{'stretch':edges[~diagonal].tolist(),'shear':edges[diagonal].tolist(),'bend':bend.tolist()}}
    metric=validate(rest,points,faces)
    uv=(points[:,:2]-low)/(high-low)
    return {'points':points,'faces':faces,'rest':rest,'uv':uv,
            'report':{'proposal_only':True,'task_approved':False,'source_outline_area_m2':area,
                      'material_area_m2':metric['rest_area_m2'],'surface_topology':topo,
                      'material_spacing_m':[float(dx),float(dy)],'crease_memory_authored':False,
                      'scope':'Complete single flat material surface; no folded pose, inferred mechanical constants or task approval'}}
