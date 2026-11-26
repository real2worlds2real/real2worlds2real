"""Thin-film optical mesh reconstructed from current simulated fluid particles.

The deposited thickness integrates particle volume; no reference footprint or
predefined water centre is used. This module never updates physical particles.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def particle_heightfield(points,table_z,spacing=.0016,cell=.0008):
    points=np.asarray(points,float)
    if not len(points):return np.empty((0,3),np.float32),np.empty((0,3),np.int32),0.
    pad=spacing*10
    low=points[:,:2].min(0)-pad;high=points[:,:2].max(0)+pad
    shape=np.ceil(np.round((high-low)/cell,8)).astype(int)+1
    if np.prod(shape)>2_000_000:raise ValueError('Film surface exceeds local render grid budget')
    ij=np.floor(np.round((points[:,:2]-low)/cell,8)).astype(int)
    volume=np.zeros(shape[::-1],float)
    np.add.at(volume,(ij[:,1],ij[:,0]),spacing**3)
    thickness=gaussian_filter(volume,sigma=2*spacing/cell,mode='constant')/cell**2
    y,x=np.meshgrid(np.arange(shape[1]),np.arange(shape[0]),indexing='ij')
    vertices=np.c_[low[0]+x.ravel()*cell,low[1]+y.ravel()*cell,
                    table_z+thickness.ravel()+.00005].astype(np.float32)
    wet=thickness>.00003
    quad=wet[:-1,:-1]&wet[:-1,1:]&wet[1:,:-1]&wet[1:,1:]
    yy,xx=np.nonzero(quad);a=yy*shape[0]+xx
    faces=np.vstack([np.c_[a,a+1,a+shape[0]],np.c_[a+1,a+shape[0]+1,a+shape[0]]]).astype(np.int32)
    return vertices,faces,float(thickness.sum()*cell**2)
