"""Finite straight shafts with explicit cross section and tip taper.

A semantic stick does not imply a circular cylinder. Source evidence selects
its section; actual triangles, not the enclosing radius, determine contacts.
"""
import numpy as np


def mesh(root,tip,radii,*,roll=0.,taper=1.,section='ellipse',sides=32):
    root,tip=np.asarray(root,float),np.asarray(tip,float);radii=np.asarray(radii,float)
    if (root.shape!=(3,) or tip.shape!=(3,) or radii.shape!=(2,)
            or not np.isfinite(np.r_[root,tip,radii,roll,taper]).all() or min(radii)<=0
            or taper<=0 or np.linalg.norm(tip-root)<=4*max(radii)
            or section not in {'ellipse','rectangle'} or sides<8):
        raise ValueError('Finite straight shaft, positive section and taper required')
    d=(tip-root)/np.linalg.norm(tip-root);u=np.eye(3)[np.argmin(abs(d))];u-=d*(u@d);u/=np.linalg.norm(u);v=np.cross(d,u)
    a=np.cos(roll)*u+np.sin(roll)*v;b=-np.sin(roll)*u+np.cos(roll)*v
    profile=(np.array([[1,1],[-1,1],[-1,-1],[1,-1]]) if section=='rectangle' else
             np.c_[np.cos(np.arange(sides)*2*np.pi/sides),np.sin(np.arange(sides)*2*np.pi/sides)])
    profile=profile*radii;vertices=[];n=len(profile)
    for point,scale in ((root,1.),(tip,taper)):
        vertices.extend(point+scale*(profile[:,0,None]*a+profile[:,1,None]*b))
    faces=[]
    for i in range(n):
        j=(i+1)%n;faces.extend([[i,j,n+j],[i,n+j,n+i]])
    for i in range(1,n-1):faces.extend([[0,i+1,i],[n,n+i,n+i+1]])
    return np.asarray(vertices),np.asarray(faces,int)
