"""Square sheet with a prepared central fold; intrinsic material remains flat."""
import numpy as np


def center_sheet(width=.12,length=.12,grip_height=.0175,rest_offset=.00085):
    radius=.00125;floor_radius=.004
    arc=np.pi*radius/2
    height=grip_height-rest_offset
    vertical=height-floor_radius
    bend_end=arc+vertical+np.pi*floor_radius/2
    if vertical<=0 or bend_end>=length/2:raise ValueError('Sheet cannot reach the desk')
    positive=np.r_[np.linspace(0,.006,7),np.linspace(.008,length/2,27)]
    s=np.r_[-positive[:0:-1],positive]
    u=np.linspace(-width/2,width/2,49)
    U,S=np.meshgrid(u,s);a=np.abs(S);sign=np.sign(S)
    y=np.zeros_like(S);z=np.zeros_like(S)
    top=a<=arc
    y[top]=sign[top]*radius*np.sin(a[top]/radius)
    z[top]=radius*np.cos(a[top]/radius)
    straight=(a>arc)&(a<=arc+vertical)
    y[straight]=sign[straight]*radius;z[straight]=-(a[straight]-arc)
    bend=(a>arc+vertical)&(a<=bend_end)
    theta=(a[bend]-arc-vertical)/floor_radius
    y[bend]=sign[bend]*(radius+floor_radius*(1-np.cos(theta)))
    z[bend]=-vertical-floor_radius*np.sin(theta)
    flat=a>bend_end
    y[flat]=sign[flat]*(radius+floor_radius+a[flat]-bend_end);z[flat]=-height
    points=np.c_[U.ravel(),y.ravel(),z.ravel()]
    material=np.c_[U.ravel(),S.ravel(),np.zeros(U.size)]
    nodes=np.flatnonzero((np.abs(U.ravel())<=.0051)&(np.abs(S.ravel())<=.0051))
    return dict(points=points,material=material,rows=len(s),columns=len(u),grasp_nodes=nodes,
                material_grip_fraction=[.5,.5],width=width,length=length)
