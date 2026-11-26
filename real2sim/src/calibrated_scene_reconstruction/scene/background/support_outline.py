"""Propose a thin supported part's concave footprint from calibrated masks.

The agent supplies a support plane/part-height hypothesis and reviews the same
part in every original frame. Two independent rays must agree. This preserves
concavities rather than filling an arbitrary rectangular envelope; hidden parts,
thickness, physical contact/stability and actual appearance remain unapproved.
"""
import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background.source_replace import select


def solve(cameras,rows,plane_z,pixel_m=.0015,simplify_m=.0015,cleanup_m=0.,cleanup_review=None):
    if len(rows)<3 or not np.isfinite(plane_z) or not .0005<=pixel_m<=.01 or not 0<=simplify_m<=.005:
        raise ValueError('Three reviewed views, finite plane and bounded outline resolution required')
    names=list(map(str,cameras['names']));points=[];seen=set()
    for row in rows:
        frame=row['frame']
        if frame in seen or frame not in names:raise ValueError('Distinct original calibrated part views required')
        seen.add(frame);i=names.index(frame);mask=np.asarray(row['pixels'],bool)
        if mask.shape!=tuple(cameras['size'][i][::-1]):raise ValueError('Calibrated original raster required')
        y,x=np.nonzero(mask);step=max(1,len(x)//10000)
        if len(x)<64:raise ValueError('Usable reviewed part silhouette required')
        rays=np.c_[x[::step],y[::step],np.ones(len(x[::step]))]@np.linalg.inv(cameras['K'][i]).T@cameras['R_world_cam'][i].T
        distance=(plane_z-cameras['centre'][i,2])/rays[:,2];valid=(abs(rays[:,2])>.05)&(distance>0)
        if valid.sum()<64:raise ValueError('Part plane must be in front of its source rays')
        points.append((cameras['centre'][i]+rays[valid]*distance[valid,None])[:,:2])
    points=np.concatenate(points);lo=points.min(0)-3*pixel_m;hi=points.max(0)+3*pixel_m
    size=np.ceil((hi-lo)/pixel_m).astype(int)+1
    if np.any(size>2048) or np.prod(size)>2000000 or np.any(size<4):raise ValueError('Resolved bounded supported-part grid required; review plane/ownership')
    x,y=np.meshgrid(np.arange(size[0]),np.arange(size[1]));P=np.c_[lo[0]+x.ravel()*pixel_m,lo[1]+y.ravel()*pixel_m,np.full(x.size,plane_z)]
    mask=select(P,cameras,rows).reshape(x.shape).astype('uint8');changed=0
    if cleanup_m:
        if not pixel_m<=cleanup_m<=.003 or not isinstance(cleanup_review,str) or not cleanup_review.strip():
            raise ValueError('Small explicitly reviewed raster completion required')
        radius=int(np.floor(cleanup_m/pixel_m));completed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2*radius+1,2*radius+1),'uint8'))
        changed=int(np.count_nonzero(completed!=mask));mask=completed
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<2:raise ValueError('No independently supported outline')
    keep=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);discarded=1-stats[keep,cv2.CC_STAT_AREA]/max(1,mask.sum())
    if discarded>.05:raise ValueError('Disconnected source parts require separate agent identities/completion')
    mask=(labels==keep).astype('uint8');contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    outer=max((k for k in range(len(contours)) if hierarchy[0,k,3]<0),key=lambda k:cv2.contourArea(contours[k]))
    holes=[k for k in range(len(contours)) if hierarchy[0,k,3]==outer]
    contour=cv2.approxPolyDP(contours[outer],simplify_m/pixel_m,True)[:,0]
    if len(contour)<3:raise ValueError('Nondegenerate source footprint required')
    outline=lo+contour*pixel_m;following=np.roll(outline,-1,axis=0)
    area=(outline[:,0]*following[:,1]-outline[:,1]*following[:,0]).sum()/2
    if abs(area)<1e-6:raise ValueError('Nondegenerate source footprint required')
    if area<0:outline=outline[::-1]
    rings=[outline]+[lo+c[:,0]*pixel_m for c in [contours[k] for k in holes]]
    checks=[]
    for row in rows:
        i=names.index(row['frame']);actual=np.zeros(row['pixels'].shape,'uint8')
        for number,ring in enumerate(rings):
            world=np.c_[ring,np.full(len(ring),plane_z)];pc=(world-cameras['centre'][i])@cameras['R_world_cam'][i];uv=pc@cameras['K'][i].T;uv=uv[:,:2]/uv[:,2,None];cv2.fillPoly(actual,[np.rint(uv).astype('int32')],int(number==0))
        target=np.asarray(row['pixels'],bool);actual=actual>0
        checks.append({'frame':row['frame'],'planar_iou':float((actual&target).sum()/max(1,(actual|target).sum()))})
    runs=[]
    for row,line in enumerate(mask):
        changes=np.diff(np.r_[0,line,0].astype(int));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
        runs.extend([[row,int(a),int(b)] for a,b in zip(starts,ends)])
    return {'approved':False,'outline_world_xy':outline.tolist(),'rings_world_xy':[r.tolist() for r in rings],
            'grid':{'origin_xy':lo.tolist(),'pixel_m':pixel_m,'shape':list(mask.shape),'occupied_runs':runs},
            'inferred_raster_cleanup':{'radius_m':cleanup_m,'changed_cells':changed,'review':cleanup_review},
            'hole_areas_m2':[float(cv2.contourArea(contours[k])*pixel_m**2) for k in holes],'plane_z':float(plane_z),
            'pixel_m':pixel_m,'simplify_m':simplify_m,'discarded_fraction':float(discarded),'source_checks':checks,'scope':__doc__.strip()}


def solid(grid,base_z,height):
    """Closed grid solid with real concavities/holes, without a convex cap fan."""
    mask=np.zeros(grid['shape'],bool)
    for y,a,b in grid['occupied_runs']:mask[y,a:b]=True
    origin=np.asarray(grid['origin_xy'],float);step=grid['pixel_m']
    if not mask.any() or not np.isfinite([*origin,base_z,height,step]).all() or not .0005<=step<=.01 or not .001<=height<=.08:raise ValueError('Finite bounded supported footprint/thickness required')
    vertices=[];faces=[];lookup={}
    def index(x,y,z):
        key=(x,y,z)
        if key not in lookup:lookup[key]=len(vertices);vertices.append([origin[0]+(x-.5)*step,origin[1]+(y-.5)*step,base_z+z*height])
        return lookup[key]
    def quad(keys):
        ids=[index(*k) for k in keys];faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
    h,w=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        quad([(x,y,1),(x+1,y,1),(x+1,y+1,1),(x,y+1,1)])
        quad([(x,y+1,0),(x+1,y+1,0),(x+1,y,0),(x,y,0)])
        if x==0 or not mask[y,x-1]:quad([(x,y,0),(x,y,1),(x,y+1,1),(x,y+1,0)])
        if x==w-1 or not mask[y,x+1]:quad([(x+1,y,0),(x+1,y+1,0),(x+1,y+1,1),(x+1,y,1)])
        if y==0 or not mask[y-1,x]:quad([(x,y,0),(x+1,y,0),(x+1,y,1),(x,y,1)])
        if y==h-1 or not mask[y+1,x]:quad([(x,y+1,0),(x,y+1,1),(x+1,y+1,1),(x+1,y+1,0)])
    from calibrated_scene_reconstruction.assets.topology import closed_oriented_surface
    V,F=np.asarray(vertices),np.asarray(faces)
    if not closed_oriented_surface(V,np.full(len(F),3),F.ravel()):raise ValueError('Raster outline is not one closed oriented solid; review diagonal contacts/disconnected parts')
    return V,F
