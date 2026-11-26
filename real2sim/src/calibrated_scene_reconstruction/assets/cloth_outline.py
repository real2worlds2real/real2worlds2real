"""Unfolded material outlines; no captured pose or supplied true dimensions."""
import cv2
import numpy as np


def describe(contours):
    loops=[np.asarray(c,float).reshape(-1,2) for c in contours]
    if not loops or any(len(c)<3 or not np.isfinite(c).all() for c in loops):
        raise ValueError('Finite closed material contours required')
    area=0.
    for i,contour in enumerate(loops):
        depth=sum(cv2.pointPolygonTest(c.astype('float32'),tuple(contour[0]),False)>0
                  for j,c in enumerate(loops) if i!=j)
        area+=(-1 if depth%2 else 1)*cv2.contourArea(contour.astype('float32'))
    points=np.concatenate(loops).astype('float32');rectangle=cv2.minAreaRect(points)
    box=cv2.boxPoints(rectangle).astype(float);edges=np.roll(box,-1,axis=0)-box
    lengths=np.linalg.norm(edges,axis=1);long=edges[np.argmax(lengths)]
    if area<=1e-14 or lengths.min()<=1e-10:raise ValueError('Positive material area and resolved extents required')
    axis=long/np.linalg.norm(long);axes=np.array([axis,[-axis[1],axis[0]]])
    shape=[((c-np.asarray(rectangle[0]))@axes.T/np.sqrt(area)).tolist() for c in loops]
    return {'area_m2':float(area),'dimensions_m':sorted(map(float,rectangle[1])),
            'aspect_ratio':float(max(rectangle[1])/min(rectangle[1])),
            'plane_centre_m':list(map(float,rectangle[0])),'plane_axes':axes.tolist(),
            'area_normalized_contours':shape,
            'scope':'Unfolded boundary and dimensions. Normalization is for shape comparison only; it does not rescale any asset.'}


def from_mask(mask,homography):
    mask=np.asarray(mask);H=np.asarray(homography,float)
    if (mask.ndim!=2 or H.shape!=(3,3) or not np.isfinite(H).all()
            or not mask.any()):raise ValueError('Nonempty source mask and finite plane camera required')
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        raise ValueError('Complete material boundary cannot touch the source image edge')
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    inverse=np.linalg.inv(H);loops=[]
    for contour in contours:
        if len(contour)<3:continue
        q=np.c_[contour[:,0,:],np.ones(len(contour))]@inverse.T
        if (not np.isfinite(q).all() or (abs(q[:,2])<1e-10).any()
                or (q[:,2]*q[0,2]<=0).any()):raise ValueError('Material outline crosses the source plane horizon')
        loops.append(q[:,:2]/q[:,2:])
    return describe(loops)


def from_material(points,faces):
    points=np.asarray(points,float);faces=np.asarray(faces)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=len(points)):
        raise ValueError('Finite intrinsic points and indexed material faces required')
    centred=points-points.mean(0);_,_,axes=np.linalg.svd(centred,full_matrices=False)
    if abs(centred@axes[2]).max()>1e-8:
        raise ValueError('Whole-sheet shape needs a planar unfolded material chart')
    planar=centred@axes[:2].T
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    unique,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    if (counts>2).any():raise ValueError('Nonmanifold material boundary cannot identify a whole sheet')
    neighbours={}
    for a,b in unique[counts==1]:
        neighbours.setdefault(int(a),[]).append(int(b));neighbours.setdefault(int(b),[]).append(int(a))
    if not neighbours or any(len(v)!=2 for v in neighbours.values()):
        raise ValueError('Closed nonbranching material boundary required')
    remaining=set(neighbours);loops=[]
    while remaining:
        start=min(remaining);sequence=[start];visited={start};previous=None;current=start
        while True:
            choices=neighbours[current];next_node=choices[0] if choices[0]!=previous else choices[1]
            if next_node==start:break
            if next_node in visited:raise ValueError('Repeated material boundary vertex')
            sequence.append(next_node);visited.add(next_node);previous,current=current,next_node
        remaining.difference_update(sequence);loops.append(planar[sequence])
    return describe(loops)


def difference(reference,candidate,*,resolution=256):
    """Compare full area-normalized boundaries under rigid planar symmetries.

    Actual area, two physical extents and aspect ratio are separate checks.
    Front/back reflection is allowed; arbitrary contour deformation is not.
    """
    a=[np.array(c,float) for c in reference['area_normalized_contours']]
    b=[np.array(c,float) for c in candidate['area_normalized_contours']]
    if type(resolution) is not int or resolution<64:raise ValueError('Resolved outline raster required')
    radius=max(abs(np.concatenate(a)).max(),abs(np.concatenate(b)).max())*1.05
    def raster(contours):
        mask=np.zeros((resolution,resolution),np.uint8)
        pixels=[np.rint((c/radius+1)*.5*(resolution-1)).astype('int32') for c in contours]
        cv2.fillPoly(mask,pixels,1);return mask.astype(bool)
    target=raster(a);best=float('inf')
    for transpose in [False,True]:
        for signs in [[1,1],[1,-1],[-1,1],[-1,-1]]:
            predicted=raster([(c[:,::-1] if transpose else c)*signs for c in b])
            best=min(best,float((predicted^target).sum()/max(1,target.sum())))
    return best
