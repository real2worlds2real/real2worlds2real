"""Exact binary triangle masks with duplicate pixel triangles drawn once."""
import cv2
import numpy as np


def triangles(points,faces,shape):
    pixels=np.rint(np.asarray(points)[np.asarray(faces)]).astype(np.int32)
    mask=np.zeros(shape,np.uint8);height,width=map(int,shape)
    if (len(pixels)>=4096 and width*height<2**21 and pixels.min()>=0
            and (pixels[...,0]<width).all() and (pixels[...,1]<height).all()):
        # Any permutation of a triangle has the same LINE_8 filled mask.
        # The pixel ID packing is injective; no triangle or small feature is
        # discarded. Coincident projected triangles need only one draw call.
        codes=np.sort(pixels[...,0].astype(np.uint64)+width*pixels[...,1].astype(np.uint64),axis=1)
        bits=max(1,(width*height-1).bit_length());keep=(1<<bits)-1
        packed=np.unique(codes[:,0] | (codes[:,1]<<bits) | (codes[:,2]<<(2*bits)))
        codes=np.column_stack([packed&keep,(packed>>bits)&keep,packed>>(2*bits)])
        pixels=np.stack([codes%width,codes//width],axis=-1).astype(np.int32)
    for face in pixels:cv2.fillConvexPoly(mask,face,255)
    return mask
