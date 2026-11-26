"""Conservative native/GS depth compatibility for appearance openings.

This uses front-most retained Gaussian centres, not a measured depth image.
Unknown or incompatible rays retain the complete native appearance. Actual
opaque-field coverage and source/free-view checks are still required.
"""
import numpy as np


def compatible(native_depth,pixels,field_depth,band,cell=2):
    if isinstance(band,bool) or not isinstance(band,(float,int)) or not .005<=band<=.08:
        raise ValueError('Bounded metric GS depth compatibility required')
    xy=np.asarray(pixels,int)//cell;depth=np.asarray(native_depth,float)
    field=np.asarray(field_depth,float)
    if xy.shape!=(len(depth),2) or field.ndim!=2 or np.any(xy<0) or np.any(xy[:,0]>=field.shape[1]) or np.any(xy[:,1]>=field.shape[0]):
        raise ValueError('Aligned source-ray depths and pixel grid required')
    observed=field[xy[:,1],xy[:,0]]
    return np.isfinite(observed)&np.isfinite(depth)&(abs(depth-observed)<=band)
