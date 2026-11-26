"""Pixels actually observed after camera undistortion, independent of RGB/classes.

Black remap padding is missing data, not a negative robot/table/background label.
The validity map follows the same OpenCV transform as capture extraction.
"""
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.paths import rel, resolve


def undistorted_valid(K, dist, size):
    w,h=map(int,size);K=np.asarray(K,float);dist=np.asarray(dist,float)
    if min(w,h)<1 or K.shape!=(3,3) or not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError('Finite camera calibration and positive image dimensions required')
    coverage=cv2.undistort(np.ones((h,w),np.float32),K,dist)
    return coverage>=1.-1e-6


def resize_valid(mask, shape):
    """A target cell is observed only when its entire source footprint is valid."""
    return cv2.resize(mask.astype('float32'),tuple(shape),interpolation=cv2.INTER_AREA)>=1.-1e-6


class ValidPixels:
    def __init__(self, session):
        self.session=session;self.inputs={};self.cache={}

    def read(self, path):
        path=Path(path)
        self.inputs[rel(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def get(self, name, shape):
        session=self.session;folder=session/'splat';shape=tuple(map(int,shape))
        path=folder/'data/images'/(Path(name).stem+'_valid.png')
        if path.exists():
            mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape!=shape:raise ValueError('Rectification validity dimensions differ from image')
            self.inputs[rel(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            return mask>128
        match=re.fullmatch(r'(\d+)_\d+\.[A-Za-z]+',str(name))
        if match is None:return np.ones(shape,bool)
        index=int(match[1]);key=(index,shape)
        if key in self.cache:return self.cache[key]
        camera=None;manifest=folder/'rectification.json'
        if manifest.exists():camera=self.read(manifest).get('cameras',{}).get(str(index))
        config_dir=getattr(session,'config_dir',None)
        if camera is None and index==0 and config_dir is not None:
            source=Path(config_dir)/'camera.json'
            if source.exists():camera=self.read(source)
        if camera is None and (folder/'frames.json').exists():
            videos=self.read(folder/'frames.json').get('videos',[])
            if index<len(videos):
                video=videos[index]
                source=Path(config_dir)/'splat.json' if config_dir is not None else None
                if source is not None and source.exists():camera=self.read(source).get('cameras',{}).get(video)
                if camera is None and (folder/'cameras.json').exists():
                    cache=self.read(folder/'cameras.json');raw=resolve(video)
                    identity=f'{video}|{raw.stat().st_size}' if raw.exists() else None
                    if identity in cache:camera=cache[identity]
        if camera is None:
            mask=np.ones(shape,bool)  # Legacy/unknown camera: do not invent missing pixels from colour.
        else:
            size=camera.get('size',[shape[1],shape[0]])
            if tuple(size)!=(shape[1],shape[0]):raise ValueError('Rectification camera dimensions differ from image')
            mask=undistorted_valid(camera['K'],camera.get('dist',np.zeros(5)),size)
        self.cache[key]=mask
        return mask
