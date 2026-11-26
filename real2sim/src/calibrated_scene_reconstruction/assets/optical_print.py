"""Reviewed opaque print over transmitting surfaces, without changing collisions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def declaration(look, base):
    if (look or {}).get('pending_optical_print'):
        raise ValueError('Reprojected optical print requires source-region review before use')
    value=(look or {}).get('optical_print')
    if value is None:return None
    keys={'opaque_mask','transmission_mask','albedo','review'}
    if not isinstance(value,dict) or set(value)!=keys:
        raise ValueError('Optical print needs two complementary masks, ink albedo and source review')
    base=Path(base).resolve();paths={}
    for key,name in value.items():
        if not isinstance(name,str) or Path(name).is_absolute():raise ValueError('Optical print files must be session-relative')
        path=(base/name).resolve()
        if not path.is_relative_to(base) or not path.is_file():raise ValueError('Missing local optical print file')
        paths[key]=path
    review=json.loads(paths['review'].read_text())
    if (review.get('status')!='resolved' or not review.get('basis')
            or review.get('source_mesh_sha256')!=look.get('source_mesh_sha256')):
        raise ValueError('Optical print requires a source review bound to this physical mesh')
    if not review.get('evidence_files'):
        raise ValueError('Optical print review needs hashed source evidence')
    for relative,expected in review['evidence_files'].items():
        evidence=(base/relative).resolve()
        if (Path(relative).is_absolute() or not evidence.is_relative_to(base) or not evidence.is_file()
                or hashlib.sha256(evidence.read_bytes()).hexdigest()!=expected):
            raise ValueError('Optical print source evidence changed or escaped the session')
    for key in keys-{'review'}:
        if review.get('sha256',{}).get(key)!=hashlib.sha256(paths[key].read_bytes()).hexdigest():
            raise ValueError('Optical print changed since source review')
    mask=np.asarray(Image.open(paths['opaque_mask']).convert('L'))
    inverse=np.asarray(Image.open(paths['transmission_mask']).convert('L'))
    if mask.shape!=inverse.shape or not np.all(abs(mask.astype(int)+inverse.astype(int)-255)<=1):
        raise ValueError('Opaque and transmission masks must be complementary')
    return paths


def opaque_faces(look, base, count):
    paths=declaration(look,base)
    if paths is None:return None
    with np.load(Path(base)/look['mesh']) as atlas:uv=atlas['face_uv']
    if uv.shape!=(count,3,2) or not np.isfinite(uv).all() or uv.min()<0 or uv.max()>1:
        raise ValueError('Optical print must retain a normalized UV for every actual triangle')
    image=np.asarray(Image.open(paths['opaque_mask']).convert('L'))>2
    h,w=image.shape;pixels=uv*np.array([w-1,h-1]);pixels[:,:,1]=(h-1)-pixels[:,:,1]
    lo=np.floor(pixels.min(1)).astype(int);hi=np.ceil(pixels.max(1)).astype(int)+1
    integral=np.pad(image.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    covered=(integral[hi[:,1],hi[:,0]]-integral[lo[:,1],hi[:,0]]
             -integral[hi[:,1],lo[:,0]]+integral[lo[:,1],lo[:,0]])>0
    # Bounding rectangles conservatively block partial ink triangles. They can
    # reject an ambiguous ray, but cannot declare an opaque printed pixel clear.
    return covered
