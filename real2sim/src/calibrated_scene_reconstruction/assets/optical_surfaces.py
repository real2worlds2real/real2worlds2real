"""Face-level optical metadata from the actual generated Blender materials."""
import json

import numpy as np


def annotate(data,parts,opaque_faces=None):
    if not parts:return parts
    required={'face_material','material_names','materials_json'}
    if not required<=set(data):return parts
    names=list(map(str,data['material_names']));materials=json.loads(str(data['materials_json']))
    ids=np.asarray(data['face_material'])
    if ids.shape!=(len(data['F']),) or ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=len(names):
        raise ValueError('Optical face materials must preserve actual triangle correspondence')
    transmitting=[]
    for name in names:
        value=materials.get(name,{})
        transmission,opacity=value.get('transmission',0.),value.get('opacity',1.)
        if not np.isfinite([transmission,opacity]).all() or not 0<=transmission<=1 or not 0<=opacity<=1:
            raise ValueError('Actual optical material transmission/opacity must be finite0..1')
        transmitting.append(transmission>0 or opacity<1)
    optical=np.asarray(transmitting,bool)[ids]
    if opaque_faces is not None:
        opaque_faces=np.asarray(opaque_faces)
        if opaque_faces.shape!=optical.shape or opaque_faces.dtype!=bool:
            raise ValueError('Printed opacity must preserve actual triangle correspondence')
        optical &= ~opaque_faces
    for part in parts:
        start,count=part['face_start'],part['triangle_count']
        part['optically_transmitting_faces']=np.flatnonzero(optical[start:start+count]).tolist()
    return parts
