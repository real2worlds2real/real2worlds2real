"""Source-verified transmitting shells affect optical rays, never collisions."""
import numpy as np


def transmitting_faces(obj,observation,parts,count):
    refs=observation.get('transmitting_links',[])
    if not refs:return None
    result=np.zeros(count,bool)
    for ref in refs:
        owner,name=ref.split('/')
        if owner!=obj:raise ValueError('Cross-assembly transmission requires a joint optical scene; local bypass refused')
        part=next((p for p in parts or [] if p['name']==name),None)
        if part is None or not part.get('optically_transmitting_faces'):
            raise ValueError('Source transmitter must have transmitting faces in the actual modeled material')
        indices=np.asarray(part['optically_transmitting_faces'])
        if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or indices.min()<0
                or indices.max()>=part['triangle_count']):raise ValueError('Invalid transmitting material face correspondence')
        result[part['face_start']+indices]=True
    return result
