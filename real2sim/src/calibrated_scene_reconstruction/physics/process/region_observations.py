"""Shared region measurements and source/native semantic correspondence checks."""
import cv2
import numpy as np

from .contract import require


def measure(mask,*,opening=(1,1),offset=(0,0)):
    """Same pixel operator for semantic source masks and projected material masks.

    This cannot certify that a source proposal includes the intended complete
    part. That needs an explicit source review; a color subset is not silently
    equivalent to the complete material projection.
    """
    kernel=tuple(opening);require(len(kernel)==2 and all(type(k) is int and k>0 for k in kernel),
        'Positive integer region morphology dimensions required')
    mask=np.asarray(mask);require(mask.ndim==2,'A binary region image is required')
    mask=cv2.morphologyEx((mask>0).astype(np.uint8),cv2.MORPH_OPEN,np.ones(kernel,np.uint8))
    y,x=np.nonzero(mask);require(len(x)>0,'Observed material region disappeared')
    center=np.array([x.mean(),y.mean()])+np.asarray(offset)
    return center,np.ptp(np.c_[x,y],axis=0)


def definition(part,scope,feature,opening):
    require(isinstance(part,str) and part and scope in {'complete_part','visible_surface','photometric_subset'}
            and feature in {'centroid','extent'},'Explicit semantic part, region scope and measurement required')
    require(isinstance(opening,(tuple,list)) and len(opening)==2
            and all(type(k) is int and k>0 for k in opening),'Explicit positive pixel morphology required')
    return {'part':part,'scope':scope,'feature':feature,'opening':list(opening)}


def validate_bindings(plan,source_review):
    """Opt-in contract for region channels; legacy point/trajectory plans remain valid."""
    definitions=plan.get('model',{}).get('observation_definitions',{})
    require(isinstance(definitions,dict),'Observation definitions must be a mapping')
    used={c['predicted'] for c in plan['channels']}
    require(set(definitions)<=used,'Declare only scored observation definitions')
    for key,item in definitions.items():
        require(item==definition(**item),'Invalid region observation definition')
        for c in plan['channels']:
            if c['predicted']==key:
                require(source_review.get('definitions',{}).get(c['observed'])==item,
                    'Source and native region scope/operator differ: '+c['id'])
    return definitions


def validate_native(expected,advertised):
    require(all(advertised.get(k)==v for k,v in expected.items()),
            'Native observer does not implement the declared region scope/operator')
    return {k:advertised[k] for k in expected}
