"""Select only named native surface parts; never borrow a neighbour's body."""
import json
from pathlib import Path
import sys

import numpy as np
from mathutils.bvhtree import BVHTree

job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
geometry=np.load(job['geometry']);cloud=np.load(job['cloud'])
points,sigma=cloud['points'],cloud['sigma'];margin=float(job['margin_m'])
if not .01<=margin<=.10:raise ValueError('Bounded native replacement margin required')
removed=np.zeros(len(points),bool)
for entity in job['entities']:
    vertices=[];faces=[];offset=0;selected=job.get('replacement_parts',{}).get(entity)
    for i,row in enumerate(job['parts']):
        if row['entity']!=entity or row['external'] or selected is not None and row['part'] not in selected:continue
        v,f=geometry[f'v{i}'],geometry[f'f{i}'];vertices.append(v);faces.append(f+offset);offset+=len(v)
    if not vertices:raise ValueError('Explicit native replacement parts missing: '+entity)
    V,F=np.concatenate(vertices),np.concatenate(faces);tree=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True)
    distance=margin+np.minimum(sigma*2,.10)
    ids=np.flatnonzero(np.all((points>=V.min(0)-distance[:,None])&(points<=V.max(0)+distance[:,None]),axis=1))
    for i in ids:
        if tree.find_nearest(tuple(points[i]),float(distance[i]))[0] is not None:removed[i]=True
    print(entity,len(ids),'nearby candidates;',int(removed.sum()),'removed so far',flush=True)
np.save(job['output'],removed)
