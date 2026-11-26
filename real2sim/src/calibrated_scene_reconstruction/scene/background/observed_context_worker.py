"""Remove captured Gaussians near explicitly replaced native background surfaces."""
import json
from pathlib import Path
import sys

import numpy as np
from mathutils.bvhtree import BVHTree

job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text());z=np.load(job['geometry']);cloud=np.load(job['cloud'])
P,sigma=cloud['points'],cloud['sigma'];default_margin=float(job['margin_m'])
if not 0<default_margin<=.20:raise ValueError('Small metric native replacement margin required')
remove=np.zeros(len(P),bool)
for entity in job['entities']:
    margin=float(job.get('margins_m',{}).get(entity,default_margin))
    if not 0<margin<=.30:raise ValueError('Bounded native surface replacement band required')
    vertices=[];faces=[];offset=0
    for i,row in enumerate(job['parts']):
        if row['entity']!=entity:continue
        v,f=z[f'v{i}'],z[f'f{i}'];vertices.append(v);faces.append(f+offset);offset+=len(v)
    V,F=np.concatenate(vertices),np.concatenate(faces)
    if entity in job.get('volume_entities',[]):
        # An enclosed native replacement must not retain the old appearance in
        # its interior. Neighbor semantic preservation still applies afterward.
        interior=np.all((P>=V.min(0)-margin)&(P<=V.max(0)+margin),axis=1)
        remove|=interior
        print(entity,int(interior.sum()),'volume replacement candidates',flush=True)
    tree=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True)
    distance=margin+np.minimum(sigma*2,.10)
    ids=np.flatnonzero(np.all((P>=V.min(0)-distance[:,None])&(P<=V.max(0)+distance[:,None]),axis=1))
    for i in ids:
        nearest=tree.find_nearest(tuple(P[i]),float(distance[i]))
        if nearest[0] is not None:remove[i]=True
    print(entity,len(ids),'nearby candidates;',int(remove.sum()),'removed so far',flush=True)
np.save(job['output'],remove)
if job.get('quality_distance_output'):
    V=[];F=[];offset=0
    for i,row in enumerate(job['parts']):
        if row['external']:continue
        v,f=z[f'v{i}'],z[f'f{i}'];V.extend(v);F.extend(f+offset);offset+=len(v)
    tree=BVHTree.FromPolygons(np.asarray(V).tolist(),np.asarray(F).tolist(),all_triangles=True)
    distances=np.full(len(P),np.inf,np.float32);limit=float(job['quality_distance_limit'])
    for i,point in enumerate(P):
        nearest=tree.find_nearest(tuple(point),limit)
        if nearest[0] is not None:distances[i]=nearest[3]
    np.save(job['quality_distance_output'],distances)
    print('Native-surface support:',int(np.isfinite(distances).sum()),'of',len(P),flush=True)
