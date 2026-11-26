"""Blender BVH audit of architectural coverage, not GS opacity or furniture."""
import json
from pathlib import Path
import sys

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def inside(points, polygon, margin=0.):
    p=np.asarray(points);poly=np.asarray(polygon);edge=np.roll(poly,-1,axis=0)-poly;q=p[...,None,:2]-poly
    return np.all(edge[:,0]*q[...,1]-edge[:,1]*q[...,0]>=margin*np.linalg.norm(edge,axis=1),axis=-1)


def audit(job):
    spec=job['specification'];data=np.load(job['geometry']);V=[];F=[];offset=0;roles=[];entities=[]
    lookup={row['entity']:row['role'] for row in spec['surfaces']}
    for i,part in enumerate(job['parts']):
        v=data[f'v{i}'];f=data[f'f{i}'];V.extend(v);F.extend(f+offset);offset+=len(v)
        roles.extend([lookup[part['entity']]]*len(f));entities.extend([part['entity']]*len(f))
    tree=BVHTree.FromPolygons(np.asarray(V).tolist(),np.asarray(F).tolist(),all_triangles=True)
    poly=np.asarray(spec['footprint']);floor=spec['floor_z'];ceiling=spec['ceiling_z']
    # Test the whole declared floor plan, plus the actual capture corridor.
    lo=poly.min(0);hi=poly.max(0);xy=np.array([[x,y] for x in np.linspace(lo[0],hi[0],11)[1:-1] for y in np.linspace(lo[1],hi[1],11)[1:-1]])
    xy=xy[inside(xy,poly,.05)]
    origins=np.concatenate([np.c_[xy,np.full(len(xy),floor+(ceiling-floor)*z)] for z in [.15,.5,.85]])
    cameras=np.load(job['cameras'])['centre'];good=inside(cameras,poly,.05)&(cameras[:,2]>floor+.05)&(cameras[:,2]<ceiling-.05)
    if not good.all():raise ValueError('Completion footprint/height excludes registered capture cameras')
    origins=np.r_[origins,cameras[::max(1,len(cameras)//40)]]
    k=np.arange(192);z=1-2*(k+.5)/len(k);az=k*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
    directions=np.r_[np.c_[r*np.cos(az),r*np.sin(az),z],[[0,0,1],[0,0,-1]]]
    limit=float(np.linalg.norm(hi-lo)+(ceiling-floor)*2);misses=[];hits={key:0 for key in ['floor','wall','ceiling']}
    # Thin missing strips can evade a finite ray grid. Also demand densely
    # sampled vertical closure with the correct floor/ceiling role and height.
    vertical_failures=[]
    for o in origins:
        for direction in directions:
            hit,normal,index,distance=tree.ray_cast(Vector(o),Vector(direction),limit)
            if hit is None:
                if len(misses)<30:misses.append({'origin':o.tolist(),'direction':direction.tolist()})
            else:hits[roles[index]]+=1
        for direction,role,target in [([0,0,1],'ceiling',ceiling),([0,0,-1],'floor',floor)]:
            # Ceiling beams and suspended fixtures can precede the roof slab.
            # Continue through their surfaces, still requiring a real hit at
            # the declared plane; a lower fixture cannot disguise a roof hole.
            start=Vector(o);ray=Vector(direction);closed=False;hit=None
            for _ in range(128):
                hit,normal,index,distance=tree.ray_cast(start,ray,limit)
                if hit is None:break
                if roles[index]==role and abs(hit.z-target)<=.025:
                    closed=True;break
                if (hit.z-target)*direction[2]>.025:break
                start=hit+ray*.0001
            if not closed:
                if len(vertical_failures)<30:vertical_failures.append({'origin':o.tolist(),'role':role,'hit':list(hit) if hit else None})
    return {'pass':not misses and not vertical_failures,'origins':len(origins),'rays':len(origins)*len(directions),
            'role_hits':hits,'escape_examples':misses,'vertical_failures':vertical_failures,'floor_z':floor,'ceiling_z':ceiling,
            'surfaces':spec['surfaces'],'photorealism_approved':False}


if __name__=='__main__':
    job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
    Path(job['output']).write_text(json.dumps(audit(job),indent=2)+'\n')
