"""Actual-surface penetration across every saved native rigid timestep.

Native manifold separation is retained separately by the caller. This finite
surface sampling is not continuous collision detection or an exhaustive solid
intersection proof; report spacing and temporal sampling with every result.
"""
import numpy as np

from .rigid_initial_state import surface_samples


import warp as wp


@wp.kernel
def _sample_pair(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                 rotations: wp.array(dtype=wp.mat33), translations: wp.array(dtype=wp.vec3),
                 output_rows: wp.array(dtype=int),
                 lower: wp.vec3, upper: wp.vec3, pair: int,
                 maxima: wp.array2d(dtype=float), errors: wp.array(dtype=int)):
    state, index = wp.tid()
    frame=output_rows[state]
    point=rotations[state]*points[index]+translations[state]
    if (point[0]>=lower[0] and point[1]>=lower[1] and point[2]>=lower[2]
            and point[0]<=upper[0] and point[1]<=upper[1] and point[2]<=upper[2]):
        query=wp.mesh_query_point_sign_winding_number(mesh,point,10000.)
        if query.result:
            closest=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
            depth=-query.sign*wp.length(point-closest)
            if depth>0.:
                wp.atomic_max(maxima,frame,pair,depth)
        else:
            wp.atomic_add(errors,0,1)


def swept_region_samples(points,rotations,translations,bounds):
    """Conservative source-frame region of all target boxes in a trace.

    This only removes samples that cannot enter the kernel's target box at any
    recorded pose. The actual signed-surface kernel and thresholds are unchanged.
    A float32 arithmetic guard retains samples at numerical box boundaries.
    """
    from itertools import product
    points=np.asarray(points,float);R=np.asarray(rotations,float);p=np.asarray(translations,float)
    bounds=np.asarray(bounds,float)
    corners=np.array(list(product(*zip(bounds[0],bounds[1]))))
    swept=np.einsum('fji,fcj->fci',R,corners[None]-p[:,None])
    magnitude=1+np.max(abs(points),initial=0.)+np.max(abs(p),initial=0.)+np.max(abs(bounds),initial=0.)
    guard=64*np.finfo(np.float32).eps*magnitude
    lower=swept.min(axis=(0,1))-guard;upper=swept.max(axis=(0,1))+guard
    return points[((points>=lower)&(points<=upper)).all(1)]


def measure(parts, poses, times, *, spacing_m, tolerance_m, device='cpu', progress=None, batch_frames=64,
            spatial_prefilter=True,cache_exact_poses=True):
    from scipy.spatial.transform import Rotation
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface
    poses=np.asarray(poses,float);times=np.asarray(times,float)
    if (poses.shape!=(len(times),len(parts),7) or len(times)<2
            or not np.isfinite(poses).all() or not np.isfinite(times).all()
            or (np.diff(times)<=0).any() or not np.isfinite(tolerance_m) or tolerance_m<0):
        raise ValueError('Every saved rigid pose and increasing native time are required')
    names=[p['name'] for p in parts]
    if len(set(names))!=len(names):raise ValueError('Distinct rigid link names required')
    quaternions=poses[:,:,3:]
    if np.max(abs(np.linalg.norm(quaternions,axis=2)-1))>1e-4:
        raise ValueError('Native rigid orientations must be unit quaternions')
    rotations=Rotation.from_quat(quaternions[:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(times),len(parts),3,3)
    surfaces=[SignedSurface(p['V'],p['F'],device=device) for p in parts]
    if type(batch_frames) is not int or not 1<=batch_frames<=1024:
        raise ValueError('Bounded native-pose batch size required')
    if type(cache_exact_poses) is not bool:raise ValueError('Explicit boolean exact-pose cache policy required')
    sampled=[surface_samples(p['V'],p['F'],spacing_m) for p in parts]
    counts=list(map(len,sampled));pair_ids=[(i,j) for i in range(len(parts)) for j in range(i)]
    if not pair_ids:raise ValueError('At least two independent bodies required')
    queries={};query_counts={}
    for i,j in pair_ids:
        for source,target in [(i,j),(j,i)]:
            inverse=rotations[:,target].transpose(0,2,1)
            relative_R=inverse@rotations[:,source]
            relative_p=np.einsum('nij,nj->ni',inverse,poses[:,source,:3]-poses[:,target,:3])
            selected=(swept_region_samples(sampled[source],relative_R,relative_p,surfaces[target].bounds)
                      if spatial_prefilter else sampled[source])
            queries[source,target]=(wp.array(selected.astype(np.float32),dtype=wp.vec3,device=device),relative_R,relative_p)
            query_counts[f'{names[source]} -> {names[target]}']=len(selected)
    caches=[]
    for i,j in pair_ids:
        if cache_exact_poses:
            # Both directed transformations are keys: neither inverse-roundoff
            # differences nor global-coordinate motion may be guessed away.
            key=np.c_[queries[i,j][1].reshape(len(times),9),queries[i,j][2],
                      queries[j,i][1].reshape(len(times),9),queries[j,i][2]]
            _,first,inverse=np.unique(key,axis=0,return_index=True,return_inverse=True)
        else:
            first=inverse=np.arange(len(times))
        caches.append({'inverse':inverse,'values':np.full(len(first),np.nan),'evaluated':0})
    trace=np.zeros((len(times),len(pair_ids)))
    for start in range(0,len(times),batch_frames):
        end=min(len(times),start+batch_frames);size=end-start
        maxima=wp.zeros((size,len(pair_ids)),dtype=float,device=device)
        errors=wp.zeros(1,dtype=int,device=device)
        active=[]
        for column,(i,j) in enumerate(pair_ids):
            cache=caches[column];ids=cache['inverse'][start:end]
            unknown=np.flatnonzero(np.isnan(cache['values'][ids]))
            missing,first=np.unique(ids[unknown],return_index=True)
            rows=unknown[first];active.append((missing,rows))
            if not len(rows):continue
            output_rows=wp.array(rows.astype(np.int32),dtype=int,device=device)
            for source,target in [(i,j),(j,i)]:
                points,relative_R,relative_p=queries[source,target]
                count=len(points)
                if not count:continue
                R=wp.array(relative_R[start+rows].astype(np.float32),dtype=wp.mat33,device=device)
                P=wp.array(relative_p[start+rows].astype(np.float32),dtype=wp.vec3,device=device)
                surface=surfaces[target]
                wp.launch(_sample_pair,dim=(len(rows),count),inputs=[surface.mesh.id,points,R,P,output_rows,
                    wp.vec3(*surface.bounds[0]),wp.vec3(*surface.bounds[1]),column,maxima,errors],device=device)
        measured=maxima.numpy()
        if errors.numpy()[0]:raise ValueError('Actual surface query failed at a saved native pose')
        for column,cache in enumerate(caches):
            missing,rows=active[column]
            cache['values'][missing]=measured[rows,column];cache['evaluated']+=len(missing)
            trace[start:end,column]=cache['values'][cache['inverse'][start:end]]
        if progress:progress(end,len(times),float(trace[:end].max()))
    pairs=[]
    for column,(i,j) in enumerate(pair_ids):
        frame=int(trace[:,column].argmax());maximum=float(trace[frame,column])
        pairs.append({'parts':[names[i],names[j]],'maximum_sampled_penetration_m':maximum,
                      'frame':frame,'time_s':float(times[frame]),'pass':maximum<=tolerance_m})
    return {'pass':all(r['pass'] for r in pairs),'pairs':pairs,
            'frames':len(times),'maximum_step_s':float(np.diff(times).max()),
            'spacing_m':spacing_m,'tolerance_m':tolerance_m,'sample_counts':dict(zip(names,counts)),
            'swept_region_query_counts':query_counts,'spatial_prefilter':spatial_prefilter,
            'exact_pose_cache':{'enabled':cache_exact_poses,'all_recorded_frames_accounted_for':True,
                'pairs':[{'parts':[names[i],names[j]],'evaluated_states':c['evaluated'],
                          'reused_states':len(times)-c['evaluated']} for (i,j),c in zip(pair_ids,caches)],
                'basis':'Reuse only exactly equal transforms in both directions within one fixed-source geometry check; no pose rounding, interpolation or native step removal.'},
            'maximum_sampled_penetration_m':float(trace.max()),'per_frame_maximum_m':trace.max(1).tolist(),
            'scope':'Actual closed source surfaces in every saved native pose; finite spatial/temporal sampling, not an exhaustive intersection proof'}
