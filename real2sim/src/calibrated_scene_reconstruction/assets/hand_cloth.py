"""Measurements for a real two-finger cloth pinch, independent of cloth forces."""
import numpy as np


def material_tangent(points, rest, patch, axis):
    """Map a declared sheet-chart direction onto the actual local cloth patch."""
    points,rest=np.asarray(points,float),np.asarray(rest,float)
    patch,axis=np.asarray(patch),np.asarray(axis,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or rest.shape!=points.shape
            or patch.ndim!=1 or patch.dtype.kind not in 'iu' or len(patch)<3
            or patch.min()<0 or patch.max()>=len(points) or axis.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [points,rest,axis]) or np.linalg.norm(axis)<1e-8):
        raise ValueError('Actual sheet, material chart, interior patch and resolved chart axis required')
    a=rest[patch]-rest[patch].mean(0);b=points[patch]-points[patch].mean(0)
    _,values,basis=np.linalg.svd(a,full_matrices=False)
    if values[1]<1e-8 or values[2]>values[1]*1e-5:
        raise ValueError('A resolved planar local material chart is required')
    projected=(axis@basis[:2].T)@basis[:2]
    if np.linalg.norm(axis-projected)>1e-5*np.linalg.norm(axis):
        raise ValueError('Declared material axis leaves the sheet chart')
    gradient=np.linalg.lstsq(a,b,rcond=1e-6)[0]
    tangent=axis@gradient;length=np.linalg.norm(tangent)
    if length<1e-8:raise ValueError('Actual material direction collapsed')
    return tangent/length


def pinch_axis_alignment(pads, direction, *, up=(0., 0., 1.), minimum=.94):
    """Check actual pad opposition against a source-declared projected axis.

    A fixture frame or an inside/outside assignment alone does not establish
    the direction of an asymmetric hand's fingers. Reject unresolved axes;
    this geometric prerequisite grants no friction or task approval.
    """
    pads=np.asarray(pads,float);direction=np.asarray(direction,float);spatial=up is None
    up=np.array([0.,0.,1.]) if spatial else np.asarray(up,float)
    if (pads.shape!=(2,3) or direction.shape!=(3,) or up.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [pads,direction,up])
            or not np.isfinite(minimum) or not 0<minimum<=1 or np.linalg.norm(up)<1e-12):
        raise ValueError('Finite pad pair, source direction, up and cosine in (0,1] required')
    up=up/np.linalg.norm(up)
    axis=pads[1]-pads[0]
    target=direction.copy()
    if not spatial:
        axis=axis-(axis@up)*up
        target=direction-(direction@up)*up
    lengths=[np.linalg.norm(axis),np.linalg.norm(target)]
    resolved=bool(min(lengths)>1e-8)
    cosine=float(abs(axis@target)/np.prod(lengths)) if resolved else 0.
    return {'pass':bool(resolved and cosine>=minimum),'resolved':resolved,
            'absolute_cosine':min(cosine,1.),'minimum_cosine':float(minimum),
            'axis_projection':'spatial' if spatial else 'opening_plane',
            'pad_axis_projected':axis.tolist(),'source_axis_projected':target.tolist(),
            'scope':'Actual pad geometry prerequisite; not a friction proof'}


def clearance_offsets(thickness, *, maximum=.01, budget=129):
    """Bounded placement proposals at the scale of the physical sheet.

    A fixed2mm grid can skip an entire accessible thin-sheet pinch corridor.
    These offsets only propose wrist positions; source-solid, full-path and
    native contact checks retain their own unchanged tolerances.
    """
    if (not np.isfinite([thickness, maximum]).all() or min(thickness, maximum)<=0
            or type(budget) is not int or not 2<=budget<=513):
        raise ValueError('Positive physical thickness/range and bounded proposal budget required')
    count=min(budget, max(2, int(np.ceil(maximum/(thickness/2)))+1))
    values=np.linspace(0.,maximum,count)
    return values, {'spacing_m':float(values[1]-values[0]), 'physical_thickness_m':float(thickness),
                    'maximum_m':float(maximum),'count':count,
                    'budget_limited':bool(values[1]-values[0]>thickness/2+1e-12),
                    'scope':'Placement proposals only; no contact or acceptance tolerance changes.'}


def opening_straddle(pads,opening,outer_radius,*,margin=.0005):
    """Measure the two real contact pads in the container's local frame."""
    pads=np.asarray(pads,float);centre=np.asarray(opening['xy'],float)
    inner=float(opening['radius'])
    if (pads.shape!=(2,3) or centre.shape!=(2,) or not np.isfinite(pads).all()
            or not np.isfinite(centre).all() or not np.isfinite([inner,outer_radius,margin]).all()
            or not 0<margin<inner<=outer_radius):
        raise ValueError('Two finite contact pads and ordered physical opening radii required')
    radii=np.linalg.norm(pads[:,:2]-centre,axis=1)
    return {'pass':bool(radii.min()<inner-margin and radii.max()>outer_radius+margin),
            'open_finger_pad_radii_m':radii.tolist(),'inner_radius_m':inner,'outer_radius_m':float(outer_radius)}


def same_hand_plan(first,second):
    """Match the free-hand experiment, independently of the object task.

    The preflight uses its own fixed 4cm wrist motion, not the object's lift,
    approach or material-distribution requirements. Hand geometry, poses,
    actuators, friction and unknown fields stay bound. The object assay hashes
    the complete task plan and its executed motion separately.
    """
    task_only={'native_fixture_check','source','not_task_approved',
               'approach_height_m','lift_height_m','lift_speed_m_s','hold_s',
               'initial_material_distribution'}
    return ({k:v for k,v in first.items() if k not in task_only}
            =={k:v for k,v in second.items() if k not in task_only})


def opening_assignment(pads,opening,outer_radius,inside_finger='either'):
    """Use actual open pads to select the same inside finger before descent."""
    if inside_finger not in {'either','thumb','index'}:
        raise ValueError('Inside finger must be either, thumb or index')
    result=opening_straddle(pads,opening,outer_radius)
    radii=result['open_finger_pad_radii_m']
    result['inside_finger']='thumb' if radii[0]<radii[1] else 'index'
    result['assignment_matches']=inside_finger in ['either',result['inside_finger']]
    result['pass'] &= result['assignment_matches']
    return result


def rim_target(points, rest, faces, opening, *, outer_radius, pad_width=.008, feasible=None, max_attempts=32, roll_angles=(0.,),align_surface=False,minimum_normal_alignment=0.,pinch_axis=None,material_axis=None,material_axis_frame='spatial'):
    """Choose accessible interior material at a rim, not an arbitrary free tip.

    This is an explicit geometry-based fixture hypothesis, not a detected human
    grasp or proof that this hand can reach the chosen target.
    """
    from scipy.spatial import cKDTree
    if material_axis_frame not in {'spatial','opening_plane'}:raise ValueError('Explicit spatial or opening-plane material direction required')
    if type(max_attempts) is not int or not 1<=max_attempts<=256:raise ValueError('Bounded grasp proposal search required')
    rolls=np.asarray(roll_angles,float)
    if (rolls.ndim!=1 or not 1<=len(rolls)<=9 or not np.isfinite(rolls).all()
            or (abs(rolls)>np.pi/3+1e-10).any()):
        raise ValueError('One to nine finite pinch-axis rolls within60degrees required')
    points,rest,faces=np.asarray(points,float),np.asarray(rest,float),np.asarray(faces,int)
    if pinch_axis is not None and material_axis is not None:raise ValueError('Choose a body axis or a material-chart axis')
    if type(align_surface) is not bool:raise ValueError('Explicit surface-alignment policy required')
    if (not np.isfinite(minimum_normal_alignment) or not 0<=minimum_normal_alignment<=1
            or (minimum_normal_alignment>0 and not align_surface)):
        raise ValueError('Normal alignment requires a surface-aware proposal and a cosine within0..1')
    normals=np.zeros_like(points)
    if align_surface:
        triangles=points[faces]
        face_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        for column in range(3):np.add.at(normals,faces[:,column],face_normals)
    centre=np.asarray(opening['xy']);rim=float(opening['rim_z']);radius=float(opening['radius'])
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[counts==1])
    material_clearance=cKDTree(rest[boundary]).query(rest)[0]
    radial=points[:,:2]-centre;rho=np.linalg.norm(radial,axis=1)
    selected=(rho>radius-.002)&(rho<outer_radius+.002)&(points[:,2]>=rim-.001)&(points[:,2]<rim+3*pad_width)&(material_clearance>pad_width)
    ids=np.flatnonzero(selected)
    if len(ids)<3:raise ValueError('No resolved interior cloth patch crossing the container rim; revise physical packing')
    score=material_clearance[ids]+.5*(points[ids,2]-rim)-.3*abs(rho[ids]-(radius+outer_radius)/2)
    tried=[]
    for index in ids[np.argsort(-score)]:
        index=int(index);target=points[index].copy()
        # Never project a real cloth point onto an invented rim-height target.
        # A fixture may reject a point using the actual hand/solid geometry.
        if any(np.linalg.norm(target-old)<pad_width/2 for old in tried):continue
        if len(tried)>=max_attempts:break
        tried.append(target)
        patch=np.flatnonzero(np.linalg.norm(rest-rest[index],axis=1)<pad_width)
        if len(patch)<3:continue
        normal=radial[index]/rho[index]
        direction=None
        if material_axis is not None:
            direction=material_tangent(points,rest,patch,material_axis)
            if np.linalg.norm(direction[:2])<1e-8:continue
            normal=direction[:2]/np.linalg.norm(direction[:2])
        if pinch_axis is not None:
            direction=np.asarray(pinch_axis,float)
            if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction[:2])<1e-8:
                raise ValueError('Source pinch axis must have a finite horizontal projection')
            normal=direction[:2]/np.linalg.norm(direction[:2])
        angle=float(np.arctan2(normal[1],normal[0]));c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        pitches=[0.]
        if material_axis is not None:
            preferred=(float(np.arctan2(direction[2],np.linalg.norm(direction[:2])))
                if material_axis_frame=='spatial' else 0.)
            if abs(preferred)>np.pi/3:continue
            pitches=[preferred]
        elif align_surface and np.linalg.norm(normals[index])>1e-12:
            local=rotation.T@normals[index]
            if local[0]<0:local=-local
            preferred=float(np.clip(np.arctan2(local[2],local[0]),-np.pi/3,np.pi/3))
            # Partial alignment can preserve aperture access when the full
            # surface-normal pitch would place both pads outside the mouth.
            pitches=list(dict.fromkeys([preferred,.75*preferred,.5*preferred,0.]))
        # A real hand is asymmetric. Both thumb-inside/index-outside and the
        # reversed assignment satisfy a radial pinch, but have different paths.
        for reverse in (False,True):
            side=rotation@np.diag([-1.,-1.,1.]) if reverse else rotation
            for pitch in pitches:
                angle=-pitch if reverse else pitch;c,s=np.cos(angle),np.sin(angle)
                pitched=side@np.array([[c,0,-s],[0,1.,0],[s,0,c]])
                for roll in rolls:
                    c,s=np.cos(roll),np.sin(roll)
                    orientation=pitched@np.array([[1.,0,0],[0,c,-s],[0,s,c]])
                    alignment=(float(abs(normals[index]@orientation[:,0])/np.linalg.norm(normals[index]))
                               if np.linalg.norm(normals[index])>1e-12 else 0.)
                    if alignment<minimum_normal_alignment:continue
                    if feasible is not None and not feasible(target,orientation):continue
                    return {'point':target,'rotation':orientation,'pinch_axis_roll_rad':float(roll),
                            'surface_normal_alignment':alignment,'minimum_normal_alignment':float(minimum_normal_alignment),
                            'pinch_axis_pitch_rad':float(angle),'pinch_axis_reversed':reverse,
                            'material_node':index,'material_patch':patch,
                            'source_material_axis':None if material_axis is None else np.asarray(material_axis).tolist(),
                            'material_axis_frame':material_axis_frame,
                            'material_tangent_body':None if material_axis is None else direction.tolist(),
                            'on_actual_material':True,'height_over_rim_m':float(target[2]-rim),
                            'minimum_selected_material_boundary_distance_m':float(material_clearance[patch].min()),
                            'basis':'Actual rim-crossing interior material, bounded surface-normal pitch/roll and both finger assignments. Full-hand aperture/approach checks and native validation required.'}
    raise ValueError('No resolved collision-free grasp region on the actual rim cloth')


def pinch_contact(points,patch,pads,near_surface_samples,*,maximum_patch_distance_m=.012):
    """Require the intended material at both actual distal fingers before lift.

    These geometric witnesses are necessary, not proof of contact force or
    friction. The matched zero-friction lift remains required.
    """
    points=np.asarray(points,float);patch=np.asarray(patch);pads=np.asarray(pads,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or patch.ndim!=1 or patch.dtype.kind not in 'iu' or len(patch)<3
            or patch.min()<0 or patch.max()>=len(points) or pads.shape!=(2,3)
            or not np.isfinite(pads).all() or not np.isfinite(maximum_patch_distance_m)
            or maximum_patch_distance_m<=0):
        raise ValueError('Actual finite cloth, interior patch and two native finger pads required')
    counts={finger:sum(n for link,n in near_surface_samples.items()
            if link.startswith('right_'+finger) and ('_DP' in link or '_elastomer' in link))
            for finger in ['thumb','index']}
    distances=np.linalg.norm(points[patch,None]-pads[None],axis=2).min(0)
    return {'pass':bool(min(counts.values())>0 and distances.max()<maximum_patch_distance_m),
            'distal_near_surface_samples':counts,'patch_to_pad_distances_m':distances.tolist(),
            'maximum_patch_distance_m':maximum_patch_distance_m,
            'scope':'Pre-lift geometric contact witnesses only; no friction or task approval.'}


def capture_target(points,rest,faces,centre,*,patch_radius,search_radius,feasible,budget=256):
    """Associate current open-jaw geometry with interior material before closing.

    A spatial pinch need not capture the node selected before a sliding drape.
    Source identity and intrinsic boundary clearance remain explicit; this
    function never changes material coordinates or native positions.
    """
    from scipy.spatial import cKDTree
    points,rest,faces,centre=np.asarray(points,float),np.asarray(rest,float),np.asarray(faces),np.asarray(centre,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or rest.shape!=points.shape or centre.shape!=(3,)
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu' or not len(faces)
            or faces.min()<0 or faces.max()>=len(points) or not all(np.isfinite(x).all()for x in [points,rest,centre])
            or not np.isfinite([patch_radius,search_radius]).all() or not 0<patch_radius<=search_radius
            or type(budget)is not int or not 1<=budget<=1024):
        raise ValueError('Finite full sheet, intrinsic chart and bounded actual-capture search required')
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[counts==1])
    if not len(boundary):raise ValueError('Resolved sheet material boundary required')
    clearance=cKDTree(rest[boundary]).query(rest)[0];distance=np.linalg.norm(points-centre,axis=1)
    ids=np.flatnonzero((clearance>patch_radius)&(distance<=search_radius));tried=[]
    for index in ids[np.argsort(distance[ids])]:
        point=points[index]
        if any(np.linalg.norm(point-old)<patch_radius/4 for old in tried):continue
        if len(tried)>=budget:break
        tried.append(point.copy());patch=np.flatnonzero(np.linalg.norm(rest-rest[index],axis=1)<patch_radius)
        if len(patch)<3 or not feasible(int(index),patch):continue
        return {'material_node':int(index),'material_patch':patch,'point':point.copy(),
            'on_actual_material':True,'minimum_selected_material_boundary_distance_m':float(clearance[patch].min()),
            'basis':'Current source-indexed interior material in the actual open-jaw corridor; no material motion or task approval.'}
    raise ValueError('No interior material is currently captured by the actual open jaws')


def lift_measurement(before, after, patch, wrist_before, wrist_after, *, minimum_lift_m=.05):
    before,after=np.asarray(before,float),np.asarray(after,float);patch=np.asarray(patch,int)
    if before.shape!=after.shape or before.ndim!=2 or before.shape[1]!=3 or not np.isfinite([before,after]).all():
        raise ValueError('Matching actual native cloth states required')
    delta=after[patch]-before[patch];lift=float(np.median(delta[:,2]))
    wrist=float(np.asarray(wrist_after)[2]-np.asarray(wrist_before)[2])
    return {'pass':bool(lift>=minimum_lift_m and wrist>=minimum_lift_m and abs(lift-wrist)<.02),
            'material_patch_lift_m':lift,'wrist_lift_m':wrist,'vertical_slip_m':abs(lift-wrist),
            'minimum_lift_m':minimum_lift_m}


def friction_control(positive, negative, *, maximum_initial_difference_m=.001):
    """A successful zero-friction lift invalidates the friction claim."""
    a,b=np.asarray(positive['prelift_points']),np.asarray(negative['prelift_points'])
    if a.shape!=b.shape or not np.isfinite([a,b]).all():raise ValueError('Matching native pre-lift states required')
    mismatch=float(np.max(np.linalg.norm(a-b,axis=1)))
    same_motion=positive['motion_sha256']==negative['motion_sha256']
    positive_ok=positive.get('lift',{}).get('pass') is True
    negative_failed=negative.get('lift',{}).get('pass') is False
    update=negative.get('native_zero_friction',{}) or {}
    zero=(negative.get('friction_zero_readback') is True
          and update.get('method')=='existing_usd_material'
          and update.get('combine_mode')=='multiply'
          and update.get('native_readback_after_step') is True)
    return {'pass':bool(positive_ok and negative_failed and zero and same_motion and mismatch<=maximum_initial_difference_m),
            'positive_lift_pass':positive_ok,'zero_friction_lift_pass':negative.get('lift',{}).get('pass'),
            'same_motion':same_motion,'maximum_prelift_state_difference_m':mismatch,
            'zero_friction_readback':zero,
            'scope':'Causal friction screen; both trials still require independent material/contact validity.'}
