"""Task-neutral measurements from native rigid poses, never task approval.

The caller supplies source geometry and explicit task tolerances. Solid
intersection, native collider fidelity, bounded forces and source agreement
remain independent required checks.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def rotations(poses):
    poses=np.asarray(poses,float)
    if poses.shape[-1]!=7 or not np.isfinite(poses).all():
        raise ValueError('Finite native position/wxyz poses required')
    q=poses[...,3:]
    if not np.allclose(np.linalg.norm(q,axis=-1),1.,atol=1e-5,rtol=0):
        raise ValueError('Native quaternions must be unit length')
    return Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*poses.shape[:-1],3,3)


def release_stability(parts,poses,times,*,release_s,window_s,tolerance_m):
    poses=np.asarray(poses,float);times=np.asarray(times,float)
    if (poses.shape!=(len(times),len(parts),7) or len(times)<2 or (np.diff(times)<=0).any()
            or not np.isfinite([release_s,window_s,tolerance_m]).all() or min(window_s,tolerance_m)<=0
            or times[-1]-window_s<release_s):
        raise ValueError('A complete unforced final window and positive tolerance required')
    selected=times>=times[-1]-window_s;R=rotations(poses[selected]);rows=[]
    for index,part in enumerate(parts):
        v=np.asarray(part['V'],float)
        # Displacement is a norm of an affine function of a source point; its
        # maximum over the source convex hull occurs at a hull vertex.
        v=v[ConvexHull(v).vertices]
        world=np.einsum('tij,nj->tni',R[:,index],v)+poses[selected,index,None,:3]
        maximum=float(np.linalg.norm(world-world[-1],axis=2).max())
        rows.append({'name':part['name'],'maximum_surface_motion_m':maximum,'pass':maximum<=tolerance_m})
    return {'pass':all(row['pass'] for row in rows),'bodies':rows,'window_s':window_s,
            'tolerance_m':tolerance_m,'scope':'Native final-window geometric stability; caller must also prove zero applied wrench.'}


def insertion(shaft_pose,owner_pose,*,tip,root,radius,opening,minimum_depth,tolerance_m):
    shaft_pose=np.asarray(shaft_pose,float);owner_pose=np.asarray(owner_pose,float)
    tip,root=np.asarray(tip,float),np.asarray(root,float)
    if (shaft_pose.shape!=(7,) or owner_pose.shape!=(7,) or tip.shape!=(3,) or root.shape!=(3,)
            or not np.isfinite(np.r_[tip,root,radius,minimum_depth,tolerance_m]).all()
            or min(radius,minimum_depth)<=0 or tolerance_m<0):
        raise ValueError('Finite shaft endpoints, native poses and explicit positive task depth required')
    Rs,Ro=rotations(shaft_pose),rotations(owner_pose)
    ends=((np.array([tip,root])@Rs.T+shaft_pose[:3]-owner_pose[:3])@Ro)
    lower,upper=ends;axis=upper-lower;length=np.linalg.norm(axis)
    if length<=0:raise ValueError('Nonzero finite shaft required')
    unit=axis/length;dz=axis[2]
    if dz<=1e-12:
        return {'pass':False,'reason':'Shaft does not extend upward through the opening'}
    fraction=(opening['rim_z']-lower[2])/dz
    at_mouth=lower+fraction*axis
    distance=float(np.linalg.norm(at_mouth[:2]-np.asarray(opening['xy'])))
    projected_radius=radius/abs(unit[2])
    depth=float(opening['rim_z']-lower[2]);floor=opening['rim_z']-opening['depth']
    tip_minimum_z=float(lower[2]-radius*np.sqrt(max(0.,1-unit[2]**2)))
    checks={'finite_shaft_crosses_mouth':bool(0<=fraction<=1),
            'opening_envelope':bool(distance+projected_radius<=opening['radius']+tolerance_m),
            'required_insertion_depth':bool(depth>=minimum_depth),
            'tip_above_floor':bool(tip_minimum_z>=floor-tolerance_m)}
    return {'pass':all(checks.values()),'checks':checks,'tip_depth_m':depth,
            'tip_surface_height_above_floor_m':tip_minimum_z-floor,
            'axis_at_mouth_radius_m':distance,'projected_shaft_radius_m':float(projected_radius),
            'minimum_depth_m':minimum_depth,'tolerance_m':tolerance_m,
            'scope':'Conservative straight-shaft/opening envelope; all-body solid contacts and free release remain separate.'}


def cylindrical_insertion_release(shaft_poses,owner_poses,times,*,tip,root,radius,opening,
                                 minimum_depth,release_s,window_s,motion_tolerance_m,
                                 contact_tolerance_m):
    """Retain a source-verified round insertion interface without locking azimuth.

    The caller must establish a smooth circular shaft/round opening with no
    task-relevant orientation constraint. This measures its task state, not whole-object
    rest, material plausibility, actual surface contacts or task approval.
    """
    shaft_poses=np.asarray(shaft_poses,float);owner_poses=np.asarray(owner_poses,float)
    times=np.asarray(times,float)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any()
            or shaft_poses.shape!=(len(times),7) or owner_poses.shape!=shaft_poses.shape
            or not np.isfinite([release_s,window_s,motion_tolerance_m,contact_tolerance_m]).all()
            or min(window_s,motion_tolerance_m)<=0 or contact_tolerance_m<0
            or times[-1]-window_s<release_s):
        raise ValueError('Complete unforced native window and explicit positive motion tolerance required')
    selected=times>=times[-1]-window_s
    if selected.sum()<2:raise ValueError('At least two native states in the unforced window required')
    a,b=shaft_poses[selected],owner_poses[selected]
    Ra,Rb=rotations(a),rotations(b)
    endpoints=np.asarray([tip,root],float)
    if endpoints.shape!=(2,3) or not np.isfinite(endpoints).all() or np.linalg.norm(endpoints[1]-endpoints[0])<=0:
        raise ValueError('Distinct source-derived shaft axis endpoints required')
    world=np.einsum('tij,nj->tni',Ra,endpoints)+a[:,None,:3]
    local=np.einsum('tji,tnj->tni',Rb,world-b[:,None,:3])
    motion=np.linalg.norm(local-local[-1],axis=2).max(0)
    states=[insertion(x,y,tip=tip,root=root,radius=radius,opening=opening,
                      minimum_depth=minimum_depth,tolerance_m=contact_tolerance_m) for x,y in zip(a,b)]
    held=all(row['pass'] for row in states)
    return {'pass':bool(held and (motion<=motion_tolerance_m).all()),
            'interface_held_every_sample':held,'samples':len(a),'window_s':window_s,
            'tip_motion_m':float(motion[0]),'root_motion_m':float(motion[1]),
            'motion_tolerance_m':motion_tolerance_m,'contact_tolerance_m':contact_tolerance_m,
            'minimum_tip_depth_m':min((row['tip_depth_m'] for row in states if 'tip_depth_m' in row),default=None),
            'first_interface_failure':next((row for row in states if not row['pass']),None),
            'allowed_motion':'Axial rotation only; insertion depth and both source axis endpoints remain bounded.',
            'scope':'Explicit round-interface task state only. Full rigid surfaces, contacts, zero wrench, semantic/physical plausibility and other bodies remain separate checks.'}
