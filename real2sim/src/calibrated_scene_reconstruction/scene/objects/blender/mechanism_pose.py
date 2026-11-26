"""Bake a proposed leaf-hinge initial pose without changing its rigid shape.

The caller supplies a source-supported angle interval. This does not infer an
allowed closure gap, deform an interposed sheet, or grant physical acceptance.
"""
from copy import deepcopy
import json
import numpy as np
from scipy.spatial.transform import Rotation
from calibrated_scene_reconstruction.assets.mechanisms import joint_frames,quaternion_matrix


def rotate_physics(settings,rotation):
    """Carry geometry-local collision directions through a baked rigid pose."""
    r=np.asarray(rotation,float)
    if (r.shape!=(3,3) or not np.isfinite(r).all()
            or not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1.,atol=1e-8)):
        raise ValueError('Proper rigid rotation required for collision metadata')
    result=deepcopy(settings)
    if 'revolution_axis' in result:
        axis=np.asarray(result['revolution_axis'],float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<=1e-12:
            raise ValueError('Finite nonzero source revolution axis required')
        result['revolution_axis']=(r@axis).tolist()
    return result


def hinge_candidate(mesh,info,index,delta_rad):
    """Propose a leaf pose and its collision metadata together, without approval."""
    joint=json.loads(str(mesh['joints_json']))[index]
    posed=hinge_proposal(mesh,index,delta_rad);candidate=deepcopy(info)
    child=joint['child'];settings=candidate.get('link_physics',{}).get(child)
    if settings is not None:
        rotation=Rotation.from_rotvec(np.asarray(joint['axis'])*delta_rad).as_matrix()
        candidate['link_physics'][child]=rotate_physics(settings,rotation)
    candidate.update(proposal_only=True,requires_agent_rebuild=True,task_approved=False,
                     native_approved=False,source_pose_approved=False)
    candidate.pop('blender_build_evidence_sha256',None)
    return posed,candidate


def fixed_leaf_candidate(mesh,info,index,rotation,pivot):
    """Propose a source-reviewed relative placement of an integrated rigid leaf.

    All meshes use the assembly frame. Rebase both sides of the fixed
    constraint together so native reset retains the new geometry placement;
    rotating only the child's constraint would undo it. No shape or material
    changes, and every source/contact/native check must run again.
    """
    r,p=np.asarray(rotation,float),np.asarray(pivot,float)
    rotate_physics({},r)
    if p.shape!=(3,)or not np.isfinite(p).all():raise ValueError('Finite rigid-placement pivot required')
    result={k:deepcopy(v)for k,v in mesh.items()};joints=json.loads(str(result['joints_json']))
    if type(index)is not int or not 0<=index<len(joints):raise ValueError('Valid fixed attachment index required')
    joint=joints[index]
    if joint['type']!='fixed'or any(j['parent']==joint['child']for j in joints):
        raise ValueError('Only a leaf fixed rigid part can be placed independently')
    part=next(x for x in json.loads(str(result['links_json']))if x['name']==joint['child'])
    if part['kind']!='rigid':raise ValueError('Fixed leaf must be rigid')
    a,n=part['vertex_start'],part['vertex_count'];result['V']=np.asarray(result['V'],float)
    result['V'][a:a+n]=(result['V'][a:a+n]-p)@r.T+p
    if 'normals'in result:
        b,k=part['face_start'],part['triangle_count'];result['normals'][b:b+k]=result['normals'][b:b+k]@r.T
    for key in ['anchor0','anchor1']:joint[key]=((np.asarray(joint[key])-p)@r.T+p).tolist()
    for i,frame in enumerate(joint_frames(joint)):
        q=Rotation.from_matrix(r@quaternion_matrix(frame)).as_quat()
        joint['frame'+str(i)]=q[[3,0,1,2]].tolist()
    result['joints_json']=np.array(json.dumps(joints));candidate=deepcopy(info)
    child=joint['child'];settings=candidate.get('link_physics',{}).get(child)
    if settings is not None:candidate['link_physics'][child]=rotate_physics(settings,r)
    candidate.update(proposal_only=True,requires_agent_rebuild=True,task_approved=False,native_approved=False,source_pose_approved=False)
    candidate.pop('blender_build_evidence_sha256',None)
    return result,candidate


def hinge_proposal(mesh,index,delta_rad):
    if not np.isfinite(delta_rad):raise ValueError('Finite hinge proposal angle required')
    result={k:deepcopy(v) for k,v in mesh.items()}
    joints=json.loads(str(result['joints_json']));parts=json.loads(str(result['links_json']))
    if type(index) is not int or not 0<=index<len(joints):raise ValueError('Valid hinge index required')
    joint=joints[index]
    if joint['type']!='revolute' or any(j['parent']==joint['child'] for j in joints):
        raise ValueError('Only a leaf revolute link can be posed independently')
    part=next(p for p in parts if p['name']==joint['child'])
    if part['kind']!='rigid':raise ValueError('Hinge child must be rigid')
    frames=joint_frames(joint);f0,f1=map(quaternion_matrix,frames)
    joint['frame0']=np.asarray(frames[0]).tolist()
    initial=float(np.arctan2((f0.T@f1)[2,1]-(f0.T@f1)[1,2],(f0.T@f1)[1,1]+(f0.T@f1)[2,2]))
    if 'initial_position' in joint:
        declared=float(joint['initial_position'])
        if abs(np.arctan2(np.sin(initial-declared),np.cos(initial-declared)))>1e-6:
            raise ValueError('Captured joint frames disagree with the declared initial position')
        initial=declared
    if not joint['limits'][0]<=initial+delta_rad<=joint['limits'][1]:raise ValueError('Initial pose exceeds joint limits')
    rotation=Rotation.from_rotvec(np.asarray(joint['axis'])*delta_rad).as_matrix()
    anchor=np.asarray(joint['anchor0']);a,n=part['vertex_start'],part['vertex_count']
    result['V'][a:a+n]=(result['V'][a:a+n]-anchor)@rotation.T+anchor
    if 'normals' in result:
        first,count=part['face_start'],part['triangle_count']
        normals=result['normals']
        if normals.shape!=(len(result['F']),3,3):raise ValueError('Face-corner source normals required')
        result['normals'][first:first+count]=normals[first:first+count]@rotation.T
    joint['anchor1']=((np.asarray(joint['anchor1'])-anchor)@rotation.T+anchor).tolist()
    xyzw=Rotation.from_matrix(rotation@f1).as_quat();joint['frame1']=xyzw[[3,0,1,2]].tolist()
    joint['initial_position']=initial+float(delta_rad)
    result['joints_json']=np.array(json.dumps(joints))
    return result


def hinge_axis_proposal(mesh,index,offset,*,reference_position=0.):
    """Propose a leaf pivot while preserving its reference-pose rigid surfaces.

    This is a geometry hypothesis, never an inferred measurement or permission
    to pass source/path/native checks. No source mesh vertices are reshaped.
    """
    offset=np.asarray(offset,float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():
        raise ValueError('Finite three-dimensional hinge-axis offset required')
    joints=json.loads(str(mesh['joints_json']))
    if type(index) is not int or not 0<=index<len(joints):raise ValueError('Valid hinge index required')
    joint=joints[index]
    f0,f1=map(quaternion_matrix,joint_frames(joint));relative=f0.T@f1
    initial=float(joint.get('initial_position',np.arctan2(relative[2,1]-relative[1,2],relative[1,1]+relative[2,2])))
    result=hinge_proposal(mesh,index,reference_position-initial)
    proposed=json.loads(str(result['joints_json']))
    for key in ['anchor0','anchor1']:proposed[index][key]=(np.asarray(proposed[index][key])+offset).tolist()
    result['joints_json']=np.array(json.dumps(proposed))
    return hinge_proposal(result,index,initial-reference_position)
