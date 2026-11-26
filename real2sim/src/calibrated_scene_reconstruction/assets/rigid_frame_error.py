"""Conservative source-surface discrepancy of equivalent native rigid frames."""
import numpy as np


def attachment_radii(stage, carriers):
    """Bound actual points carried by collider-free numerical frames.

    Such frames do not carry the owner's entire rigid surface. Its geometry
    is checked separately at the actual source body's pose.
    """
    points = {str(path): [] for path in carriers}
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'OmniPhysicsVtxXformAttachment':
            relation, attribute = 'omniphysics:src1', 'omniphysics:localPositionsSrc1'
        elif prim.GetTypeName() == 'PhysxPhysicsAttachment':
            relation, attribute = 'actor1', 'points1'
        else:
            continue
        targets = prim.GetRelationship(relation).GetTargets()
        if len(targets) != 1 or str(targets[0]) not in points:
            continue
        value = np.asarray(prim.GetAttribute(attribute).Get(), float)
        if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
            raise ValueError('Actual finite attachment-local points required for a numerical frame')
        points[str(targets[0])].append(value)
    if any(not rows for rows in points.values()):
        raise ValueError('Numerical frame has no verified carried attachment points')
    return np.array([np.linalg.norm(np.concatenate(points[str(path)]), axis=1).max() for path in carriers])


def discrepancy(parent_position,parent_wxyz,frame_position,frame_wxyz,centres,radii):
    """Bound every source point by translation error plus rotational motion.

    Frame origins are at centres in the parent source coordinates. Both poses
    must be actual native readbacks; this function never corrects either pose.
    """
    p=np.asarray(parent_position,float);q=np.asarray(parent_wxyz,float)
    fp=np.asarray(frame_position,float);fq=np.asarray(frame_wxyz,float)
    c=np.asarray(centres,float);r=np.asarray(radii,float)
    if (p.shape!=(3,) or q.shape!=(4,) or fp.ndim!=2 or fp.shape[1:]!=(3,)
            or fq.shape!=(len(fp),4) or c.shape!=fp.shape or r.shape!=(len(fp),)
            or not all(np.isfinite(x).all() for x in [p,q,fp,fq,c,r])
            or (r<0).any() or not np.isclose(q@q,1.,atol=1e-4)
            or not np.allclose(np.linalg.norm(fq,axis=1),1.,atol=1e-4)):
        raise ValueError('Finite unit native poses and source-frame bounds required')
    q=q/np.linalg.norm(q);fq=fq/np.linalg.norm(fq,axis=1)[:,None]
    w,x,y,z=q
    R=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    translation=np.linalg.norm(fp-(c@R.T+p),axis=1)
    # 2*sin(theta/2) <= 2*min(||q1-q2||,||q1+q2||).
    rotation=2*np.minimum(np.linalg.norm(fq-q,axis=1),np.linalg.norm(fq+q,axis=1))*r
    return translation+rotation
