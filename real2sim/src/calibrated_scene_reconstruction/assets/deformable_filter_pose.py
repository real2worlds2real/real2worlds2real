"""Reference geometry for collision filtering, separate from physical state."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def material_pose(reference,cells,*,separation_m):
    """Keep material neighbours and separate disconnected material islands.

    A loaded/assembled pose can put independent folds into contact. It must
    not silently define persistent self-collision exclusions. These points
    affect filtering only: no rest metric, bending reference, mass or current
    simulation coordinates are changed.
    """
    points=np.asarray(reference,float);indices=np.asarray(cells)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not len(points) or not np.isfinite(points).all()
            or indices.ndim!=2 or indices.shape[1] not in (3,4) or not len(indices)
            or indices.dtype.kind not in 'iu' or indices.min()<0 or indices.max()>=len(points)
            or not np.isfinite(separation_m) or separation_m<=0
            or len(np.unique(indices))!=len(points)):
        raise ValueError('Complete finite material reference, indexed faces/cells and positive island separation required')
    first=np.repeat(indices[:,0],indices.shape[1]-1);other=indices[:,1:].ravel()
    graph=coo_matrix((np.ones(len(first)),(first,other)),shape=(len(points),len(points)))
    count,labels=connected_components(graph,directed=False)
    result=points.copy()
    if count>1:
        width=int(np.ceil(count**(1/3)));stride=float(np.ptp(points,axis=0).max()+separation_m)
        for i in range(count):
            selected=labels==i;local=points[selected]
            centre=(local.min(0)+local.max(0))/2
            grid=np.array([i%width,(i//width)%width,i//(width*width)],float)
            result[selected]=local-centre+stride*grid
    return result,{'material_components':int(count),'points':len(points),
        'collision_filter_purpose':'selfCollisionFilterPose','current_state_changed':False,
        'material_reference_changed':False,'task_approved':False,
        'scope':'Immutable material-reference filtering pose, with disconnected islands separated. Nonlocal contacts in the loaded geometry remain enabled; native controls and reload validation are still required.'}
