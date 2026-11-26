"""Explicit particle spring-network proposals, separate from continuum FEM."""
import numpy as np

from .cloth_rest import triangle_edges,validate


def spring_proposal(youngs_modulus,thickness,bend_stiffness):
    """Bulk small-strain target for a balanced square central-force lattice.

    Its Poisson ratio is1/3. Distance bending is nonlinear and does not reproduce
    a specified continuum flexural rigidity. Native response tests are required.
    """
    if not np.isfinite([youngs_modulus,thickness,bend_stiffness]).all() or min(youngs_modulus,thickness,bend_stiffness)<=0:
        raise ValueError('Positive finite membrane and distance-bend proposal required')
    et=youngs_modulus*thickness;stretch=.75*et-4*bend_stiffness
    if stretch<=0:raise ValueError('Distance-bend stiffness prevents a positive membrane lattice')
    return {'settings':{'stretch_stiffness':float(stretch),'shear_stiffness':float(.375*et),
                        'bend_stiffness':float(bend_stiffness)},
            'target_membrane_Et_N_m':float(et),'linear_lattice_poissons_ratio':1/3,
            'continuum_bending_matched':False,'native_response_qualified':False,
            'material_identified':False,'task_approved':False}


def calibrated_membrane_proposal(proposal,scale):
    """A numerical spring hypothesis against the unchanged continuum target.

    Native iterations and timestep can make a nominal lattice much softer than
    its analytic limit. Scaling is a candidate for an independent traction test,
    not a material fit or a way to change the expected test response.
    """
    from copy import deepcopy
    if not np.isfinite(scale) or not 0<scale<=1000:
        raise ValueError('Finite bounded native membrane calibration factor required')
    result=deepcopy(proposal);original=dict(result['settings'])
    for key in ['stretch_stiffness','shear_stiffness']:
        result['settings'][key]*=float(scale)
    result['uncalibrated_settings']=original
    result['native_membrane_scale']=float(scale)
    result['native_response_qualified']=False
    result['task_approved']=False
    return result


def topology(points,faces,*,model='balanced_square_lattice'):
    """Preserve a clipped regular chart; balance both diagonals in full cells.

    Render diagonal choice must not change the interior membrane law. No spring
    fills an absent cell. Axis bending follows two existing material edges.
    """
    if model not in {'balanced_square_lattice','triangle_surface'}:
        raise ValueError('Explicit supported material lattice required')
    rest=np.asarray(points,float);faces=np.asarray(faces)
    if (rest.ndim!=2 or rest.shape[1:]!=(3,) or not np.isfinite(rest).all() or np.ptp(rest[:,2])>1e-9):
        raise ValueError('Planar finite material coordinates required')
    if (faces.ndim!=2 or faces.shape[1:]!=(3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min()<0 or faces.max()>=len(rest) or len(np.unique(np.sort(faces,axis=1),axis=0))!=len(faces)):
        raise ValueError('Distinct valid material triangles required')
    lattice=[];steps=[]
    for axis in [0,1]:
        values=np.unique(np.round(rest[:,axis],12));step=float(np.median(np.diff(values)))
        if not np.isfinite(step) or step<=0:raise ValueError('Resolved regular material chart required')
        indices=np.rint((rest[:,axis]-values[0])/step).astype(int)
        if abs(rest[:,axis]-values[0]-indices*step).max()>max(1e-10,step*1e-5):
            raise ValueError('Particle lattice requires an explicitly regular chart')
        lattice.append(indices);steps.append(step)
    xy=np.stack(lattice,axis=1);nx,ny=xy.max(0)+1
    if nx*ny>8*len(rest):raise ValueError('Sparse chart needs a different explicit spring topology')
    grid=np.full((ny,nx),-1,int)
    if len(np.unique(xy,axis=0))!=len(rest):raise ValueError('Duplicate material nodes are not a regular lattice')
    grid[xy[:,1],xy[:,0]]=np.arange(len(rest))
    edges=triangle_edges(faces);delta=abs(xy[edges[:,1]]-xy[edges[:,0]])
    axis_edge=(delta.sum(1)==1);diagonal=(delta==1).all(1)
    if not (axis_edge|diagonal).all():raise ValueError('Triangle edges exceed one regular material cell')
    stretch=edges[axis_edge];shear=[edges[diagonal]]
    tri=xy[faces];minimum=tri.min(1);span=np.ptp(tri,axis=1)
    if not (span==1).all():raise ValueError('Triangles must occupy a regular material half-cell')
    cell=minimum[:,1]*(nx-1)+minimum[:,0];counts=np.bincount(cell,minlength=(nx-1)*(ny-1)).reshape(ny-1,nx-1)
    if (counts>2).any():raise ValueError('Overlapping material cells are not a lattice')
    for axis in [0,1]:
        balance=np.bincount(cell,weights=tri[:,:,axis].mean(1)-minimum[:,axis],
                            minlength=(nx-1)*(ny-1)).reshape(ny-1,nx-1)
        if not np.allclose(balance[counts==2],1.,atol=1e-9):
            raise ValueError('Paired half-cells must tile the material without overlaps or gaps')
    y,x=np.nonzero(counts==2)
    if len(x):
        shear.extend([np.c_[grid[y,x],grid[y+1,x+1]],np.c_[grid[y,x+1],grid[y+1,x]]])
    shear=np.unique(np.sort(np.concatenate(shear),axis=1),axis=0)
    horizontal=np.zeros((ny,nx-1),bool);vertical=np.zeros((ny-1,nx),bool)
    for ax,array in [(0,horizontal),(1,vertical)]:
        selected=stretch[abs(xy[stretch[:,1],ax]-xy[stretch[:,0],ax])==1]
        lo=np.minimum(xy[selected[:,0]],xy[selected[:,1]])
        array[lo[:,1],lo[:,0]]=True
    yh,xh=np.nonzero(horizontal[:,:-1]&horizontal[:,1:]);yv,xv=np.nonzero(vertical[:-1]&vertical[1:])
    bend=np.concatenate([np.c_[grid[yh,xh],grid[yh,xh+2]],np.c_[grid[yv,xv],grid[yv+2,xv]]])
    if model=='triangle_surface':
        # The opposite diagonal is a bending chord when its two triangles
        # rotate isometrically about the rendered diagonal. Treating both
        # diagonals as membrane springs artificially loads a pure fold.
        actual=edges[diagonal]
        code=lambda pairs:np.sort(pairs,axis=1)[:,0]*len(rest)+np.sort(pairs,axis=1)[:,1]
        opposite=shear[~np.isin(code(shear),code(actual))]
        bend=np.unique(np.sort(np.concatenate([bend,opposite]),axis=1),axis=0)
        shear=actual
    springs={'stretch':stretch.tolist(),'shear':shear.tolist(),'bend':bend.tolist()}
    return springs,{'spacing_xy_m':steps,'balanced_cells':int((counts==2).sum()),
        'partial_boundary_cells':int((counts==1).sum()),'material_points_changed':False,
        'spring_model':model,'native_response_qualified':False,'task_approved':False}


def chart(points,faces,*,model='balanced_square_lattice'):
    springs,report=topology(points,faces,model=model)
    result={'points':np.asarray(points,float).tolist(),'springs':springs,
            'spring_model':model}
    validate(result,points,faces)
    return result,report
