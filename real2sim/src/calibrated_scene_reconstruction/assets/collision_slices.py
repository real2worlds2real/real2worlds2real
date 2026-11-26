"""Closed spatial collider pieces of the same rigid solid.

Planar cuts preserve the solid union, including holes. They change neither
rigid ownership nor mass properties. Source/render geometry is never replaced.
"""
import numpy as np


def partition(part,max_span_m,*,axis=None,maximum_pieces=32,dimensions=1):
    if type(dimensions) is not int or dimensions not in {1,3}:
        raise ValueError('Spatial partition supports one axial span or three bounded Cartesian spans')
    if dimensions==3:
        if axis is not None:raise ValueError('Three-dimensional collision blocks use the actual cooking coordinate axes')
        return blocks(part,max_span_m,maximum_pieces=maximum_pieces)
    import trimesh
    if (type(max_span_m) not in (int,float) or not np.isfinite(max_span_m) or max_span_m<=0
            or type(maximum_pieces) is not int or not 1<=maximum_pieces<=128):
        raise ValueError('Positive spatial span and bounded collider count required')
    source=trimesh.Trimesh(part['V'],part['F'],process=True)
    if not source.is_volume:raise ValueError('Spatial collision cuts require a closed outward solid')
    if axis is None:
        # Area/vertex density must not choose the axis: use the actual solid's
        # mass moment, so tessellation changes do not rotate the partition.
        _,basis=np.linalg.eigh(np.asarray(source.moment_inertia));axis=basis[:,0]
    axis=np.asarray(axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<=0:
        raise ValueError('Finite nonzero spatial cutting axis required')
    axis/=np.linalg.norm(axis)
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    lo,hi=(float(v) for v in (np.min(source.vertices@axis),np.max(source.vertices@axis)))
    count=int(np.ceil((hi-lo)/max_span_m))
    if count>maximum_pieces:raise ValueError('Spatial collider count exceeds the declared budget')
    # Avoid planes through numerically coincident voxel vertices without
    # changing the solid. Every accepted cut remains closed and conserves
    # volume; a failed triangulation never becomes an approved collider.
    for attempt_count in range(count,min(maximum_pieces,count+4)+1):
        pending=source;meshes=[];failed=False
        slack=max(0.,max_span_m-(hi-lo)/attempt_count)
        offsets=[0.]+[sign*min(epsilon,slack/4) for epsilon in (1e-9,1e-8,1e-7,1e-6) for sign in (-1,1)]
        for position in np.linspace(lo,hi,attempt_count+1)[1:-1]:
            accepted=None
            for offset in dict.fromkeys(offsets):
                point=(position+offset)*axis
                try:
                    left=trimesh.intersections.slice_mesh_plane(pending,-axis,point,cap=True)
                    right=trimesh.intersections.slice_mesh_plane(pending,axis,point,cap=True)
                except (ValueError,IndexError):
                    continue
                if (left.is_volume and right.is_volume
                        and np.isclose(left.volume+right.volume,pending.volume,rtol=1e-6,atol=abs(pending.volume)*1e-10)):
                    accepted=left,right;break
            if accepted is None:failed=True;break
            left,pending=accepted;meshes.append(left)
        if not failed:
            meshes.append(pending);break
    else:raise ValueError('Spatial cut did not preserve closed outward material')
    before=float(source.volume);after=sum(float(m.volume) for m in meshes)
    if not np.isclose(before,after,rtol=1e-6,atol=before*1e-10):
        raise ValueError('Spatial cuts changed source material volume')
    result=[]
    for index,mesh in enumerate(meshes):
        if np.ptp(mesh.vertices@axis)>max_span_m*(1+1e-7):
            raise ValueError('Spatial collider exceeds its requested span')
        result.append({'name':f'spatial_{index:03d}','V':np.array(mesh.vertices),
                       'F':np.array(mesh.faces),'materials':part.get('material_names',[]),
                       'approximation':'sdf'})
    return result,{'pieces':len(result),'axis':axis.tolist(),'max_span_m':float(max_span_m),
                   'volume_before':before,'volume_after':after,
                   'scope':'Closed source-solid union preserved by planar cuts; actual native contact precision still requires validation'}


def blocks(part,max_span_m,*,maximum_pieces=128):
    """Bound all three cooking-coordinate extents, including broad end caps.

    Cutting only along a long shaft does not resolve thin details on its broad
    head. Exact closed cuts in all axes keep the same solid and rigid actor.
    """
    import trimesh
    import manifold3d
    if (type(max_span_m) not in (int,float) or not np.isfinite(max_span_m) or max_span_m<=0
            or type(maximum_pieces) is not int or not 1<=maximum_pieces<=128):
        raise ValueError('Positive spatial span and bounded collider count required')
    source=trimesh.Trimesh(part['V'],part['F'],process=True)
    if not source.is_volume:raise ValueError('Spatial collision blocks require a closed outward solid')
    before=float(source.volume)
    solid=manifold3d.Manifold(manifold3d.Mesh64(
        np.ascontiguousarray(source.vertices,dtype=np.float64),
        np.ascontiguousarray(source.faces,dtype=np.uint64)))
    if solid.status()!=manifold3d.Error.NoError or not np.isclose(solid.volume(),before,rtol=1e-6):
        raise ValueError('Exact solid conversion failed or changed source material')
    solids=[solid]
    for axis_index,axis in enumerate(np.eye(3)):
        divided=[]
        for item in solids:
            bounds=np.asarray(item.bounding_box());lo,hi=bounds[axis_index],bounds[axis_index+3]
            count=int(np.ceil((hi-lo)/max_span_m));pending=item
            if count>maximum_pieces:raise ValueError('Three-dimensional collider count exceeds the declared budget')
            for position in np.linspace(lo,hi,count+1)[1:-1]:
                right,left=pending.split_by_plane(axis.tolist(),float(position))
                if left.status()!=manifold3d.Error.NoError or right.status()!=manifold3d.Error.NoError:
                    raise ValueError('Exact solid cut failed')
                if not left.is_empty():divided.append(left)
                pending=right
            if not pending.is_empty():divided.append(pending)
            if len(divided)>maximum_pieces:raise ValueError('Three-dimensional collider count exceeds the declared budget')
        solids=divided
    groups=[]
    for item in solids:
        mesh=item.to_mesh64()
        group={'V':np.asarray(mesh.vert_properties)[:,:3].copy(),'F':np.asarray(mesh.tri_verts).copy(),
               'materials':part.get('material_names',[]),'approximation':'sdf'}
        if not trimesh.Trimesh(group['V'],group['F'],process=False).is_volume:
            raise ValueError('Exported collision block lost closed material topology')
        groups.append(group)
    after=sum(float(trimesh.Trimesh(p['V'],p['F'],process=False).volume) for p in groups)
    maximum=max(float(np.ptp(p['V'],axis=0).max()) for p in groups)
    if not np.isclose(before,after,rtol=1e-6,atol=before*1e-10) or maximum>max_span_m*(1+1e-7):
        raise ValueError('Three-dimensional cuts did not preserve material and spatial bounds')
    for index,piece in enumerate(groups):piece['name']=f'spatial_block_{index:03d}'
    return groups,{'pieces':len(groups),'dimensions':3,'max_span_m':float(max_span_m),
                   'maximum_cooking_extent_m':maximum,'engine':'manifold3d_float64','volume_before':before,'volume_after':after,
                   'scope':'Same closed solid union and actor, with three bounded cooking extents; native contact accuracy remains unverified'}
