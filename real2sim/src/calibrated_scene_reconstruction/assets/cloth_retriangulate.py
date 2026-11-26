"""Remesh a structured surface proposal without moving material or vertices."""
import json
import numpy as np

from .cloth_rest import triangulate_grid


def propose(mesh,link,shape,*,preserve_cells=()):
    parts=json.loads(str(mesh['links_json']));charts=json.loads(str(mesh['cloth_rest_json']))
    part=next(p for p in parts if p['name']==link)
    if part['kind']!='cloth' or 'bend_angles' in charts[link]:
        raise ValueError('A cloth proposal without unremapped independent curvature indices required')
    a,n=part['vertex_start'],part['vertex_count'];b,count=part['face_start'],part['triangle_count']
    rows,cols=shape
    if n!=rows*cols or count!=2*(rows-1)*(cols-1):raise ValueError('Explicit matching rectangular material chart required')
    old=charts[link]
    faces,chart,report=triangulate_grid(mesh['V'][a:a+n],old['points'],rows,cols)
    preserve=np.asarray(preserve_cells,dtype=int)
    if preserve.ndim!=1 or (len(preserve) and (preserve.min()<0 or preserve.max()>=(rows-1)*(cols-1))):
        raise ValueError('Preserved quadrilateral indices must belong to this material grid')
    if len(preserve):
        faces=faces.reshape(-1,2,3)
        faces[preserve]=(mesh['F'][b:b+count]-a).reshape(-1,2,3)[preserve]
        grid=np.arange(rows*cols).reshape(rows,cols);start=grid[:-1,:-1].ravel()
        positive=np.c_[start,start+cols+1];negative=np.c_[start+1,start+cols]
        is_positive=faces[:,0,2]==start+cols+1
        shear=np.where(is_positive[:,None],positive,negative)
        opposite=np.where(is_positive[:,None],negative,positive)
        bend=np.r_[opposite,np.c_[grid[:,:-2].ravel(),grid[:,2:].ravel()],
                   np.c_[grid[:-2].ravel(),grid[2:].ravel()]]
        chart['springs']['shear']=shear.tolist();chart['springs']['bend']=bend.tolist()
        faces=faces.reshape(-1,3)
        report['preserved_cells']=np.unique(preserve).tolist()
    if 'bend_memory' in old:
        # New diagonal chords have different identities. Do not pretend that
        # their old numeric indices still encode the same particle history.
        if part.get('backend')!='surface_fem':
            raise ValueError('Particle bend-memory remapping is not implemented')
        report['removed_unused_particle_memory']=True
    if 'bend_reference' in old:chart['bend_reference']=old['bend_reference']
    from .cloth_rest import validate
    validate(chart,mesh['V'][a:a+n],faces)
    if 'face_material' in mesh and len(np.unique(mesh['face_material'][b:b+count]))!=1:
        raise ValueError('Remeshing across material boundaries needs explicit face transfer')
    result={k:v.copy() for k,v in mesh.items()};result['F'][b:b+count]=faces+a
    if 'UV' in mesh and mesh['UV'].shape==(len(mesh['F']),3,2):
        oldfaces=mesh['F'][b:b+count]-a;uv=mesh['UV'][b:b+count]
        ids,first=np.unique(oldfaces,return_index=True);vertex_uv=np.zeros((n,2));vertex_uv[ids]=uv.reshape(-1,2)[first]
        if not np.allclose(vertex_uv[oldfaces],uv,atol=1e-7):
            raise ValueError('A UV seam crosses remeshed cloth; explicit texture transfer required')
        result['UV'][b:b+count]=vertex_uv[faces]
    charts[link]=chart;result['cloth_rest_json']=np.array(json.dumps(charts))
    from .cloth_rest import material_inverse,principal_strains
    original_faces=mesh['F'][b:b+count]-a
    points=mesh['V'][a:a+n];rest=np.asarray(old['points'])
    report['old_maximum_principal_strain']=float(abs(principal_strains(points,original_faces,material_inverse(rest,original_faces))).max())
    report['new_maximum_principal_strain']=float(abs(principal_strains(points,faces,material_inverse(rest,faces))).max())
    report.update(changed_triangles=int(np.any(mesh['F'][b:b+count]!=faces+a,axis=1).sum()),
                  source_approved=False,task_approved=False)
    return result,report
