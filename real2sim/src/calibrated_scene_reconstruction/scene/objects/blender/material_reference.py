"""Image-derived unfolded-sheet constraints, separate from target silhouettes.

An independently identified reference sheet may constrain a folded sheet's
intrinsic area. Its plane projection uses current measured cameras/support;
uncertainty and the same-sheet relationship must be declared explicitly.
No supplied metric dimensions or candidate-rendered measurements are read.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def declaration(s,obj):
    if not hasattr(s,'config_dir'):return []
    path=Path(s.config_dir)/'agent/material_references.json'
    plan=(s.config('agent/material_references.json',{}) if hasattr(s,'config') else
          json.loads(path.read_text()) if path.is_file() else {})
    if not plan:return []
    if set(plan)!={'version','references'} or plan['version'] not in {1,2} or not isinstance(plan['references'],list):
        raise ValueError('Material references need a version1 or2 reference list')
    result=[]
    for item in plan['references']:
        base={'link','region','relationship','area_factors','basis'}
        whole=isinstance(item,dict) and item.get('relationship')=='same_sheet_geometry'
        required=base|({'source_extent','linear_factors','aspect_factors','outline_difference_max'} if whole else set())
        if (not isinstance(item,dict) or set(item)-{'instance_relationship'}!=required
                or any(not isinstance(item[k],str) or not item[k].strip() for k in ['link','region','basis'])
                or len(item['link'].split('/'))!=2 or item['relationship'] not in {'same_sheet_area','same_sheet_geometry'}
                or whole and (plan['version']!=2 or item['source_extent']!='complete')):
            raise ValueError('Declare source identity and an explicit same-sheet area relationship')
        if 'instance_relationship' in item and (not isinstance(item['instance_relationship'],str)
                or item['instance_relationship'] not in {'same_instance','same_type'}):
            raise ValueError('Reference must name the same physical instance or a separate specimen of the same type')
        factors=item['area_factors']
        if (not isinstance(factors,list) or len(factors)!=2 or any(type(v) not in (int,float) for v in factors)
                or not np.isfinite(factors).all() or not 0<factors[0]<=1<=factors[1] or factors[0]>=factors[1]):
            raise ValueError('Source area needs finite uncertainty factors bracketing one')
        if whole:
            for name in ['linear_factors','aspect_factors']:
                values=item[name]
                if (not isinstance(values,list) or len(values)!=2 or any(type(v) not in (int,float) for v in values)
                        or not np.isfinite(values).all() or not 0<values[0]<=1<=values[1] or values[0]>=values[1]):
                    raise ValueError('Whole-sheet geometry needs explicit finite uncertainty factors')
            limit=item['outline_difference_max']
            if type(limit) not in (float,int) or not np.isfinite(limit) or not 0<=limit<=.25:
                raise ValueError('Whole-sheet normalized outline tolerance must be in0..0.25')
        if item['link'].split('/')[0]==obj:result.append(item)
    if len({item['link'] for item in result})!=len(result):raise ValueError('Duplicate material reference link')
    return result


def footprint(mask,H):
    """Source contour area on the actual measured support plane."""
    contours,hierarchy=cv2.findContours(np.asarray(mask,'uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:raise ValueError('Material reference has no source footprint')
    inverse=np.linalg.inv(H);total=0.
    for index,contour in enumerate(contours):
        pixels=np.c_[contour[:,0,:],np.ones(len(contour))]
        world=pixels@inverse.T
        if (not np.isfinite(world).all() or np.any(abs(world[:,2])<1e-10)
                or np.any(world[:,2]*world[0,2]<=0)):
            raise ValueError('Reference footprint crosses the support horizon')
        xy=(world[:,:2]/world[:,2:]).astype('float32')
        depth=0;parent=hierarchy[0,index,3]
        while parent>=0:depth+=1;parent=hierarchy[0,parent,3]
        total+=(-1 if depth%2 else 1)*cv2.contourArea(xy)
    if not np.isfinite(total) or total<=0:raise ValueError('Source reference has no positive plane area')
    return float(total)


def attach(s,obj,cameras,frames):
    requested=declaration(s,obj)
    if not requested:return []
    from calibrated_scene_reconstruction.scene.objects.observation_exclusions import source_inputs,load,plane_camera
    # Reuse independently reviewed identity/visibility, not a raw projected
    # mask or a candidate model that happens to overlap the reference.
    _,files=load(s,obj,cameras,frames)
    sources={item['region']['id']:item for item in source_inputs(s,obj,cameras)}
    result=[]
    for item in requested:
        source=sources.get(item['region'])
        if source is None:raise ValueError('Material reference needs a reviewed source identity')
        a=source['anchor'];height=source['top_z']
        H=plane_camera(cameras['K'][a],cameras['R_world_cam'][a],cameras['centre'][a],height)
        area=footprint(source['mask'],H)
        row={**item,'footprint_area_m2':area,
                       'area_interval_m2':(area*np.asarray(item['area_factors'])).tolist(),
                       'source':source['region'],'support_fix':source['support_fix']}
        if item['relationship']=='same_sheet_geometry':
            from calibrated_scene_reconstruction.assets.cloth_outline import from_mask
            shape=from_mask(source['mask'],H)
            row.update(source_unfolded_geometry=shape,
                       dimensions_interval_m=(np.asarray(shape['dimensions_m'])[:,None]*item['linear_factors']).tolist(),
                       aspect_ratio_interval=(shape['aspect_ratio']*np.asarray(item['aspect_factors'])).tolist())
        result.append(row)
    cameras['_source_material']=result
    cameras.setdefault('_object_domain_inputs',{}).update(files)
    return result


def intrinsic_geometry(rest,faces,area):
    result={'material_area_m2':float(area)}
    from calibrated_scene_reconstruction.assets.cloth_outline import from_material
    try:
        outline=from_material(rest['points'],faces)
        if not np.isclose(outline['area_m2'],area,rtol=1e-5,atol=1e-12):
            raise ValueError('Unfolded boundary and triangle areas disagree; inspect overlapping material charts')
        result['material_outline']=outline
    except ValueError as error:
        result['material_outline_unavailable']=str(error)
    return result


def mesh_parts(data):
    """Recompute intrinsic areas from saved material charts for final CPU QA."""
    parts=json.loads(str(data['links_json'])) if 'links_json' in data else None
    rest=json.loads(str(data['cloth_rest_json'])) if 'cloth_rest_json' in data else {}
    for part in parts or []:
        if part['name'] not in rest:continue
        from calibrated_scene_reconstruction.assets.cloth_rest import validate
        a,n=part['vertex_start'],part['vertex_count'];b,count=part['face_start'],part['triangle_count']
        material=validate(rest[part['name']],data['V'][a:a+n],data['F'][b:b+count]-a)
        part.setdefault('geometry',{}).update(intrinsic_geometry(rest[part['name']],data['F'][b:b+count]-a,material['rest_area_m2']))
    from calibrated_scene_reconstruction.assets.optical_surfaces import annotate
    from calibrated_scene_reconstruction.assets.surface_regions import annotate as regions
    return regions(data,annotate(data,parts))


def score(cameras,parts):
    result=[]
    for item in cameras.get('_source_material',[]):
        name=item['link'].split('/')[1];part=next((p for p in parts or [] if p['name']==name),None)
        area=(part or {}).get('geometry',{}).get('material_area_m2')
        if area is None or not np.isfinite(area) or area<=0:
            raise ValueError('Referenced sheet needs its actual intrinsic material area')
        low,high=item['area_interval_m2'];ok=bool(low<=area<=high)
        deviation=max(low/area-1,area/high-1,0.)
        extra={}
        if item.get('relationship')=='same_sheet_geometry':
            shape=part['geometry'].get('material_outline')
            if shape is None:
                ok=False;extra['whole_sheet_failure']=part['geometry'].get('material_outline_unavailable','Missing unfolded boundary metadata; rebuild source evidence')
            else:
                from calibrated_scene_reconstruction.assets.cloth_outline import difference
                dimensions=np.asarray(shape['dimensions_m']);interval=np.asarray(item['dimensions_interval_m'])
                aspect=shape['aspect_ratio'];lo,hi=item['aspect_ratio_interval']
                outline=difference(item['source_unfolded_geometry'],shape)
                conditions={'area':ok,'dimensions':bool(((dimensions>=interval[:,0])&(dimensions<=interval[:,1])).all()),
                            'aspect_ratio':bool(lo<=aspect<=hi),'normalized_outline':outline<=item['outline_difference_max']}
                ok=all(conditions.values())
                extra.update(checks=conditions,material_dimensions_m=dimensions.tolist(),material_aspect_ratio=aspect,
                             normalized_outline_difference=outline)
                deviation+=max(0,outline-item['outline_difference_max'])
        result.append({**item,'material_area_m2':float(area),'ok':ok,**extra,
                       'loss':0. if ok else float(10.+deviation),
                       'scope':'Image-derived unfolded reference with declared uncertainty; no ground-truth dimensions, geometry scaling or captured-pose rest metric'})
    return result
