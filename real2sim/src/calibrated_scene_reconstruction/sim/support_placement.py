"""Case-specific shell support correction, before native physics starts."""
import json

import numpy as np
from pxr import Gf,UsdGeom

from calibrated_scene_reconstruction.assets.support_placement import offset
from calibrated_scene_reconstruction.core.paths import resolve


def apply(stage,ts,report):
    adjustments={}
    priors=ts.inputs.physical_priors
    for name,spec in ts.s.scene_graph['dynamic_objects'].items():
        fix=spec.get('fix') or {}
        # Only renewed assets with an explicit physical-support contract.
        # Existing frozen/legacy scene placements are not reinterpreted.
        if 'support_contact_offset_m' not in fix or name not in ts.objects:continue
        with np.load(resolve(fix['mesh'])) as source:
            if 'links_json' not in source:continue
            parts=json.loads(str(source['links_json']));v=source['V'].astype(float).copy()
        nominal={};sampled={};cache=UsdGeom.XformCache()
        for part in parts:
            first,count=part['vertex_start'],part['vertex_count']
            path=str(ts.objects[name].GetPath())+'/'+part['name']+'/surface'
            matrix=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
            v[first:first+count]=v[first:first+count]@matrix[:3,:3]+matrix[3,:3]
            ref=name+'/'+part['name'];parameters=priors['bodies'][ref]['parameters']
            if part['kind']=='cloth' and 'thickness' in parameters:
                nominal[part['name']]=parameters['thickness']['nominal']
                sampled[part['name']]=report['bodies'][ref]['thickness']
        mesh={'V':v,'links_json':json.dumps(parts)}
        normal=fix.get('support_plane',{}).get('normal',(0.,0.,1.))
        delta=offset(mesh,sampled,normal=normal)-offset(mesh,nominal,normal=normal)
        if abs(delta)<1e-9:continue
        prim=ts.objects[name];world=Gf.Matrix4d(cache.GetLocalToWorldTransform(prim));p=world.ExtractTranslation()
        world.SetTranslateOnly(p+Gf.Vec3d(0,0,float(delta)))
        parent=cache.GetLocalToWorldTransform(prim.GetParent());UsdGeom.Xformable(prim).MakeMatrixXform().Set(world*parent.GetInverse())
        note={'world_z_adjustment_m':float(delta),'nominal_thickness_m':nominal,'sampled_thickness_m':sampled,
              'purpose':'Preserve the same physical support at initial setup when material thickness changes; no runtime pose following'}
        prim.SetCustomDataByKey('calibrated_scene_reconstruction:sampled_support_placement',json.dumps(note));adjustments[name]=note
    report['support_placement_adjustments']=adjustments
    return adjustments
