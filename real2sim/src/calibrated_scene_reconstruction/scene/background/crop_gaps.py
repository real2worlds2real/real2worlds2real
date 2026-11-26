"""Bind native appearance to holes made by the original GS task-zone crop.

Every proposed crop row must be part of the original recorded exclusion. Only UVs
and a per-mesh alpha material change; native topology and collision stay intact.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def geometry_key(world, counts, indices):
    h=hashlib.sha256()
    for values in (np.rint(np.asarray(world)*1e6),counts,indices):
        h.update(np.asarray(values,dtype='<i8').tobytes())
    return h.hexdigest()


def atlas(world, counts, indices, inside, pixel_m,postprocess=None,face_normals=False):
    """Independent planar face tiles; face-varying UVs preserve original faces."""
    if isinstance(pixel_m,bool) or not isinstance(pixel_m,(int,float)) or not .005<=pixel_m<=.05:
        raise ValueError('Crop gap resolution must be 5–50 mm')
    tiles=[];uv=[];offset=0;x=0;y=0;row_height=0;width=2048
    for count in counts:
        face=world[indices[offset:offset+count]];offset+=count
        if count<3:raise ValueError('Polygon faces required for crop masks')
        origin=face[0];_,singular,basis=np.linalg.svd(face-origin,full_matrices=False)
        if len(singular)<2 or singular[1]<1e-9:raise ValueError('Degenerate crop mask face')
        axes=basis[:2];local=(face-origin)@axes.T;lo=local.min(0);extent=np.ptp(local,axis=0)
        if np.max(abs((face-origin)@basis[2]))>1e-4:raise ValueError('Crop mask requires planar polygons')
        wh=np.maximum(1,np.ceil(extent/pixel_m).astype(int));w,h=wh+4
        if w>width or h>8192:raise ValueError('Crop face exceeds bounded atlas size')
        if x+w>width:x=0;y+=row_height;row_height=0
        if y+h>8192:raise ValueError('Crop atlas exceeds bounded memory')
        xx,yy=np.meshgrid(np.arange(w),np.arange(h));xy=(np.stack([xx,yy],-1)-1.5)/wh*extent+lo
        samples=origin+xy.reshape(-1,2)@axes
        normal=np.cross(face[1]-face[0],face[2]-face[0]);normal/=np.linalg.norm(normal)
        tile=np.asarray(inside(samples,normal) if face_normals else inside(samples),bool).reshape(h,w)
        if postprocess is not None:tile=postprocess(tile)
        tiles.append((x,y,tile));uv.extend(np.c_[x+2+(local[:,0]-lo[0])/extent[0]*wh[0],y+2+(local[:,1]-lo[1])/extent[1]*wh[1]])
        x+=w;row_height=max(row_height,h)
    height=y+row_height;image=np.zeros((height,width),np.uint8)
    for x,y,tile in tiles:image[y:y+tile.shape[0],x:x+tile.shape[1]]=tile*255
    return image,np.asarray(uv)/[width,height]


def verified_crop(source_session,declaration):
    from plyfile import PlyData
    from calibrated_scene_reconstruction.scene.background.splat.config import sim3,task_zone
    from calibrated_scene_reconstruction.core.paths import resolve
    import calibrated_scene_reconstruction.assets.metric_mesh as metric_mesh
    from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
    root=source_session/'splat/export';selection=root/'selection.json';proof=json.loads(selection.read_text())
    cloud=root/'full_colmap.ply';masks=root/'selection_masks.npz'
    for path in (cloud,masks):
        if proof['files'].get(str(path))!=digest(path):raise ValueError('Original crop row evidence changed')
    inside,zone=task_zone(source_session,proof['crop']);v=PlyData.read(str(cloud))['vertex'].data
    scale,R,t=sim3(source_session);world=scale*np.c_[v['x'],v['y'],v['z']].astype(float)@R.T+t
    original=np.load(masks)['in_zone']
    inputs={str(p.resolve()):digest(p) for p in (selection,cloud,masks,source_session/'splat/align.json',Path(__file__),Path(__file__).with_name('splat')/'config.py')}
    if declaration.get('include_robot_surface'):
        from scipy.spatial import cKDTree
        metadata=source_session/'scene/splat/transform.json'
        record=source_session/'qa/robot_background_exclusion.json'
        robot_cloud=source_session/'parsing/robot_cloud.npz'
        tr=json.loads(metadata.read_text());audit=json.loads(record.read_text())
        robot=tr['zone'].get('capture_robot_surface_exclusion') or {}
        voxel=declaration.get('robot_surface_voxel_m',.005)
        if (not robot.get('raw_readout') or robot.get('margin_m')!=audit.get('surface_margin_m')
                or not 0<float(robot.get('margin_m',0))<=.1 or audit.get('cloud_sha256')!=digest(robot_cloud)
                or isinstance(voxel,bool) or not isinstance(voxel,(int,float)) or not .001<=voxel<=.01):
            raise ValueError('Original capture robot crop needs bound raw-readout evidence')
        points=np.load(robot_cloud)['points'].astype(float)
        _,ids=np.unique(np.floor(points/voxel).astype('int64'),axis=0,return_index=True);points=points[ids]
        if len(points)!=robot.get('points') or len(points)!=audit.get('sampled_surface_points'):
            raise ValueError('Original robot crop sampling does not match')
        graph=source_session.scene_graph;base=graph['photo_robot']['base'];yaw=np.radians(base['yaw_deg']);c,si=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-si,0],[si,c,0],[0,0,1.]])
        points=points@rotation.T+[base['xy'][0],base['xy'][1],graph['_robot_fit']['fit']['height_offset_m']]
        tree=cKDTree(points);base_inside=inside
        def inside(p):
            return base_inside(p)|((tree.query(p,workers=1)[0]<robot['margin_m'])&(p[:,2]>proof['crop']['keep_floor']))
        zone['capture_robot_surface_exclusion']=robot
        inputs.update({str(p.resolve()):digest(p) for p in (metadata,record,robot_cloud)})
    proposed=inside(world)
    if original.shape!=(len(world),) or np.any(proposed & ~original):
        raise ValueError('Proposed task crop includes rows not excluded originally; refuse gap ownership')
    if declaration.get('include_robot_surface') and not np.array_equal(proposed,original):
        raise ValueError('Full declared crop must reproduce every original row decision')
    for path in (source_session.config_dir/'scene_graph.json',source_session.config_dir/'twin.json',Path(metric_mesh.__file__)):
        if path.is_file():inputs[str(path.resolve())]=digest(path)
    for obj in source_session.scene_graph.get('dynamic_objects',{}).values():
        fix=obj.get('fix') or {}
        if fix.get('model') and fix.get('mesh'):
            path=resolve(fix['mesh']);inputs[str(path.resolve())]=digest(path)
    return inside,{'rows_verified':len(world),'excluded_rows':int(proposed.sum()),'other_original_exclusions_not_filled':int((original & ~proposed).sum()),'zone':zone,
                   'protected_source_state':protected_state(source_session)},inputs


def prepare(source_session,job,entity,declaration,out):
    from PIL import Image
    from pxr import Usd,UsdGeom
    from calibrated_scene_reconstruction.scene.review.environment import digest
    inside,crop,inputs=verified_crop(source_session,declaration)
    return bake(job,entity,declaration,out,inside,{'crop_evidence':crop},inputs)


def bake(job,entity,declaration,out,inside,evidence,inputs,postprocess=None,face_normals=False):
    from PIL import Image
    from pxr import Usd,UsdGeom
    from calibrated_scene_reconstruction.scene.review.environment import digest
    result=Path(job['output']);usd=Path(json.loads(result.read_text())['usd'])
    inputs[str(Path(__file__).resolve())]=digest(__file__)
    inputs.update({str(p.resolve()):digest(p) for p in (result,usd)})
    stage=Usd.Stage.Open(str(usd));cache=UsdGeom.XformCache();meshes={}
    for i,row in enumerate(r for r in job['parts'] if r['entity']==entity and not r['external']):
        prim=stage.GetPrimAtPath(row['path']);mesh=UsdGeom.Mesh(prim)
        if not mesh:raise ValueError('Original crop mask needs current exported meshes')
        points=np.asarray(mesh.GetPointsAttr().Get(),float);world=(np.c_[points,np.ones(len(points))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:,:3]
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int);indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
        image,uv=atlas(world,counts,indices,inside,declaration.get('pixel_m',.01),postprocess,face_normals)
        # Diagnostic layers can retain older material networks as well. Renderer
        # caches keyed by basename must never confuse masks from different
        # rounds, or different parts after Blender reorders its export.
        identity=hashlib.sha256(row['path'].encode()).hexdigest()[:12]
        pixels=hashlib.sha256(np.asarray(image.shape,dtype='<i8').tobytes()+image.tobytes()).hexdigest()[:16]
        path=out/(entity+'_gap_'+identity+'_'+pixels+'.png');Image.fromarray(np.flipud(image)).save(path)
        uv_path=path.with_suffix('.npz');np.savez_compressed(uv_path,uv=uv,world=world,counts=counts,indices=indices)
        meshes[row['path']]={'path':str(path),'sha256':digest(path),'uv_path':str(uv_path),'uv_sha256':digest(uv_path),
                             'geometry_key':geometry_key(world,counts,indices),'visible_fraction':float((image>0).mean())}
    if not meshes:raise ValueError('No native mesh parts for source crop filler')
    return {'kind':'surface_atlas','meshes':meshes,**evidence},inputs


def uncovered_surface(points,sigma,opacity,declaration):
    """Conservative local GS coverage at points on an explicit native surface."""
    from scipy.spatial import cKDTree
    radius=declaration.get('radius_m',.05);minimum=declaration.get('coverage',.9)
    if (isinstance(radius,bool) or not isinstance(radius,(int,float)) or not .01<=radius<=.08
            or isinstance(minimum,bool) or not isinstance(minimum,(int,float)) or not .5<=minimum<=.99):
        raise ValueError('Bounded metric surface neighborhood and GS coverage required')
    points=np.asarray(points,float);sigma=np.asarray(sigma,float);opacity=np.asarray(opacity,float)
    if points.shape!=(len(sigma),3) or opacity.shape!=sigma.shape or not len(points):raise ValueError('Matching retained GS rows required')
    if not np.isfinite(points).all() or not np.isfinite(sigma).all() or not np.isfinite(opacity).all() or np.any(sigma<=0) or np.any((opacity<0)|(opacity>1)):
        raise ValueError('Finite GS geometry and opacity required')
    tree=cKDTree(points);width=np.minimum(sigma,radius/2);weight=-np.log1p(-np.minimum(opacity,1-1e-6))
    def inside(samples):
        result=[]
        for start in range(0,len(samples),32768):
            distance,ids=tree.query(samples[start:start+32768],k=min(32,len(points)),distance_upper_bound=radius,workers=1)
            if distance.ndim==1:distance=distance[:,None];ids=ids[:,None]
            valid=ids<len(points);safe=np.minimum(ids,len(points)-1)
            density=(np.exp(-.5*(distance/width[safe])**2)*weight[safe]*valid).sum(1)
            result.append(-np.expm1(-density)<minimum)
        return np.concatenate(result)
    return inside


def prepare_coverage(job,entity,declaration,points,sigma,opacity,out):
    mode=declaration.get('compositing','cutout')
    if mode not in ('cutout','solid_backing'):raise ValueError('Use cutout or solid native backing')
    if mode=='solid_backing':
        return {'kind':'solid_backing','scope':'Keep the original opaque PBR mesh behind source-qualified GS. Coverage estimates cannot cut holes into unsupported novel rays.'}, {str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    inside=uncovered_surface(points,sigma,opacity,declaration)
    from scipy.ndimage import binary_closing
    close=declaration.get('close_gaps_m',0.);cell=declaration.get('pixel_m',.01)
    if isinstance(close,bool) or not isinstance(close,(int,float)) or not 0<=close<=.03:
        raise ValueError('Surface mask closing must be0–30 mm')
    size=int(np.ceil(close/cell))
    postprocess=(lambda image:binary_closing(image,structure=np.ones((size*2+1,)*2),border_value=1)) if size else None
    return bake(job,entity,declaration,out,inside,{'coverage_evidence':{'method':'Local opacity support from retained source-backed GS, up to32 neighbors; capped isotropic widths.',
                'radius_m':declaration.get('radius_m',.05),'minimum':declaration.get('coverage',.9),'close_gaps_m':close,
                'scope':'Appearance-gap hypothesis; source/free-view preservation and physical/native geometry reviews remain mandatory.'}},
                {str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},postprocess)
