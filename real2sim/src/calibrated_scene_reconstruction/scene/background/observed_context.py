"""Prepare a same-capture GS appearance context around editable native geometry.

Only background appearance is selected. Protected task/support/robot calibration
and geometry are never fitted or changed. Native proxies still need physics QA;
GS appearance still needs source/novel-view floater and coverage review.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT,resolve
from calibrated_scene_reconstruction.scene.background.blender_env import label
from calibrated_scene_reconstruction.scene.background.splat.config import sim3
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
from calibrated_scene_reconstruction.scene.session import Session


def parameter_digest(parameters):
    import hashlib
    return hashlib.sha256(json.dumps(parameters,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def replacement_mask(near_native,claims,replace_ids,preserve_ids):
    """Protect identified neighboring props from a replacement's metric margin."""
    replace=set(replace_ids);preserve=set(preserve_ids)
    if replace&preserve or any(type(k)is not int or k<0 for k in replace|preserve):
        raise ValueError('Use disjoint nonnegative replacement/preservation claim IDs')
    near=np.asarray(near_native,bool);claims=np.asarray(claims)
    if near.shape!=claims.shape:raise ValueError('Claim rows must match native proximity rows')
    semantic=np.isin(claims,list(replace));protected=np.isin(claims,list(preserve))
    return (near&~protected)|semantic,semantic,int((near&protected).sum())


def floor_uncertainty(points, stuff, seen, floor_z):
    """Estimate a ground rejection band from repeatedly observed floor rows.

    This changes rejection uncertainty, never the calibrated floor or task
    geometry. Insufficient semantic support retains the conservative default.
    """
    points=np.asarray(points,float);stuff=np.asarray(stuff);seen=np.asarray(seen)
    if points.ndim!=2 or points.shape[1]!=3 or stuff.shape!=(len(points),) or seen.shape!=(len(points),):
        raise ValueError('Aligned semantic floor rows required')
    residual=points[:,2]-floor_z
    take=(stuff==1)&(seen>=3)&np.isfinite(points).all(1)&(abs(residual)<.15)
    values=residual[take];report={'support':int(take.sum()),'minimum_support':200,'tolerance_m':.015,
                                'method':'Observed floor residual median plus three robust standard deviations, bounded to15–100mm; floor geometry unchanged.'}
    if len(values)>=200:
        median=float(np.median(values));mad=float(np.median(abs(values-median)))
        report.update(median_m=median,mad_m=mad,tolerance_m=float(np.clip(abs(median)+3*1.4826*mad,.015,.1)))
    return report


def protected_overlap(vertices, points, scale, rotation, geometry, parts):
    """Reject GS whose three-sigma support can paint a frozen task mesh.

    Use each exported part separately, not a room-sized aggregate task box.
    Oriented covariance keeps broad thin floor sheets from behaving like blobs.
    All bounds come from the current task mesh, never evaluation dimensions.
    """
    from scipy.spatial.transform import Rotation
    q=np.c_[vertices['rot_1'],vertices['rot_2'],vertices['rot_3'],vertices['rot_0']].astype(float)
    orientations=np.asarray(rotation)@Rotation.from_quat(q).as_matrix()
    axes=scale*np.exp(np.c_[vertices['scale_0'],vertices['scale_1'],vertices['scale_2']].astype(float))
    spread=3*np.sqrt(np.einsum('nij,nj->ni',orientations**2,axes**2))
    removed=np.zeros(len(points),bool);counts={}
    for i,row in enumerate(parts):
        if not row['external']:continue
        margin=row.get('margin_m',.015)
        v=geometry[f'v{i}'];hit=np.all((points+spread+margin>=v.min(0))&(points-spread-margin<=v.max(0)),axis=1)
        removed|=hit;counts[row['entity']]=counts.get(row['entity'],0)+int(hit.sum())
    return removed,{'removed':int(removed.sum()),'part_overlap_counts':counts,'sigma_support':3.,'margin_m':.015,
                    'scope':'Conservative oriented Gaussian support against current protected mesh parts. No task mesh, position, scale or calibration changes.'}


def protected_geometry(job,robot_margin=.05):
    """Add frozen robot visuals to the read-only task geometry used for GS crop.

    Background gravity audits deliberately omit robot dynamics. That omission
    must not omit the robot from appearance protection as well.
    """
    from pxr import Usd,UsdGeom
    if isinstance(robot_margin,bool) or not isinstance(robot_margin,(int,float)) or not .015<=robot_margin<=.1:
        raise ValueError('Bounded robot appearance-exclusion uncertainty required')
    result=Path(job['output']);audit=json.loads(result.read_text());usd=Path(audit['usd']).resolve()
    if audit.get('pass')is not True or audit['input_sha256'].get(str(usd))!=digest(usd):raise ValueError('Protected geometry requires unchanged audited USD')
    geometry=dict(np.load(job['geometry']));parts=list(job['parts']);stage=Usd.Stage.Open(str(usd));cache=UsdGeom.XformCache();links={}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        path=str(prim.GetPath())
        if not path.startswith('/World/Robot/') or not prim.IsA(UsdGeom.Mesh):continue
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
        if not len(points):continue
        world=(np.c_[points,np.ones(len(points))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:,:3]
        links.setdefault(path.split('/')[3],[]).append(world)
    for link,arrays in links.items():
        index=len(parts);geometry[f'v{index}']=np.concatenate(arrays)
        parts.append({'entity':'frozen_robot/'+link,'external':True,'margin_m':robot_margin})
    return geometry,parts,{str(result.resolve()):digest(result),str(usd):digest(usd)},{'links':len(links),'margin_m':robot_margin,'source_usd':str(usd),'read_only':True}


def opacity_adjustment(vertices, gain):
    """Bounded, explicit appearance hypothesis; never move or densify geometry."""
    if isinstance(gain,bool) or not isinstance(gain,(int,float)) or not np.isfinite(gain) or not 1<=gain<=4:
        raise ValueError('Opacity gain must be a finite appearance hypothesis in1..4')
    result=vertices.copy()
    if gain!=1:
        alpha=1/(1+np.exp(-np.clip(result['opacity'].astype(float),-80,80)))
        alpha=np.clip(-np.expm1(gain*np.log1p(-np.minimum(alpha,1-1e-12))),1e-12,1-1e-12)
        result['opacity']=np.log(alpha)-np.log1p(-alpha)
    return result


def clean_context(vertices, points, sigma, config, enclosure, cameras, image_root, surface_distance=None, world_rotation=None):
    """Cull unsupported appearance using the capture, independently of proxies.

    DC colour agreement is a conservative diagnostic, not a reconstruction loss.
    It cannot create detail absent from the capture. Thresholds are explicit,
    metric/capture-relative and never selected from task size truth.
    """
    import cv2
    from scipy.spatial import cKDTree
    from calibrated_scene_reconstruction.scene.background.completion import inside, specification
    specification(enclosure)
    defaults={'max_sigma_m':.065,'min_opacity':.08,'neighbor_radius_m':.10,'min_neighbors':4,
              'floor_below_tolerance_m':.015,
              'photo_views':64,'min_photo_views':2,'max_rgb_error':.35,'max_sigma_px':12.,'surface_band_m':None,
              'planar_filter':'capture_supported'}
    if not isinstance(config,dict) or set(config)-set(defaults):raise ValueError('Known context quality-filter settings required')
    cfg={**defaults,**config}
    if cfg['surface_band_m'] is not None:
        raise ValueError('Native proxy distance cannot establish GS validity; use capture evidence, not surface_band_m')
    if (cfg['planar_filter'] not in ('capture_supported','bounded_projection')
            or not .005<=cfg['max_sigma_m']<=.2 or not 0<cfg['min_opacity']<1 or not .01<=cfg['neighbor_radius_m']<=.3
            or any(type(cfg[k]) is not int for k in ['min_neighbors','photo_views','min_photo_views'])
            or not 2<=cfg['min_neighbors']<=32 or not 8<=cfg['photo_views']<=256
            or not 2<=cfg['min_photo_views']<=cfg['photo_views'] or not .03<=cfg['max_rgb_error']<=.5
            or not isinstance(cfg['floor_below_tolerance_m'],(int,float)) or isinstance(cfg['floor_below_tolerance_m'],bool)
            or not .015<=cfg['floor_below_tolerance_m']<=.1
            or not isinstance(cfg['max_sigma_px'],(int,float)) or isinstance(cfg['max_sigma_px'],bool) or not 2<=cfg['max_sigma_px']<=64
            or (cfg['surface_band_m'] is not None and (not isinstance(cfg['surface_band_m'],(int,float))
                or isinstance(cfg['surface_band_m'],bool) or not 0<cfg['surface_band_m']<=.3))):
        raise ValueError('Bounded Gaussian quality thresholds and multiple photo views required')
    finite=np.isfinite(points).all(1)&np.isfinite(sigma)&np.isfinite(vertices['opacity'])
    in_room=inside(points,enclosure['footprint'])&(points[:,2]>=enclosure['floor_z']-cfg['floor_below_tolerance_m'])&(points[:,2]<=enclosure['ceiling_z'])
    opacity=1/(1+np.exp(-np.clip(vertices['opacity'],-80,80)))
    planar=np.zeros(len(vertices),bool)
    world_axes=None
    if all(k in vertices.dtype.names for k in ['scale_0','scale_1','scale_2']):
        axis_logs=np.c_[vertices['scale_0'],vertices['scale_1'],vertices['scale_2']].astype(float)
        logs=np.sort(axis_logs,axis=1)
        # The middle/minor ratio distinguishes a thin surface from a spherical
        # blob or a long needle. Large in-plane extent is not poor geometry.
        planar=np.isfinite(logs).all(1)&((logs[:,1]-logs[:,0])>=np.log(10.))
        orientations=np.broadcast_to(np.eye(3),(len(vertices),3,3))
        if all('rot_'+str(i) in vertices.dtype.names for i in range(4)):
            from scipy.spatial.transform import Rotation
            orientations=Rotation.from_quat(np.c_[vertices['rot_1'],vertices['rot_2'],vertices['rot_3'],vertices['rot_0']]).as_matrix()
        orientations=np.asarray(np.eye(3) if world_rotation is None else world_rotation)@orientations
        axes=sigma[:,None]*np.exp(axis_logs-logs[:,-1,None])
        world_axes=orientations*axes[:,None,:]
    # The bounded proposal can reject streaks, but also real textureless walls.
    # Keep it explicit and require source-preservation QA before adopting it.
    # Capture-supported mode retains large sheets for agent visual review.
    bounded=cfg['planar_filter']=='bounded_projection'
    oversized=(sigma>cfg['max_sigma_m'])&(~planar|(bounded&(sigma>min(.3,4*cfg['max_sigma_m']))))
    keep=finite&in_room&~oversized&(opacity>=cfg['min_opacity'])
    counts={'source_rows':len(vertices),'nonfinite':int((~finite).sum()),'outside_enclosure':int((finite&~in_room).sum()),
            'oversized':int((finite&oversized).sum()),'faint':int((opacity<cfg['min_opacity']).sum()),
            'large_planar_candidates':int((planar&(sigma>cfg['max_sigma_m'])).sum())}
    ids=np.flatnonzero(keep)
    neighbors=cKDTree(points[ids]).query_ball_point(points[ids],cfg['neighbor_radius_m'],return_length=True,workers=-1) if len(ids) else np.zeros(0,int)
    isolated=(neighbors<cfg['min_neighbors'])&(~planar[ids]|bounded)
    keep[ids[isolated]]=False;counts['isolated']=int(isolated.sum())
    ids=np.flatnonzero(keep);Q=points[ids];votes=np.zeros(len(ids),np.uint16);last=np.full((len(ids),3),np.nan)
    angular_support=np.zeros(len(ids),bool)
    colors=np.clip(.5+.2820947918*np.c_[vertices['f_dc_0'][ids],vertices['f_dc_1'][ids],vertices['f_dc_2'][ids]],0,1)
    view_ids=np.unique(np.linspace(0,len(cameras['names'])-1,min(cfg['photo_views'],len(cameras['names']))).round().astype(int))
    image_inputs={}
    for i in view_ids:
        path=Path(image_root)/str(cameras['names'][i]);image_inputs[str(path.resolve())]=digest(path)
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)/255.
        c=cameras['centre'][i];pc=(Q-c)@cameras['R_world_cam'][i];uv=pc@cameras['K'][i].T;uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9)
        valid=(pc[:,2]>.15)&(uv[:,0]>=1)&(uv[:,0]<rgb.shape[1]-1)&(uv[:,1]>=1)&(uv[:,1]<rgb.shape[0]-1)
        take=np.flatnonzero(valid);xy=uv[take].astype(int)
        agreement=np.abs(rgb[xy[:,1],xy[:,0]]-colors[take]).mean(1)<=cfg['max_rgb_error']
        # Only an actual observing camera may reject angular resolution. A
        # nearby camera behind the surface must not erase a distant-view prop.
        footprint=sigma[ids[take]]*max(cameras['K'][i,0,0],cameras['K'][i,1,1])/pc[take,2]
        sheets=np.flatnonzero(planar[ids[take]])
        if len(sheets):
            basis=cameras['R_world_cam'][i].T@world_axes[ids[take[sheets]]]
            xyz=pc[take[sheets]];depth=xyz[:,2,None]
            jx=cameras['K'][i,0,0]/depth*(basis[:,0,:]-xyz[:,0,None]/depth*basis[:,2,:])
            jy=cameras['K'][i,1,1]/depth*(basis[:,1,:]-xyz[:,1,None]/depth*basis[:,2,:])
            a=(jx*jx).sum(1);b=(jx*jy).sum(1);d=(jy*jy).sum(1)
            footprint[sheets]=np.sqrt(np.maximum(0,(a+d+np.sqrt((a-d)**2+4*b*b))/2))
        resolved=(footprint<=cfg['max_sigma_px'])|((not bounded)&planar[ids[take]]);angular_support[take[resolved]]=True
        # Consecutive nearly identical cameras do not count as independent support.
        separated=(votes[take]==0)|(np.linalg.norm(last[take]-c,axis=1)>.05)
        take=take[agreement&separated&resolved];votes[take]+=1;last[take]=c
    accepted=votes>=cfg['min_photo_views'];keep[ids[~accepted]]=False
    counts.update(photo_unsupported=int((~accepted).sum()),large_angular_footprint=int((~angular_support).sum()),retained=int(keep.sum()))
    return keep,{'settings':cfg,'counts':counts,'scope':'Capture-only appearance rejection, independent of native proxies. RGB agreement is not occlusion proof or a floater/photorealism pass.'},image_inputs


def complete_ground(session,source_session,cfg,job,base_vertices,output):
    """Restore only trained near-ground rows supported by independent capture masks.

    Never restore unseen/undrawn/near-camera/huge rows. A generic free-space
    rejection can remove carpet seen at grazing angles; geometry, semantic and
    color agreement must all support its reinstatement.
    """
    import cv2
    from plyfile import PlyData
    cloud=source_session/'splat/export/full_colmap.ply';selection=source_session/'splat/export/selection_masks.npz'
    observations=Path(cfg['observations']);inputs={str(p.resolve()):digest(p) for p in [cloud,selection,observations]}
    full=PlyData.read(str(cloud))['vertex'].data;masks=np.load(selection)
    if not np.array_equal(full[masks['kept']],base_vertices):raise ValueError('Ground completion source does not match retained cloud row IDs')
    scale,R,t=sim3(source_session);P=scale*np.c_[full['x'],full['y'],full['z']].astype(float)@R.T+t
    sigma=scale*np.exp(np.c_[full['scale_0'],full['scale_1'],full['scale_2']].astype(float)).max(1)
    ground=float(job['contract']['ground_height']);band=float(cfg.get('band_m',.025));maximum=float(cfg.get('max_sigma_m',.04))
    min_views=cfg.get('min_views',2);error=float(cfg.get('max_rgb_error',.18))
    if not 0<band<=.05 or not 0<maximum<=.1 or type(min_views)is not int or min_views<2 or not 0<error<=.25:
        raise ValueError('Conservative ground band, size, multiple views and color agreement required')
    rejected=np.logical_or.reduce([masks[k] for k in ['in_zone','unseen','faint','huge','near_cameras','undrawn']])
    eligible=masks['floaters']&~rejected&(abs(P[:,2]-ground)<band)&(sigma<maximum)
    ids=np.flatnonzero(eligible);Q=P[ids];votes=np.zeros(len(ids),int)
    color=np.clip(.5+.2820947918*np.c_[full['f_dc_0'][ids],full['f_dc_1'][ids],full['f_dc_2'][ids]],0,1)
    camera=np.load(session/'parsing/posed_frames/cameras.npz');names=list(map(str,camera['names']))
    rows={r['frame']:r for r in json.loads(observations.read_text())['records'] if r['entity']==cfg.get('entity','observed_floor') and r.get('mask')}
    for frame,row in rows.items():
        for path,expected in [(row['source'],row['source_sha256']),(row['mask'],row['mask_sha256'])]:
            if digest(path)!=expected:raise ValueError('Ground observation changed: '+path)
            inputs[str(Path(path).resolve())]=expected
        mask=cv2.erode(cv2.imread(row['mask'],0),np.ones((5,5),'uint8'));rgb=cv2.cvtColor(cv2.imread(row['source']),cv2.COLOR_BGR2RGB)/255.
        i=names.index(frame);pc=(Q-camera['centre'][i])@camera['R_world_cam'][i];uv=pc@camera['K'][i].T;uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9)
        valid=(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0]);take=np.flatnonzero(valid)
        xy=uv[take].astype(int);good=(mask[xy[:,1],xy[:,0]]>128)&(np.abs(rgb[xy[:,1],xy[:,0]]-color[take]).mean(1)<=error)
        votes[take[good]]+=1
    restored=ids[votes>=min_views];np.savez_compressed(output/'ground_rows.npz',source_rows=restored,votes=votes[votes>=min_views])
    report={'eligible':len(ids),'restored':len(restored),'views':len(rows),'ground_height_m':ground,
            'band_m':band,'max_sigma_m':maximum,'min_views':min_views,'max_rgb_error':error,
            'scope':'Near the observed ground plane, originally trained/drawn rows only; no task-size truth. Appearance restoration still requires novel-view review.'}
    (output/'ground_completion.json').write_text(json.dumps(report,indent=2)+'\n')
    return full[restored],report,inputs


def validate(session, specification, assets):
    cfg=specification.get('splat_context')
    if cfg is None:return None
    if not isinstance(cfg,dict) or not {'asset','evidence'}<=set(cfg) or set(cfg)-{'asset','evidence','native_light_gain','background_color_linear'}:raise ValueError('Splat context needs declared asset/evidence IDs')
    if any(cfg[k] not in assets for k in ['asset','evidence']):raise ValueError('Declare both context assets in environment.json assets')
    gain=cfg.get('native_light_gain',1.)
    if isinstance(gain,bool) or not isinstance(gain,(int,float)) or not np.isfinite(gain) or not .05<=gain<=100:raise ValueError('Finite bounded native light calibration gain required')
    color=np.asarray(cfg.get('background_color_linear',[0.,0.,0.]),float)
    if color.shape!=(3,) or not np.isfinite(color).all() or np.any((color<0)|(color>1)):
        raise ValueError('Background display color needs three finite linear values in0..1')
    proof=json.loads(Path(assets[cfg['evidence']]).read_text())
    if proof.get('version') not in (1,2) or digest(assets[cfg['asset']])!=proof['asset_sha256']:raise ValueError('Observed context asset changed')
    camera=session/'parsing/posed_frames/cameras.npz'
    if digest(camera)!=proof['camera_sha256'] or proof['protected_task_state']!=protected_state(session):
        raise ValueError('Observed context belongs to different capture calibration/task state')
    if proof['parameters_sha256']!=parameter_digest(specification['parameters']):raise ValueError('Reprepare context after changing native geometry parameters')
    for source,expected in proof['source_sha256'].items():
        if digest(source)!=expected:raise ValueError('Observed context source changed: '+source)
    if proof.get('version')==2:
        from calibrated_scene_reconstruction.scene.background.appearance_ownership import specification as ownership_specification
        # Agent parameter schemas are arbitrary. Bind actual exported entities,
        # rather than requiring the model to expose an `entities` parameter.
        entities=set(proof.get('background_entities',[])) or ({e['id'] for e in specification['parameters'].get('entities',[])}|{r['entity'] for r in specification['completion']['surfaces']})
        ownership_specification(proof.get('appearance_ownership'),entities,
                                proof['replaced_entities'],proof['gap_fill_entities'],specification['completion'])
        for source,expected in proof['input_sha256'].items():
            if digest(source)!=expected:raise ValueError('Observed context selection input changed: '+source)
        if proof.get('source_asset') and proof['input_sha256'].get(proof['source_asset'])!=proof.get('source_asset_sha256'):
            raise ValueError('Original captured source must be bound by selection inputs')
        for masks in proof.get('source_background_scope',{}).values():
            if not masks or any(proof['input_sha256'].get(p)!=h for p,h in masks.items()):
                raise ValueError('Fixed source background masks must be bound by selection inputs')
        expected_masks={k for k,v in proof['appearance_ownership'].items() if v['role']=='gap_fill'}
        if set(proof.get('gap_masks',{}))!=expected_masks:raise ValueError('Gap fillers need bound GS coverage masks')
        for entity,declaration in proof['gap_masks'].items():
            if declaration.get('kind')=='solid_backing':
                if proof['appearance_ownership'][entity].get('mesh_coverage',{}).get('compositing')!='solid_backing':
                    raise ValueError('Solid backing must match its declared native completion policy')
                continue
            payloads=declaration.get('meshes',{}).values() if declaration.get('kind') in ('source_crop','surface_atlas') else [declaration]
            for mask in payloads:
                if mask.get('uv_path') and digest(mask['uv_path'])!=mask['uv_sha256']:raise ValueError('GS crop UV payload changed')
                if digest(mask['path'])!=mask['sha256']:raise ValueError('GS coverage mask changed')
                if mask.get('radiance') and digest(mask['radiance']['path'])!=mask['radiance']['sha256']:
                    raise ValueError('Gap radiance changed')
    matrix=np.asarray(proof['matrix'],float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]) or np.linalg.det(matrix[:3,:3])<=0:
        raise ValueError('Finite proper context coordinate transform required')
    if specification['render']['mode']!='rt':raise ValueError('NuRec observed context requires RT rendering')
    return {'asset':str(Path(assets[cfg['asset']]).resolve()),'asset_sha256':proof['asset_sha256'],
            'evidence_sha256':digest(assets[cfg['evidence']]),'matrix':proof['matrix'],
            'native_entities':proof['native_entities'],'source_layer_data':proof['source_layer_data'],'native_light_gain':gain,
            'background_color_linear':color.tolist(),
            'appearance_ownership':proof.get('appearance_ownership'), 'gap_masks':proof.get('gap_masks',{}),'appearance_plan':proof.get('appearance_plan'),
            'source_asset':proof.get('source_asset'),'source_asset_sha256':proof.get('source_asset_sha256'),
            'source_background_scope':proof.get('source_background_scope',{}),
            'scope':'Same-capture GS appearance plus native geometry/physics; no extra task truth or automatic photorealism approval.'}


def capture_reference(session,source_session,config):
    """Keep a fixed original GS reference across background retraining."""
    reference=Session(resolve(config['reference_session'])) if config.get('reference_session') else source_session
    paths=[s/'parsing/posed_frames/cameras.npz' for s in (session,source_session,reference)]
    inputs={str(p.resolve()):digest(p) for p in paths}
    if len(set(inputs.values()))!=1:raise ValueError('GS source and original reference require identical calibrated cameras')
    for name in np.load(paths[0])['names']:
        images={str((s/'splat/data/images'/str(name)).resolve()) for s in (session,source_session,reference)}
        values={p:digest(p) for p in images}
        if len(set(values.values()))!=1:raise ValueError('GS source and original reference capture pixels differ: '+str(name))
        inputs.update(values)
    return reference,inputs


def prepare(session,source_session,config,geometry_job,output):
    from plyfile import PlyData,PlyElement
    from pxr import Usd
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    cfg=json.loads(Path(config).read_text());out=Path(output).resolve()
    projection_support=cfg.get('projection_support','full_footprint')
    if projection_support not in ('full_footprint','render_candidate'):
        raise ValueError('Choose full_footprint or an explicitly unapproved render_candidate projection policy')
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Context output belongs in this session')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config,out/'config.json')
    camera=session/'parsing/posed_frames/cameras.npz'
    if digest(camera)!=digest(source_session/'parsing/posed_frames/cameras.npz'):raise ValueError('Context must use the exact same calibrated capture')
    native=session.config('environment.json');job=json.loads(Path(geometry_job).read_text())
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    verify(session)
    if (job.get('native_manifest_sha256')!=digest(session/'scene/environment/twin/current.json')
            or job.get('geometry_sha256')!=digest(job['geometry'])):
        raise ValueError('Context selection needs geometry bound to the current native build')
    names=[label(name) for name in cfg['native_entities']]
    volumes=[label(name) for name in cfg.get('volume_entities',[])]
    if len(set(volumes))!=len(volumes) or not set(volumes)<=set(names):raise ValueError('Volume replacements must be distinct selected native entities')
    margins=cfg.get('surface_margins_m',{})
    if not isinstance(margins,dict) or not set(margins)<=set(names) or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not 0<v<=.3 for v in margins.values()):
        raise ValueError('Small per-entity replacement margins require selected native entities')
    fills=[label(name) for name in cfg.get('gap_fill_entities',[])]
    available={row['entity'] for row in job['parts'] if not row['external']}
    if not names+fills or len(set(names+fills))!=len(names+fills) or not set(names+fills)<=available:raise ValueError('Select distinct existing background entities only')
    from calibrated_scene_reconstruction.scene.background.appearance_ownership import specification as ownership_specification, plane_gap_mask
    ownership=ownership_specification(cfg.get('appearance_ownership'),available,names,fills,native['completion'])
    plan=None;plan_inputs={}
    if cfg.get('appearance_plan'):
        from calibrated_scene_reconstruction.scene.background.appearance_plan import validate as validate_plan
        plan,plan_inputs=validate_plan(session,cfg['appearance_plan'],ownership)
    cloud=source_session/'splat/export/background.ply';v=PlyData.read(str(cloud))['vertex'].data
    scale,R,t=sim3(source_session);P=scale*np.c_[v['x'],v['y'],v['z']].astype(float)@R.T+t
    sigma=scale*np.exp(np.c_[v['scale_0'],v['scale_1'],v['scale_2']].astype(float)).max(1)
    from scipy.spatial.transform import Rotation
    orientations=R@Rotation.from_quat(np.c_[v['rot_1'],v['rot_2'],v['rot_3'],v['rot_0']]).as_matrix()
    world_axes=orientations*(scale*np.exp(np.c_[v['scale_0'],v['scale_1'],v['scale_2']]))[:,None,:]
    # Per-kernel 3-sigma rejection can destroy a trained field's occlusion
    # closure. This explicit candidate policy tests centre-supported primitives
    # together in the renderer; it is never proof that their tails are reliable.
    # Existing source preservation, composition and novel-view gates still apply.
    support_axes=world_axes if projection_support=='full_footprint' else None
    inputs=[Path(config),Path(geometry_job),Path(job['geometry']),cloud,camera,
            source_session/'splat/align.json',source_session/'scene/splat/background.usdz',Path(__file__)]
    worker=Path(__file__).with_name('observed_context_worker.py');inputs.append(worker)
    inputs.append(Path(__file__).with_name('appearance_ownership.py'))
    if cfg.get('segmentation'):inputs += [Path(cfg['segmentation']),Path(cfg['segmentation_world'])]
    input_sha={str(p.resolve()):digest(p) for p in inputs}
    reference,reference_inputs=capture_reference(session,source_session,cfg)
    input_sha.update(reference_inputs)
    input_sha.update(plan_inputs)
    reference_asset=reference/'scene/splat/background.usdz'
    input_sha[str(reference_asset.resolve())]=digest(reference_asset)
    background_scope={}
    for frame in np.load(camera)['names']:
        stem=Path(str(frame)).stem;base=reference/'splat/data/images'
        zone=base/(stem+'_zone.png')
        if not zone.is_file():raise ValueError('Original capture needs fixed task-zone projection masks for background QA')
        paths=[zone,*[base/(stem+suffix) for suffix in ['_sam.png','_mask.png'] if (base/(stem+suffix)).is_file()]]
        background_scope[str(frame)]={str(p.resolve()):digest(p) for p in paths}
        input_sha.update(background_scope[str(frame)])
    source_sha={str(p.resolve()):digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')}
    np.savez_compressed(out/'cloud_world.npz',points=P,sigma=sigma)
    selection={'geometry':job['geometry'],'parts':job['parts'],'cloud':str(out/'cloud_world.npz'),
               'entities':names,'volume_entities':volumes,'margin_m':cfg.get('surface_margin_m',.055),'margins_m':margins,'output':str(out/'near_native.npy')}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');shutil.copyfile(worker,out/'worker.py')
    with (out/'selection.log').open('w') as log:
        subprocess.run(['tools/run_blender.sh','--python',str(out/'worker.py'),'--',str(out/'selection.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
    remove=np.load(out/'near_native.npy');semantic=np.zeros(len(v),bool);preserved=0
    if cfg.get('segmentation'):
        seg=Path(cfg['segmentation']);world=Path(cfg['segmentation_world']);Z=np.load(world)
        if Z['X'].shape!=P.shape or not np.allclose(Z['X'],P,atol=2e-6):raise ValueError('Semantic row IDs do not match the source cloud')
        claims=np.load(seg)['thing']
        remove,semantic,preserved=replacement_mask(remove,claims,cfg.get('replace_claim_ids',[]),cfg.get('preserve_claim_ids',[]))
        unassigned_keep=np.zeros(len(claims),bool);unassigned_report=None
        background_regions=cfg.get('background_capture_regions')
        if background_regions or cfg.get('unassigned_capture_regions'):
            if plan is None:raise ValueError('Unassigned captured appearance requires an explicit routing plan')
            from calibrated_scene_reconstruction.scene.background.gs_regions import select as select_regions
            unassigned_path=Path(background_regions or cfg['unassigned_capture_regions']);record=json.loads(unassigned_path.read_text())
            if (record.get('parameters_sha256')!=parameter_digest(native['parameters'])
                    or record.get('source_asset')!=str(reference_asset.resolve())
                    or record.get('source_asset_sha256')!=digest(reference_asset)):
                raise ValueError('Unassigned capture evidence must bind this geometry and original GS field')
            opacity=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-80,80)))
            if background_regions:
                if record.get('claim_scope')!='outside_known_furniture':raise ValueError('Background support needs its explicit furniture/task exclusions')
                unassigned_keep,unassigned_report,unassigned_inputs=select_regions(session,unassigned_path,P,opacity,np.zeros(len(claims),int),np.load(camera),{'background_capture':0},support_axes)
                # Instance labels are proposals, not authority to erase a
                # repeatedly observed surface outside the excluded furniture.
                remove &= ~unassigned_keep
                ids,counts=np.unique(claims[unassigned_keep],return_counts=True)
                unassigned_report['source_claims_supported']={str(i):int(n) for i,n in zip(ids,counts)}
            else:
                unassigned_keep,unassigned_report,unassigned_inputs=select_regions(session,unassigned_path,P,opacity,claims,np.load(camera),{'unassigned_capture':-1},support_axes)
            input_sha.update(unassigned_inputs)
        if cfg.get('retain_claim_ids') is not None:
            from calibrated_scene_reconstruction.scene.background.appearance_plan import semantic_keep
            if plan is None:raise ValueError('Semantic GS retention requires an image-backed appearance plan')
            approved={e['source_instance'] for e in native['parameters']['entities'] if e.get('source_instance') is not None and plan['routes'][e['id']]['route'] in ('gs','hybrid')}
            if set(cfg['retain_claim_ids'])!=approved:raise ValueError('Retained GS claims must exactly match reviewed GS owners')
            stuff_ids=cfg.get('retain_stuff_ids',[])
            approved_stuff={e['source_stuff'] for e in native['parameters']['entities'] if e.get('source_stuff') is not None and plan['routes'][e['id']]['route'] in ('gs','hybrid')}
            if (any(type(i)is not int or i<0 for i in stuff_ids) or len(set(stuff_ids))!=len(stuff_ids)
                    or set(stuff_ids)!=approved_stuff):raise ValueError('Retained GS surfaces must exactly match reviewed surface owners')
            keep=semantic_keep(claims,cfg['retain_claim_ids']) if cfg['retain_claim_ids'] else np.zeros(len(claims),bool)
            if stuff_ids:
                stuff=np.load(seg)['stuff'];keep|=(claims<0)&np.isin(stuff,stuff_ids)
            remove |= ~(keep|unassigned_keep)
        regional_report=None
        if cfg.get('capture_regions'):
            if plan is None:raise ValueError('Regional GS requires an image-backed appearance plan')
            from calibrated_scene_reconstruction.scene.background.gs_regions import select as select_regions
            owners={e['id']:(e['source_instance'] if e.get('source_instance') is not None else -2-e['source_stuff']) for e in native['parameters']['entities'] if (e.get('source_instance') is not None or e.get('source_stuff') is not None) and plan['routes'][e['id']]['route'] in ('gs','hybrid')}
            region_claims=claims.copy();stuff=np.load(seg)['stuff'];surface=(claims<0)&(stuff>=0)
            region_claims[surface]=-2-stuff[surface]
            opacity=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-80,80)))
            regional_keep,regional_report,regional_inputs=select_regions(session,cfg['capture_regions'],P,opacity,region_claims,np.load(camera),owners,support_axes)
            remove |= ~(regional_keep|unassigned_keep);input_sha.update(regional_inputs)
    elif cfg.get('replace_claim_ids') or cfg.get('preserve_claim_ids'):
        raise ValueError('Semantic claim selection requires a matching segmentation')
    else:
        regional_report=None
        if cfg.get('capture_regions'):raise ValueError('Regional GS requires semantic source associations')
    if not cfg.get('segmentation'):
        unassigned_report=None
        if cfg.get('unassigned_capture_regions') or cfg.get('background_capture_regions'):raise ValueError('Captured background selection needs exact source semantic rows')
    retained=v[~remove];ground_report=None
    if cfg.get('ground_completion'):
        restored,ground_report,ground_inputs=complete_ground(session,source_session,cfg['ground_completion'],job,v,out)
        input_sha.update(ground_inputs);retained=np.concatenate([retained,restored])
    quality_report=None
    if cfg.get('quality_filter') is not None or native.get('completion'):
        points=scale*np.c_[retained['x'],retained['y'],retained['z']].astype(float)@R.T+t
        widths=scale*np.exp(np.c_[retained['scale_0'],retained['scale_1'],retained['scale_2']].astype(float)).max(1)
        cameras=np.load(camera)
        quality_cfg=cfg.get('quality_filter') or {};surface_distance=None;floor_report=None
        if not isinstance(quality_cfg,dict):raise ValueError('Context quality_filter must be a settings dictionary')
        quality_cfg=dict(quality_cfg)
        if cfg.get('segmentation') and 'floor_below_tolerance_m' not in quality_cfg:
            segmentation=np.load(cfg['segmentation'])
            if {'stuff','seen'}<=set(segmentation.files):
                floor_report=floor_uncertainty(P,segmentation['stuff'],segmentation['seen'],native['completion']['floor_z'])
                quality_cfg['floor_below_tolerance_m']=floor_report['tolerance_m']
        keep,quality_report,quality_inputs=clean_context(retained,points,widths,quality_cfg,native.get('completion'),cameras,session/'splat/data/images',surface_distance,world_rotation=R)
        quality_report['observed_floor_uncertainty']=floor_report
        input_sha.update(quality_inputs);np.savez_compressed(out/'quality_rows.npz',kept=keep)
        retained=retained[keep];(out/'quality_filter.json').write_text(json.dumps(quality_report,indent=2)+'\n')
    protected_points=scale*np.c_[retained['x'],retained['y'],retained['z']].astype(float)@R.T+t
    protected_mesh,protected_parts,protected_inputs,robot_report=protected_geometry(job,cfg.get('robot_exclusion_margin_m',.05))
    input_sha.update(protected_inputs)
    protected_remove,protected_report=protected_overlap(retained,protected_points,scale,R,protected_mesh,protected_parts)
    protected_report['frozen_robot']=robot_report
    np.savez_compressed(out/'protected_rows.npz',removed=protected_remove)
    retained=opacity_adjustment(retained[~protected_remove],cfg.get('opacity_gain',1.))
    points=scale*np.c_[retained['x'],retained['y'],retained['z']].astype(float)@R.T+t
    widths=scale*np.exp(np.c_[retained['scale_0'],retained['scale_1'],retained['scale_2']].astype(float)).max(1)
    opacity=1/(1+np.exp(-np.clip(retained['opacity'].astype(float),-80,80)))
    gap_masks={entity:plane_gap_mask(points,widths,row['plane'],out/(entity+'_gaps.png'),opacity)
               for entity,row in ownership.items() if row['role']=='gap_fill' and 'plane' in row}
    for entity,row in ownership.items():
        if row['role']=='gap_fill' and 'source_crop' in row:
            from calibrated_scene_reconstruction.scene.background.crop_gaps import prepare as crop_gap
            gap_masks[entity],crop_inputs=crop_gap(source_session,job,entity,row['source_crop'],out)
            input_sha.update(crop_inputs)
        if row['role']=='gap_fill' and 'mesh_coverage' in row:
            if row['mesh_coverage'].get('compositing')=='captured_projection':
                from calibrated_scene_reconstruction.scene.background.capture_gaps import bake as captured_coverage
                if not cfg.get('projected_coverage'):raise ValueError('Projected gaps require actual retained-field render evidence')
                gap_masks[entity],coverage_inputs=captured_coverage(session,job,entity,row['mesh_coverage'],retained,points,cfg['projected_coverage'],out)
            else:
                from calibrated_scene_reconstruction.scene.background.crop_gaps import prepare_coverage
                gap_masks[entity],coverage_inputs=prepare_coverage(job,entity,row['mesh_coverage'],points,widths,opacity,out)
            input_sha.update(coverage_inputs)
    for entity,mask in gap_masks.items():
        if mask.get('mesh_parts'):
            components={name for row in job['parts'] if row['entity']==entity for name in row['path'].split('/')}
            if not set(mask['mesh_parts'])<=components:raise ValueError('Gap binding names absent native mesh parts: '+entity)
    from calibrated_scene_reconstruction.scene.background.appearance_ownership import bind_planar_radiance
    for entity,row in cfg.get('gap_radiance',{}).items():
        if entity not in gap_masks:raise ValueError('Capture radiance is restricted to declared gap planes')
        radiance,radiance_inputs=bind_planar_radiance(session,entity,ownership[entity]['plane'],row['evidence'],row['surface'])
        gap_masks[entity]['radiance']=radiance;input_sha.update(radiance_inputs)
    if any(mask.get('compositing')=='depth_backing' and not mask.get('radiance') for mask in gap_masks.values()):
        raise ValueError('Depth backing requires bound capture radiance')
    PlyData([PlyElement.describe(retained,'vertex')],text=False).write(str(out/'context.ply'))
    python=ROOT/'envs/splat/3dgrut/.venv/bin/python';asset=out/'context.usdz'
    with (out/'transcode.log').open('w') as log:
        subprocess.run([str(python),'-m','threedgrut.export.scripts.transcode',str(out/'context.ply'),'-o',str(asset),'-f','nurec'],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    matrix=np.eye(4);matrix[:3,:3]=scale*R;matrix[:3,3]=t
    stage=Usd.Stage.Open(str(source_session/'scene/splat/background.usdz'))
    manifest={'version':2,'asset_sha256':digest(asset),'camera_sha256':digest(camera),'matrix':matrix.tolist(),
              'projection_support':projection_support,
              'source_asset':str(reference_asset.resolve()),
              'source_asset_sha256':digest(reference_asset),
              'source_background_scope':background_scope,
              'parameters_sha256':parameter_digest(native['parameters']),'source_sha256':source_sha,
              'protected_task_state':protected_state(session),'native_entities':names+fills,
              'replaced_entities':names,'gap_fill_entities':fills,'background_entities':sorted(available),
              'appearance_ownership':ownership,'gap_masks':gap_masks,'appearance_plan':plan,
              'volume_entities':volumes,
              'source_layer_data':dict(stage.GetRootLayer().customLayerData),
              'source_gaussians':len(v),'removed':int(remove.sum()),'semantic_removed':int(semantic.sum()),'retained':len(retained),
              'neighbor_gaussians_preserved':preserved,
              'regional_selection':regional_report,
              'unassigned_selection':unassigned_report,
              'ground_completion':ground_report,
              'quality_filter':quality_report,
              'protected_overlap':protected_report,'opacity_gain':cfg.get('opacity_gain',1.),
              'input_sha256':input_sha,
              'photorealism_approved':False,'scope':'Captured appearance, not solid geometry. Native proxies retain mandatory physical QA. Hidden context still needs visual review.'}
    for path,expected in {**input_sha,**source_sha}.items():
        if digest(path)!=expected:raise ValueError('Context preparation input changed: '+path)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Observed context:',manifest['retained'],'retained;',manifest['removed'],'native replacements')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session');parser.add_argument('source_session')
    parser.add_argument('config');parser.add_argument('geometry_job');parser.add_argument('--output',required=True)
    a=parser.parse_args();prepare(Session(a.session),Session(a.source_session),a.config,a.geometry_job,a.output)


if __name__=='__main__':main()
