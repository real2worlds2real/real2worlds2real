"""Render a static exported scene along a continuous path or capture-anchor arcs.

No reconstruction, geometry edits or robot operation. The actual native/GS USD
and its render profile are retained. Each shot moves out from, then returns to,
an existing registered review camera; cuts separate independent anchor shots.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('usd');p.add_argument('cameras');p.add_argument('--output',required=True)
p.add_argument('--seconds-per-view',type=float,default=6.)
p.add_argument('--fps',type=int,default=24);p.add_argument('--offset',type=float,default=.35)
p.add_argument('--subframes',type=int,default=4);p.add_argument('--renderer',choices=['rt','pt'],default='rt')
a=p.parse_args()
if not 2<=a.seconds_per_view<=30 or not 12<=a.fps<=60 or not 0<=a.offset<=.5 or not 1<=a.subframes<=32:p.error('Use bounded review duration, FPS, offset and samples')
usd=Path(a.usd).resolve();cameras=Path(a.cameras).resolve();manifest=json.loads(cameras.read_text());out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
W,H=manifest['width'],manifest['height']
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting' if a.renderer=='rt' else 'PathTracing','width':W,'height':H})
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image,ImageDraw  # noqa: E402
from pxr import Sdf,Usd,UsdGeom,UsdPhysics  # noqa: E402
from calibrated_scene_reconstruction.scene.review.cameras import novel_view  # noqa: E402
from calibrated_scene_reconstruction.sim.environment_stage import apply_environment_render  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import VideoWriter,grab,rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import fixed_exposure  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402


def digest(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


inputs={str(usd):digest(usd),str(cameras):digest(cameras),str(Path(__file__).resolve()):digest(__file__)}
inputs.update(manifest.get('input_sha256',{}))
for path,sha in inputs.items():
    if digest(path)!=sha:raise ValueError('Video input changed: '+path)
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.usd.schema.omni_nurec_types',True)
omni.usd.get_context().open_stage(str(usd));stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
omni.timeline.get_timeline_interface().stop()
metadata=dict(stage.GetRootLayer().customLayerData)
field=stage.GetPrimAtPath('/World/SplatBackground')
has_gs=bool(field and field.IsActive() and field.GetChildren()
            and UsdGeom.Imageable(field).ComputeVisibility()!='invisible')
native=stage.GetPrimAtPath('/World/Environment')
native_visible=sum(1 for prim in Usd.PrimRange(native)
                   if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility()!='invisible'
                   and not prim.GetAttribute('primvars:hideForCamera').Get()) if native else 0
backend='hybrid' if has_gs and native_visible else 'gs' if has_gs else 'blender' if native_visible else 'mesh'
backend_label={'hybrid':'Blender + GS environment','gs':'GS environment',
               'blender':'Blender mesh environment','mesh':'Mesh environment'}[backend]
if metadata.get('r2s2r_diagnostic'):raise ValueError('Do not present a diagnostic USD as the current scene video')
completion=metadata.get('r2s2r_environment_completion')
if metadata.get('r2s2r_environment_context') and not completion:
    raise ValueError('Hybrid presentation requires a completed enclosure and its actual-export audit')
if completion:
    from calibrated_scene_reconstruction.scene.background.completion import current as completed
    from calibrated_scene_reconstruction.scene.session import Session
    proof=completed(Session(completion['session']),completion['variant'],usd)
    inputs.update(proof['input_sha256'])
    pointer=Path(completion['session'])/'qa/environment_completion'/completion['variant']/'current.json'
    inputs[str(pointer.resolve())]=digest(pointer)
    record=Path(json.loads(pointer.read_text())['path']);inputs[str(record.resolve())]=digest(record)
for layer in stage.GetUsedLayers():
    if not layer.anonymous and Path(layer.realPath).is_file():inputs[str(Path(layer.realPath).resolve())]=digest(layer.realPath)
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        value=attr.Get()
        if isinstance(value,Sdf.AssetPath) and value.resolvedPath:
            path=Path(value.resolvedPath.split('[',1)[0])
            if path.is_file():inputs[str(path.resolve())]=digest(path)
configure_render(mode=a.renderer,auto_exposure=False)
if not apply_environment_render(stage):fixed_exposure(metadata.get('r2s2r_background_render'))
# Replicator initializes its timeline even for rendering. Keep all simulation
# disabled in the anonymous session layer; never author this into the asset.
disabled_bodies=0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False);disabled_bodies+=1
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.CreateAttribute('physxArticulation:articulationEnabled',Sdf.ValueTypeNames.Bool).Set(False)


def scene_transforms():
    cache=UsdGeom.XformCache();result={}
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/') and UsdGeom.Xformable(prim):
            result[str(prim.GetPath())]=np.asarray(cache.GetLocalToWorldTransform(prim),float)
    return result


continuous=bool(manifest.get('trajectory'))
if continuous:
    if manifest.get('fps')!=a.fps:raise ValueError('Trajectory and encoding FPS must match')
    anchors=[{'name':'Continuous free roam','source':'continuous_roam','source_size':manifest['source_size'],'K':manifest['K']}]
else:anchors=manifest['anchors']
before=scene_transforms();writer=VideoWriter(out/'stream.mp4',W,H,a.fps,crf=17);annotator=None;poses=[];posters=[]
np.savez_compressed(out/'source_transforms.npz',**before)
frames_per_view=round(a.seconds_per_view*a.fps)
try:
    for number,anchor in enumerate(anchors):
        K=np.asarray(anchor['K'],float).copy();K[0]*=W/anchor['source_size'][0];K[1]*=H/anchor['source_size'][1]
        if continuous:frames_per_view=len(manifest['trajectory'])
        else:
            base=anchor['views'][0];C=np.asarray(base['centre']);R=np.asarray(base['R_world_cam'])
        for frame in range(frames_per_view):
            t=frame/max(1,frames_per_view-1);offset=0. if continuous else a.offset*np.sin(np.pi*t)**2
            view=manifest['trajectory'][frame] if continuous else novel_view(C,R,anchor['look_distance_m'],float(offset))
            T=np.eye(4);T[:3,:3]=view['R_world_cam'];T[:3,3]=view['centre']
            camera=add_camera(stage,'/ReviewVideo/Camera',K,W,H,T_world_cam_cv=T)
            camera.CreateHorizontalApertureOffsetAttr(float((W/2-K[0,2])/K[0,0]*18.))
            camera.CreateVerticalApertureOffsetAttr(float((K[1,2]-H/2)/K[1,1]*18.))
            if annotator is None:annotator=rgb_annotator(camera.GetPath(),W,H)
            for _ in range(40 if frame==0 else 2):app.update()
            rep.orchestrator.step(rt_subframes=a.subframes,delta_time=0.)
            image=Image.fromarray(grab(annotator,render=app.update));draw=ImageDraw.Draw(image)
            label=anchor.get('label',anchor['name'])
            caption=f'Continuous free roam | {frame/a.fps:.1f} s | {backend_label}' if continuous else f'{label} | Free-view offset {offset:.2f} m | {backend_label}'
            draw.rectangle((0,0,W,27),fill=(18,18,18));draw.text((12,8),caption,fill='white')
            writer.write(np.asarray(image))
            poses.append({'shot':number,'frame':frame,'offset_m':float(offset),'centre':view['centre'],'R_world_cam':view['R_world_cam']})
            if (continuous and frame%max(1,frames_per_view//8)==0) or (not continuous and frame==frames_per_view//2):
                path=out/(f'roam_{frame:05d}.jpg' if continuous else Path(anchor['source']).stem+'_poster.jpg');image.save(path,quality=95);posters.append(path)
            if frame%(a.fps*3)==0:print(f'[review.video] segment {number+1}/{len(anchors)} frame {frame}/{frames_per_view}',flush=True)
finally:writer.close()
after=scene_transforms()
changes=[k for k in before if k not in after or not np.allclose(before[k],after[k],atol=1e-9,rtol=0)]
(out/'transform_audit.json').write_text(json.dumps({'changed':changes,'added':sorted(after.keys()-before.keys()),'disabled_bodies':disabled_bodies},indent=2)+'\n')
if changes:
    app.close()
    raise ValueError('Static scene transforms changed during camera-only video: '+str(changes[:5]))
for path,sha in inputs.items():
    if digest(path)!=sha:raise ValueError('Video input changed: '+path)
video=out/'current_environment.mp4'
subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out/'stream.mp4'),'-c','copy','-movflags','+faststart',str(video)],check=True)
(out/'camera_poses.json').write_text(json.dumps(poses)+'\n')
panel=Image.new('RGB',(W*2,H*((len(posters)+1)//2)))
for i,path in enumerate(posters):panel.paste(Image.open(path),((i%2)*W,(i//2)*H))
panel.save(out/'preview.jpg',quality=92)
(out/'video.json').write_text(json.dumps({'usd':str(usd),'camera_manifest':str(cameras),'video':str(video),'sha256':digest(video),
    'fps':a.fps,'frames':len(poses),'seconds':len(poses)/a.fps,'width':W,'height':H,'static_scene_preserved':True,
    'physics_stepped':False,'dynamics_disabled_in_session_layer':disabled_bodies,'environment':metadata.get('r2s2r_environment'),'environment_context':metadata.get('r2s2r_environment_context'),
    'input_sha256':inputs,'continuous_take':continuous,'camera_cuts':0 if continuous else len(anchors)-1,'path_stats':manifest.get('path_stats'),
    'background_representation':{'backend':backend,'gs_present':has_gs,'visible_native_environment_meshes':native_visible,
       'scope':'Actual composed representations, not a pixel-contribution or quality measurement.'},
    'completion_audited':bool(completion),
    'scope':'Actual static USD, camera-only free view; continuous trajectory without cuts.' if continuous else 'Actual static USD; shots cut between registered anchors.'},indent=2)+'\n')
print(video,flush=True);app.close()
