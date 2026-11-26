"""Render a recorded Isaac process trace; delivery requires independent approval.

Rendering replays native states; no new physics or observed response poses drive
this visualization. The accepted round and frozen source stay unchanged.
"""
import argparse
from pathlib import Path

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('recording');parser.add_argument('--output',required=True)
parser.add_argument('--width',type=int,default=1280)
parser.add_argument('--subframes',type=int,default=4)
parser.add_argument('--limit',type=int)
parser.add_argument('--camera-input',default='process_camera')
parser.add_argument('--video-input',default='video')
parser.add_argument('--round',dest='round_number',type=int,
                    help='Inspect this completed failed/rejected round with --diagnostic; never changes the active round.')
mode=parser.add_mutually_exclusive_group()
mode.add_argument('--candidate',action='store_true',help='Render an intact numerically passed candidate for review; does not approve it.')
mode.add_argument('--diagnostic',action='store_true',help='Render an intact failed native result with a failure label; never approves or delivers it.')
a=parser.parse_args()
from calibrated_scene_reconstruction.physics.process import workflow  # noqa: E402
from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require  # noqa: E402
from calibrated_scene_reconstruction.physics.process import render_gate  # noqa: E402

root=render_gate.select(a.recording,candidate=a.candidate,diagnostic=a.diagnostic,round_number=a.round_number)
initial_state=workflow.status(a.recording)['state']
plan=read(root/'plan.json')
result=read(root/'result.json')
trial=root/'trials'/f"{result['best']['index']:04d}";output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
inputs=read(root/'inputs.json');camera_path=inputs[a.camera_input]['path'];video_path=inputs[a.video_input]['path']
source_paths=[trial/'native_initial.usdc',trial/'native_states.npz',trial/'bindings.json',Path(camera_path),Path(video_path),root/'result.json',root/'inputs.json',root/'plan.json',root/'result_hashes.json',Path(__file__),Path(render_gate.__file__)]
if not(a.candidate or a.diagnostic):source_paths.append(root/'review.json')
source_hashes={str(p.resolve()):digest(p) for p in source_paths}
import cv2  # noqa: E402
source_capture=cv2.VideoCapture(video_path)
source_width=source_capture.get(cv2.CAP_PROP_FRAME_WIDTH);source_height=source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
source_capture.release();require(source_width>0 and source_height>0,'Cannot read original image size')
W=a.width;H=round(W*source_height/source_width/2)*2
require(W%2==0 and H>0 and 640<=W<=1920,'Even bounded output size required')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','width':W,'height':H,
                   'multi_gpu':False,'limit_cpu_threads':8})
try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import omni.timeline
    from PIL import Image
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,Vt,Semantics
    from scipy.spatial.transform import Rotation
    from calibrated_scene_reconstruction.sim.runtime import rgb_annotator,grab
    from calibrated_scene_reconstruction.sim.scene_stage import configure_render
    from calibrated_scene_reconstruction.sim.usd import add_camera
    from calibrated_scene_reconstruction.core.video import VideoWriter
    from calibrated_scene_reconstruction.physics.process.resources import watch
    monitor=watch(output)
    trace=dict(np.load(trial/'native_states.npz',allow_pickle=False));camera=dict(np.load(camera_path,allow_pickle=False))
    render_gate.validate_timing(trace['t'],camera['t'],plan['simulation']['dt'])
    omni.usd.get_context().open_stage(str(trial/'native_initial.usdc'))
    stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    omni.timeline.get_timeline_interface().stop()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        for attr in ['physxParticleSystem:particleSystemEnabled','physxParticle:particleEnabled','physxDeformable:deformableEnabled']:
            if prim.GetAttribute(attr):prim.GetAttribute(attr).Set(False)
    configure_render(mode='rt',auto_exposure=False)
    # Reset temporal caches for the recorded discrete states; never use motion
    # interpolation or temporal denoising to synthesize flexible geometry.
    import carb
    settings=carb.settings.get_settings();settings.set('/rtx/ecoMode/enabled',False)
    bindings=read(trial/'bindings.json')['bodies'];actors={};operations={}
    for name,binding in bindings.items():
        if binding['kind']=='static':continue
        prim=stage.GetPrimAtPath(binding['path']);actors[name]=(prim,binding['kind'])
        sem=Semantics.SemanticsAPI.Apply(prim,'ProcessResponse')
        sem.CreateSemanticTypeAttr('class');sem.CreateSemanticDataAttr('process_response')
        if binding['kind']=='rigid':
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();operations[name]=xf.AddTransformOp()
        elif prim.IsA(UsdGeom.Mesh):
            # Rest-state authored normals do not follow recorded cloth points.
            # Let Hydra derive the current surface normals in this render layer.
            prim.GetAttribute('normals').Block()
    K=camera['K'].copy();K[0]*=W/source_width;K[1]*=H/source_height
    camera_prim=add_camera(stage,'/ProcessVisualization/Camera',K,W,H,T_world_cam_cv=camera['T_world_cam'][0])
    rgb,seg=rgb_annotator(camera_prim.GetPath(),W,H,semantic=True)
    fps=1/np.median(np.diff(camera['t']));writer=VideoWriter(output/'simulation_replay.mp4',W,H,float(fps),crf=17)
    rgb_dir=output/'rgb';mask_dir=output/'masks';rgb_dir.mkdir();mask_dir.mkdir()
    count=min(len(trace['t']),a.limit or len(trace['t']));rendered=[]
    try:
        for i in range(count):
            xforms=UsdGeom.XformCache()
            for name,(prim,kind) in actors.items():
                value=trace[f'{name}.{i:05d}'];parent=np.asarray(xforms.GetLocalToWorldTransform(prim.GetParent())).T
                if kind=='rigid':
                    M=np.eye(4);M[:3,:3]=Rotation.from_quat(value[[4,5,6,3]]).as_matrix();M[:3,3]=value[:3]
                    operations[name].Set(Gf.Matrix4d((np.linalg.inv(parent)@M).T.tolist()))
                else:
                    M=np.asarray(xforms.GetLocalToWorldTransform(prim)).T
                    local=(value-M[:3,3])@np.linalg.inv(M[:3,:3]).T
                    prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(local,np.float32)))
                    UsdGeom.Mesh(prim).GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([local.min(0),local.max(0)],np.float32)))
            add_camera(stage,'/ProcessVisualization/Camera',K,W,H,T_world_cam_cv=camera['T_world_cam'][i])
            for _ in range(24 if i==0 else 2):app.update()
            rep.orchestrator.step(rt_subframes=a.subframes,delta_time=0.)
            image=grab(rgb,render=app.update)
            if a.diagnostic:
                image=image.copy()
                cv2.rectangle(image,(0,H-36),(W,H),(16,16,16),-1)
                cv2.putText(image,'FAILED CHECKS - native replay for diagnosis',(14,H-12),0,.6,(255,100,100),1,cv2.LINE_AA)
            raw=seg.get_data();labels=raw['info']['idToLabels']
            ids=[int(k) for k,v in labels.items() if v.get('class')=='process_response']
            require(ids,'Rendered response semantic labels are missing')
            mask=np.isin(np.asarray(raw['data']).reshape(H,W),ids).astype(np.uint8)*255
            require(mask.any(),'Rendered response disappeared')
            Image.fromarray(image).save(rgb_dir/f'{i:05d}.png')
            Image.fromarray(mask).save(mask_dir/f'{i:05d}.png');writer.write(image)
            if i%15==0:print(f'[process.render] {i+1}/{count}',flush=True)
            rendered.append({'frame':i,'source_t':float(camera['t'][i]),'native_t':float(trace['t'][i]),'mask_pixels':int((mask>0).sum())})
    finally:writer.close()
    for path,expected in source_hashes.items():require(digest(path)==expected,'Render source changed: '+path)
    require(render_gate.select(a.recording,candidate=a.candidate,diagnostic=a.diagnostic,
                              round_number=a.round_number)==root,'Selected process changed during render')
    if a.round_number is None:
        state=workflow.status(a.recording)
        expected=initial_state if a.diagnostic else 'awaiting_review'
        require(state['state']==expected if a.candidate or a.diagnostic else state['passed'],'Process approval state changed during render')
    write(output/'render.json',{'recording':str(Path(a.recording).resolve()),'round':str(root),'fps':float(fps),'width':W,'height':H,
        'source_width':int(source_width),'source_height':int(source_height),
        'frames':rendered,'sha256':source_hashes,'renderer':'Isaac Sim 5.1 RTX','new_physics_run':False,
        'candidate_for_review':a.candidate,'diagnostic_failed_result':a.diagnostic,
        'historical_round':a.round_number is not None,'source_video':str(Path(video_path).resolve()),
        'scope':'Actual native rigid/PBD states and recorded camera rendered on the recorded initial USD; no pose or shape interpolation. Render-session dynamics disabled; frozen inputs unchanged. '+
                ('Failed native result for diagnosis; NOT approved.' if a.diagnostic else
                 'Numerically passed candidate awaiting independent review; this render does not approve the process.' if a.candidate else 'Independently accepted process.')})
    monitor.set();print(str(output/'simulation_replay.mp4'),flush=True)
finally:app.close()
