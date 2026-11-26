"""Diagnostic native rendering of a prepared initial state; never a physics pass."""
import argparse
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument('usd');p.add_argument('camera');p.add_argument('--output',required=True)
p.add_argument('--frames',type=int,nargs='+',default=[0]);p.add_argument('--width',type=int,default=1280)
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
from calibrated_scene_reconstruction.physics.process.contract import digest,write  # noqa: E402
source={str(Path(f).resolve()):digest(f) for f in [a.usd,a.camera,__file__]}
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','width':a.width,'height':round(a.width*9/16),'multi_gpu':False,'limit_cpu_threads':8})
try:
 import numpy as np
 import omni.usd,omni.replicator.core as rep,omni.timeline
 from pxr import UsdGeom,UsdPhysics
 from PIL import Image
 from calibrated_scene_reconstruction.sim.scene_stage import configure_render
 from calibrated_scene_reconstruction.sim.runtime import rgb_annotator,grab
 from calibrated_scene_reconstruction.sim.usd import add_camera
 omni.usd.get_context().open_stage(str(Path(a.usd).resolve()));stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
 omni.timeline.get_timeline_interface().stop()
 for prim in stage.Traverse():
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
  for key in ['physxParticleSystem:particleSystemEnabled','physxParticle:particleEnabled','physxDeformable:deformableEnabled']:
   if prim.GetAttribute(key):prim.GetAttribute(key).Set(False)
  if 'ParticleCloth' in str(prim.GetMetadata('apiSchemas')):prim.GetAttribute('normals').Block()
 configure_render(mode='rt',auto_exposure=False)
 camera=np.load(a.camera);K=camera['K'].copy();W=a.width;H=round(W*9/16);K[0]*=W/1920;K[1]*=H/1080
 cam=add_camera(stage,'/InitialReview/Camera',K,W,H,T_world_cam_cv=camera['T_world_cam'][a.frames[0]])
 ann=rgb_annotator(cam.GetPath(),W,H)
 for i in a.frames:
  add_camera(stage,'/InitialReview/Camera',K,W,H,T_world_cam_cv=camera['T_world_cam'][i])
  for _ in range(24):app.update()
  rep.orchestrator.step(rt_subframes=4,delta_time=0.)
  Image.fromarray(grab(ann,render=app.update)).save(out/f'{i:05d}.png');print('frame',i,flush=True)
 write(out/'preview.json',{'diagnostic':True,'physics_stepped':False,'source':source,'frames':a.frames,'scope':'Prepared geometry and initial poses at source cameras. Not accepted/native motion evidence.'})
finally:app.close()
