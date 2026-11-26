"""Native RTX transmission control, never scene appearance approval."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
args=parser.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'renderer':'PathTracing','width':512,'height':512})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom,UsdLux,UsdShade  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose,color_material,add_camera  # noqa: E402
from calibrated_scene_reconstruction.sim.object_materials import surface_material  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import rgb_annotator,grab  # noqa: E402

out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
try:
    stage=omni.usd.get_context().get_stage()
    def box(path,position,scale,material):
        cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
        set_pose(cube.GetPrim(),position,scale=scale)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        return cube
    for i in range(8):
        color=(.02,.03,.04) if i%2 else (.85,.2,.04)
        box('/World/Stripe'+str(i),[-.07+i*.02,.018,.1],[.02,.004,.16],
            color_material(stage,'/World/Looks/Stripe'+str(i),color,roughness=1.))
    props={'color':[.97,.98,.97],'roughness':.08,'metallic':0.,'opacity':1.,'ior':1.48,'transmission':.9}
    pane=box('/World/Pane',[0,-.008,.1],[.16,.002,.16],surface_material(stage,'/World/Looks/Pane',props))
    dome=UsdLux.DomeLight.Define(stage,'/World/Light');dome.CreateIntensityAttr(700.)
    configure_render(mode='pt',auto_exposure=False,spp=128,bounces=8)
    camera=add_camera(stage,'/World/Camera',np.array([[700,0,256],[0,700,256],[0,0,1]]),512,512,
                      eye=[0,-.5,.1],target=[0,0,.1])
    annotator=rgb_annotator(camera.GetPath(),512,512)
    images={}
    for name in ['reference','transmission','opaque']:
        pane.GetVisibilityAttr().Set('invisible' if name=='reference' else 'inherited')
        if name=='opaque':
            UsdShade.MaterialBindingAPI(pane.GetPrim()).Bind(surface_material(stage,'/World/Looks/Opaque',{**props,'transmission':0.}))
        for _ in range(30):app.update()
        rep.orchestrator.step(rt_subframes=64,delta_time=0.)
        images[name]=grab(annotator,render=app.update)
        Image.fromarray(images[name]).save(out/(name+'.png'))
    # Same pane and background; the clear material must preserve stripe contrast
    # whereas the opaque negative control must hide it. No scene-fit threshold.
    signals={name:float(im[235:277,155:357].astype(float).mean((0,2)).std()) for name,im in images.items()}
    result={'pass':bool(signals['reference']>10 and signals['transmission']>.3*signals['reference']
                         and signals['opaque']<.2*signals['reference']),
            'stripe_signal':signals,'synthetic_control':True,'not_scene_approval':True,
            'material':props,'renderer':'Isaac Sim5.1 RTX path tracing'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    stage.GetRootLayer().Export(str(out/'fixture.usda'))
    print('OPTICS_RESULT',json.dumps(result),flush=True)
finally:
    app.close()
