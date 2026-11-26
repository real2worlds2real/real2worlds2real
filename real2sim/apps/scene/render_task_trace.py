"""Visualize a checked native-contact trace on the delivered USD/Z asset.

This is explicitly a render replay of recorded physics, not a new physical test.
The delivered package and the source trace remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('session')
p.add_argument('usd')
p.add_argument('trace', help='checked motion.npz, with phases.json and trace.json beside it')
p.add_argument('--output', required=True)
p.add_argument('--seconds', type=float, default=16.)
p.add_argument('--fps', type=int, default=24)
a = p.parse_args()
if not 4 <= a.seconds <= 120 or not 12 <= a.fps <= 60:
    p.error('Use bounded video duration and frame rate')

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.mechanisms import fingerprint  # noqa: E402

s = Session(a.session)
trace, usd, output = Path(a.trace).resolve(), Path(a.usd).resolve(), Path(a.output).resolve()
qa = s.read_json('qa/task_check.json')
if qa.get('pass') is not True or qa.get('input_fingerprint') != fingerprint(s):
    raise ValueError('Render only current passing task evidence')
package = s.read_json(f'qa/package_{usd.stem}.json')
if (package.get('pass') is not True or package.get('passive') is not True
        or package.get('source_input_fingerprint') != qa['input_fingerprint']
        or package.get('asset_sha256') != hashlib.sha256(usd.read_bytes()).hexdigest()):
    raise ValueError('Render only the reopened, current audited delivery package')
proofs = {**qa.get('mechanism_evidence_sha256', {}), **qa.get('sensitivity_evidence_sha256', {})}
relative = str(trace.relative_to(s.dir.resolve()))
if hashlib.sha256(trace.read_bytes()).hexdigest() != proofs.get(relative):
    raise ValueError('Trace is not bound to the current passing task result')
metadata = json.loads(trace.with_name('trace.json').read_text())
phases = json.loads(trace.with_name('phases.json').read_text())
W, H = 1280, 720

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'width': W, 'height': H})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

from calibrated_scene_reconstruction.sim.runtime import VideoWriter, grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, look_at_quat, set_pose  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import body_path  # noqa: E402

omni.usd.get_context().open_stage(str(usd))
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
omni.timeline.get_timeline_interface().stop()
configure_render(mode='rt', auto_exposure=False)
with np.load(trace) as data:
    values = data['values']
if not len(values) or values.ndim != 2 or values.shape[1] != 24 or not np.isfinite(values).all():
    raise ValueError('Unsupported or nonfinite native contact trace')


def actor(name):
    ref = name if '/' in name else name+'/model'
    prim = stage.GetPrimAtPath(body_path(stage, ref))
    if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Trace body absent from delivered USD: '+name)
    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    return prim


base, moving = actor(metadata['base']), actor(metadata['moving'])
cache = UsdGeom.XformCache()


def pose(prim, position, q):
    cache.Clear()
    parent = cache.GetLocalToWorldTransform(prim.GetParent())
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
    matrix = matrix*Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, position)))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(matrix*parent.GetInverse())


bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(base).ComputeAlignedBox()
extent = np.array(bounds.GetMax())-np.array(bounds.GetMin())
height = float(extent.max())
target = values[0, 1:4]+np.array([0, 0, float(extent[2])*.7])
radius = max(.25, height*2.8)
K = np.array([[W/(2*np.tan(np.radians(22.5))), 0, W/2],
              [0, W/(2*np.tan(np.radians(22.5))), H/2], [0, 0, 1.]])
camera = add_camera(stage, '/World/Cameras/task_trace', K, W, H,
                    eye=target+np.array([radius, -radius, radius*.6]), target=target)
annotator = rgb_annotator(camera.GetPath(), W, H)
writer = VideoWriter(output, W, H, a.fps)
indices = np.linspace(0, len(values)-1, round(a.seconds*a.fps)).astype(int)
phase_posters = {name: int(np.argmin(abs(indices-(lo+hi-1)/2)))
                 for name, (lo, hi) in phases.items() if hi > lo}
try:
    for frame, index in enumerate(indices):
        row = values[index]
        pose(base, row[1:4], row[4:8])
        pose(moving, row[8:11], row[11:15])
        theta = -.9+.6*frame/max(1, len(indices)-1)
        eye = target+radius*np.array([np.cos(theta), np.sin(theta), .55])
        set_pose(camera.GetPrim(), eye, look_at_quat(eye, target))
        for _ in range(30 if frame == 0 else 2):
            app.update()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.)
        image = Image.fromarray(grab(annotator, render=app.update))
        phase = next((name for name, (lo, hi) in phases.items() if lo <= index < hi), '')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, W, 42), fill=(12, 12, 12))
        draw.text((12, 8), f'Isaac Sim 5.1 checked contact replay | {phase} | physics t={row[0]:.2f}s', fill='white')
        writer.write(np.asarray(image))
        if frame in (0, len(indices)//2, len(indices)-1):
            image.save(output.with_name(output.stem+f'_{frame:04d}.jpg'))
        for name, poster_frame in phase_posters.items():
            if frame == poster_frame:
                image.save(output.with_name(output.stem+'_'+name+'.jpg'))
finally:
    writer.close()
output.with_suffix('.json').write_text(json.dumps({'replay_only': True, 'usd': str(usd), 'trace': relative,
    'input_sha256': {str(usd): hashlib.sha256(usd.read_bytes()).hexdigest(), relative: proofs[relative]},
    'source_task_check': 'qa/task_check.json', 'frames': len(indices), 'fps': a.fps}, indent=2)+'\n')
print('saved '+str(output), flush=True)
app.close()
