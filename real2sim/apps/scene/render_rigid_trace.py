"""Render exact recorded native rigid poses and source-indexed flexible nodes.

The renderer does not simulate, interpolate object poses, or approve physics.
Camera movement is independent of the recorded object motion.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest');args=parser.parse_args()
manifest_path=Path(args.manifest).resolve();manifest=json.loads(manifest_path.read_text())
out=Path(manifest['output']).resolve();out.mkdir(parents=True,exist_ok=False)
video_name=manifest.get('video_name','motion.mp4')
if Path(video_name).name!=video_name or not video_name.endswith('.mp4'):
    raise ValueError('Playback video must have a local MP4 filename')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
inputs={str(manifest_path):sha(manifest_path),**manifest['sha256']}
for path,expected in inputs.items():
    if sha(path)!=expected:raise ValueError('Recorded render input changed: '+path)
if not manifest.get('diagnostic',False):
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.scene.twin.reviews import task_result,delivery_package
    session=Session(manifest['session']);valid,reason=task_result(session,True)
    if not valid:raise ValueError('Current passing native task required: '+reason)
    package=session.dir/('qa/package_'+Path(manifest['usd']).stem+'.json')
    delivery_package(session,{'package_report':str(package.relative_to(session.dir))})
    proof=json.loads(package.read_text())
    if Path(proof['asset']).resolve()!=Path(manifest['usd']).resolve():raise ValueError('Playback asset differs from checked package')
    task=session.dir/'qa/task_check.json';result=json.loads(task.read_text())
    evidence={**result.get('mechanism_evidence_sha256',{}),**result.get('sensitivity_evidence_sha256',{})}
    for name in ['trace','native_result']:
        path=Path(manifest[name]).resolve();relative=str(path.relative_to(session.dir.resolve()))
        if sha(path)!=evidence.get(relative):raise ValueError('Trace is not bound to the current passing task')
    inputs.update({str(package.resolve()):sha(package),str(task.resolve()):sha(task)})

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
W,H=manifest.get('size',[1280,720]);fps=manifest.get('fps',30)
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','width':W,'height':H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image,ImageDraw  # noqa: E402
from pxr import Gf,Sdf,UsdGeom,UsdPhysics,Vt  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera,set_pose,look_at_quat  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import rgb_annotator,grab,VideoWriter  # noqa: E402

try:
    if not omni.usd.get_context().open_stage(str(Path(manifest['usd']).resolve())):
        raise ValueError('Cannot open assembled USD')
    stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    omni.timeline.get_timeline_interface().stop()
    configure_render(mode='rt',auto_exposure=False)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.CreateAttribute('physxArticulation:articulationEnabled',Sdf.ValueTypeNames.Bool).Set(False)
    native=json.loads(Path(manifest['native_result']).read_text())
    with np.load(manifest['trace']) as trace:
        poses=trace['poses'].copy();times=trace['times_s'].copy();names=trace['names'].tolist()
    if names!=native['names'] or poses.shape!=(len(times),len(names),7) or not np.isfinite(poses).all():
        raise ValueError('Trace body order or pose data differs from native record')
    paths=native['body_paths'];prims=[stage.GetPrimAtPath(path) for path in paths]
    if not all(prims):raise ValueError('Recorded physical bodies absent from scene')
    if len(times)<2 or not np.all(np.diff(times)>0):raise ValueError('Strictly increasing native time required')
    flexible=None;surfaces=[]
    flex=native.get('flexible_trace')
    if flex:
        # A mixed assembly must never be presented with static petals while
        # only its rigid carrier is replayed. Use the actual source-indexed
        # nodes and the exact rigid pose from that same native sample.
        node_path=Path(manifest['trace']).parent/flex['positions']
        source_path=Path(manifest['geometry'])
        for path in [node_path,source_path]:
            if str(path.resolve()) not in manifest['sha256']:
                raise ValueError('Mixed replay requires hash-bound nodes and source topology')
        if sha(node_path)!=flex['positions_sha256']:
            raise ValueError('Flexible nodes differ from native recording')
        flexible=np.load(node_path,mmap_mode='r')
        steps=np.asarray(flex['sample_steps'],int)
        if (steps.ndim!=1 or not len(steps) or steps[0]!=0 or steps[-1]!=len(times)-1
                or (np.diff(steps)<=0).any() or not np.isfinite(flexible).all()
                or len(flexible)!=len(steps)):
            raise ValueError('Complete finite source-indexed flexible trace required')
        with np.load(source_path) as geometry:
            links={p['name']:p for p in json.loads(str(geometry['links_json']))}
            faces=geometry['F'];vertices=geometry['V']
            offset=0
            for name in flex['names']:
                obj,link=name.split('/');part=links[link];start,count=flex['vertex_ranges'][name]
                prim=stage.GetPrimAtPath('/World/Objects/'+name+'/surface');mesh=UsdGeom.Mesh(prim)
                a,n,b,k=(part[key] for key in ['vertex_start','vertex_count','face_start','triangle_count'])
                if (not prim or part['kind']!='cloth' or start!=offset or count!=n
                        or len(mesh.GetPointsAttr().Get())!=count
                        or not np.array_equal(np.asarray(mesh.GetFaceVertexCountsAttr().Get()),np.full(k,3))
                        or not np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),faces[b:b+k]-a)):
                    raise ValueError('Replay source topology differs from the displayed flexible surface: '+name)
                if not manifest.get('diagnostic',False):
                    fix=session.scene_graph['dynamic_objects'][obj]['fix']
                    if sha(source_path)!=sha(fix['mesh']):
                        raise ValueError('Flexible source differs from the checked current candidate')
                mesh.GetNormalsAttr().Block()
                surfaces.append((mesh,start,count));offset+=count
            if flexible.shape[1:]!=(offset,3):raise ValueError('Unowned or missing flexible trace vertices')
        flexible_times=times[steps]
        requested=np.arange(0,times[-1]+1e-8,1/fps)
        node_indices=np.abs(flexible_times[:,None]-requested[None,:]).argmin(0)
        indices=steps[node_indices]
        if np.max(abs(times[indices]-requested))>np.max(np.diff(flexible_times))/2+1e-7:
            raise ValueError('Flexible playback sampling exceeds native trace resolution')
    else:
        if manifest.get('geometry'):raise ValueError('Flexible geometry supplied without a native node trace')
        requested=np.arange(0,times[-1]+1e-8,1/fps)
        indices=np.abs(times[:,None]-requested[None,:]).argmin(0)
        if np.max(abs(times[indices]-requested))>np.max(np.diff(times))/2+1e-7:
            raise ValueError('Playback time sampling differs from native recording')
    target=np.asarray(manifest['camera']['look'],float);start=np.asarray(manifest['camera']['eye'],float)
    orbit=np.radians(manifest['camera'].get('orbit_deg',0.))
    K=np.array([[W/(2*np.tan(np.radians(manifest['camera'].get('horizontal_fov_deg',48))/2)),0,W/2],
                [0,W/(2*np.tan(np.radians(manifest['camera'].get('horizontal_fov_deg',48))/2)),H/2],[0,0,1]])
    cam=add_camera(stage,'/Review/Camera',K,W,H,eye=start,target=target,near=.005)
    ann=rgb_annotator(cam.GetPath(),W,H);ops=[]
    for prim in prims:
        xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();ops.append(xf.AddTransformOp(opSuffix='nativeReplay'))
    writer=VideoWriter(out/video_name,W,H,fps=fps)
    snapshots={0,len(indices)//2,len(indices)-1}
    for frame,index in enumerate(indices):
        cache=UsdGeom.XformCache()
        for prim,op,pose in zip(prims,ops,poses[index]):
            position,q=pose[:3],pose[3:]
            world=Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))))
            world.SetTranslateOnly(Gf.Vec3d(*map(float,position)))
            parent=cache.GetLocalToWorldTransform(prim.GetParent())
            op.Set(world*parent.GetInverse());cache.Clear()
        if flexible is not None:
            for mesh,node_start,count in surfaces:
                matrix=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()));inverse=np.linalg.inv(matrix)
                world=np.asarray(flexible[node_indices[frame],node_start:node_start+count],float)
                local=world@inverse[:3,:3]+inverse[3,:3]
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(local,np.float32)))
                mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray([local.min(0),local.max(0)],np.float32)))
        angle=orbit*frame/max(len(indices)-1,1);c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        eye=target+R@(start-target);set_pose(cam.GetPrim(),eye,look_at_quat(eye,target))
        if frame==0:
            for _ in range(30):app.update()
        rep.orchestrator.step(rt_subframes=4,delta_time=0.)
        rgb=grab(ann,render=app.update);image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        probe=native.get('probe') or {};release=probe.get('start_s',0)+probe.get('duration_s',float('inf'))
        phase='Free settling (no tool force)' if times[index]>=release else manifest.get('actuation_label','Bounded-force motion')
        draw.rectangle((0,0,W,29),fill=(24,24,24));draw.text((10,8),f'Isaac Sim recorded physics | {times[index]:.2f}s | {phase}',fill='white')
        writer.write(np.asarray(image))
        if frame in snapshots:image.save(out/f'frame_{frame:04d}.jpg',quality=95)
        if frame%30==0:print(f'[render.rigid] {frame}/{len(indices)}',flush=True)
    writer.close()
    for path,expected in inputs.items():
        if sha(path)!=expected:raise ValueError('Input changed during recorded playback')
    (out/'render.json').write_text(json.dumps({'task_approved':False,
        'mode':'Exact native rigid poses and flexible nodes; no object interpolation' if flex else 'Exact native rigid pose replay; no object interpolation',
        'flexible_surfaces':len(surfaces),
        'source_indices':indices.tolist(),'source_times_s':times[indices].tolist(),'input_sha256':inputs,
        'video':video_name,'video_sha256':sha(out/video_name)},indent=2)+'\n')
except Exception:
    # SimulationApp.close may terminate Python before an exception propagates.
    # Preserve a visible failure instead of a misleading successful exit.
    import traceback
    (out/'render_failure.txt').write_text(traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    app.close()
