"""Render a checked contact trajectory with lid-interior and actual-surface cutaway views.

The cutaway copies the delivered triangles and discards front-facing half-space
triangles for inspection. It never changes source geometry, contacts or motion.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('session'); p.add_argument('usd'); p.add_argument('trace')
p.add_argument('--output', required=True); p.add_argument('--preview', action='store_true')
p.add_argument('--ablation-root', help='Optional completed on/off native contact diagnostics')
a = p.parse_args()
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.trace_evidence import checked  # noqa: E402
from calibrated_scene_reconstruction.scene.review.environment import digest  # noqa: E402
s = Session(a.session); proof = checked(s, a.usd, a.trace)
out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=False)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True, 'renderer': 'PathTracing', 'width': 800, 'height': 700})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux, Vt  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import VideoWriter, grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, look_at_quat, set_pose  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import body_path  # noqa: E402
from calibrated_scene_reconstruction.sim.receptacle import look_material  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import rot  # noqa: E402
from pxr import UsdShade  # noqa: E402

omni.usd.get_context().open_stage(str(Path(a.usd).resolve()))
stage = omni.usd.get_context().get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
omni.timeline.get_timeline_interface().stop()
configure_render(mode='pt', auto_exposure=False,spp=64,bounces=16)
values = np.load(a.trace)['values']; meta, phases = proof['metadata'], proof['phases']
if values.ndim != 2 or values.shape[1] != 24 or not np.isfinite(values).all():
    raise ValueError('Finite 24-column checked contact trace required')
names = [meta['base'], meta['moving']]
actors = [stage.GetPrimAtPath(body_path(stage, n+'/model')) for n in names]
cache = UsdGeom.XformCache(); copies = []
offset = np.array([-2., 0., 0.]); world_surfaces = []
for index, actor in enumerate(actors):
    if not actor or not actor.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Recorded body absent from delivery')
    UsdPhysics.RigidBodyAPI(actor).CreateRigidBodyEnabledAttr(False)
    body_inv = cache.GetLocalToWorldTransform(actor).GetInverse()
    material = look_material(stage, '/World/Inspection/Looks/body_'+str(index),
                             {'color': (.38, .50, .58) if index == 0 else (.89, .50, .66), 'roughness': .38})
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(str(actor.GetPath())+'/') or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim); counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if np.any(counts != 3):
            raise ValueError('Inspection requires original triangulated surfaces')
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)*body_inv)
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        local = (np.c_[points, np.ones(len(points))]@transform)[:, :3]
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        clone = UsdGeom.Mesh.Define(stage, '/World/Inspection/'+names[index]+'/'+prim.GetName())
        clone.CreateSubdivisionSchemeAttr('none'); clone.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(clone.GetPrim()).Bind(material)
        copies.append((index, local, faces, clone))
    if not any(item[0]==index for item in copies):
        raise ValueError('No actual delivered surface found below the checked rigid body')

def pose(prim, position, q):
    cache.Clear(); parent = cache.GetLocalToWorldTransform(prim.GetParent())
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
    matrix *= Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, position)))
    xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(matrix*parent.GetInverse())

def update(row, interior=False, full=False, clone_row=None):
    world_surfaces.clear()
    for i, actor in enumerate(actors):
        start = 1 if i == 0 else 8
        pose(actor, row[start:start+3], row[start+3:start+7])
    clone_row = row if clone_row is None else clone_row
    for i, local, faces, clone in copies:
        start = 1 if i == 0 else 8
        points = local@rot(clone_row[start+3:start+7]).T+clone_row[start:start+3]+offset
        world_surfaces.append((i, points, faces))
        kept = faces if interior or full else faces[points[faces].mean(1)[:, 0] <= clone_row[1]+offset[0]]
        clone.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
        clone.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(kept), 3, 'int32')))
        clone.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(kept.astype('int32').ravel()))
        clone.CreateVisibilityAttr('invisible' if interior and i == 0 else 'inherited')

W, H = 800, 700
K = np.array([[1300., 0, W/2], [0, 1300., H/2], [0, 0, 1.]])
base_position = values[0, 1:4]
neck = base_position+np.array([0., 0., float(np.mean(meta['thread_bands']['base']))])
neck_points = np.concatenate([local for i, local, _, _ in copies if i == 0])
neck_points = neck_points[(neck_points[:, 2] >= meta['thread_bands']['base'][0])
                          & (neck_points[:, 2] <= meta['thread_bands']['base'][1])]
neck_radius = float(np.linalg.norm(neck_points[:, :2], axis=1).max())
whole_target = base_position+np.array([0., 0., .105])
whole = add_camera(stage, '/World/Cameras/thread_whole', K, W, H,
                   eye=whole_target+np.array([.35, -.36, .17]), target=whole_target)
detail_camera = add_camera(stage, '/World/Cameras/thread_section', K, W, H,
                           eye=neck+offset+np.array([.14, 0., .004]), target=neck+offset, near=.001)
annotators = [rgb_annotator(c.GetPath(), W, H) for c in [whole, detail_camera]]
# Inspection-only lighting reaches the open cut and the underside cavity.
for i, direction in enumerate([(1., -.3, -.4), (1., .5, .7)]):
    light = UsdLux.DistantLight.Define(stage, '/World/Inspection/light_'+str(i))
    light.CreateIntensityAttr(900. if i == 0 else 500.)
    light.CreateAngleAttr(15.)
    eye = neck+offset+np.array(direction)
    set_pose(light.GetPrim(), eye, look_at_quat(eye, neck+offset))
font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_file, 24); small = ImageFont.truetype(font_file, 20)
tiny = ImageFont.truetype(font_file, 16)

def section_image(row):
    """Intersect original triangles with the same display cut plane; no idealized teeth."""
    image = Image.new('RGB', (300, 400), (248, 248, 248)); draw = ImageDraw.Draw(image)
    plane = row[1]+offset[0]
    y0, y1 = -neck_radius-.0095, -neck_radius+.0045
    z0, z1 = np.array(meta['thread_bands']['base'])+row[3]+np.array([-.001, .001])
    scale = min(265/(y1-y0), 340/(z1-z0))
    for i, points, faces in world_surfaces:
        tri = points[faces]; d = tri[:, :, 0]-plane
        d[abs(d) < 1e-10] = 1e-10
        active = (d.min(1) < 0) & (d.max(1) > 0); tri, d = tri[active], d[active]
        hits = [[] for _ in range(len(tri))]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            ids = np.flatnonzero(d[:, a]*d[:, b] < 0)
            fraction = d[ids, a]/(d[ids, a]-d[ids, b])
            crossings = tri[ids, a]+fraction[:, None]*(tri[ids, b]-tri[ids, a])
            for index, point in zip(ids, crossings):
                hits[index].append((20+(point[1]-row[2]-offset[1]-y0)*scale, 365-(point[2]-z0)*scale))
        for line in hits:
            if len(line) == 2:
                draw.line(line, fill=(20, 110, 160) if i == 0 else (205, 50, 110), width=3)
    draw.rectangle((0, 0, 300, 30), fill=(248, 248, 248))
    draw.text((8, 5), 'Actual mesh section; same pose', font=tiny, fill=(25, 25, 25))
    draw.line((18, 381, 18+.001*scale, 381), fill=(30, 30, 30), width=3)
    draw.text((25+.001*scale, 372), '1 mm', font=tiny, fill=(25, 25, 25))
    return image
fps = 24
schedule = [('inside', 4), ('screw_on', 7), ('passive_retention', 2), ('pull_without_rotation', 2), ('screw_off', 7), ('release', 2)]
ablation, ablation_inputs = {}, {}
if a.ablation_root:
    for mode in ('on', 'off'):
        folder = Path(a.ablation_root)/('ablation_'+mode)
        report = json.loads((folder/'report.json').read_text())
        if (report['input_sha256'] != proof['input_sha256']
                or report['source_input_fingerprint'] != proof['source_input_fingerprint']
                or report['contact_enabled'] != (mode == 'on') or report['net_upward_pull_N'] != 1.
                or report['axial_pose_or_velocity_constraint'] is not False):
            raise ValueError('Comparison must use this delivery and the same unrestrained 1N pull')
        array = np.load(folder/'motion.npz')['values']
        if array.ndim != 2 or array.shape[1] != 15 or not np.isfinite(array).all():
            raise ValueError('Finite native ablation motion required')
        ablation[mode] = report, array
        for name in ('motion.npz', 'report.json'):
            file = folder/name; ablation_inputs[str(file.resolve())] = digest(file)
    if (ablation['on'][0]['source_closed_pose'] != ablation['off'][0]['source_closed_pose']
            or ablation['on'][0]['max_axial_change_m'] > .0008
            or ablation['off'][0]['final_axial_change_m'] < .005
            or ablation['on'][0]['max_contact_force_N'] < .001
            or ablation['off'][0]['max_contact_force_N'] > 1e-6):
        raise ValueError('Native contact counterfactual did not separate as expected')
    schedule.append(('ablation', 4))
labels = {'inside': 'Lid interior: actual delivered thread surface', 'screw_on': 'Tightening: applied torque + gentle preload',
          'passive_retention': 'Hands off: applied forces and all cap controllers OFF',
          'pull_without_rotation': 'Pull test: 1 N upward; rotation held by bounded torque',
          'screw_off': 'Unscrewing: reverse torque + gentle lift', 'release': 'Disengaged: cap lifted clear of thread',
          'ablation': 'Control: identical closed pose and 1 N upward pull; only pair contact changed'}
jobs = []
for phase, seconds in schedule:
    lo, hi = phases.get(phase, (0, 1))
    indices = np.linspace(lo, hi-1, seconds*fps).astype(int)
    jobs.extend((phase, int(i), k/max(1, len(indices)-1)) for k, i in enumerate(indices))
if a.preview:
    jobs = [('inside', 0, .35), ('screw_on', int(phases['screw_on'][1]*.65), .6)]
    if ablation:
        jobs.append(('ablation', 0, 1.))
writer = None if a.preview else VideoWriter(out/'thread_detail.mp4', 1600, 900, fps)
posters = set()
try:
    for frame, (phase, index, progress) in enumerate(jobs):
        row = values[index]; interior = phase == 'inside'; clone_row = None
        if phase == 'ablation':
            pair = []
            for mode in ('on', 'off'):
                report, array = ablation[mode]
                sample = array[round(progress*(len(array)-1))]
                r = np.array(report['source_closed_pose']); r[0] = sample[0]
                r[8:15] = sample[1:8]; r[15] = sample[8]; r[17:23] = sample[9:15]
                r[23] = float((rot(r[4:8]).T@(r[8:11]-r[1:4]))[2]); pair.append(r)
            row, clone_row = pair
        update(row, interior, full=phase == 'ablation', clone_row=clone_row)
        if interior:
            centre = row[8:11]+offset+np.array([0., 0., .010])
            angle = -.5+progress*.9
            eye = centre+np.array([.035*np.cos(angle), .035*np.sin(angle), -.085])
            set_pose(detail_camera.GetPrim(), eye, look_at_quat(eye, centre))
        elif phase == 'ablation':
            target = base_position+np.array([0., 0., .135])
            eye = target+np.array([.42, -.43, .21])
            set_pose(whole.GetPrim(), eye, look_at_quat(eye, target))
            set_pose(detail_camera.GetPrim(), eye+offset, look_at_quat(eye+offset, target+offset))
        else:
            focus = float(np.clip((progress-.12)/.18, 0., 1.)) if phase == 'screw_on' else 1.
            target = neck+offset+np.array([0., -neck_radius*focus, .001])
            eye = target+np.array([.14-.085*focus, 0., .001])
            set_pose(detail_camera.GetPrim(), eye, look_at_quat(eye, target))
        for _ in range(35 if frame == 0 else 0):
            app.update()
        # Path tracing already completes64 samples with denoising per render.
        # Additional real-time subframes repeat that full work at the same pose.
        rep.orchestrator.step(rt_subframes=1, delta_time=0.)
        panels = [Image.fromarray(grab(an, render=app.update)) for an in annotators]
        canvas = Image.new('RGB', (1600, 900), (22, 24, 28)); draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), labels[phase], font=font, fill='white')
        if interior:
            image = panels[1].resize((800, 700)); canvas.paste(image, (800, 80)); canvas.paste(panels[0], (0, 80))
            draw.text((20, 49), 'Delivered rigid bodies', font=small, fill=(205, 215, 225))
            draw.text((820, 49), 'Underside inspection; display copy isolated', font=small, fill=(245, 180, 205))
        elif phase == 'ablation':
            for i, image in enumerate(panels):
                canvas.paste(image, (800*i, 80))
            draw.text((20, 49), 'Contact ON: original thread surfaces', font=small, fill=(160, 235, 190))
            draw.text((820, 49), 'Contact OFF: diagnostic only', font=small, fill=(255, 170, 140))
        else:
            for i, image in enumerate(panels):
                canvas.paste(image, (800*i, 80))
            canvas.paste(section_image(row), (1285, 365))
            note = 'Base clamped; cap guides OFF; gravity remains' if phase == 'passive_retention' else 'Base clamped; bounded lateral/tilt guidance; recorded motion'
            draw.text((20, 49), note, font=small, fill=(205, 215, 225))
            draw.text((820, 49), 'Cutaway: same poses; original surface triangles', font=small, fill=(245, 180, 205))
        if phase == 'ablation':
            initial = np.array(ablation['on'][0]['source_closed_pose'])
            draw.text((20, 790), 'Native time %.3f s  |  lift ON %.4f mm  |  lift OFF %.2f mm' % (row[0], (row[10]-initial[10])*1000, (clone_row[10]-initial[10])*1000), font=font, fill='white')
        else:
            draw.text((20, 790), 'Physics time %.2f s  |  cap contact force %.3f N  |  axial height %.2f mm' % (row[0], row[15], row[23]*1000), font=font, fill='white')
        draw.text((20, 830), 'Blue-gray: base. Pink: lid. No helical joint or prescribed axial trajectory.', font=small, fill=(205, 215, 225))
        draw.text((20, 862), 'Checked reconstruction hypothesis; hidden thread dimensions are not measured ground truth.', font=small, fill=(180, 188, 195))
        if writer:
            writer.write(np.asarray(canvas))
        if phase not in posters and (a.preview or progress >= .4):
            canvas.save(out/(phase+'.jpg')); panels[1].save(out/(phase+'_detail.jpg')); posters.add(phase)
        if frame % 72 == 0:
            print('Thread detail frame', frame, phase, flush=True)
finally:
    if writer:
        writer.close()
(out/'render.json').write_text(json.dumps({'replay_only': True, 'new_physical_test': False, 'renderer': 'path_tracing', 'refraction_bounces': 16,
    'frozen_source': proof['frozen'], 'source_input_fingerprint': proof['source_input_fingerprint'],
    'input_sha256': {**proof['input_sha256'], **ablation_inputs, str(Path(__file__).resolve()): digest(__file__)},
    'cutaway': 'Display copy uses original surface vertices/faces at recorded poses. Front half-space triangles hidden; flat inspection colors, no added teeth or motion.',
    'frames': len(jobs), 'fps': fps, 'source_geometry_modified': False}, indent=2)+'\n')
app.close()
