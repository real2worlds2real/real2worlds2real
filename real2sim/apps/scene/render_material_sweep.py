"""Compare native material proposals with one calibrated source view.

This is an unapplied appearance experiment. It never changes model geometry,
source masks, simulation inputs or a QA verdict. Variant inputs must already
exist on the material's native MDL shader.
"""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('session');p.add_argument('object');p.add_argument('--frame',required=True)
p.add_argument('--variants',required=True);p.add_argument('--output',required=True)
p.add_argument('--mode',choices=['rt','pt'],default='pt')
p.add_argument('--atlas',help='Optional source-conforming UV atlas; triangle positions must match')
p.add_argument('--smooth',action='store_true',help='Preview smooth rigid shading without changing triangles')
p.add_argument('--object-only',action='store_true',help='Static isolated object with neutral diagnostic lighting; no full scene or physics approval')
a=p.parse_args()
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'renderer':'PathTracing' if a.mode=='pt' else 'RaytracedLighting'})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image,ImageDraw  # noqa: E402
from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,UsdLux,UsdPhysics,Vt  # noqa: E402
import trimesh  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.blender.evidence import load  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.object_refine import _frame  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant,photo_goal  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene,prepare_background  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene,configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import add_background  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import rgb_annotator,grab  # noqa: E402

s=Session(a.session);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
try:
    z,frames,masks,windows=load(s,a.object)
    i=list(map(str,z['names'])).index(a.frame)
    k=list(frames).index(i);window=windows[k];mask=masks[k];H,W=mask.shape
    source=_frame(s,z,i,window,mask.shape);Image.fromarray(source).save(out/'source.png')
    K=z['K'][i].copy();x,y,_,_,factor=window
    K=np.array([[factor,0.,-x*factor],[0.,factor,-y*factor],[0.,0.,1.]])@K
    T=np.eye(4);T[:3,:3]=z['R_world_cam'][i];T[:3,3]=z['centre'][i]
    stage=omni.usd.get_context().get_stage()
    if a.object_only:
        from calibrated_scene_reconstruction.sim.mechanism_stage import add_links
        from calibrated_scene_reconstruction.sim.physics import physics_material
        from calibrated_scene_reconstruction.sim.usd import set_pose,add_mesh,color_material,quat_from_matrix
        from calibrated_scene_reconstruction.assets.object_pose import rotation
        spec=s.scene_graph['dynamic_objects'][a.object];fix=spec['fix'];kit=asset(fix)
        material=physics_material(stage,'/World/Looks/diagnostic_physics',.4,.3,0.)
        root=add_links(stage,'/World/Objects/'+a.object,kit,fix,spec,material,look_base=s.dir)
        set_pose(root,[*fix['xy'],fix['support_z']],quat_from_matrix(rotation(fix)))
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        floor=trimesh.creation.box(extents=[4,4,.03]);floor.apply_translation([*fix['xy'],fix['support_z']-.015])
        add_mesh(stage,'/World/DiagnosticSupport',floor.vertices,floor.faces,None,
                 color_material(stage,'/World/Looks/neutral_support',[.35,.35,.35],roughness=.6))
        UsdLux.DomeLight.Define(stage,'/World/Lights/diagnostic_dome').CreateIntensityAttr(700.)
    else:
        build_scene(stage,prepare_scene(s),photo_goal(s.dir,prepare_variant(s.dir,{})))
        add_background(stage,prepare_background(s),'zone')
    configure_render(mode=a.mode,auto_exposure=False,spp=128,bounces=12)
    camera=add_camera(stage,'/World/Cams/material_sweep',K,W,H,T_world_cam_cv=T)
    ann=rgb_annotator(camera.GetPath(),W,H)
    if a.atlas:
        atlas=np.load(a.atlas);start=0
        parts=asset(s.scene_graph['dynamic_objects'][a.object]['fix'])['parts']
        for part in parts:
            surface=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Objects/'+a.object+'/'+part['name']+'/surface'))
            n=len(part['F']);original=part['V'][part['F']]
            if not np.allclose(atlas['V'][atlas['F'][start:start+n]],original,atol=1e-7,rtol=0):
                raise ValueError('Appearance atlas changes actual triangle positions/order')
            uv=atlas['face_uv'][start:start+n];start+=n
            st=UsdGeom.PrimvarsAPI(surface).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying)
            st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv,np.float32).reshape(-1,2)))
    if a.smooth:
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Objects/'+a.object)):
            if not prim.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(prim);vertices=np.array(mesh.GetPointsAttr().Get());faces=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
            normals=trimesh.Trimesh(vertices,faces,process=False).vertex_normals
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals,np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    shaders={}
    root=stage.GetPrimAtPath('/World/Objects/'+a.object)
    for prim in Usd.PrimRange(root):
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):continue
        mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not mat:continue
        output=mat.GetSurfaceOutput('mdl')
        connected=output.GetConnectedSource() if output else None
        if connected:
            shader=UsdShade.Shader(connected[0].GetPrim());shaders[str(shader.GetPath())]=shader
    if not shaders:raise ValueError('No native MDL surface on this object')
    variants=json.loads(Path(a.variants).read_text());rows=[];tiles=[('source',source)]
    original={};texture_names={'diffuse_reflection_weight_image','diffuse_reflection_color_image','specular_transmission_weight_image'}
    referenced_textures=[]
    for path,shader in shaders.items():
        for name in {key for v in variants for key in v['inputs']}|{key for v in variants for key in v.get('textures',{})}:
            inp=shader.GetInput(name)
            if not inp and name in texture_names:inp=shader.CreateInput(name,Sdf.ValueTypeNames.Asset);inp.Set(Sdf.AssetPath())
            if not inp:raise ValueError('Unsupported native material input: '+name)
            original[path,name]=inp.Get()
    # Give each proposal its own material network. Native MDL compilation can
    # cache a network's original texture signature after its first render.
    templates=dict(shaders)
    bindings=[]
    for prim in Usd.PrimRange(root):
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):continue
        mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bindings.append((prim,str(mat.GetPath())))
    for number,v in enumerate(variants):
        shaders={}
        for path,template in templates.items():
            old_mat=path.rsplit('/',1)[0];new_mat=old_mat+f'_proposal_{number}'
            Sdf.CopySpec(stage.GetRootLayer(),old_mat,stage.GetRootLayer(),new_mat)
            shader=UsdShade.Shader(stage.GetPrimAtPath(new_mat+'/'+template.GetPrim().GetName()))
            UsdShade.Material(stage.GetPrimAtPath(new_mat)).CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
            shaders[path]=shader
            for prim,material in bindings:
                if material==old_mat:UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(new_mat)))
        for path,shader in shaders.items():
            for (owner,name),value in original.items():
                if owner==path:shader.GetInput(name).Set(value)
            for name,value in v['inputs'].items():
                inp=shader.GetInput(name)
                inp.Set(Gf.Vec3f(*map(float,value)) if isinstance(value,list) else value)
            for name,value in v.get('textures',{}).items():
                if name not in texture_names:raise ValueError('Unsupported optical texture input: '+name)
                file=Path(value).resolve()
                if not file.is_file():raise ValueError('Missing optical texture')
                shader.GetInput(name).Set(Sdf.AssetPath(str(file)));referenced_textures.append(file)
                if name.endswith('weight_image'):
                    shader.CreateInput(name+'_alpha_mode',Sdf.ValueTypeNames.Int).Set(1)
        for _ in range(25):app.update()
        rep.orchestrator.step(rt_subframes=64,delta_time=0.)
        image=grab(ann,render=app.update);Image.fromarray(image).save(out/f'variant_{number}.png')
        selected=mask
        rows.append({'id':v['id'],'inputs':v['inputs'],'textures':v.get('textures',{}),'source_mask_mean_rgb':source[selected].mean(0).tolist(),
                     'render_mask_mean_rgb':image[selected].mean(0).tolist(),
                     'scope':'Appearance proposal in unchanged source mask; lighting/context remain separate uncertainty'})
        tiles.append((v['id'],image));print('[material_sweep]',v['id'],flush=True)
    board=Image.new('RGB',(320*len(tiles),340))
    for j,(label,image) in enumerate(tiles):
        tile=Image.fromarray(image);tile.thumbnail((320,320));board.paste(tile,(320*j,20));ImageDraw.Draw(board).text((320*j+3,2),label,fill='white')
    board.save(out/'comparison.jpg')
    files=[Path(a.variants),s/'splat/object_fit'/f'{a.object}.npz',s/'parsing/posed_frames/cameras.npz',s.config_dir/'scene_graph.json',Path(__file__)]
    files.extend(set(referenced_textures))
    if a.atlas:files.append(Path(a.atlas))
    (out/'result.json').write_text(json.dumps({'approved':False,'completed':True,'object_only':a.object_only,'scope':'Native appearance proposals only; isolated mode disables dynamics and uses neutral lighting; no geometry/physics/source approval','renderer':a.mode,'frame':a.frame,'rows':rows,'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
except Exception as error:
    import traceback
    failure={'approved':False,'completed':False,'error':str(error),'traceback':traceback.format_exc()}
    (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    print(failure['traceback'],flush=True)
    app.app.post_quit(1)
    raise
finally:
    app.close()
