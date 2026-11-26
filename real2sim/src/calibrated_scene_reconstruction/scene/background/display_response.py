"""Measure and invert the actual renderer response for captured appearance.

Synthetic radiance patches calibrate display reproduction, never reconstruction
scale or true object dimensions. Calibration scenes are diagnostic-only. The
result preserves captured lighting, not relightable surface reflectance.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
from calibrated_scene_reconstruction.scene.session import Session


def profile(render,context):
    from calibrated_scene_reconstruction.scene.background.blender_env import zero_specular_emission
    names={'tonemap_op':'op','film_iso':'filmIso','exposure_time_s':'exposureTime','f_number':'fNumber','cm2_factor':'cm2Factor'}
    settings={'/rtx/post/tonemap/'+names[k]:v for k,v in render['photometry'].items()}
    if context:
        # This scene-side contract describes native display response only. It
        # does not import runtime adapters or Gaussian-only inversion policies.
        for name,value in context['source_layer_data'].get('renderSettings',{}).items():
            key='/'+name.replace(':','/') if name.startswith('rtx:') else name
            if key.startswith('/rtx/post/tonemap/') or key=='/rtx/post/histogram/enabled':
                if not isinstance(value,(bool,int,float)) or not np.isfinite(value):raise ValueError('Finite display settings required')
                settings[key]=value
    return {'renderer':render['mode'],'auto_exposure':False,'settings':settings,
            'shader_contract_sha256':hashlib.sha256(inspect.getsource(zero_specular_emission).encode()).hexdigest()}


def validate_profile(response,render,context):
    if response.get('version')!=1 or response.get('profile')!=profile(render,context):
        raise ValueError('Captured display response belongs to different renderer settings/material semantics')
    if (response.get('colour_validation') or {}).get('pass') is not True:
        raise ValueError('Neutral-ramp calibration needs independent RGB validation before scene use')


def prepare(session,usd,cameras,output):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,Vt
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    built=verify(session);out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Display calibration belongs in this session')
    out.mkdir(parents=True,exist_ok=False)
    usd=Path(usd).resolve();camera_path=Path(cameras).resolve();manifest=json.loads(camera_path.read_text())
    anchor=copy.deepcopy(manifest['anchors'][-1]);anchor['views']=[anchor['views'][0]]
    W,H=manifest['width'],manifest['height'];K=np.asarray(anchor['K'],float);K[0]*=W/anchor['source_size'][0];K[1]*=H/anchor['source_size'][1]
    C=np.asarray(anchor['views'][0]['centre']);R=np.asarray(anchor['views'][0]['R_world_cam']);cols,rows,tile=16,8,32
    values=np.r_[0.,np.geomspace(1e-8,1.,cols*rows-1)];image=np.repeat(values.reshape(rows,cols),tile,0);image=np.repeat(image,tile,1)
    texture=out/'radiance.hdr';assert cv2.imwrite(str(texture),np.repeat(image[...,None],3,2).astype('float32'))
    old=Usd.Stage.Open(str(usd));stage=Usd.Stage.CreateNew(str(out/'scene.usda'));stage.GetRootLayer().subLayerPaths=[str(usd)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    data=dict(old.GetRootLayer().customLayerData);data['r2s2r_diagnostic']='Synthetic captured-radiance response chart; not a deliverable environment';stage.GetRootLayer().customLayerData=data
    x0,y0,x1,y1=W*.15,H*.15,W*.85,H*.85
    screen=np.array([[x0,y1],[x1,y1],[x1,y0],[x0,y0]])
    # A small synthetic chart in front of the camera prevents room occlusion.
    # Its arbitrary dimensions never participate in metric scene calibration.
    points=np.c_[screen,np.ones(4)]@np.linalg.inv(K).T*.2@R.T+C
    mesh=UsdGeom.Mesh.Define(stage,'/DisplayResponse/Chart');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
    st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex');st.Set([(0,0),(1,0),(1,1),(0,1)])
    material=UsdShade.Material.Define(stage,'/DisplayResponse/Material');shader=UsdShade.Shader.Define(stage,'/DisplayResponse/Material/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    for key,value in [('diffuseColor',(0.,0.,0.)),('specularColor',(0.,0.,0.))]:shader.CreateInput(key,Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1);shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.95)
    tex=UsdShade.Shader.Define(stage,'/DisplayResponse/Material/Texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
    reader=UsdShade.Shader.Define(stage,'/DisplayResponse/Material/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput('result',Sdf.ValueTypeNames.Float2))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface',Sdf.ValueTypeNames.Token));UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    manifest['anchors']=[anchor];manifest['novel_offsets_m']=[];manifest['purpose']='Diagnostic radiance chart; not source appearance or task-scale evidence'
    (out/'cameras.json').write_text(json.dumps(manifest,indent=2)+'\n')
    xy=np.array([[x0+(x1-x0)*(x+.5)/cols,y0+(y1-y0)*(y+.5)/rows] for y in range(rows) for x in range(cols)])
    job={'version':1,'profile':profile(built['render'],built.get('splat_context')),'radiance':values.tolist(),'pixels':xy.tolist(),
         'render_name':Path(anchor['source']).stem+'_offset00.png','input_sha256':{str(p):digest(p) for p in [usd,camera_path,texture,out/'scene.usda',out/'cameras.json',Path(__file__).resolve()]},
         'scope':'Synthetic display response only. No object dimensions, task geometry or reconstructed scale fit.'}
    (out/'job.json').write_text(json.dumps(job,indent=2)+'\n');return out


def fit(session,folder):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    folder=Path(folder).resolve();job=json.loads((folder/'job.json').read_text());built=verify(session)
    if job['profile']!=profile(built['render'],built.get('splat_context')):raise ValueError('Renderer changed during display calibration')
    for p,sha in job['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('Display calibration input changed: '+p)
    review=folder/'render/review.json';record=json.loads(review.read_text())
    if not record.get('diagnostic'):raise ValueError('Response chart must be explicitly diagnostic')
    image=np.asarray(Image.open(folder/'render'/job['render_name']).convert('RGB'))/255.;display=[]
    for x,y in np.rint(job['pixels']).astype(int):display.append(np.median(image[y-3:y+4,x-3:x+4],axis=(0,1)))
    display=np.asarray(display);values=np.asarray(job['radiance'])
    if np.max(display[0])>.02 or np.min(display[-1])<.98:raise ValueError('Chart black/white response is occluded or insufficiently bracketed')
    inputs=dict(job['input_sha256']);inputs[str(review)]=digest(review);inputs[str(folder/'render'/job['render_name'])]=digest(folder/'render'/job['render_name'])
    result={'version':1,'profile':job['profile'],'radiance':values.tolist(),'display_rgb':display.tolist(),'input_sha256':inputs,
            'colour_validation':{'pass':False,'reason':'Neutral response measured only; independent RGB chart validation is required before use.'},
            'scope':'Measured display reproduction for zero-specular captured appearance; not inverse lighting or reflectance.'}
    (folder/'response.json').write_text(json.dumps(result,indent=2)+'\n');return result


def invert(rgb,response):
    values=np.asarray(response['radiance'],float);display=np.asarray(response['display_rgb'],float)
    if values.ndim!=1 or len(values)<16 or display.shape!=(len(values),3) or not np.isfinite(display).all() or not np.isfinite(values).all() or (values<0).any() or (np.diff(values)<=0).any():
        raise ValueError('Finite ordered calibrated response required')
    rgb=np.asarray(rgb,float);output=np.empty_like(rgb,dtype=np.float32)
    transform=response.get('colour_transform')
    if transform is not None:
        from calibrated_scene_reconstruction.scene.background.surface_textures import linear
        weights=np.asarray(transform.get('weights'),float)
        if (transform.get('mode')!='luminance' or weights.shape!=(3,) or not np.isfinite(weights).all()
                or np.any(weights<=0) or not np.isclose(weights.sum(),1.)):
            raise ValueError('Measured positive luminance weights required')
        target=linear(np.clip(rgb,0,1));luminance=target@weights
        curve=np.maximum.accumulate(linear(display.mean(1)));unique,index=np.unique(curve,return_index=True)
        if len(unique)<16 or unique[0]>.002 or unique[-1]<.95:raise ValueError('Insufficient measured display response range')
        emitted=np.interp(luminance,unique,values[index])
        return (target*(emitted/np.maximum(luminance,1e-12))[...,None]).astype('float32')
    for c in range(3):
        curve=np.maximum.accumulate(display[:,c]);unique,index=np.unique(curve,return_index=True)
        if len(unique)<16 or unique[0]>.02 or unique[-1]<.98:raise ValueError('Insufficient measured display response range')
        output[...,c]=np.interp(np.clip(rgb[...,c],0,1),unique,values[index])
    return output


def texture(session,source,response_path,output):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    source=Path(source).resolve();response_path=Path(response_path).resolve();out=Path(output).resolve();b=verify(session)
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Converted capture texture belongs in this session')
    response=json.loads(response_path.read_text());validate_profile(response,b['render'],b.get('splat_context'));out.mkdir(parents=True,exist_ok=False)
    import tifffile
    rgb=np.asarray(Image.open(source).convert('RGB'))/255.;radiance=invert(rgb,response).astype('float32')
    pixels_sha=hashlib.sha256(radiance.tobytes(order='C')).hexdigest()
    hdr=out/(source.stem+'_'+pixels_sha[:16]+'_radiance.tiff')
    # RGBE's shared exponent loses weak channels after inverse tone mapping.
    tifffile.imwrite(hdr,radiance,photometric='rgb')
    proof={'version':1,'source_texture':str(source),'source_sha256':digest(source),'radiance':str(hdr),'radiance_sha256':digest(hdr),
           'response':str(response_path),'response_sha256':digest(response_path),'profile':response['profile'],'scope':response['scope']}
    (out/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


def mesh(session,source_manifest,response_path,output):
    original=json.loads(Path(source_manifest).read_text())
    if original['protected_task_state']!=protected_state(session) or digest(original['texture'])!=original['texture_sha256'] or digest(original['mesh'])!=original['mesh_sha256']:
        raise ValueError('Native appearance/geometry proof changed')
    response=texture(session,original['texture'],response_path,output);out=Path(output).resolve();proof=copy.deepcopy(original)
    shutil.copyfile(original['mesh'],out/'geometry.usdc');proof.update(mesh=str(out/'geometry.usdc'),radiance=response['radiance'],radiance_sha256=response['radiance_sha256'],display_response=response)
    proof['input_sha256'].update({str(Path(source_manifest).resolve()):digest(source_manifest),str(Path(response_path).resolve()):digest(response_path)})
    (out/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('usd');q.add_argument('cameras');q.add_argument('--output',required=True)
    q=sub.add_parser('fit');q.add_argument('folder')
    for name in ['texture','mesh']:
        q=sub.add_parser(name);q.add_argument('source');q.add_argument('response');q.add_argument('--output',required=True)
    a=p.parse_args();s=Session(a.session)
    if a.command=='prepare':result=prepare(s,a.usd,a.cameras,a.output)
    elif a.command=='fit':result=fit(s,a.folder)
    else:result=globals()[a.command](s,a.source,a.response,a.output)
    print(result if isinstance(result,Path) else json.dumps(result,indent=2))


if __name__=='__main__':main()
