"""Independent RGB chart validation for measured captured-radiance display.

A neutral ramp cannot establish reproduction of saturated colours. Validate an
independent RGB cube through the actual renderer before converting source
textures. The chart is diagnostic-only and never alters scene geometry/scale.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from calibrated_scene_reconstruction.scene.background.display_response import invert, profile
from calibrated_scene_reconstruction.scene.review.environment import digest


def errors(target, actual):
    target,actual=np.asarray(target,float),np.asarray(actual,float)
    if target.shape!=actual.shape or target.ndim!=2 or target.shape[1]!=3 or len(target)<64 or not np.isfinite(np.r_[target,actual]).all():
        raise ValueError('Independent finite RGB target and measured patches required')
    delta=abs(target-actual);mae=float(delta.mean());maximum=float(delta.max())
    return {'pass':mae<=.025 and maximum<=.08,'rgb_mae':mae,'maximum_channel_error':maximum,
            'patches':len(target),'thresholds':{'rgb_mae':.025,'maximum_channel_error':.08}}


def prepare(session, neutral, output, response_path=None, independent=False):
    from pxr import Sdf,Usd,UsdGeom
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    neutral=Path(neutral).resolve();out=Path(output).resolve();b=verify(session)
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen display diagnostic required')
    response_path=Path(response_path).resolve() if response_path else neutral/'response.json'
    response=json.loads(response_path.read_text());job=json.loads((neutral/'job.json').read_text())
    if response['profile']!=profile(b['render'],b.get('splat_context')):raise ValueError('Neutral response settings changed')
    for p,sha in response['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('Neutral measurement changed')
    out.mkdir(parents=True);manifest=json.loads((neutral/'cameras.json').read_text());W,H=manifest['width'],manifest['height'];cols,rows,tile=18,12,24
    levels=[0.,.06,.18,.38,.72,1.] if independent else [0.,.1,.25,.5,.75,1.]
    targets=np.array([[r,g,b] for r in levels for g in levels for b in levels]);emitted=invert(targets,response)
    # Radiance RGBE shares an 8-bit mantissa/exponent across RGB. Bright saturated
    # pixels can quantize another channel's small inverse-display radiance to
    # zero. Float TIFF retains independent channel precision; validate its actual
    # renderer path rather than assuming the format is faithfully consumed.
    import tifffile
    image=np.repeat(np.repeat(emitted.reshape(rows,cols,3),tile,0),tile,1);texture=out/'radiance.tiff'
    tifffile.imwrite(texture,image.astype('float32'),photometric='rgb')
    stage=Usd.Stage.CreateNew(str(out/'scene.usda'));stage.GetRootLayer().subLayerPaths=[str(neutral/'scene.usda')];stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.GetPrimAtPath('/DisplayResponse/Material/Texture').GetAttribute('inputs:file').Set(Sdf.AssetPath(str(texture)))
    # Root-layer custom data does not compose through a sublayer. Preserve the
    # real scene's native/context photometry contract in this diagnostic root.
    data=dict(Usd.Stage.Open(str(neutral/'scene.usda')).GetRootLayer().customLayerData)
    data['r2s2r_diagnostic']='Independent RGB response validation; not a deliverable environment';stage.GetRootLayer().customLayerData=data;stage.GetRootLayer().Save()
    (out/'cameras.json').write_text(json.dumps(manifest,indent=2)+'\n')
    x0,y0,x1,y1=W*.15,H*.15,W*.85,H*.85;xy=[[x0+(x1-x0)*(x+.5)/cols,y0+(y1-y0)*(y+.5)/rows] for y in range(rows) for x in range(cols)]
    paths=[response_path,neutral/'scene.usda',neutral/'render/review.json',out/'scene.usda',out/'cameras.json',texture,Path(__file__).resolve()]
    record={'version':1,'profile':response['profile'],'neutral_response':str(response_path),'render_name':job['render_name'],
            'emitted_rgb':emitted.tolist(),'independent':independent,
            'targets':targets.tolist(),'pixels':xy,
            'effective_photometry':json.loads((neutral/'render/review.json').read_text())['effective_photometry'],
            'input_sha256':{str(p):digest(p) for p in paths}}
    (out/'job.json').write_text(json.dumps(record,indent=2)+'\n');return record


def fit(session,folder):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    folder=Path(folder).resolve()
    if not folder.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('Use an unfrozen display diagnostic')
    job=json.loads((folder/'job.json').read_text());b=verify(session)
    if job['profile']!=profile(b['render'],b.get('splat_context')):raise ValueError('Display settings changed')
    for p,sha in job['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('RGB validation evidence changed')
    review=folder/'render/review.json';render=json.loads(review.read_text());image_path=folder/'render'/job['render_name']
    if not render.get('diagnostic') or Path(render['usd']).resolve()!=folder/'scene.usda':raise ValueError('Use the actual diagnostic RGB chart')
    if render.get('effective_photometry')!=job['effective_photometry']:raise ValueError('RGB validation must use the measured neutral photometry')
    image=np.asarray(Image.open(image_path).convert('RGB'))/255;actual=[]
    for x,y in np.rint(job['pixels']).astype(int):actual.append(np.median(image[y-2:y+3,x-2:x+3],axis=(0,1)))
    result=errors(job['targets'],actual);result.update(actual_rgb=np.asarray(actual).tolist(),target_rgb=job['targets'],input_sha256={**job['input_sha256'],str(review):digest(review),str(image_path):digest(image_path)})
    response=copy.deepcopy(json.loads(Path(job['neutral_response']).read_text()))
    if response.get('colour_transform') and not job.get('independent'):raise ValueError('Fitted colour response needs an independent validation cube')
    response['colour_validation']=result;response['input_sha256'].update(result['input_sha256'])
    # A failed diagnostic remains explicit; validate_profile will refuse it.
    (folder/'response.json').write_text(json.dumps(response,indent=2)+'\n');return result


def fit_weights(emitted, actual, response):
    """Fit positive luminance coupling; independent RGB validation is separate."""
    from scipy.optimize import least_squares
    from calibrated_scene_reconstruction.scene.background.surface_textures import linear,srgb
    emitted,actual=np.asarray(emitted,float),np.asarray(actual,float)
    if emitted.shape!=actual.shape or emitted.ndim!=2 or emitted.shape[1]!=3 or len(actual)<64:
        raise ValueError('Measured calibration colour cube required')
    radiance=np.asarray(response['radiance']);curve=linear(np.asarray(response['display_rgb']).mean(1))
    def weights(x):
        v=np.exp(np.r_[x,0.]);return v/v.sum()
    def predicted(x):
        luminance=emitted@weights(x);shown=np.interp(luminance,radiance,curve)
        return srgb(emitted*(shown/np.maximum(luminance,1e-12))[:,None])
    result=least_squares(lambda x:(predicted(x)-actual).ravel(),[.96,1.64],bounds=(-5.,5.))
    return {'mode':'luminance','weights':weights(result.x).tolist(),
            'training_rgb_mae':float(abs(predicted(result.x)-actual).mean()),
            'scope':'Measured colour coupling hypothesis; not usable until independent RGB chart validation passes.'}


def colour_model(session, folder, output):
    folder=Path(folder).resolve();out=Path(output).resolve();job=json.loads((folder/'job.json').read_text())
    if job.get('independent') or out.exists() or not out.is_relative_to(session.dir.resolve()):raise ValueError('Use a calibration cube and a new proposal path')
    measured=fit(session,folder);response=json.loads(Path(job['neutral_response']).read_text())
    response['colour_transform']=fit_weights(job['emitted_rgb'],measured['actual_rgb'],response)
    response['colour_validation']={'pass':False,'reason':'Colour coupling fitted; independent source colours must still be rendered.'}
    response['input_sha256'].update(measured['input_sha256']);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(response,indent=2)+'\n');return response['colour_transform']


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('neutral');q.add_argument('--output',required=True);q.add_argument('--response');q.add_argument('--independent',action='store_true')
    q=sub.add_parser('colour-model');q.add_argument('folder');q.add_argument('--output',required=True)
    q=sub.add_parser('fit');q.add_argument('folder');a=p.parse_args();s=Session(a.session)
    if a.command=='prepare':r=prepare(s,a.neutral,a.output,a.response,a.independent)
    elif a.command=='colour-model':r=colour_model(s,a.folder,a.output)
    else:r=fit(s,a.folder)
    print('RGB validation proposal' if a.command=='prepare' else {k:v for k,v in r.items() if k not in ['actual_rgb','target_rgb','input_sha256']})


if __name__=='__main__':main()
