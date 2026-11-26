"""Compare a new GS field against a fixed original field; never approve a scene."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from calibrated_scene_reconstruction.scene.review.environment import digest


def error(rgb,source,mask):
    delta=(rgb.astype(float)-source)/255.
    if int(mask.sum())<1024:raise ValueError('At least1024 fixed observed background pixels required')
    return {'mae':float(abs(delta[mask]).mean()),'psnr':float(-10*np.log10(max(np.square(delta[mask]).mean(),1e-12)))}


def compare(trial,reference,output):
    trial,reference=Path(trial).resolve(),Path(reference).resolve();new,old=[json.loads(p.read_text()) for p in (trial,reference)]
    if (new.get('diagnostic')!='context_only' or not new.get('background_field_override')
            or old.get('diagnostic')!='context_source' or not old.get('source_context_override')):
        raise ValueError('Explicit new-field ablation and original GS reference required')
    keys=('usd','camera_manifest','environment','environment_context','effective_photometry','context_background_override')
    if any(new.get(k)!=old.get(k) for k in keys) or not all(r.get('static_scene_preserved') for r in (new,old)):
        raise ValueError('Field comparison requires identical static scene, cameras and render profile')
    if any(r.get(k) for r in (new,old) for k in ('render_settings','background_transform_override')):
        raise ValueError('Field trial cannot also change photometry or coordinate transforms')
    asset=Path(old['source_context_override']).resolve();context=new['environment_context']
    if str(asset)!=context.get('source_asset') or digest(asset)!=context.get('source_asset_sha256'):
        raise ValueError('Field trial requires the fixed original GS asset')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={str(p):digest(p) for p in (trial,reference,Path(__file__).resolve())};inputs.update(new['input_sha256'])
    for p,expected in old['input_sha256'].items():
        if p in inputs and inputs[p]!=expected:raise ValueError('Shared field comparison input changed')
        inputs[p]=expected
    scope=json.loads(new['environment_context']['source_background_scope_json']);refs={r['source']:r for r in old['views']};rows=[]
    for view in new['views']:
        source=Path(view['source']).resolve();ref=refs[view['source']]
        raw=cv2.imread(str(source));candidate=cv2.imread(view['renders'][0]);baseline=cv2.imread(ref['renders'][0])
        if candidate.shape!=baseline.shape:raise ValueError('Aligned render rasters required')
        mask=np.ones(raw.shape[:2],bool)
        for p,sha in scope[source.name].items():
            if digest(p)!=sha:raise ValueError('Fixed comparison scope changed')
            region=cv2.imread(p,0)
            if region is None or region.shape!=mask.shape:raise ValueError('Fixed source mask dimensions differ')
            mask &= region>128;inputs[p]=sha
        size=(candidate.shape[1],candidate.shape[0]);raw=cv2.resize(raw,size,interpolation=cv2.INTER_AREA)
        mask=cv2.erode(cv2.resize(mask.astype('uint8'),size,interpolation=cv2.INTER_NEAREST),np.ones((5,5),'uint8'))>0
        row={'source':str(source),'pixels':int(mask.sum()),'original':error(baseline,raw,mask),'trial':error(candidate,raw,mask)}
        rows.append(row)
        files=[source,Path(ref['renders'][0]),*[Path(p) for p in view['renders']]]
        sheet=Image.new('RGB',(2048,312));draw=ImageDraw.Draw(sheet)
        for i,(path,label) in enumerate(zip(files,['Source','Original GS','Trial GS: captured camera','Trial GS: free view'])):
            inputs[str(path.resolve())]=digest(path);sheet.paste(Image.open(path).convert('RGB').resize((512,288)),(i*512,24));draw.text((i*512+6,6),label,fill='white')
        sheet.save(out/(source.stem+'_comparison.jpg'),quality=90)
    if len({r['source'] for r in rows})<2:raise ValueError('At least two captured camera comparisons required')
    result={'pass':False,'photorealism_approved':False,'views':rows,'input_sha256':inputs,
            'mean':{key:{metric:float(np.mean([r[key][metric] for r in rows])) for metric in ('mae','psnr')} for key in ('original','trial')},
            'scope':'Fixed captured-background RGB comparison plus actual free-view images. No novel-view ground truth, task-size truth or automatic scene acceptance.'}
    for p,expected in inputs.items():
        if digest(p)!=expected:raise ValueError('Field comparison input changed: '+p)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trial');p.add_argument('reference');p.add_argument('--output',required=True)
    a=p.parse_args();q=compare(a.trial,a.reference,a.output);print(q['mean'])


if __name__=='__main__':main()
