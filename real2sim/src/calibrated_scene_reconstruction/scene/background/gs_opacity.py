"""Measure whether retained GS appearance is invariant to known backdrops.

Colour agreement on one background cannot certify opaque coverage. Only pixels
stable on black and white backgrounds may replace native backing; task/robot
pixels and actual GS contribution are excluded/tested by capture_gaps as well.
This is a conservative display-space invariance test, not recovered material
opacity or photorealism approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from plyfile import PlyData
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state

MEASUREMENT_KEYS=('usd','camera_manifest','environment','environment_context','effective_photometry',
                  'background_field_override','background_field_kind','background_world_matrix')


def validate_field(render,context,manifest):
    """Allow an actual complete-field diagnostic, never an empty USDZ payload."""
    shown=render.get('environment_context') or {};override=render.get('background_field_override')
    if render.get('static_scene_preserved')is not True or render.get('source_context_override') or render.get('render_settings') or render.get('background_transform_override'):
        raise ValueError('Unchanged static field diagnostic required')
    if override:
        path=Path(override).resolve()
        matrix=np.asarray(render.get('background_world_matrix',[]))
        if (render.get('background_field_kind')!='usd_reference' or digest(path)!=context['asset_sha256']
                or render.get('input_sha256',{}).get(str(path))!=digest(path)
                or matrix.shape!=(4,4) or not np.allclose(matrix,np.asarray(context['matrix']),rtol=0,atol=1e-8)):
            raise ValueError('Verified complete USD field and calibrated transform required')
    elif shown.get('asset_sha256')!=context['asset_sha256'] or shown.get('evidence_sha256')!=digest(manifest):
        raise ValueError('Diagnostic must show this exact retained field')


def stable_mask(black,white,threshold=8):
    if black.shape!=white.shape or black.ndim!=3 or black.shape[2]!=3:raise ValueError('Aligned RGB backdrop diagnostics required')
    if type(threshold)is not int or not 2<=threshold<=12:raise ValueError('Bounded backdrop-invariance threshold required')
    good=np.max(abs(black.astype(float)-white.astype(float)),axis=2)<=threshold
    return cv2.erode(good.astype('uint8'),np.ones((3,3),'uint8'))>0


def prepare(session,context_manifest,black_review,white_review,output,threshold=8):
    out=Path(output).resolve();paths=list(map(lambda p:Path(p).resolve(),[context_manifest,black_review,white_review]));context,black,white=[json.loads(p.read_text()) for p in paths]
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen opacity diagnostic required')
    for render,color in [(black,[0.,0.,0.]),(white,[1.,1.,1.])]:
        validate_field(render,context,paths[0])
        if (render.get('diagnostic')!='context_only' or render.get('context_background_override')!=color
                ):raise ValueError('Use this exact field on known black/white diagnostic backgrounds')
    if any(black.get(k)!=white.get(k) for k in MEASUREMENT_KEYS):raise ValueError('Opacity diagnostics need the same scene, cameras and photometry')
    camera=session/'parsing/posed_frames/cameras.npz';cloud=paths[0].parent/'context.ply'
    if context['protected_task_state']!=protected_state(session) or context['camera_sha256']!=digest(camera):raise ValueError('Opacity field belongs to another capture/protected state')
    field=PlyData.read(str(cloud))['vertex'].data;inputs={str(p):digest(p) for p in [*paths,camera,cloud,Path(__file__).resolve()]}
    for record in [context,black,white]:
        for p,sha in record['input_sha256'].items():
            if p in inputs and inputs[p]!=sha:raise ValueError('Conflicting opacity evidence')
            inputs[p]=sha
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Stale opacity diagnostics; rerender this exact field')
    out.mkdir(parents=True);rows=[];others={r['source']:r for r in white['views']}
    for row in black['views']:
        other=others[row['source']];a,b=[cv2.imread(r['renders'][0]) for r in [row,other]];mask=stable_mask(a,b,threshold);h,w=mask.shape;width=min(640,w);height=round(h*width/w);path=out/(Path(row['source']).stem+'.png');cv2.imwrite(str(path),cv2.resize(mask.astype('uint8')*255,(width,height),interpolation=cv2.INTER_NEAREST))
        for p in [row['source'],row['renders'][0],other['renders'][0],str(path)]:inputs[str(Path(p).resolve())]=digest(p)
        rows.append({'frame':Path(row['source']).name,'mask':str(path),'mask_sha256':digest(path),'stable_pixels':int(mask.sum()),'total_pixels':h*w})
    result={'version':2,'measurement':{k:black.get(k) for k in MEASUREMENT_KEYS},'rows':rows,'field_rows_sha256':hashlib.sha256(field.tobytes()).hexdigest(),'asset_sha256':context['asset_sha256'],
            'camera_sha256':digest(camera),'protected_task_state':protected_state(session),'input_sha256':inputs,'threshold_u8':threshold,'pass':False,
            'scope':'Backdrop-invariant coverage candidate only. Require actual GS contribution, frozen foreground exclusion, repeated source visibility, complete native backing and final composite/free-view review.'}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Opacity diagnostic input changed')
    (out/'opacity.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('context_manifest');p.add_argument('black_review');p.add_argument('white_review');p.add_argument('--output',required=True);a=p.parse_args();r=prepare(Session(a.session),a.context_manifest,a.black_review,a.white_review,a.output);print('Backdrop-invariant coverage proposals',len(r['rows']))


if __name__=='__main__':main()
