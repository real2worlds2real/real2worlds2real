"""Cache capture-backed background masks and crops for agent geometry/material iteration."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.background.blender_env import label
from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session


def geometry_review(row):
    """A semantic mask can depict one door rather than the complete cabinet."""
    if (row.get('geometry_fit_scope') not in ('whole_object','occluded_object')
            or not isinstance(row.get('geometry_fit_review'),str) or not row['geometry_fit_review'].strip()):
        raise ValueError('Review whole-object/occluded-object silhouette scope before fitting geometry: '+str((row.get('entity'),row.get('frame'))))
    if row.get('inference_clipped_edges') and row.get('geometry_fit_scope') == 'whole_object':
        raise ValueError('Crop-clipped segmentation cannot establish a whole-object silhouette')


def inference_image(image, crop):
    """Use original source pixels for a small object, retaining camera coordinates."""
    h, w = image.shape[:2]
    rect = [0, 0, w, h] if crop is None else crop
    if (not isinstance(rect, list) or len(rect) != 4 or any(type(v) is not int for v in rect)
            or not 0 <= rect[0] < rect[2] <= w or not 0 <= rect[1] < rect[3] <= h
            or min(rect[2]-rect[0], rect[3]-rect[1]) < 32):
        raise ValueError('Inference crop must be an in-bounds source-pixel rectangle at least 32px wide and high')
    return image[rect[1]:rect[3], rect[0]:rect[2]], rect


def source_masks(masks, shape, crop):
    masks = np.asarray(masks, bool); h, w = shape
    x0, y0, x1, y1 = crop
    if masks.ndim != 3 or masks.shape[1:] != (y1-y0, x1-x0):
        raise ValueError('Segmentation must match the actual inference crop')
    result = np.zeros((len(masks), h, w), bool)
    result[:, y0:y1, x0:x1] = masks
    return result


def select_mask(masks, scores, roi, allow_large=False):
    masks, scores = np.asarray(masks, bool), np.asarray(scores, float)
    if len(masks) != len(scores) or masks.ndim != 3:
        raise ValueError('One calibrated raster per proposed mask required')
    h, w = masks.shape[1:]
    x0, y0, x1, y1 = np.asarray(roi, int)
    x0,y0,x1,y1=max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Nonempty source ROI required')
    expected = np.zeros((h,w),bool);expected[y0:y1,x0:x1]=True
    candidates=[]
    for i,mask in enumerate(masks):
        area=int(mask.sum())
        if area<64 or scores[i]<.3 or area>.9*h*w and not allow_large:
            continue
        intersection=int((mask&expected).sum())
        if intersection==0:continue
        overlap=intersection/max(1,int((mask|expected).sum()))
        candidates.append((overlap*float(scores[i]),i,overlap))
    if not candidates:return None
    rank,index,overlap=max(candidates)
    return {'index':index,'roi_iou':overlap,'score':float(scores[index]),'rank':rank}


def prepare(session, output, config=None):
    import cv2
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen task delivery: use an environment derivative')
    config=Path(config) if config else session.config_dir/'environment_observations.json'
    requests=json.loads(config.read_text())['requests']
    camera=session/'parsing/posed_frames/cameras.npz'
    z=np.load(camera);names=list(map(str,z['names']))
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Observation cache belongs in this session')
    out.mkdir(parents=True,exist_ok=False)
    inputs={str(p.resolve()):digest(p) for p in [camera,config,Path(__file__)]}
    model=Sam3Model.from_pretrained('facebook/sam3',local_files_only=True).to('cuda').eval()
    processor=Sam3Processor.from_pretrained('facebook/sam3',local_files_only=True)
    cached={};records=[];validity=ValidPixels(session)
    for number,request in enumerate(requests):
        entity=label(request['entity']);frame=request['frame'];prompt=request['prompt']
        if frame not in names or Path(frame).name!=frame:raise ValueError('Use registered capture basenames')
        source=session/'splat/data/images'/frame
        inputs[str(source.resolve())]=digest(source)
        image=np.asarray(Image.open(source).convert('RGB'));h,w=image.shape[:2]
        index=names.index(frame)
        if list(map(int,z['size'][index])) != [w,h]:raise ValueError('Source image differs from calibrated raster')
        model_image, crop = inference_image(image, request.get('inference_crop'))
        key=(frame,prompt,tuple(crop))
        if key not in cached:
            data=processor(images=Image.fromarray(model_image),text=prompt,return_tensors='pt').to('cuda')
            with torch.no_grad():result=model(**data)
            proposed=processor.post_process_instance_segmentation(result,threshold=.3,mask_threshold=.5,
                target_sizes=data.get('original_sizes').tolist())[0]
            masks=proposed['masks'].cpu().numpy().astype(bool);scores=proposed['scores'].cpu().numpy()
            if not len(masks):masks=np.zeros((0,crop[3]-crop[1],crop[2]-crop[0]),bool)
            masks = source_masks(masks, (h, w), crop)
            cached[key]=(masks,scores)
        masks,scores=cached[key]
        masks=masks & validity.get(frame,(h,w))[None]
        inputs.update({str(resolve(p).resolve()):expected for p,expected in validity.inputs.items()})
        roi=request.get('roi',[0,0,w,h])
        chosen=select_mask(masks,scores,roi,request.get('allow_large',False))
        row={'entity':entity,'frame':frame,'prompt':prompt,'source':str(source.resolve()),'source_sha256':digest(source),
             'geometry_fit_scope':'unknown','geometry_fit_review':'',
             'selection':chosen,'roi':roi,'approved':False}
        if chosen is not None:
            folder=out/entity;folder.mkdir(exist_ok=True)
            mask=masks[chosen['index']]
            if request.get('inference_crop') is not None:
                row['inference_crop'] = crop
                local = mask[crop[1]:crop[3], crop[0]:crop[2]]
                row['inference_clipped_edges'] = [name for name, edge in zip(
                    ['left', 'top', 'right', 'bottom'], [local[:, 0], local[0], local[:, -1], local[-1]]) if edge.any()]
            path=folder/(Path(frame).stem+'_mask.png');Image.fromarray(mask.astype('uint8')*255).save(path)
            row['mask']=str(path);row['mask_sha256']=digest(path)
            yy,xx=np.nonzero(mask);crop=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];row['crop']=crop
            overlay=image.copy();overlay[mask]=(overlay[mask]*.6+np.array([0,200,100])*.4).astype('uint8')
            overlay=cv2.resize(overlay,(960,round(h*960/w)))
            Image.fromarray(overlay).save(folder/(Path(frame).stem+'_overlay.jpg'),quality=92)
        records.append(row)
        (out/'observations.json').write_text(json.dumps({'version':1,'records':records,'input_sha256':inputs,
            'scope':'Capture-backed masks for agent review; no task dimensions/CAD and no automatic shape/photorealism acceptance.'},indent=2)+'\n')
        print(f'[environment.observations] {number+1}/{len(requests)} {entity} {frame} '+('mask' if chosen else 'no reliable proposal'),flush=True)
    for path,expected in inputs.items():
        if digest(path)!=expected:raise ValueError('Observation input changed: '+path)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session');parser.add_argument('--output',required=True)
    parser.add_argument('--config')
    args=parser.parse_args();prepare(Session(args.session),args.output,args.config)


if __name__=='__main__':main()
