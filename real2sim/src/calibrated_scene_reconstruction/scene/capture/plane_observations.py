"""Add actual-view surface observations when the primary photo has too few tracks."""
import argparse
import json

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.session import Session


def select_floor(candidates, supports, *, allow_empty=False):
    """Reject floor proposals that actually describe an independently labelled table."""
    if not candidates:
        raise ValueError('No observed floor candidates')
    shape = candidates[0]['mask'].shape
    support = np.zeros(shape, bool)
    for mask in supports:
        if mask.shape != shape:
            raise ValueError('Surface exclusion masks have inconsistent sizes')
        support |= mask
    excluded = cv2.dilate(support.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    output, decisions = np.zeros(shape, bool), []
    for proposal in candidates:
        mask = np.asarray(proposal['mask'], bool)
        overlap = float((mask & support).sum()/max(mask.sum(), 1))
        accepted = bool(mask.any() and overlap <= .25)
        clean = mask & ~excluded
        accepted = accepted and bool(clean.any())
        if accepted:
            output |= clean
        decisions.append({'prompt':proposal['prompt'],'confidence':proposal['confidence'],
                          'pixels':int(mask.sum()),'support_overlap':overlap,'accepted':accepted,
                          'reason':'Independent support labels contradict floor' if not accepted else 'Observed floor excluding support'})
    if not output.any() and not allow_empty:
        raise ValueError('All floor candidates contradict independently labelled support; acquire another view')
    return output, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--role', choices=['floor', 'support'])
    parser.add_argument('--frames', nargs='+', help='registered frame filenames of this capture')
    parser.add_argument('--request', action='store_true', help='use the insufficient-tracks request from alignment')
    args = parser.parse_args()
    s = Session(args.session)
    if args.request:
        request = json.loads((s/'qa/placement_plane_request.json').read_text())
        args.role, args.frames = request['role'], request['frames']
    if not args.role or not args.frames:
        parser.error('A surface role and current registered frames are required')
    configured = s.config('placement_planes.json', {}).get(args.role, {})
    prompts = configured.get('prompts', ['floor' if args.role == 'floor' else 'wooden tabletop'])
    exclusions = configured.get('support_exclusion_prompts', ['tabletop']) if args.role == 'floor' else []
    if any(not isinstance(value, list) or not value or any(not isinstance(p,str) or not p.strip() for p in value)
           for value in [prompts, *([exclusions] if args.role == 'floor' else [])]):
        raise ValueError('Plane observation prompts must be nonempty lists of individual descriptions')
    directory = s/'parsing/placement_plane_masks'
    directory.mkdir(parents=True, exist_ok=True)
    from calibrated_scene_reconstruction.scene.objects.object_hull import CropSam3
    model = CropSam3(threshold=.3)
    tiles, inputs = [], []
    for name in args.frames:
        if '/' in name or not name.endswith('.jpg'):
            parser.error('Frame names must be local capture jpg filenames')
        image = cv2.imread(str(s/'splat/data/images'/name))
        if image is None:
            parser.error('Missing current-capture frame '+name)
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        candidates = [{'mask':mask,'confidence':float(confidence),'prompt':prompt}
                      for prompt in prompts for mask,confidence in model.instances(rgb,prompt,w/2,h/2,max(w,h))]
        if args.role == 'floor':
            supports = [mask for prompt in exclusions
                        for mask,_ in model.instances(rgb,prompt,w/2,h/2,max(w,h))]
            try:
                mask, decisions = select_floor(candidates, supports)
            except ValueError as error:
                # Some registered images do not observe this surface. They
                # contribute no plane evidence; do not promote a table instead.
                dst = directory/(args.role+'__'+name[:-4]+'.png')
                if dst.exists():
                    raise ValueError('Preserve earlier plane masks before revising rejected observations') from error
                inputs.append({'frame':name,'prompts':prompts,'accepted':False,'reason':str(error),
                               'support_exclusion_prompts':exclusions})
                rejected=(image*.3).astype(np.uint8)
                cv2.putText(rejected,name+' rejected: no independent floor',(10,30),0,.6,(0,0,255),2)
                tiles.append(cv2.resize(rejected,(640,360)))
                continue
            confidence = max(c['confidence'] for c in decisions if c['accepted'])
        else:
            if not candidates:
                raise ValueError(f'No actual {prompts} instance in {name}')
            chosen = max(candidates, key=lambda item:item['mask'].sum()*item['confidence'])
            mask, confidence, decisions = chosen['mask'], chosen['confidence'], []
        dst = directory/(args.role+'__'+name[:-4]+'.png')
        cv2.imwrite(str(dst), np.where(mask, 0, 255).astype(np.uint8))
        overlay = (image*.3).astype(np.uint8)
        overlay[mask] = image[mask]
        cv2.drawContours(overlay, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 220, 0), 2)
        cv2.putText(overlay, name, (10, 30), 0, .8, (0, 255, 255), 2)
        tiles.append(cv2.resize(overlay, (640, 360)))
        inputs.append({'frame': name, 'prompts': prompts, 'confidence': confidence, 'pixels': int(mask.sum()),
                       'accepted':True,'support_exclusion_prompts':exclusions,'candidate_decisions':decisions})
    (directory/(args.role+'_inputs.json')).write_text(json.dumps(inputs, indent=2)+'\n')
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(directory/(args.role+'_review.jpg')), np.vstack([
        np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))
    if not any(row.get('accepted') for row in inputs):
        raise ValueError('No independently observed surface in the requested frames; review evidence and acquire other views')
    print(json.dumps(inputs, indent=2))


if __name__ == '__main__':
    main()
