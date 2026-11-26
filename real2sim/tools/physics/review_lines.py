"""Export source-only line measurements after explicit agent fragment review."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require
from calibrated_scene_reconstruction.physics.process.line_features import reviewed_groups
from calibrated_scene_reconstruction.physics.process.stripe_centers import refine


def run(args):
    review=read(args.review);proposals=read(review['proposals'])
    for path,expected in proposals['source_sha256'].items():
        require(digest(path)==expected,'Changed source proposal input: '+path)
    require(review.get('reason') and review['frames'],'Explain source fragment selection')
    require(review.get('centerline_refinement') or getattr(args,'edge_group_proposals',False),
            'Centerline observations need original-pixel refinement. Use --edge-group-proposals only for explicitly uncertain boundary proposals, not precise object centers.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    regions=np.load(review['regions'],allow_pickle=False)['features']
    cap=cv2.VideoCapture(review['video']);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    require(len(regions)==count,'Source region/video timebase mismatch')
    times=np.arange(count)/fps
    y=regions[:,1,None]+regions[:,review['unit_feature_index'],None]*np.asarray(review['probe_y_offsets'])
    values=np.full((count,review['part_count'],2),np.nan);records=[]
    bank={r['frame']:r for r in proposals['frames']}
    for selected in review['frames']:
        frame=selected['frame'];require(frame in bank and selected.get('reason'),'Explain each source frame selection')
        values[frame],groups=reviewed_groups(bank[frame]['lines'],selected['groups'],y[frame],review['part_count'],
            max_spread_px=review['max_edge_spread_px'],max_extrapolation_ratio=review['max_extrapolation_ratio'])
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();require(ok,'Source frame unavailable')
        if review.get('centerline_refinement'):
            refined,groups=refine(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),groups,y[frame],review['centerline_refinement'])
            values[frame,:len(refined)]=refined
        x1,y1,x2,y2=bank[frame]['crop'];image=image[y1:y2,x1:x2].copy()
        for group in groups:
            for endpoints in group['source_segments']:
                a,b=np.rint(np.asarray(endpoints)-[x1,y1]).astype(int)
                cv2.line(image,tuple(a),tuple(b),(60,245,100),1,cv2.LINE_AA)
            if group.get('centerline_segment'):
                a,b=np.rint(np.asarray(group['centerline_segment'])-[x1,y1]).astype(int)
                cv2.line(image,tuple(a),tuple(b),(255,100,30),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f'review_{frame:05d}.jpg'),image)
        records.append({'frame':frame,'groups':groups,'probe_y':y[frame].tolist(),'reason':selected['reason']})
    cap.release()
    np.savez_compressed(out/'line_observations.npz',t=times,**{review['channel']:values})
    np.savez_compressed(out/'line_probe_coordinates.npz',t=times,y=y)
    paths=[args.review,review['proposals'],review['video'],review['regions'],__file__,
           'src/calibrated_scene_reconstruction/physics/process/line_features.py','src/calibrated_scene_reconstruction/physics/process/stripe_centers.py']
    write(out/'review.json',{'pass':True,'reason':review['reason'],'frames':records,
        'measurement_kind':'source_pixel_stripe_centerline' if review.get('centerline_refinement') else 'edge_group_proposal',
        'source_sha256':{str(Path(p).resolve()):digest(p) for p in paths},
        'scope':'Reviewed source-visible segments only. Probe intercepts encode line geometry; unseen endpoints/part identities are not asserted.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    p.add_argument('--edge-group-proposals',action='store_true',help='Export explicitly uncertain boundary groups; these are not object centerlines')
    run(p.parse_args())
