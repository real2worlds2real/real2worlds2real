"""Explain rejected source landmark intervals without consulting simulation."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require
from calibrated_scene_reconstruction.physics.process.surface_features import annotated_tracks


def audit(plan):
    capture=cv2.VideoCapture(plan['video']);rows=[]
    try:
        for interval in plan['intervals']:
            images=[];capture.set(cv2.CAP_PROP_POS_FRAMES,interval['start'])
            for frame in range(interval['start'],interval['end']+1):
                ok,image=capture.read();require(ok,'Missing source frame during track audit')
                images.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            kwargs={k:plan[k] for k in ['max_fb_px','max_lk_error','max_consistency_px']}
            forward=annotated_tracks(images,interval['pixels'],**kwargs)
            checked=annotated_tracks(images,interval['pixels'],end_pixels=interval.get('end_pixels'),**kwargs)
            for j,name in enumerate(plan['ids']):
                endpoint=forward[-1,j];declared=interval.get('end_pixels')
                error=None if declared is None or not np.isfinite(endpoint).all() else float(np.linalg.norm(endpoint-declared[j]))
                rows.append({'id':name,'start':interval['start'],'end':interval['end'],
                    'forward_valid_frames':int(np.isfinite(forward[:,j]).all(1).sum()),
                    'reviewed_valid_frames':int(np.isfinite(checked[:,j]).all(1).sum()),
                    'source_tracked_endpoint':endpoint.tolist() if np.isfinite(endpoint).all() else None,
                    'independent_endpoint_error_px':error,
                    'endpoint_limit_px':plan['max_consistency_px'],
                    'endpoint_disagrees':error is not None and error>plan['max_consistency_px']})
    finally:capture.release()
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    plan=read(args.plan);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result={'intervals':audit(plan),'source_sha256':{str(Path(p).resolve()):digest(p) for p in [args.plan,plan['video'],__file__]},
        'scope':'Source-only tracking/annotation diagnostics. A rejected interval requires raw source review; never replace annotations using native predictions or automatically relax the consistency limit.'}
    write(out/'report.json',result);print(result['intervals'])
