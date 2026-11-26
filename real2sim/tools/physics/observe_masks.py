"""Measure reviewed source semantic masks with the native region operator."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from calibrated_scene_reconstruction.physics.process.contract import digest,read,write,require
from calibrated_scene_reconstruction.physics.process.region_observations import measure,definition


def observe(source,review_path,output):
    source=Path(source);review_path=Path(review_path);output=Path(output)
    require(not output.exists(),'Use a fresh source observation directory')
    manifest=read(source/'masks.json');review=read(review_path)
    require(review.get('pass') is True and review.get('reason') and review.get('mask_manifest_sha256')==digest(source/'masks.json'),
            'Bind explicit source identity/completeness review to the segmentation manifest')
    for path,expected in manifest['source_sha256'].items():require(digest(path)==expected,'Changed source segmentation evidence')
    scope=definition(review['semantic_part'],review['scope'],'centroid',review['opening'])
    rows=manifest['frames'];require([r['frame'] for r in rows]==list(range(len(rows))),
                                   'Source mask frames must cover the video in order')
    accepted=review['trusted_frames'];complete=review['complete_frames'];features=[];hashes={str(p.resolve()):digest(p) for p in [source/'masks.json',review_path,Path(__file__)]}
    overrides={}
    for variant in review.get('variants',[]):
        folder=Path(variant['source_masks']);other=read(folder/'masks.json');video=str(Path(review['video']).resolve())
        require(video in manifest['source_sha256'] and other['source_sha256'].get(video)==manifest['source_sha256'][video]
                and other['fps']==manifest['fps'],'Source mask variants must share the original video/timebase')
        for path,expected in other['source_sha256'].items():require(digest(path)==expected,'Changed source mask variant')
        hashes[str((folder/'masks.json').resolve())]=digest(folder/'masks.json')
        frames={r['frame']:r for r in other['frames']}
        for i in variant['frames']:
            require(i not in overrides and i in frames and 0<=i<len(rows) and frames[i]['t']==rows[i]['t'],
                    'Unique source-reviewed variant frames required')
            overrides[i]=folder/variant.get('part',review['part'])/f'{i:05d}.png'
    from calibrated_scene_reconstruction.physics.process import region_observations
    hashes[str(Path(region_observations.__file__).resolve())]=digest(region_observations.__file__)
    def includes(frames,i):return frames=='all' or isinstance(frames,list) and i in frames
    for row in rows:
        i=row['frame'];path=overrides.get(i,source/review['part']/f'{i:05d}.png');mask=cv2.imread(str(path),0)
        require(mask is not None,'Missing source mask file');hashes[str(path.resolve())]=digest(path)
        if not includes(accepted,i) or scope['scope']=='complete_part' and not includes(complete,i) or not np.any(mask):
            features.append([np.nan]*5);continue
        center,extent=measure(mask,opening=scope['opening']);features.append([*center,float(np.count_nonzero(mask)),*extent])
    output.mkdir(parents=True);features=np.asarray(features);times=np.array([r['t'] for r in rows])
    np.savez_compressed(output/'observations.npz',t=times,features=features,centroid=features[:,:2],extent=features[:,3:])
    write(output/'observations.json',{'definition':scope,'source_sha256':hashes,'valid_frames':np.flatnonzero(np.isfinite(features).all(1)).tolist(),
        'scope':'Reviewed source-only mask measurements; missing/partial regions are never filled from simulation. This is not native calibration or acceptance.'})
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--source-masks',required=True);ap.add_argument('--review',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();print(observe(args.source_masks,args.review,args.output))
