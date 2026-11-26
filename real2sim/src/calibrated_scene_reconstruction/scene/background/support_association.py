"""Associate sparse validation tracks with the observed support patch.

Sharing a height and projecting inside an occluded semantic mask does not make
a distant tabletop track part of this support. Preserve unassociated tracks for
review; they must not expand a different object's footprint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def local_tracks(gs, sparse, radius):
    gs=np.asarray(gs,float);sparse=np.asarray(sparse,float)
    if (gs.ndim!=2 or gs.shape[1]!=3 or len(gs)<10 or sparse.ndim!=2 or sparse.shape[1]!=3
            or not np.isfinite(gs).all() or not np.isfinite(sparse).all()
            or isinstance(radius,bool) or not .03<=radius<=.2):
        raise ValueError('Finite source patches and bounded association radius required')
    distance=cKDTree(gs).query(sparse)[0];accepted=distance<=radius
    if accepted.sum()<3:raise ValueError('Fewer than three independently reconstructed tracks associated with this support')
    return accepted,distance


def prepare(session, proposal, output, radius=.12):
    proposal=Path(proposal).resolve();output=Path(output).resolve()
    if (output.exists() or not output.is_relative_to(session.dir.resolve())
            or (session/'qa/independent_freeze.json').exists()):raise ValueError('New unfrozen support evidence required')
    q=json.loads(proposal.read_text());samples=Path(q['samples']).resolve()
    if (q['protected_task_state']!=protected_state(session) or digest(samples)!=q['samples_sha256']
            or not samples.is_relative_to(session.dir.resolve())):raise ValueError('Current source support evidence required')
    inputs={**q['input_sha256'],str(proposal):digest(proposal),str(samples):digest(samples),
            str(Path(__file__).resolve()):digest(__file__)}
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Source support evidence changed: '+p)
    cloud=np.load(samples);gs=cloud['gs'];sparse=cloud['sparse'];accepted,distance=local_tracks(gs,sparse,radius)
    output.mkdir(parents=True);np.savez_compressed(output/'samples.npz',gs=gs,sparse=sparse[accepted])
    np.savez_compressed(output/'association.npz',all_sparse=sparse,associated=accepted,nearest_gs_distance_m=distance)
    audit=output/'association.npz';inputs[str(audit)]=digest(audit)
    result={**q,'samples':str(output/'samples.npz'),'samples_sha256':digest(output/'samples.npz'),
            'sparse_samples':int(accepted.sum()),'sparse_height_m':float(np.median(sparse[accepted,2])),
            'input_sha256':inputs,'approved':False,
            'association':{'radius_m':radius,'all_sparse_samples':len(sparse),
                           'unassociated_samples':int((~accepted).sum()),'audit':str(audit)},'scope':__doc__.strip()}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Support association input changed')
    (output/'proposal.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('proposal');p.add_argument('--output',required=True)
    p.add_argument('--radius-m',type=float,default=.12);a=p.parse_args();q=prepare(Session(a.session),a.proposal,a.output,a.radius_m)
    print('Unapproved local support association',q['association'])


if __name__=='__main__':main()
