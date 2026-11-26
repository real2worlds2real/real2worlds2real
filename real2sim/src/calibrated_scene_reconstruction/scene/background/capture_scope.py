"""Reject cross-object GS evidence for native appearance cutouts.

Opaque background somewhere on a source ray is not evidence that the intended
object is reconstructed there. Masks must be associated with that same owner.
"""
import argparse
import json
from pathlib import Path


def audit(ownership, masks):
    checks=[]
    for entity,owner in ownership.items():
        declaration=owner.get('mesh_coverage',{})
        if owner.get('role')!='gap_fill' or declaration.get('compositing')!='captured_projection':continue
        declared=declaration.get('coverage_entity',entity)
        measured=masks.get(entity,{}).get('capture_coverage',{}).get('coverage_entity')
        checks.append({'entity':entity,'declared_source_owner':declared,'measured_source_owner':measured,
                       'pass':declared==entity and measured==entity})
    return {'pass':all(c['pass'] for c in checks),'checks':checks,
            'scope':__doc__.strip()+' Source masks still require semantic review, independent rays, opaque actual GS and full-scene appearance checks.'}


def validate(ownership, masks):
    result=audit(ownership,masks)
    if not result['pass']:
        raise ValueError('GS cutouts need source evidence scoped to the same owner: '+
                         ', '.join(c['entity'] for c in result['checks'] if not c['pass']))
    return result


def main():
    from calibrated_scene_reconstruction.scene.review.environment import digest
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.manifest).resolve();output=Path(a.output).resolve()
    if output.exists():raise ValueError('Use a new source-scope audit output')
    raw=json.loads(source.read_text());context=raw.get('splat_context') or raw
    result=audit(context.get('appearance_ownership',{}),context.get('gap_masks',{}))
    result['input_sha256']={str(source):digest(source),str(Path(__file__).resolve()):digest(__file__)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print('Per-owner GS source scope','PASS' if result['pass'] else 'FAIL')


if __name__=='__main__':main()
