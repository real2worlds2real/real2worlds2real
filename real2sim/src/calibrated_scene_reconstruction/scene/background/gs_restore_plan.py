"""Make clear captured furniture visible in the agent's next work queue.

A mesh proposal is not evidence that captured GS should be erased. Source
scores only recommend review; instance correspondence, novel views and the
actual composed field remain mandatory and are never automatically approved.
"""
import argparse
import json
from pathlib import Path
from calibrated_scene_reconstruction.scene.review.environment import digest


def recommend(plan,composition=None):
    scores=(composition or {}).get('scores',[]);result=[]
    for entity,row in plan['routes'].items():
        good={r['source']:r for r in row.get('scores',[]) if r.get('detail_pixels',0)>=32 and r.get('blur_fraction',1)<=.2
              and r.get('rgb_mae')is not None and r['rgb_mae']<=.12}
        views=sorted(good.values(),key=lambda r:r['rgb_mae']+r['blur_fraction']*.05)
        overpaint=[r for r in scores if r['entity']==entity and r.get('scored') and r.get('overpainted_pixels',0)>0]
        lost=sum(r['overpainted_pixels'] for r in overpaint)
        result.append({'entity':entity,'action':'review_gs_regions' if len(good)>=2 else 'agent_mesh_refinement',
            'source_views':views,'overpainted_pixels':lost,'reviewed':False,
            'required':['verify source instance correspondence','inspect novel views','compare actual retained field against original/source','review actual mesh/GS composite'],
            'basis':'Two source views resolve captured detail; inspect preservation before replacing photographed furniture.' if len(good)>=2
                else 'Source evidence insufficient or blurry; refine/complete native geometry and appearance.'})
    return sorted(result,key=lambda r:(r['action']!='review_gs_regions',-r['overpainted_pixels'],-len(r['source_views'])))


def prepare(plan_path,output,composition_path=None):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve();plan=json.loads(plan_path.read_text());inputs={str(plan_path):digest(plan_path),str(Path(__file__).resolve()):digest(__file__)}
    composition=None
    if composition_path:
        composition_path=Path(composition_path).resolve();composition=json.loads(composition_path.read_text());inputs[str(composition_path)]=digest(composition_path)
    if output.exists():raise ValueError('Use a new GS preservation work queue')
    result={'version':1,'work':recommend(plan,composition),'input_sha256':inputs,'pass':False,
            'scope':'Agent recommendations, not GS/shape/photorealism approval. Existing native geometry does not justify erasing reliable captured appearance.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan');p.add_argument('--composition');p.add_argument('--output',required=True);a=p.parse_args()
    r=prepare(a.plan,a.output,a.composition);print('GS source-review candidates',sum(w['action']=='review_gs_regions' for w in r['work']))


if __name__=='__main__':main()
