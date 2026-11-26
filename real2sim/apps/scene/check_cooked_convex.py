"""Read-only SDK convex cooking audit of a preserved native input stage."""
import argparse
from pathlib import Path
import hashlib,json,traceback
p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage');p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--tolerance-m',type=float,required=True);a=p.parse_args()
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_bytes(Path(__file__).read_bytes())
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import omni.usd  # noqa: E402
from calibrated_scene_reconstruction.sim.cooked_convex import audit  # noqa: E402
try:
    source=Path(a.stage).resolve();checksum=hashlib.sha256(source.read_bytes()).hexdigest()
    context=omni.usd.get_context();context.open_stage(str(source))
    for _ in range(3):app.update()
    result=audit(context.get_stage(),a.root,out/'cooking',tolerance_m=a.tolerance_m)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=checksum:raise ValueError('Preserved stage changed')
    (out/'result.json').write_text(json.dumps({'pass':result['pass'],'stage_sha256':checksum,'task_approved':False,'cells':len(result['cells']),'maximum_error_m':max(row['sampled_surface_error_m'] for row in result['cells']),'simulation_stepped':False},indent=2)+'\n')
    print('COOKING_RESULT',json.loads((out/'result.json').read_text()),flush=True)
except Exception as e:
    (out/'error.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__,'traceback':traceback.format_exc()},indent=2)+'\n');traceback.print_exc()
finally:app.close()
