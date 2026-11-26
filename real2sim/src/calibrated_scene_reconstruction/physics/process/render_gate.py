"""Choose intact native evidence for delivery, candidate review or diagnosis."""
from pathlib import Path

from . import workflow
from .contract import digest,read,require


def select(recording,*,candidate=False,diagnostic=False,round_number=None):
    require(not(candidate and diagnostic),'Choose one native render mode')
    require(round_number is None or diagnostic,
            'Historical rounds are diagnostic only; candidate and delivery use the current round')
    if not(candidate or diagnostic):return Path(workflow.accepted(recording)['round'])
    if round_number is None:
        expected={'failed','rejected'} if diagnostic else {'awaiting_review'}
        require(workflow.status(recording)['state'] in expected,
                'Diagnostic rendering requires an intact failed or rejected result' if diagnostic else
                'Candidate rendering requires an intact numerically passed result awaiting review')
        root=workflow.current(recording)
    else:
        require(type(round_number) is int and 1<=round_number<=9999,'A valid process round number is required')
        root=Path(recording).resolve()/'physics_process'/f'round_{round_number:04d}'
        require(root.is_dir() and root.resolve().parent==root.parent.resolve(),
                'Historical round must belong to this recording')
        require(workflow.intact(root),'Historical diagnostic requires intact inputs and evidence')
    require((root/'result_hashes.json').is_file(),'A sealed native result is required for rendering')
    result=read(root/'result.json')
    if diagnostic:
        rejected=(root/'review.json').is_file() and read(root/'review.json').get('pass') is False
        require(result.get('pass') is False or rejected,'Diagnostic rendering requires a failed or rejected result')
    launches=sorted((root/'launches').glob('[0-9][0-9][0-9][0-9].json'))
    require((root/'started.json').is_file() and launches and read(launches[-1]).get('returncode')==0
            and not (root/'execution_failure.json').exists(),
            'Completed native execution is required; partial trials cannot be rendered as a completed round')
    require(result.get('best') and result.get('engine',{}).get('name')=='isaacsim'
            and str(result['engine'].get('version','')).startswith('5.1'),
            'Completed native Isaac 5.1 best-candidate evidence required')
    trial=root/'trials'/f"{result['best']['index']:04d}" if 'index' in result['best'] else None
    # Never guess which native trial belongs to a result.
    require(trial is not None and all((trial/name).is_file() for name in ['native_initial.usdc','native_states.npz','bindings.json']),
            'Native trial, source geometry, states and bindings required for rendering')
    artifacts=result['best'].get('artifacts',{})
    for name in ['native_initial.usdc','native_states.npz','bindings.json']:
        relative=str((trial/name).relative_to(root))
        require(artifacts.get(relative)==digest(trial/name),'Best-candidate render evidence must match the sealed result')
    return root


def validate_timing(native_t, camera_t, dt):
    """Recorded states use the nearest native step, independent of video rate."""
    import numpy as np

    native=np.asarray(native_t,float);source=np.asarray(camera_t,float)
    require(np.isfinite(dt) and dt>0,'Positive simulation timestep required')
    require(native.ndim==source.ndim==1 and len(native)==len(source) and len(source)>=2,
            'Trace and camera need matching timestamp arrays with at least two samples')
    require(np.isfinite(native).all() and np.isfinite(source).all()
            and np.all(np.diff(native)>0) and np.all(np.diff(source)>0),
            'Finite, increasing native and camera timestamps required')
    require(np.max(abs(native-source))<=dt/2+1e-8,
            'Trace and camera timebase differ by more than half a native simulation step')
