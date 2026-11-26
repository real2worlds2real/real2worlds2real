"""Launch Isaac with an exclusive native log for each process/run.

Kit's timestamp-only default can collide when validations start in parallel.
Keep startup configuration here; simulator APIs are imported only at launch.
"""
import os
from pathlib import Path
import tempfile


from calibrated_scene_reconstruction.core.environment import setting

def configuration(launch_config=None, *, log_root=None):
    config=dict(launch_config or {})
    extra=config.get('extra_args',[])
    if not isinstance(extra,list) or any(not isinstance(x,str) for x in extra):
        raise ValueError('Isaac extra_args must be a list of strings')
    if any(x.startswith('--/log/file') for x in extra):
        raise ValueError('Configure the native log folder, not a shared Kit log file')
    root=log_root or setting('NATIVE_LOG_DIR')
    if root is not None:Path(root).mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='r2s2r_isaac_',dir=root)).resolve()
    config['extra_args']=[*extra,'--/log/file='+str(folder/'kit.log')]
    return config


def SimulationApp(launch_config=None,*args,**kwargs):
    """Return the native SimulationApp, retaining its arguments and behavior."""
    from isaacsim import SimulationApp as NativeApp
    return NativeApp(configuration(launch_config),*args,**kwargs)
