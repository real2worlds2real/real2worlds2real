"""Monitor the current worker's shared GPU headroom, never another process."""
import os
import subprocess
import threading

from .contract import write


from calibrated_scene_reconstruction.core.environment import setting

def watch(directory):
    """Return a stop event; on exhaustion record evidence and exit this worker only."""
    done=threading.Event()
    if setting('SHARED_GPU_ACTIVE')!='1':return done
    floor=int(setting('SHARED_GPU_RESERVE_MIB','8192'))
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    attempt=setting('PROCESS_ATTEMPT','0001')

    def monitor():
        while not done.is_set():
            reason=None;free=None
            try:
                free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True,timeout=3).strip())
                if free<floor:reason='Shared GPU headroom fell below the declared reserve'
            except (OSError,ValueError,subprocess.SubprocessError) as exc:reason='GPU headroom query failed: '+str(exc)
            if reason:
                write(directory/'launches'/f'{int(attempt):04d}_resource_stop.json',
                      {'reason':reason,'free_mib':free,'reserve_mib':floor})
                # An immediate exit avoids a stuck native callback retaining VRAM.
                # OS teardown destroys this worker's context; no external PID is signalled.
                os._exit(75)
            done.wait(.5)
    threading.Thread(target=monitor,daemon=True).start()
    return done
