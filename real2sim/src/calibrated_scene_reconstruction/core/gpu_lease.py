"""Optional admission checks between serial native trials on a shared GPU."""
import os
import subprocess
import time


from calibrated_scene_reconstruction.core.environment import setting

def wait_for_capacity():
    """Wait before creating a native app; never stop or alter another job.

    Enabled only by run_shared_isaac.sh. This is an admission estimate, not
    a memory cap or a reservation respected by unrelated GPU processes.
    """
    if setting('SHARED_GPU_ACTIVE') != '1':
        return
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    if not gpu.isdecimal():
        raise ValueError('Shared trials require one physical GPU index')
    expected = int(setting('ISAAC_EXPECTED_VRAM_MIB', '8192'))
    reserve = int(setting('SHARED_GPU_RESERVE_MIB', '8192'))
    if expected <= 0 or reserve < 0:
        raise ValueError('Positive expected VRAM and nonnegative headroom required')
    attempts = 0
    while True:
        value = subprocess.check_output([
            'nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True)
        free = int(value.strip())
        if free >= expected + reserve:
            print(f'[shared_gpu] Trial admitted: {free}MiB free, '
                  f'{expected}MiB expected plus {reserve}MiB headroom.', flush=True)
            return
        if attempts % 6 == 0:
            print(f'[shared_gpu] Waiting: {free}MiB free; need '
                  f'{expected + reserve}MiB before next trial.', flush=True)
        attempts += 1
        time.sleep(5)
