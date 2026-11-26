"""Sharing the card with another job: SCENE_RECONSTRUCTION_GPU_FRACTION=0.28 caps this process's CUDA memory to that fraction of it
(torch's per-process limit: an overrun raises OutOfMemoryError here instead of starving the other job). The GPU
stages call `share()` before their first allocation; unset, nothing changes. SCENE_RECONSTRUCTION_DEVICE=cpu runs them on the CPU
instead (`device()`), when the card is taken altogether: MoGe-2 and SAM 3 take minutes there, DA3 tens of minutes."""
import os


from calibrated_scene_reconstruction.core.environment import setting

def device():
    import torch

    return setting("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def share():
    frac = setting("GPU_FRACTION")
    if frac and device() == "cuda":
        import torch

        torch.cuda.set_per_process_memory_fraction(float(frac))
        print(f"[gpu] this process capped at {float(frac):.2f} of the card")
