# Sandbox start-up hook: cap each run's share of this machine's GPU (it also serves the real robot's live pipeline).
#
# The cap is applied when the run itself imports torch, not here. Importing torch at interpreter start would load the
# system libstdc++ before a policy can load its own: an ACT candidate preloads the newer libstdc++ its point-ops
# extension needs and only then imports torch, and with two libstdc++ copies mixed the first cuDNN convolution crashes.
import os
import sys

_FRACTION = os.environ.get("STATION_GPU_FRACTION")


def _cap(torch):
    try:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(_FRACTION))
    except Exception:  # noqa: BLE001 - a cap that cannot be set must not stop the run
        pass


class _CapWhenTorchLoads:
    """Meta-path finder that only wraps the loading of `torch`: once torch has executed, apply the cap."""

    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        import importlib.util
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            _cap(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "" and _FRACTION:
    if "torch" in sys.modules:
        _cap(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _CapWhenTorchLoads())
