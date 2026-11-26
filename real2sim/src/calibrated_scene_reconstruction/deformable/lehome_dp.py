"""Legacy registered LeHome policy; new apps inject this client explicitly."""
import warnings
from calibrated_scene_reconstruction.policy.dp_official_client import DPOfficialClient
from calibrated_scene_reconstruction.tasks.lehome.policy import DPOfficialPolicy, register_policy

warnings.warn("Use the tasks.lehome adapter with an injected client", DeprecationWarning, stacklevel=2)
register_policy(DPOfficialClient)
__all__ = ["DPOfficialPolicy", "register_policy"]
