"""Legacy demonstration CLI adapter; scene construction stays at this compatibility boundary."""
import warnings
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.datagen.demo import generate as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]

warnings.warn("Use tools/datagen/generate.py or the prepared-input datagen API", DeprecationWarning, stacklevel=2)


def __getattr__(name):
    return getattr(_implementation, name)


def main(argv=None):
    return _implementation.main(argv, session_factory=Session)


if __name__ == "__main__":
    main()
