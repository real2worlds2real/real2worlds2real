"""Compose scene inputs for demonstration retargeting."""
from calibrated_scene_reconstruction.datagen.demo.generate import main
from calibrated_scene_reconstruction.scene.session import Session

if __name__ == "__main__":
    main(session_factory=Session)
