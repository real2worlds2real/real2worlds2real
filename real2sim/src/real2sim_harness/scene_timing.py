"""Fixed-step scene clock shared by physics, actions and capture scheduling."""

import math
from dataclasses import dataclass


def override_frequency(default_hz, default_steps, physics_hz):
    """Keep a runner's physical duration when an agent requests a finer clock."""
    previous = SceneTiming(default_hz, default_steps)
    SceneTiming(physics_hz)
    steps = previous.duration_s * physics_hz
    if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
        raise ValueError("Configured duration must align with the requested physics timestep")
    if round(steps) < 1:
        raise ValueError("Configured duration is shorter than the requested physics timestep")
    return physics_hz, round(steps)


@dataclass(frozen=True)
class SceneTiming:
    physics_hz: int = 120
    steps: int = 240
    capture_fps: int = 0

    def __post_init__(self):
        for name in ("physics_hz", "steps", "capture_fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "capture_fps" else 1):
                raise ValueError(f"{name} must be an {'nonnegative' if name == 'capture_fps' else 'positive'} integer")
        if self.capture_fps and self.physics_hz % self.capture_fps:
            raise ValueError("Capture FPS must divide the configured physics frequency")

    @property
    def dt(self):
        return 1 / self.physics_hz

    @property
    def duration_s(self):
        return self.steps * self.dt

    def action_step(self, seconds):
        """Actions occur before stepping, exactly once at a solver boundary."""
        if not math.isfinite(seconds) or not 0 <= seconds < self.duration_s:
            raise ValueError("Action time must be finite and within the simulation")
        index = round(seconds * self.physics_hz)
        if not math.isclose(index, seconds * self.physics_hz, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Action time must align with the configured physics timestep")
        return index

    def capture_after_step(self, step):
        # Preserve the existing first capture after one step, then exact FPS.
        return bool(self.capture_fps and step % (self.physics_hz // self.capture_fps) == 0)
