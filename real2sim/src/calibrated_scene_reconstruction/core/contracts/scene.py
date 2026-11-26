"""Prepared scene inputs consumed by asset geometry, grasping and simulation adapters."""
from pathlib import Path
from typing import Any, Protocol


class SceneInputs(Protocol):
    dir: Path
    name: str
    config_dir: Path

    @property
    def scene_graph(self) -> dict: ...

    @property
    def manifest(self) -> dict: ...

    def __truediv__(self, path: str) -> Path: ...
    def config(self, name: str, default: Any = None) -> Any: ...
    def read_json(self, path: str) -> Any: ...
