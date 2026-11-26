"""Read/write physical calibration documents without changing their nested layouts."""
import json
from pathlib import Path
from calibrated_scene_reconstruction.core.contracts.physics import PhysicalParameters


def read_document(path) -> PhysicalParameters:
    return json.loads(Path(path).read_text())


def write_document(path, document, **options):
    return Path(path).write_text(json.dumps(document, **options))


def scene_values(document: PhysicalParameters) -> PhysicalParameters:
    return document.get("physics", document)
