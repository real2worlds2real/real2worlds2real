"""Re-export a saved Blender asset with transform ancestors and joint metadata intact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "real2sim_harness"))
from blender_harness import export_asset_glb  # noqa: E402
from blender_surface import is_asset_object  # noqa: E402


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(bpy.data.filepath)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and is_asset_object(obj)
    ]
    report = export_asset_glb(output / "object.glb", meshes)
    report.update(source_scene=str(source), source_sha256=digest)
    (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
