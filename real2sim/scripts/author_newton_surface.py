"""Author the accepted Blender control mesh as a Newton-ready USD surface."""

import json
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


def main():
    folder = Path(sys.argv[1]).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["asset_kind"] not in {"deformable", "hybrid"}:
        raise ValueError("expected deformable asset")
    mesh_data = json.loads((folder / manifest["sim_mesh"]).read_text())
    points = [Gf.Vec3f(*values) for values in mesh_data["vertices"]]
    faces = mesh_data["faces"]
    output = folder / "newton_surface.usda"
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/SimMesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    mesh.GetPrim().CreateAttribute("remo:sourceObject", Sdf.ValueTypeNames.String).Set(mesh_data["object_name"])
    mesh.GetPrim().CreateAttribute("remo:pinIndices", Sdf.ValueTypeNames.IntArray).Set(mesh_data["pin_indices"])
    root.GetPrim().CreateAttribute("remo:sourcePhysics", Sdf.ValueTypeNames.String).Set(json.dumps(mesh_data.get("source_physics", [])))
    root.GetPrim().CreateAttribute("remo:constraintPolicy", Sdf.ValueTypeNames.String).Set("Pins are optional test or grasp constraints, enabled by the runtime loader")
    stage.GetRootLayer().Save()
    print(
        f"NEWTON_SURFACE_AUTHORED points={len(points)} triangles={len(faces)} "
        f"pins={len(mesh_data['pin_indices'])} path={output}"
    )


if __name__ == "__main__":
    main()
