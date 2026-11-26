"""Version-independent scene descriptions and portable USD composition.

Import OpenUSD lazily: normal object generation does not require the scene extra.
No simulator or Isaac Lab version is selected here.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .measurement import rigid_transform
from .scene_physics import plan_scene_physics
from .sim_bundle import verify_bundle


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An input path must be a nonempty string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def validate_scene(spec, base):
    """Resolve a scene request without silently inventing metric evidence."""
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Expected scene schema_version 1")
    if set(spec) - {"schema_version", "frame", "backend", "objects", "backgrounds", "supports", "support_requirements", "notes", "review_camera", "review_camera_path", "review_render", "actions"}:
        raise ValueError("Unknown scene fields")
    frame = spec.get("frame", {})
    if frame.get("meters_per_unit") != 1 or frame.get("up_axis") != "Z":
        raise ValueError("The assembly interface uses metres and Z-up; convert inputs explicitly")
    if not isinstance(frame.get("scale_source"), str) or not frame["scale_source"].strip():
        raise ValueError("Describe the scale source, including an estimated scale when unmeasured")
    if not isinstance(frame.get("scale_verified"), bool):
        raise ValueError("Explicitly state whether scale is verified")
    backend = spec.get("backend", "auto")
    if backend not in {"auto", "physx", "newton"}:
        raise ValueError("Choose a scene backend, auto, physx or newton")
    result = {"schema_version": 1, "frame": frame, "backend": backend,
              "notes": spec.get("notes", "")}
    if "review_render" in spec:
        render = spec["review_render"]
        if not isinstance(render, dict) or set(render)-{"authored_light_exposure", "default_lighting"}:
            raise ValueError("Invalid review render settings")
        exposure = _number(render.get("authored_light_exposure", 0), "authored light exposure")
        default = render.get("default_lighting", True)
        if not -16 <= exposure <= 16 or type(default) is not bool:
            raise ValueError("Use light exposure from -16 to 16 stops and a boolean default lighting policy")
        result["review_render"] = {"authored_light_exposure":exposure, "default_lighting":default}
    if spec.get("review_camera") is not None:
        from .scene_motion import view_spec
        result['review_camera'] = view_spec(spec['review_camera'])
    ids = set()
    for group in ("objects", "backgrounds", "supports"):
        entries = spec.get(group, [])
        if not isinstance(entries, list) or len(entries) > 200:
            raise ValueError(f"{group} must be a bounded list")
        result[group] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Scene entries must be objects")
            name = entry.get("id", "")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) or name in ids:
                raise ValueError("Scene IDs must be unique USD identifiers")
            ids.add(name)
            allowed = {"id", "transform", "notes"}
            allowed |= ({"bundle"} if group == "objects" else
                        {"path", "format", "root_prim", "observed", "editable_source"} if group == "backgrounds" else
                        {"size_m", "static_friction", "dynamic_friction", "restitution", "friction_combine_mode"})
            if set(entry) - allowed:
                raise ValueError(f"Unknown fields on {name}")
            item = dict(entry)
            item["transform"] = rigid_transform(entry.get("transform")).tolist()
            if group == "objects":
                item["bundle"] = str(_path(entry.get("bundle"), base))
            elif group == "backgrounds":
                item["path"] = str(_path(entry.get("path"), base))
                if entry.get("editable_source") is not None:
                    editable = _path(entry["editable_source"], base)
                    if entry.get("format") != "mesh" or editable.suffix.lower() != ".blend" or not editable.is_file():
                        raise ValueError("Editable background source must be a Blender file for a mesh background")
                    item["editable_source"] = str(editable)
                if entry.get("format") not in {"mesh", "nurec_volume", "particle_field"}:
                    raise ValueError("Declare the background format for capability negotiation")
                if not isinstance(entry.get("observed"), bool):
                    raise ValueError("Backgrounds must distinguish observed capture from generated completion")
                root = entry.get("root_prim")
                if root is not None and (not isinstance(root, str) or not re.fullmatch(r"/[A-Za-z_][A-Za-z0-9_]*", root)):
                    raise ValueError("A background root_prim must name a root-level prim")
            else:
                size = entry.get("size_m")
                if not isinstance(size, list) or len(size) != 3:
                    raise ValueError("Support boxes need three positive sizes")
                item["size_m"] = [_number(v, "size_m", positive=True) for v in size]
                for key, default in (("static_friction", 0.6), ("dynamic_friction", 0.5), ("restitution", 0.0)):
                    item[key] = _number(entry.get(key, default), key)
                    if item[key] < 0 or (key == "restitution" and item[key] > 1):
                        raise ValueError(f"Invalid {key}")
                if "friction_combine_mode" in entry:
                    mode = entry["friction_combine_mode"]
                    if backend != "physx":
                        raise ValueError("Support friction_combine_mode requires an explicit physx backend")
                    if not isinstance(mode, str) or mode not in {"average", "min", "multiply", "max"}:
                        raise ValueError("Invalid support friction_combine_mode")
            result[group].append(item)
    if not result["objects"]:
        raise ValueError("An operable scene needs at least one generated asset")
    if "support_requirements" in spec:
        from .scene_support import support_requirements
        result["support_requirements"] = support_requirements(
            spec["support_requirements"], {r["id"] for r in result["objects"]}, {r["id"] for r in result["supports"]})
        by_id = {r["id"]: r for r in result["supports"]}
        for requirement in result["support_requirements"]:
            if not np.allclose(np.asarray(by_id[requirement["support_id"]]["transform"])[:3, 2], [0, 0, 1], atol=1e-5):
                raise ValueError("Drop/settle support requirements need horizontal static support boxes")
    if spec.get("review_camera_path") is not None:
        from .scene_motion import camera_path
        result["review_camera_path"] = camera_path(spec["review_camera_path"])
    if spec.get("actions") is not None:
        from .scene_motion import actions
        result["actions"] = actions(spec["actions"], [item["id"] for item in result["objects"]])
    return result


def _usd():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    except ImportError as exc:
        raise RuntimeError("Scene composition requires the optional real2sim[scene] dependencies") from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


def _dependency_file(value):
    from pxr import Ar
    value = str(value)
    while Ar.IsPackageRelativePath(value):
        value, _ = Ar.SplitPackageRelativePathOuter(value)
    return Path(value).resolve()


def _packaged_dependency_failures(folder, files):
    """Resolve USD dependencies from the delivery, not from source inventories."""
    _, _, _, _, _, _, UsdUtils = _usd()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(folder / 'scene.usda'))
    failures = [f'unresolved_packaged_dependencies:{len(unresolved)}'] if unresolved else []
    for value in [*(layer.realPath for layer in layers), *assets]:
        path = _dependency_file(value)
        if not path.is_relative_to(folder):
            failures.append('external_packaged_dependency:' + path.name)
        elif path.relative_to(folder).as_posix() not in files:
            failures.append('unretained_packaged_dependency:' + path.relative_to(folder).as_posix())
    return sorted(set(failures))


def _inventory(prim, Usd, UsdPhysics):
    prims = list(Usd.PrimRange(prim))
    return {
        "types": dict(Counter(str(p.GetTypeName()) for p in prims)),
        "schemas": dict(Counter(s for p in prims for s in p.GetAppliedSchemas())),
        "joints": sum(p.IsA(UsdPhysics.Joint) for p in prims),
        "rigid_bodies": sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),
        "colliders": sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims),
    }


def _check_stage(source, root_path=None):
    _, _, Usd, UsdGeom, _, _, UsdUtils = _usd()
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise ValueError(f"Cannot open USD {source.name}")
    root = stage.GetPrimAtPath(root_path) if root_path else stage.GetDefaultPrim()
    if not root or root.GetPath().GetParentPath() != stage.GetPseudoRoot().GetPath():
        raise ValueError("USD input needs a valid root-level default prim or explicit root_prim")
    if not UsdGeom.Xformable(root):
        raise ValueError("The referenced asset root must be transformable")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-9 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("USD input frame is not metres/Z-up; adapt it before assembly")
    for prim in Usd.PrimRange(root):
        if UsdGeom.Xformable(prim) and UsdGeom.Xformable(prim).GetResetXformStack():
            raise ValueError("resetXformStack would discard the scene placement transform")
        if prim.IsInstance() or prim.IsInstanceProxy():
            raise ValueError("Instanceable input roots need an explicit runtime adapter")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
    if unresolved:
        raise ValueError(f"Unresolved USD dependencies: {len(unresolved)}")
    return stage, root, layers, assets


def _relationship_failures(prim, Usd):
    failures = []
    stage = prim.GetStage()
    for p in Usd.PrimRange(prim):
        for prop in p.GetProperties():
            paths = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
            for path in paths:
                if path.IsAbsolutePath() and not stage.GetObjectAtPath(path):
                    failures.append(f"unresolved_target:{prop.GetPath()}->{path}")
    return failures


def _relocate_world_anchors(prim, matrix, Gf, Usd, UsdPhysics):
    """USD references remap links, but a joint's world-side anchor needs relocation."""
    rotation = matrix.ExtractRotationQuat()
    count = 0
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(p)
        for side in (0, 1):
            if getattr(joint, f"GetBody{side}Rel")().GetTargets():
                continue
            pos = getattr(joint, f"GetLocalPos{side}Attr")()
            quat = getattr(joint, f"GetLocalRot{side}Attr")()
            if pos.GetTimeSamples() or quat.GetTimeSamples():
                raise ValueError("Animated world anchors need an explicit adapter")
            value = matrix.Transform(Gf.Vec3d(pos.Get()))
            orientation = rotation * Gf.Quatd(quat.Get())
            pos.Set(Gf.Vec3f(value))
            quat.Set(Gf.Quatf(orientation))
            count += 1
    return count


def compose_scene(manifest, output):
    """Build a self-contained scene without flattening interactive assets."""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    spec = validate_scene(json.loads(manifest.read_text()), manifest.parent)
    inputs = [Path(v["bundle"]) for v in spec["objects"]] + [Path(v["path"]) for v in spec["backgrounds"]]
    inputs.extend(Path(v["editable_source"]) for v in spec["backgrounds"] if v.get("editable_source"))
    if output.exists() or any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError("Choose a fresh output directory separate from all source assets")
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = _usd()
    output.mkdir(parents=True)
    dependencies = {str(manifest): sha256_file(manifest)}
    checks, capabilities = [], {"usd_mesh"}
    try:
        participants, bundles = [], {}
        for entry in spec["objects"]:
            bundles[entry["id"]] = json.loads((Path(entry["bundle"]) / "asset_bundle.json").read_text())
            participants.append({"id": entry["id"], "kind": bundles[entry["id"]].get("asset_kind")})
        plan = plan_scene_physics(objects=participants, preferred_backend=spec["backend"])
        spec["backend"] = plan["backend"]
        atomic_json(output / "request.json", spec)
        atomic_json(output / "physics_plan.json", plan)
        capabilities.add(spec["backend"])
        from .scene_newton import materialization
        bindings = {name: materialization(bundle) for name, bundle in bundles.items()} if spec["backend"] == "newton" else {}
        conversions = [{"id": entry["id"], "asset_kind": bundles[entry["id"]]["asset_kind"],
                        "present_backend": bundles[entry["id"]].get("backend"), "target_backend": spec["backend"]}
                       for entry in spec["objects"] if
                       (spec["backend"] == "newton" and not bindings[entry["id"]]["ok"]) or
                       (spec["backend"] != "newton" and bundles[entry["id"]].get("backend") != spec["backend"])]
        if conversions:
            report = {"schema_version": 1, "ok": False, "status": "backend_conversion_required",
                      "physics_plan": plan, "conversion_requests": conversions, "materialization": bindings,
                      "simulation_verified": False,
                      "files": {str(p.relative_to(output)): sha256_file(p) for p in output.iterdir() if p.is_file()}}
            atomic_json(output / "scene_bundle.json", report)
            return report
        if spec["backend"] == "newton":
            capabilities.update(plan["required_capabilities"])
            atomic_json(output / "newton_bindings.json", {"schema_version": 1, "assets": bindings,
                        "scope": "Retained native inputs; runtime verification is required after materialization"})
        if spec["backend"] == "physx" and any(p["kind"] == "deformable" for p in participants):
            capabilities.update(plan["required_capabilities"])
            capabilities.add("physx_particle_cloth")
        stage = Usd.Stage.CreateNew(str(output / "scene.usda"))
        if spec.get("support_requirements"):
            capabilities.add("rigid_support_response")
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        if spec.get("review_camera"):
            view = spec["review_camera"]
            camera = UsdGeom.Camera.Define(stage, "/World/Cameras/Review")
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, view["eye_m"])), Gf.Vec3d(*map(float, view["target_m"])), Gf.Vec3d(*map(float, view["up"]))).GetInverse()
            camera.AddTransformOp().Set(matrix)
            camera.CreateHorizontalApertureAttr(36.0)
            camera.CreateFocalLengthAttr(18 / math.tan(math.radians(view["horizontal_fov_deg"]) / 2))
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
            if view.get('calibration'):
                from .scene_camera import author_intrinsics, raster
                w, h, k = raster(view['calibration'])
                author_intrinsics(camera, k, w, h)
                capabilities.add('calibrated_camera')
        # A runtime adapter owns its solver and simulation context. No engine scene is embedded.
        for support in spec["supports"]:
            path = f"/World/Environment/{support['id']}"
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTransformOp().Set(Gf.Matrix4d(np.array(support["transform"]).T.tolist()))
            box = UsdGeom.Cube.Define(stage, path + "/Collider")
            box.CreateSizeAttr(1.0)
            box.AddScaleOp().Set(Gf.Vec3f(support["size_m"]))
            box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            material = UsdShade.Material.Define(stage, path + "/ContactMaterial")
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(support["static_friction"])
            physics.CreateDynamicFrictionAttr(support["dynamic_friction"])
            physics.CreateRestitutionAttr(support["restitution"])
            if "friction_combine_mode" in support:
                material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
                material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(
                    support["friction_combine_mode"])
            UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material, materialPurpose="physics")
        for entry in spec["objects"]:
            folder = Path(entry["bundle"])
            verification = verify_bundle(folder)
            if not verification["ok"]:
                raise ValueError(f"{entry['id']}: invalid asset bundle: {verification['failures']}")
            bundle = json.loads((folder / "asset_bundle.json").read_text())
            if any(r["asset_id"] == entry["id"] for r in spec.get("support_requirements", [])) and bundle.get("asset_kind") != "rigid":
                raise ValueError("Declared drop/settle support response currently requires a single rigid asset")
            if spec["backend"] == "physx" and bundle.get("asset_kind") == "deformable":
                from .scene_physx_cloth import read_configuration
                read_configuration(folder, bundle)
            elif spec["backend"] != "newton" and (bundle.get("backend") != spec["backend"] or bundle.get("asset_kind") not in {"rigid", "articulated"}):
                raise ValueError("PhysX composition needs rigid/articulated or bound particle-cloth bundles")
            capabilities.add(bundle["asset_kind"])
            entrypoint = bindings[entry["id"]]["visual_entrypoint"] if bindings else bundle["entrypoint"]
            source = folder / entrypoint
            if not source.resolve().is_relative_to(folder) or entrypoint not in bundle["files"]:
                raise ValueError("Entrypoint must be a retained bundle file")
            original, source_root, layers, assets = _check_stage(source)
            needed = {_dependency_file(layer.realPath) for layer in layers}
            needed.update(_dependency_file(asset) for asset in assets)
            retained = {(folder / name).resolve() for name in bundle["files"]}
            if not needed <= retained:
                raise ValueError("USD dependency is outside the verified bundle inventory")
            destination = output / "assets" / entry["id"]
            destination.mkdir(parents=True)
            for name, digest in {**bundle["files"], "asset_bundle.json": sha256_file(folder / "asset_bundle.json")}.items():
                src, dst = (folder / name).resolve(), destination / name
                if not src.is_relative_to(folder) or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Bundle files must stay inside their directory")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if sha256_file(dst) != digest:
                    raise ValueError("Asset changed during scene packaging")
                dependencies[str(src)] = digest
            editable = False
            source_blend = bundle.get("source_blend")
            if isinstance(source_blend, str):
                blend = _path(source_blend, folder)
                if blend.suffix == ".blend" and blend.is_file() and sha256_file(blend) == bundle.get("source_blend_sha256"):
                    target = destination / "editable" / "scene.blend"
                    target.parent.mkdir()
                    shutil.copy2(blend, target)
                    if sha256_file(target) != bundle["source_blend_sha256"]:
                        raise ValueError("Editable source changed during packaging")
                    dependencies[str(blend)] = bundle["source_blend_sha256"]
                    editable = True
            path = f"/World/Objects/{entry['id']}"
            xf = UsdGeom.Xform.Define(stage, path)
            matrix = Gf.Matrix4d(np.array(entry["transform"]).T.tolist())
            xf.AddTransformOp().Set(matrix)
            placed = stage.DefinePrim(path + "/Asset")
            placed.GetReferences().AddReference(f"./assets/{entry['id']}/{entrypoint}")
            anchors = _relocate_world_anchors(placed, matrix, Gf, Usd, UsdPhysics) if not bindings else 0
            before, after = _inventory(source_root, Usd, UsdPhysics), _inventory(placed, Usd, UsdPhysics)
            failures = _relationship_failures(placed, Usd)
            if before != after:
                failures.append("asset_structure_changed")
            checks.append({"id": entry["id"], "ok": not failures, "failures": failures,
                           "inventory": after, "world_anchors_relocated": anchors,
                           "editable_blender_source_included": editable,
                           "prim_path": path + "/Asset", "asset_kind": bundle["asset_kind"]})
            # Keep the source stage alive until composition checks finish.
            del original
        for entry in spec["backgrounds"]:
            source = Path(entry["path"])
            original, source_root, layers, assets = _check_stage(source, entry.get("root_prim"))
            if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdPhysics.Scene)
                   or p.IsA(UsdPhysics.Joint) for p in Usd.PrimRange(source_root)):
                raise ValueError("Background is visual-only; express environment collisions separately")
            for path in [*(_dependency_file(v.realPath) for v in layers), *(_dependency_file(v) for v in assets)]:
                dependencies[str(path)] = sha256_file(path)
            destination = output / "backgrounds" / f"{entry['id']}.usdz"
            destination.parent.mkdir(exist_ok=True)
            if entry.get("editable_source"):
                editable = Path(entry["editable_source"])
                dependencies[str(editable)] = sha256_file(editable)
                shutil.copy2(editable, destination.with_suffix(".blend"))
            if source.suffix.lower() == ".usdz":
                shutil.copy2(source, destination)
            elif not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(destination)):
                raise RuntimeError("Failed to package background USD dependencies")
            path = f"/World/Backgrounds/{entry['id']}"
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTransformOp().Set(Gf.Matrix4d(np.array(entry["transform"]).T.tolist()))
            prim = stage.DefinePrim(path + "/Content")
            prim.GetReferences().AddReference(f"./backgrounds/{entry['id']}.usdz", source_root.GetPath())
            if entry["format"] != "mesh":
                capabilities.add(entry["format"])
            if entry["format"] == "mesh" and not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim)):
                raise ValueError("A mesh background must contain mesh geometry")
            if entry["format"] == "particle_field" and not any(p.GetTypeName().startswith("ParticleField") for p in Usd.PrimRange(prim)):
                raise ValueError("Declared ParticleField background has no ParticleField prim")
            if entry["format"] == "nurec_volume" and not any(p.GetTypeName() in {"OmniNuRecFieldAsset", "Volume"} for p in Usd.PrimRange(prim)):
                raise ValueError("Declared NuRec background has no neural volume prim")
            checks.append({"id": entry["id"], "ok": True, "observed": entry["observed"], "format": entry["format"]})
            del original
        stage.GetRootLayer().Save()
        if any(sha256_file(Path(path)) != digest for path, digest in dependencies.items()):
            raise ValueError("Scene inputs changed during assembly")
        report = {"schema_version": 1, "ok": all(v["ok"] for v in checks), "status": "composed",
                  "checks": checks, "required_capabilities": sorted(capabilities),
                  "backend": spec["backend"], "entrypoint": "scene.usda", "simulation_verified": False,
                  "frame": spec["frame"], "source_dependencies": dependencies,
                  "physics_plan": plan,
                  "support_requirements": spec.get("support_requirements", []),
                  "limits": ["Composition does not establish runtime contacts or visual acceptance.",
                             "Background removal, exposed regions and view coverage require scene review."]}
        if not report["ok"]:
            report["status"] = "composition_failed"
    except (OSError, ValueError, RuntimeError) as exc:
        report = {"schema_version": 1, "ok": False, "status": "composition_failed", "error": str(exc),
                  "checks": checks, "simulation_verified": False}
    report["files"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    if report.get('ok'):
        closure_failures = _packaged_dependency_failures(output.resolve(), report['files'])
        if closure_failures:
            report.update(ok=False, status='composition_failed',
                          error='Packaged USD is not self-contained', dependency_failures=closure_failures)
    atomic_json(output / "scene_bundle.json", report)
    return report


def verify_scene(folder, *, check_sources=False):
    folder = Path(folder).resolve()
    report = json.loads((folder / "scene_bundle.json").read_text())
    failures = []
    if not report.get("ok") or not report.get("files") or "scene.usda" not in report.get("files", {}):
        failures.append("no_successful_composition")
    for name, digest in report.get("files", {}).items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != digest:
            failures.append(f"missing_or_changed:{name}")
    if not failures:
        try:
            failures.extend(_packaged_dependency_failures(folder, report['files']))
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append('packaged_dependency_check_failed:' + str(exc))
    if check_sources:
        for name, digest in report.get("source_dependencies", {}).items():
            path = Path(name)
            if not path.is_file() or sha256_file(path) != digest:
                failures.append(f"scene_input_changed:{path.name}")
    return {"ok": not failures, "failures": failures, "checked_files": len(report.get("files", {})),
            "simulation_verified": False}
