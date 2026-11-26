"""Native RTX Gaussian setup, separate from physics and portable USD authoring.

Isaac Sim 5.1 uses NuRec volumes. Standard ParticleField assets require a newer
renderer; they are never silently replaced by meshes or flattened to images.
"""
from __future__ import annotations

GAUSSIAN_FORMATS = frozenset({"particle_field", "nurec_volume"})


def requested_formats(spec):
    return sorted({item["format"] for item in spec.get("backgrounds", [])
                   if item["format"] in GAUSSIAN_FORMATS})


def startup_arguments(spec, kit_args=""):
    """Enable geometry processing before Lab 2 constructs SimulationApp."""
    if not requested_formats(spec):
        return kit_args or ""
    return (kit_args or "") + " --/renderer/multiGpu/enabled=false --/UJITSO/geometry=true"


def configure_gaussian_rendering(stage, root, *, legacy, settings):
    """Inspect composed fields and bind the renderer's matching representation.

    The returned receipt describes setup, not visual acceptance. Review rendered
    frames for visibility, occlusion and appearance before accepting the scene.
    """
    from pxr import Sdf, Usd, UsdShade, UsdVol

    if not root or not root.IsValid():
        raise ValueError("Gaussian scene root does not exist")
    prims = list(Usd.PrimRange(root))
    if any(p.GetAttribute("info:spg:sourceAsset").Get() for p in prims):
        raise ValueError("SPG/PPISP requires a dedicated renderer adapter")
    particles = [p for p in prims if p.GetTypeName().startswith("ParticleField")]
    volumes = [p for p in prims if p.GetTypeName() == "Volume"
               and p.GetAttribute("omni:nurec:isNuRecVolume").Get() is True]
    if legacy and particles:
        raise ValueError("Isaac Sim 5.1 requires nurec_volume backgrounds; transcode the "
                         "Gaussian source with the configured 3DGRUT adapter. "
                         "Standard ParticleField rendering needs a compatible newer runtime.")
    if not particles and not volumes:
        raise ValueError("No native Gaussian fields found; an ordinary USD Volume is not a NuRec background")

    payloads = []
    for prim in volumes:
        volume = UsdVol.Volume(prim)
        for name in ("density", "emissiveColor"):
            target = volume.GetFieldPath(name)
            field = stage.GetPrimAtPath(target) if target else None
            if not field or field.GetTypeName() != "OmniNuRecFieldAsset":
                raise ValueError(f"NuRec {prim.GetPath()} needs an OmniNuRecFieldAsset for {name}")
            asset = field.GetAttribute("filePath").Get()
            resolved = asset.resolvedPath if isinstance(asset, Sdf.AssetPath) else ""
            if not resolved:
                raise ValueError(f"Unresolved NuRec payload on {field.GetPath()}")
            payloads.append({"field": str(field.GetPath()), "role": name, "resolved": resolved})

    applied = {"/renderer/multiGpu/enabled": False, "/rtx-transient/dlssg/enabled": False}
    if legacy:
        # NuRec's captured radiance is already display-referred. These settings
        # match the plain NuRec export, without modifying mesh material assets.
        applied.update({"/rtx/post/histogram/enabled": False,
                        "/rtx/post/registeredCompositing/invertToneMap": True,
                        "/rtx/post/registeredCompositing/invertColorCorrection": True,
                        "/rtx/post/tonemap/op": 2})
    for key, value in applied.items():
        settings.set(key, value)

    material = None
    bound = []
    for prim in particles:
        if UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]:
            continue
        if material is None:
            material = UsdShade.Material.Define(stage, "/World/ReviewMaterials/GaussianRadiance")
            shader = UsdShade.Shader.Define(stage, "/World/ReviewMaterials/GaussianRadiance/Shader")
            shader.SetSourceAsset(Sdf.AssetPath("ParticleFieldEmissive.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("ParticleFieldEmissive", "mdl")
            out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            out.GetAttr().SetMetadata("renderType", "material")
            for kind in ("surface", "volume", "displacement"):
                material.CreateOutput("mdl:" + kind, Sdf.ValueTypeNames.Token).ConnectToSource(out)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        bound.append(str(prim.GetPath()))
    return {"ok": True, "mode": "native_nurec_volume" if legacy else "native_plain_field",
            "particle_fields": [str(p.GetPath()) for p in particles],
            "nurec_volumes": [str(p.GetPath()) for p in volumes], "payloads": payloads,
            "settings": {key: settings.get(key) for key in applied},
            "default_emissive_fields": bound, "visual_acceptance": False}
