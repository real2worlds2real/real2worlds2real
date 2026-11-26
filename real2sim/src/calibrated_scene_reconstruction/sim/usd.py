"""Small USD authoring helpers: poses, materials, meshes, colliders, cameras."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt


def _precision(prim, name, default):
    attr = prim.GetAttribute(name)
    if not attr:
        return default
    t = str(attr.GetTypeName())
    if t.endswith("d") or t.startswith("double"):
        return UsdGeom.XformOp.PrecisionDouble
    if t.endswith("h") or t.startswith("half"):
        return UsdGeom.XformOp.PrecisionHalf
    return UsdGeom.XformOp.PrecisionFloat


def set_pose(prim, pos, quat_wxyz=(1, 0, 0, 0), scale=None):
    """Translate/orient(/scale), reusing the precision of xform ops a referenced asset may already author."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    pt = _precision(prim, "xformOp:translate", UsdGeom.XformOp.PrecisionDouble)
    po = _precision(prim, "xformOp:orient", UsdGeom.XformOp.PrecisionFloat)
    xf.AddTranslateOp(precision=pt).Set(
        (Gf.Vec3d if pt == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f)(*[float(v) for v in pos]))
    Q = Gf.Quatd if po == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf
    xf.AddOrientOp(precision=po).Set(Q(float(quat_wxyz[0]), *[float(v) for v in quat_wxyz[1:]]))
    if scale is not None:
        ps = _precision(prim, "xformOp:scale", UsdGeom.XformOp.PrecisionFloat)
        xf.AddScaleOp(precision=ps).Set((Gf.Vec3d if ps == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f)(
            *[float(v) for v in np.broadcast_to(scale, 3)]))


def quat_from_matrix(R):
    q = Gf.Matrix3d(*np.asarray(R).T.ravel().tolist()).ExtractRotation().GetQuat()
    return (q.GetReal(), *q.GetImaginary())


def quat_between(a, b):
    q = Gf.Rotation(Gf.Vec3d(*[float(v) for v in a]), Gf.Vec3d(*[float(v) for v in b])).GetQuat()
    return (q.GetReal(), *q.GetImaginary())


def _uv_texture(stage, path, reader, file, color_space, wrap, scale=(1, 1, 1, 1), bias=(0, 0, 0, 0)):
    from calibrated_scene_reconstruction.core.paths import resolve
    t = UsdShade.Shader.Define(stage, path)
    t.CreateIdAttr("UsdUVTexture")
    # Authoring inputs are repository filesystem paths. A relative value in
    # USD would instead resolve next to the output layer and silently go black
    # after save/reload. Packaging later copies and rebases these absolute files.
    source = str(file)
    if '://' not in source:
        source = str(resolve(source))
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(source)
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
    modes=[wrap,wrap] if isinstance(wrap,str) else list(wrap)
    if len(modes)!=2 or any(mode not in {'repeat','clamp','mirror','black','useMetadata'} for mode in modes):
        raise ValueError('Two valid USD texture wrap modes required')
    t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(modes[0])
    t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(modes[1])
    t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(v) for v in scale]))
    t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(v) for v in bias]))
    t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    return t


def _surface(stage, path):
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat, pbr, reader


def textured_material(stage, path, tex_file, roughness=0.7, metallic=0.0, tint=(1, 1, 1), emissive=0.0,
                      emissive_tint=None, wrap="clamp"):
    mat, pbr, reader = _surface(stage, path)
    diff = _uv_texture(stage, f"{path}/tex", reader, tex_file, "sRGB", wrap, (*tint, 1))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diff.ConnectableAPI(), "rgb")
    if emissive > 0:
        et = tint if emissive_tint is None else emissive_tint
        emi = _uv_texture(stage, f"{path}/tex_emissive", reader, tex_file, "sRGB", "clamp",
                          (emissive * et[0], emissive * et[1], emissive * et[2], 1))
        pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(emi.ConnectableAPI(), "rgb")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    return mat


def pbr_material(stage, path, tex_dir, tint=(1, 1, 1)):
    """Poly Haven texture set (diff / rough / nor_gl) with repeat wrapping."""
    mat, pbr, reader = _surface(stage, path)
    diff = _uv_texture(stage, f"{path}/diff", reader, tex_dir / "diff.jpg", "sRGB", "repeat", (*tint, 1))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diff.ConnectableAPI(), "rgb")
    if (tex_dir / "rough.jpg").exists():
        r = _uv_texture(stage, f"{path}/rough", reader, tex_dir / "rough.jpg", "raw", "repeat")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r.ConnectableAPI(), "r")
    if (tex_dir / "nor_gl.jpg").exists():
        n = _uv_texture(stage, f"{path}/normal", reader, tex_dir / "nor_gl.jpg", "raw", "repeat", (2, 2, 2, 1),
                        (-1, -1, -1, 0))
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n.ConnectableAPI(), "rgb")
    return mat


def glass_material(stage, path, rgb=(0.95, 0.97, 0.96), ior=1.5, roughness=0.0, thin_walled=False):
    """Clear glass: the RTX renderer's OmniGlass MDL (a UsdPreviewSurface with opacity renders as a mirror-like
    dark shell there), with a translucent UsdPreviewSurface for other renderers."""
    mat = color_material(stage, path, rgb, max(roughness, 0.02), 0.0, opacity=0.15, ior=ior)
    mdl = UsdShade.Shader.Define(stage, f"{path}/Glass")
    mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
    mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(bool(thin_walled))
    for out in ("surface", "volume", "displacement"):
        mat.CreateOutput(f"mdl:{out}", Sdf.ValueTypeNames.Token).ConnectToSource(mdl.ConnectableAPI(), "out")
    return mat


def color_material(stage, path, rgb, roughness=0.6, metallic=0.0, emissive=None, opacity=None, ior=None):
    """A flat colour; `opacity` < 1 with an `ior` is see-through (glass_material is the clear glass of RTX)."""
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if emissive is not None:
        pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in emissive]))
    if opacity is not None:
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if ior is not None:
        pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def add_mesh(stage, path, V, F, UV=None, material=None, double_sided=True, normals=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).ravel()))
    if UV is not None:
        corner_uv = np.asarray(UV).shape == (len(F), 3, 2)
        interpolation = UsdGeom.Tokens.faceVarying if corner_uv else UsdGeom.Tokens.vertex
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(UV, np.float32).reshape(-1, 2)))
    m.CreateSubdivisionSchemeAttr("none")
    if normals is not None:
        values=np.asarray(normals,float)
        if values.shape!=(len(F),3,3) or not np.isfinite(values).all():
            raise ValueError('Corner normals must match the authored triangles')
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(values,np.float32).reshape(-1,3)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.CreateDoubleSidedAttr(double_sided)
    V = np.asarray(V)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def add_box_collider(stage, path, lo, hi):
    """Invisible (guide purpose) static box collider."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
    xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo).tolist()))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.CreatePurposeAttr("guide")
    return cube


def look_at_quat(eye, target, up=(0, 0, 1)):
    """USD camera orientation (wxyz), preserving the USD quaternion conversion."""
    from calibrated_scene_reconstruction.core.geometry import look_at_matrix

    return quat_from_matrix(look_at_matrix(eye, target, up))


def pinhole(width, height, focal_frac=0.62):
    return np.array([[width * focal_frac, 0, width / 2], [0, width * focal_frac, height / 2], [0, 0, 1]])


def add_camera(stage, path, K, width, height, eye=None, target=None, T_world_cam_cv=None, near=0.02, far=100.0):
    cam = UsdGeom.Camera.Define(stage, path)
    focal = 18.0
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(float(width / K[0][0] * focal))
    cam.CreateVerticalApertureAttr(float(height / K[1][1] * focal))
    cam.CreateHorizontalApertureOffsetAttr(float((width/2-K[0][2])*focal/K[0][0]))
    cam.CreateVerticalApertureOffsetAttr(float((K[1][2]-height/2)*focal/K[1][1]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(near, float(far)))
    if T_world_cam_cv is not None:
        T = np.asarray(T_world_cam_cv)
        set_pose(cam.GetPrim(), T[:3, 3], quat_from_matrix(T[:3, :3] @ np.diag([1, -1, -1])))  # OpenCV -> USD camera
    elif eye is not None:
        set_pose(cam.GetPrim(), eye, look_at_quat(eye, target))
    return cam
