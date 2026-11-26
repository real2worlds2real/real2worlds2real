"""The LeHome Challenge's garment-folding env (two SO-101 arms, PhysX particle cloth, envs/lehome, apps/tasks/lehome_run.py)
with a randomized, realistic look, its physics untouched: at every reset the garment wears a real fabric (an
ambientCG PBR set: colour, roughness and normal maps, at a woven fabric's scale) in a colour clothes come in, the table
one of the challenge's surface textures that reads as a table top, the dome light a plausible brightness and colour
temperature. The garment's own material stays in the draw. Each reset's look goes to $SCENE_RECONSTRUCTION_LOOK_LOG (JSON lines)
when set; $SCENE_RECONSTRUCTION_LOOK_SEED seeds the draws (default 0).

Registered (apps/tasks/lehome_run.py) as R2S2R-LeHome-Garment-Look-v0 on the challenge's v2 config:

    ../../../tools/run_isaac.sh ../../../apps/tasks/lehome_run.py -m scripts.dataset_sim replay --task R2S2R-LeHome-Garment-Look-v0 ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.core.environment import setting

FABRICS = ROOT / "envs" / "lehome" / "textures" / "ambientcg"
TASK_ID = "R2S2R-LeHome-Garment-Look-v0"

# ambientCG fabrics that read as garment cloth (no upholstery weaves, tablecloth ginghams, felts, prints or neon
# solids), by what they suit: knits and jerseys for tops, denims and twills for trousers, shirtings, plain weaves
DENIM = ["Fabric003", "Fabric012", "Fabric069", "Fabric075", "Fabric081A", "Fabric005"]
TWILL = ["Fabric006", "Fabric026", "Fabric044", "Fabric062", "Fabric064", "Fabric065", "Fabric066", "Fabric061"]
KNIT = ["Fabric016", "Fabric018", "Fabric019", "Fabric034", "Fabric073", "Fabric079", "Fabric036", "Fabric038"]
SHIRTING = ["Fabric048", "Fabric049", "Fabric050", "Fabric051", "Fabric052", "Fabric053", "Fabric054", "Fabric057",
            "Fabric058", "Fabric067", "Fabric071", "Fabric072", "Fabric076", "Fabric080", "Fabric007"]
PLAIN = ["Fabric001", "Fabric002", "Fabric017", "Fabric020", "Fabric022", "Fabric023", "Fabric029", "Fabric030",
         "Fabric031", "Fabric032", "Fabric037", "Fabric039", "Fabric040", "Fabric041", "Fabric042", "Fabric045",
         "Fabric047", "Fabric060", "Fabric063", "Fabric068", "Fabric074", "Fabric077", "Fabric081C", "Fabric082A",
         "Fabric084", "Fabric085", "Fabric086"]
BY_KIND = {"top": {"knit": 0.4, "plain": 0.35, "shirting": 0.25},
           "pant": {"denim": 0.45, "twill": 0.35, "plain": 0.2}}
POOLS = {"denim": DENIM, "twill": TWILL, "knit": KNIT, "shirting": SHIRTING, "plain": PLAIN}
# colours clothes come in (sRGB): a grey fabric is dyed one of these, a coloured one keeps its own
PALETTE = {"black": (0.07, 0.07, 0.08), "charcoal": (0.22, 0.22, 0.24), "heather": (0.55, 0.55, 0.56),
           "white": (0.9, 0.9, 0.88), "cream": (0.88, 0.84, 0.74), "navy": (0.11, 0.15, 0.3),
           "denim": (0.24, 0.34, 0.52), "light_blue": (0.58, 0.68, 0.82), "khaki": (0.6, 0.53, 0.38),
           "beige": (0.78, 0.7, 0.58), "olive": (0.33, 0.36, 0.21), "forest": (0.13, 0.26, 0.17),
           "burgundy": (0.38, 0.1, 0.13), "brick": (0.55, 0.22, 0.16), "brown": (0.33, 0.23, 0.15),
           "dusty_pink": (0.8, 0.6, 0.62), "mustard": (0.68, 0.52, 0.2), "slate": (0.36, 0.42, 0.48)}
# the challenge's table-top textures (Assets/textures/surface/<i>.png) that read as a table: woods, laminates, stone,
# concrete; not the cracked earth, tiles, cork, fabrics, checkers, vignetted metals, ripples or saturated colours
TABLES = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 16, 17, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 46, 49, 50, 51, 52, 53, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75,
          76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100]
TABLE_SHADER = "/World/Scene/scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7"
MATERIAL = "/World/Looks/R2S2RGarment"


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class GarmentLook:
    """The look drawn at each reset: `apply(env)` binds the garment material (or restores the garment's own), sets
    the table's texture and the dome light, and returns what it drew."""

    def __init__(self, seed=0, keep_own_p=0.15, uv_scale=(6.0, 10.0), log_path=None):
        self.rng = np.random.default_rng(seed)
        self.keep_own_p, self.uv_scale = keep_own_p, uv_scale
        self.log_path = log_path
        self.own = {}  # garment prim path -> {submesh path: its own material path}
        self.mean = {}  # fabric -> the colour map's mean (sRGB), its saturation

    def fabric_maps(self, fid):
        d = FABRICS / fid
        return {k: d / f"{fid}_1K-JPG_{k}.jpg" for k in ("Color", "Roughness", "NormalGL")}

    def fabric_stats(self, fid):
        if fid not in self.mean:
            from PIL import Image

            im = np.asarray(Image.open(self.fabric_maps(fid)["Color"]).convert("RGB").resize((64, 64)), float) / 255.0
            m = im.reshape(-1, 3).mean(0)
            self.mean[fid] = (m, float(m.max() - m.min()))
        return self.mean[fid]

    def material(self, stage):
        """The garment material, made once: UsdPreviewSurface fed by a colour (scaled: the tint), a roughness and a
        normal map, all through one st transform (the weave's scale)."""
        from pxr import Gf, Sdf, UsdShade

        mat = UsdShade.Material.Get(stage, MATERIAL)
        if mat:
            return mat
        mat = UsdShade.Material.Define(stage, MATERIAL)
        surf = UsdShade.Shader.Define(stage, f"{MATERIAL}/Surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        surf.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04, 0.04, 0.04))
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        reader = UsdShade.Shader.Define(stage, f"{MATERIAL}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        xf = UsdShade.Shader.Define(stage, f"{MATERIAL}/st_scale")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        for name, out, kind in (("color", "rgb", Sdf.ValueTypeNames.Float3), ("roughness", "r", Sdf.ValueTypeNames.Float),
                                ("normal", "rgb", Sdf.ValueTypeNames.Float3)):
            tex = UsdShade.Shader.Define(stage, f"{MATERIAL}/{name}")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if name == "color" else "raw")
            tex.CreateOutput(out, kind)
            target = {"color": "diffuseColor", "roughness": "roughness", "normal": "normal"}[name]
            surf.CreateInput(target, kind).ConnectToSource(tex.ConnectableAPI(), out)
            if name == "normal":
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
                tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
        return mat

    def garment_kind(self, env):
        name = str(getattr(env.cfg, "garment_name", "") or "")
        return "pant" if name.lower().startswith("pant") else "top"

    def submeshes(self, stage, garment):
        """The garment's mesh prims: its sub-meshes (mesh, mesh1, ...) or the one mesh."""
        root = stage.GetPrimAtPath(str(garment.mesh_prim_path))
        if not root.IsValid():
            return []
        kids = [p for p in root.GetChildren() if p.GetTypeName() == "Mesh"]
        return kids or [root]

    def apply(self, env):
        from pxr import Gf, Sdf, UsdShade

        stage, garment, rng = env.scene.stage, env.object, self.rng
        look = {}
        meshes = self.submeshes(stage, garment) if garment is not None else []
        key = str(garment.mesh_prim_path) if meshes else ""
        if meshes and key not in self.own:  # the garment's own materials, to put back
            self.own[key] = {str(p.GetPath()): str(UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterialPath())
                             for p in meshes}
        if meshes:
            if rng.random() < self.keep_own_p:
                for p in meshes:
                    own = self.own[key].get(str(p.GetPath()))
                    if own:
                        UsdShade.MaterialBindingAPI(p).Bind(UsdShade.Material.Get(stage, own))
                look["garment"] = {"material": "own"}
            else:
                kind = self.garment_kind(env)
                groups, w = zip(*BY_KIND[kind].items())
                group = groups[rng.choice(len(groups), p=np.asarray(w) / sum(w))]
                fid = POOLS[group][rng.integers(len(POOLS[group]))]
                mean, sat = self.fabric_stats(fid)
                if sat < 0.08 and group != "shirting":  # a grey or white fabric: dyed a garment colour (a shirting
                    # keeps its print: a dyed windowpane reads as a tablecloth), its brightness kept relative
                    cname = list(PALETTE)[rng.integers(len(PALETTE))]
                    tint = srgb_to_linear(PALETTE[cname]) / np.maximum(srgb_to_linear(mean).mean(), 1e-3)
                else:  # a coloured one keeps its colour, a little lighter or darker
                    cname = None
                    tint = np.full(3, rng.uniform(0.8, 1.15))
                tint = np.clip(tint * rng.uniform(0.9, 1.1, 3) ** 0.3, 0.0, 4.0)
                s = float(rng.uniform(*self.uv_scale))
                mat = self.material(stage)
                maps = self.fabric_maps(fid)
                for name, tex in (("color", "Color"), ("roughness", "Roughness"), ("normal", "NormalGL")):
                    sh = UsdShade.Shader.Get(stage, f"{MATERIAL}/{name}")
                    (sh.GetInput("file") or sh.CreateInput("file", Sdf.ValueTypeNames.Asset)).Set(Sdf.AssetPath(str(maps[tex])))
                col = UsdShade.Shader.Get(stage, f"{MATERIAL}/color")
                (col.GetInput("scale") or col.CreateInput("scale", Sdf.ValueTypeNames.Float4)).Set(
                    Gf.Vec4f(*[float(v) for v in tint], 1.0))
                xf = UsdShade.Shader.Get(stage, f"{MATERIAL}/st_scale")
                (xf.GetInput("scale") or xf.CreateInput("scale", Sdf.ValueTypeNames.Float2)).Set(Gf.Vec2f(s, s))
                (xf.GetInput("rotation") or xf.CreateInput("rotation", Sdf.ValueTypeNames.Float)).Set(
                    float(rng.choice([0.0, 90.0])))
                for p in meshes:
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                look["garment"] = {"material": fid, "group": group, "colour": cname, "tint": np.round(tint, 3).tolist(),
                                   "uv_scale": round(s, 2)}
        # the table top
        shader = UsdShade.Shader(stage.GetPrimAtPath(TABLE_SHADER))
        tex = (shader.GetInput("file") or shader.GetInput("diffuse_texture")) if shader else None
        if tex:
            i = int(TABLES[rng.integers(len(TABLES))])
            tex.Set(Sdf.AssetPath(os.path.join(os.getcwd(), "Assets", "textures", "surface", f"{i}.png")))
            look["table"] = i
        # the dome light: brightness and a colour temperature between warm and cool white
        light = stage.GetPrimAtPath("/World/Light")
        if light.IsValid():
            warm = float(rng.uniform(-1.0, 1.0))
            colour = np.clip(np.array([1.0 + 0.08 * warm, 1.0, 1.0 - 0.1 * warm]) * 0.85, 0.0, 1.0)
            intensity = float(rng.uniform(900.0, 1700.0))
            light.GetAttribute("inputs:intensity").Set(intensity)
            light.GetAttribute("inputs:color").Set(Gf.Vec3f(*[float(v) for v in colour]))
            look["light"] = {"intensity": round(intensity, 1), "colour": np.round(colour, 3).tolist()}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"garment_name": str(getattr(env.cfg, "garment_name", "")), **look}) + "\n")
        return look


def make_env_class():
    """The challenge's v2 env with the look drawn after each reset (imported once the app runs)."""
    from lehome.tasks.bedroom.garment_bi_v2 import GarmentEnv

    class LookGarmentEnv(GarmentEnv):
        def __init__(self, cfg, render_mode=None, **kwargs):
            self.look = GarmentLook(seed=int(setting("LOOK_SEED", "0")),
                                    log_path=setting("LOOK_LOG") or None)
            super().__init__(cfg, render_mode, **kwargs)

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            self.last_look = self.look.apply(self)

    return LookGarmentEnv


def LookGarmentEnv(cfg, render_mode=None, **kwargs):  # noqa: N802  (gym's entry point: a callable making the env)
    return make_env_class()(cfg, render_mode, **kwargs)
