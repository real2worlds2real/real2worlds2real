"""Build a twin (or a cousin variant) of a session into a USD stage.

/World/Room      structure planes + furniture boxes (albedo textures or PBR overrides) + invisible box colliders
/World/Static    background objects: SAM 3D meshes, templates, library swaps or the agent's set pieces (static
                 colliders; the scene graph's object fixes, sim.setpieces)
/World/Support   movable furniture (shared SAM 3D reconstruction or library models)
/World/Synthetic room objects (recon.room_objects): their meshes are their colliders (static triangle meshes)
/World/Objects   manipulable rigid objects (twin meshes, YCB scans, Poly Haven models)
/World/Lights    calibrated relighting rig;  /World/Robot (added by robots.articulation.add_robot)

A *variant* is a dict layered over scene/twin.json (see cousins.py):
  room_overrides {part-prefix: {pbr, tile_m, tint} | {tex} | {tint} | {white}}, objects [...],
  stool_models [{recon: group | model: library name, center, yaw, height?}],
  static_swaps {name: {model, height, yaw, xy?}}, skip_static [...], set_piece_overrides {name: {kit params}},
  synthetic_furniture [{name, source: {recon | model | proxy, color}, box, center, yaw, height, synthetic: true}],
  lights {scale, temp, sun, bounce, front, disk, window, hdri (an equirect .hdr for the sky dome)}
"""
import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from ..assets.physprops import ROLLING_DAMPING, mass_props, rest_pose, rolls
from ..assets.object_geometry import collision_parts, level, mesh_key, object_label
from calibrated_scene_reconstruction.core.paths import resolve as repo_path
from calibrated_scene_reconstruction.assets.support_geometry import support_top
from calibrated_scene_reconstruction.assets.metric_mesh import sdf_settings
from .assets import add_library_model, planar_uv, polyhaven_material
from .clearance import resolve, summarize
from .setpiece_stage import add_set_piece, snap
from .collision_geometry import solid_supports, surface_points, top_slabs, fit_footprint  # noqa: F401 -- public helpers
from .scene_physics import apply_scene_physics
from calibrated_scene_reconstruction.assets.material_mass import DENSITY, kit_mass  # noqa: F401 -- public constants
from calibrated_scene_reconstruction.assets.templates import TEMPLATES
from .usd import (add_box_collider, add_mesh, color_material, quat_between, quat_from_matrix, set_pose,
                  textured_material)

SINK_FLOOR = 0.01  # m a static object's bottom may lie under the floor before it is put on it


def configure_render(gi=True, auto_exposure=True, mode="rt", spp=64, bounces=6):
    """Real-time RTX is direct-light dominated: enable indirect diffuse / AO and auto exposure for interiors.
    mode "pt": RTX path tracing instead (multi-bounce light, soft and contact shadows; the app must be started with
    renderer "PathTracing"), `spp` samples per rendered frame (all of them: a frame is complete, not accumulated
    over the next ones), the OptiX denoiser on."""
    import carb

    st = carb.settings.get_settings()
    if mode == "pt":
        st.set("/rtx/rendermode", "PathTracing")
        st.set("/rtx/pathtracing/spp", int(spp))
        st.set("/rtx/pathtracing/totalSpp", int(spp))
        st.set("/rtx/pathtracing/maxBounces", int(bounces))
        st.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", int(bounces))
        st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        if auto_exposure:
            st.set("/rtx/post/histogram/enabled", True)
            st.set("/rtx/post/histogram/whiteScale", 9.0)
        return
    if gi:
        st.set("/rtx/indirectDiffuse/enabled", True)
        st.set("/rtx/indirectDiffuse/scalingFactor", 1.6)
        st.set("/rtx/ambientOcclusion/enabled", True)
    st.set("/rtx/post/histogram/enabled", bool(auto_exposure))
    if auto_exposure:
        st.set("/rtx/post/histogram/whiteScale", 9.0)
    # anti-aliasing at the native resolution (DLAA): Kit's default DLSS performance mode renders at a fraction of it
    # and upscales -- the splat background came out a fifth softer (variance of the Laplacian 189 -> 224 on a walk frame)
    st.set("/rtx/post/aa/op", 4)
    st.set("/rtx/reflections/enabled", True)
    st.set("/rtx/translucency/enabled", True)
    st.set("/rtx/translucency/maxRefractionBounces", 12)  # a stack of tumblers is a dozen glass surfaces deep


SDF_RESOLUTION = 256  # a task object's collider: its signed distance field's cells along its longest side


class TwinScene:
    def __init__(self, stage, session):
        self.stage = stage
        self.inputs = session
        self.s = session.session
        from ..assets import kits as _kits

        _kits.use_session(self.s.config_dir)  # the agent's own kits for this session's objects
        self.manifest = self.s.manifest
        self.T_wc = np.asarray(self.manifest["photo_camera"]["T_world_cam_cv"])
        self.objects, self.stool_prims, self.light_cfg = {}, [], {}
        self.support_parts = self.inputs.support_parts  # stone / wood tops: less rough
        self.windows = self.inputs.windows
        self.kit_furniture = self.inputs.furniture_kits  # furniture the agent rebuilt as a set piece
        self.model_furniture = self.inputs.support_models

    # --- world ----------------------------------------------------------------------------------------------
    def build_physics(self):
        # World usually already authored its PhysicsScene. A second scene has
        # its own integration frequency and can silently defeat World's dt.
        existing = next((p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
        scene = UsdPhysics.Scene(existing) if existing is not None else UsdPhysics.Scene.Define(
            self.stage, '/World/PhysicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        mechanisms = getattr(self.inputs, 'mechanisms', None)
        flexible = mechanisms and any(p['kind'] != 'rigid' for o in mechanisms['objects'].values()
                                      for p in o['links'].values())
        physics.CreateEnableCCDAttr(not bool(flexible))
        from .simulation import apply as apply_simulation
        apply_simulation(self.stage, getattr(self.inputs, 'simulation', None), flexible=bool(flexible))
        if flexible:
            physics.CreateEnableGPUDynamicsAttr(True)
            physics.CreateBroadphaseTypeAttr('GPU')

    def build_room(self, overrides=None):
        overrides = overrides or {}
        keys = sorted(overrides, key=len, reverse=True)  # longest matching prefix wins
        UsdGeom.Scope.Define(self.stage, "/World/Room")
        UsdGeom.Scope.Define(self.stage, "/World/Looks")
        for name, part in self.manifest["parts"].items():
            d = np.load(self.s / "scene" / part["npz"])
            ov = next((overrides[k] for k in keys if name.startswith(k)), {})
            look = f"/World/Looks/{name}"
            uv = d["UV"]
            if any(name == k or name.startswith(k + "_") for k in (*self.kit_furniture, *self.model_furniture)):
                continue  # rebuilt as a set piece (sim.setpieces)
            if part.get("material") == "ceramic":
                mat = color_material(self.stage, look, (0.85, 0.85, 0.83), roughness=0.15)
            elif "color" in ov:  # the agent's plain finish (a laminate top)
                mat = color_material(self.stage, look, ov["color"], roughness=ov.get("roughness", 0.5))
            elif ov.get("white"):  # irradiance probe
                mat = color_material(self.stage, look, (0.8, 0.8, 0.8), roughness=0.95)
            elif "tex" in ov:  # recon.materials composite: photo where observed, PBR fill elsewhere
                mat = textured_material(self.stage, look, self.s / "scene" / ov["tex"], roughness=0.7,
                                        tint=ov.get("tint", (1, 1, 1)))
            elif "pbr" in ov:
                mat = polyhaven_material(self.stage, look, ov["pbr"], ov.get("tint", (1, 1, 1)))
                uv = planar_uv(d["V"], ov.get("tile_m", 0.8))
            else:
                metal = part.get("material") == "steel"
                rough = 0.3 if metal else 0.45 if any(name.startswith(k) for k in self.support_parts) else 0.7
                tex = part.get("tex_albedo", part["tex_baked"])
                mat = textured_material(self.stage, look, self.s / "scene" / tex, roughness=rough,
                                        metallic=0.25 if metal else 0.0, tint=ov.get("tint", (1, 1, 1)),
                                        emissive=part.get("emissive", 0.0))
            mesh = add_mesh(self.stage, f"/World/Room/{name}", d["V"], d["F"], uv, mat, double_sided=False)
            if part.get("collide"):
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        for name, (lo, hi) in solid_supports(self.manifest["collision_boxes"]).items():
            if any(name == k or name.startswith(k + "_") for k in (*self.kit_furniture, *self.model_furniture)):
                continue
            add_box_collider(self.stage, f"/World/Room/collider_{name}", lo, hi)
        for name, (kind, params, origin, yaw) in self.kit_furniture.items():
            add_set_piece(self.stage, f"/World/Room/{name}", kind, params, origin, yaw)
        for name, model in self.model_furniture.items():
            from .support_stage import add_support
            add_support(self.stage, f'/World/Room/{name}', model)

    def object_mesh_world(self, name):
        mesh = trimesh.load(self.s / "objects" / name / "mesh_cam.glb", force="mesh")
        return mesh, np.asarray(mesh.vertices) @ self.T_wc[:3, :3].T + self.T_wc[:3, 3]

    def _top_of(self, piece):
        """The top height of a piece of the scene: a furniture box, a room object (synthetic_furniture), or None."""
        box = self.manifest["collision_boxes"].get(piece)
        if box is not None:
            return float(box[1][2])
        sf = self.s.scene_graph.get("synthetic_furniture", {}).get(piece)
        return float(np.asarray(sf["box"], float)[1, 2]) if sf else None

    def build_static_objects(self, skip=(), swaps=None):
        UsdGeom.Scope.Define(self.stage, "/World/Static")
        swaps = swaps or {}
        fixes = {n: sp.get("fix") or {} for n, sp in self.s.scene_graph.get("static_objects", {}).items()}
        self.fixed = {}
        for name, spec in self.manifest["static_objects"].items():
            fix = fixes.get(name, {})
            if name in skip or fix.get("drop"):
                continue
            path = f"/World/Static/{name}"
            if fix.get("kit") and name not in swaps:  # the agent's set piece in place of the reconstruction
                kind, params, origin, yaw = self.inputs.object_kit(name, fix, self.object_mesh_world(name)[1], getattr(self, "synthetic_boxes", {}))
                add_set_piece(self.stage, path, kind, params, origin, yaw, seed=fix.get("seed", 0))
                self.fixed[name] = f"kit {kind}"
                continue
            if fix.get("swap") and name not in swaps:
                swaps = {**swaps, name: fix["swap"]}
            if name in swaps:  # library model in place of the reconstruction (retrieval fallback or cousin)
                _, V = self.object_mesh_world(name)
                sw = swaps[name]
                z = float(V[:, 2].min())
                if fix.get("snap"):  # the reconstruction floats: its swap stands on what is under it
                    z = support_top(self.manifest["collision_boxes"], V.min(0)[:2], V.max(0)[:2], z + 0.02)
                    self.fixed[name] = "swap snapped"
                add_library_model(self.stage, path, sw["model"], sw.get("xy", ((V.min(0) + V.max(0)) / 2)[:2]),
                                  sw.get("support_z", z), yaw=sw.get("yaw", 0.0), height=sw.get("height"))
                continue
            if spec.get("template"):
                fn, look = TEMPLATES[spec["template"]]
                V, F = fn(spec["base"], spec["spout"], spec["height"])
                mat = color_material(self.stage, f"/World/Looks/static_{name}", look["color"], look["roughness"],
                                     look["metallic"])
                geom = add_mesh(self.stage, path, V, F, None, mat)
            else:
                mesh, V = self.object_mesh_world(name)
                on = self._top_of(fix.get("on")) if fix.get("on") else None
                if on is not None:  # the agent named what it stands on: its bottom on that piece's top (a snap's ray
                    V = V + [0.0, 0.0, on - float(V[:, 2].min())]  # through a library desk's open top found a box under it)
                    self.fixed[name] = f"on {fix['on']} at {on:.3f} m"
                elif V[:, 2].min() < -SINK_FLOOR:
                    self.fixed[name] = f"onto the floor from {100 * V[:, 2].min():.1f} cm"  # under the desk, 11 cm in it:
                    V = V - [0.0, 0.0, float(V[:, 2].min())]  # the depth of its hidden base)
                mat = textured_material(self.stage, f"/World/Looks/static_{name}",
                                        self.s / "objects" / name / "albedo.png", roughness=0.6)
                geom = add_mesh(self.stage, path, V, np.asarray(mesh.faces), mesh.visual.uv, mat)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
            if fix.get("snap") and not fix.get("on"):
                self.fixed[name] = f"snapped {100 * snap(self, geom.GetPrim()):+.1f} cm"

    def build_set_pieces(self, pieces, overrides=None):
        """Set pieces the agent added where the photo shows something the scene graph has no instance for; a variant's
        `set_piece_overrides` re-dress them (a cousin's cloth on a folding screen)."""
        for name, p in (pieces or {}).items():
            p = {**p, **(overrides or {}).get(name, {})}
            box = self.manifest["collision_boxes"].get(p.get("on"))
            z = (float(box[1][2]) if box else float(p.get("z", 0.0))) - p.get("inset", 0.0)  # set into the top
            params = {k: v for k, v in p.items() if k not in ("kit", "center", "yaw_deg", "on", "z", "inset", "seed")}
            UsdGeom.Scope.Define(self.stage, "/World/Static")
            add_set_piece(self.stage, f"/World/Static/{name}", p["kit"], params, [*p["center"], z],
                          np.radians(p.get("yaw_deg", 0.0)), seed=p.get("seed", 0))

    def flat_pieces(self, top, band=0.012):
        """World AABBs of the set pieces lying flat on a top at height `top` (a cooktop: < band proud of it), as
        resolve() will leave them."""
        from .clearance import world_points

        out = []
        root = self.stage.GetPrimAtPath("/World/Static")
        for prim in (root.GetChildren() if root else []):
            if not prim.GetCustomDataByKey("calibrated_scene_reconstruction:kit"):
                continue
            P = world_points(prim)  # the points: a fresh mesh has no extent for a bounds cache yet
            if not len(P):
                continue
            lo, hi = P.min(0), P.max(0)
            hi[2] += max(0.0, top - lo[2])  # set into the top, resolve() lifts it onto the top: where it ends up
            if top - band < hi[2] < top + band:
                out.append((lo, hi))
        return out

    def build_stools(self, stools):
        UsdGeom.Scope.Define(self.stage, "/World/Support")
        self.stool_prims = [self._recon_furniture(f"/World/Support/stool_{k}", st) if st.get("recon") else
                            add_library_model(self.stage, f"/World/Support/stool_{k}", st["model"], st["center"], 0.0,
                                              yaw=st.get("yaw", 0.0), height=st.get("height", 0.72))[0]
                            for k, st in enumerate(stools)]

    def _recon_furniture(self, path, st):
        """One member of a shared reconstruction (recon.furniture), or a room object reconstructed from the completion
        (recon.room_objects, `mesh`: its folder): mesh_local is centred on its footprint."""
        d = self.s / st["mesh"] if st.get("mesh") else self.s / "objects" / st["recon"]
        st = {**st, "recon": st.get("recon") or st["mesh"].replace("/", "_")}
        if not hasattr(self, "_furniture"):
            self._furniture = {}
        if st["recon"] not in self._furniture:
            self._furniture[st["recon"]] = trimesh.load(d / "mesh_local.glb", force="mesh", process=False)
        mesh = self._furniture[st["recon"]]
        V = np.asarray(mesh.vertices)
        if st.get("flip"):  # the review's fix of one SAM 3D laid upside down: half a turn
            # about its own x or y axis through its middle, back down on its base
            c = (V.min(0) + V.max(0)) / 2
            s = np.array([1.0, -1.0, -1.0]) if st["flip"] == "x" else np.array([-1.0, 1.0, -1.0])
            V = (V - c) * s + c
            V = V - [0.0, 0.0, V[:, 2].min()]
        yaw = st.get("yaw", 0.0)
        xform = UsdGeom.Xform.Define(self.stage, path)
        xform.GetPrim().SetCustomDataByKey("calibrated_scene_reconstruction:reconstructed", True)  # stands as the photo shows it (sim.semantics)
        set_pose(xform.GetPrim(), [*st["center"], 0.0], (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        mat = textured_material(self.stage, f"/World/Looks/furniture_{st['recon']}", d / "albedo.png", roughness=0.6,
                                tint=st.get("tint", (1, 1, 1)))
        geom = add_mesh(self.stage, f"{path}/geom", V, np.asarray(mesh.faces), mesh.visual.uv, mat)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexDecomposition")
        return xform.GetPrim()

    def _panel(self, path, name, src, box):
        """A flat room object (recon.room_objects: a whiteboard, a door, a poster) as a thin box on its wall, the face
        towards the room textured from the completion (rectified onto the wall plane)."""
        lo, hi = box
        ax, sign = int(src["axis"]), float(src["normal"])
        o = 1 - ax
        xform = UsdGeom.Xform.Define(self.stage, path)
        m = trimesh.creation.box(extents=hi - lo)
        m.apply_translation((lo + hi) / 2)
        add_mesh(self.stage, f"{path}/body", np.asarray(m.vertices), np.asarray(m.faces), None,
                 color_material(self.stage, f"/World/Looks/{name}_edge", (0.5, 0.5, 0.5), roughness=0.6))
        f = hi[ax] + 0.001 if sign > 0 else lo[ax] - 0.001  # the face towards the room
        q = np.zeros((4, 3))
        q[:, ax] = f
        q[:, o] = [lo[o], hi[o], hi[o], lo[o]]
        q[:, 2] = [lo[2], lo[2], hi[2], hi[2]]
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)  # panel.png spans the wall coordinate lo -> hi, top row up
        add_mesh(self.stage, f"{path}/face", q, np.array([[0, 1, 2], [0, 2, 3]]), uv,
                 textured_material(self.stage, f"/World/Looks/{name}", self.s / src["panel"], roughness=0.5))
        return xform.GetPrim()

    def build_synthetic(self, pieces):
        """Room objects (recon.room_objects), where their depth puts them: what is seen of a piece is what collides
        (its meshes, static triangle meshes); its top slabs (top_slabs) are what a snapped item stands on. Outside /World/Support: clearance, semantics and the sweep leave it alone
        (it stands in the part of the room the photo did not see, away from the task)."""
        UsdGeom.Scope.Define(self.stage, "/World/Synthetic")
        self.synthetic_prims, self.synthetic_boxes = [], {}
        fixes = {n: f.get("fix") or {} for n, f in self.s.scene_graph.get("synthetic_furniture", {}).items()}
        for p in pieces:
            fix = p.get("fix") or fixes.get(p["name"], {})  # a cousin's piece carries its own review fix
            if fix.get("drop"):  # the agent's object review removed it
                continue
            path, src, box = f"/World/Synthetic/{p['name']}", p["source"], np.asarray(p["box"], float)
            if "center" in fix:  # moved by the review (a stool clear of a table's base)
                d = np.r_[np.asarray(fix["center"], float) - np.asarray(p["center"], float), 0.0]
                p, box = {**p, "center": list(fix["center"])}, box + d
            if "model" in fix and src.get("model"):  # the review's library model for a stand-in (DINOv2's pick on a
                src = {**src, "model": fix["model"]}  # failed fit's crop: a rustic table with a low shelf for a desk)
            if "box" in fix and src.get("model"):  # a library stand-in's measured box set by the review (a failed
                box = np.asarray(fix["box"], float)  # fit's box ends short of where the piece reaches in the frames)
                p = {**p, "center": ((box[0, :2] + box[1, :2]) / 2).tolist(), "height": float(box[1, 2] - box[0, 2])}
            if src.get("recon") or src.get("mesh"):
                prim = self._recon_furniture(path, {**src, "center": p["center"], "yaw": p["yaw"],
                                                    "flip": fix.get("flip")})
            elif src.get("panel"):
                prim = self._panel(path, p["name"], src, box)
            elif src.get("model"):
                yaw, ext = float(p["yaw"]), box[1, :2] - box[0, :2]
                if ext.max() >= 1.3 * ext.min():  # a long measured footprint lies along the room's axes (its box is
                    # axis-aligned): the stand-in's own yaw, from a loose silhouette fit, turned to the nearest axis
                    yaw = round(yaw / (np.pi / 2)) * np.pi / 2
                prim = add_library_model(self.stage, path, src["model"], p["center"], float(box[0, 2]), yaw=yaw,
                                         height=p["height"])[0]
                fit_footprint(prim, box, yaw)
            else:
                prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
                mat = color_material(self.stage, f"/World/Looks/{p['name']}", src["color"], roughness=0.7)
                parts = [box]
                if src.get("proxy") == "table":  # a top slab on four legs
                    lo, hi, t, w = box[0], box[1], 0.04, 0.05
                    parts = [np.array([[lo[0], lo[1], hi[2] - t], hi])] + [
                        np.array([[x, y, lo[2]], [x + w, y + w, hi[2] - t]])
                        for x in (lo[0] + 0.03, hi[0] - 0.03 - w) for y in (lo[1] + 0.03, hi[1] - 0.03 - w)]
                for k, b in enumerate(parts):
                    m = trimesh.creation.box(extents=b[1] - b[0])
                    m.apply_translation((b[0] + b[1]) / 2)
                    add_mesh(self.stage, f"{path}/part_{k}", np.asarray(m.vertices), np.asarray(m.faces), None, mat)
            for q in Usd.PrimRange(prim):  # what is seen is what collides: its meshes as static triangle meshes
                if q.IsA(UsdGeom.Mesh):  # (one box per piece held things on ledges nobody sees: a desk's box topped
                    UsdPhysics.CollisionAPI.Apply(q).CreateCollisionEnabledAttr(True)  # by the lamp on it)
                    UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr("none")
                elif q.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
            prim.SetCustomDataByKey("calibrated_scene_reconstruction:synthetic", True)
            prim.SetCustomDataByKey("calibrated_scene_reconstruction:mesh_collider", True)
            self.synthetic_prims.append(prim)
            self.synthetic_boxes.update({f"{p['name']}#{k}": b for k, b in enumerate(top_slabs(prim))})

    def _add_kit_object(self, path, spec, support_z):
        """A kit or Blender solid at its fitted pose, with surface SDF collision and material-based mass."""
        from ..assets import kits
        from .receptacle import look_material

        if spec.get("model"):
            from calibrated_scene_reconstruction.assets.metric_mesh import asset
            kit = asset(spec)
        else:
            kit = kits.build(spec["kit"], spec.get("params", {}))
        xform = UsdGeom.Xform.Define(self.stage, path)
        xy = spec.get("xy", [0.0, 0.0])
        from calibrated_scene_reconstruction.assets.object_pose import rotation
        set_pose(xform.GetPrim(), [xy[0], xy[1], support_z if spec.get("model") else support_z + 0.002], quat_from_matrix(rotation(spec)))
        if kit.get('links'):
            from .mechanism_stage import add_links
            return add_links(self.stage, path, kit, spec,
                             self.s.scene_graph['dynamic_objects'][path.split('/')[-1]], self._grip_material(),
                             look_base=self.s.dir)
        look = spec.get("look")
        if look and look.get('preserves_surface'):
            import hashlib
            from calibrated_scene_reconstruction.core.paths import resolve
            from .object_materials import add_surface
            if hashlib.sha256(resolve(spec['mesh']).read_bytes()).hexdigest() != look.get('source_mesh_sha256'):
                raise ValueError('Object appearance belongs to a different physical surface')
            d = np.load(self.s / look['mesh'])
            offset = 0
            for part in kit['parts']:
                n = len(part['F'])
                surface = {**part, 'UV': d['face_uv'][offset:offset+n]}
                add_surface(self.stage, f"{path}/{part['name']}", surface, look=look, look_base=self.s.dir)
                offset += n
            if offset != len(d['face_uv']):
                raise ValueError('Object texture face count differs from physical geometry')
        elif look:  # legacy appearance atlas
            d = np.load(self.s / look["mesh"])
            mat = textured_material(self.stage, f"/World/Looks/obj_{path.split('/')[-1]}", self.s / look["albedo"],
                                    roughness=look.get("roughness", 0.5), tint=(float(look.get("gain", 1.0)),) * 3)
            add_mesh(self.stage, f"{path}/look", d["V"], d["F"], d["UV"], mat)
        else:
            for k, part in enumerate(kit["parts"]):
                if part.get('materials'):
                    from .object_materials import add_surface
                    add_surface(self.stage, f"{path}/{part['name']}", part)
                else:
                    mat = look_material(self.stage, f"/World/Looks/obj_{path.split('/')[-1]}_{k}", part["material"])
                    add_mesh(self.stage, f"{path}/{part['name']}", part["V"], part["F"], part["UV"], mat)
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid.CreateEnableCCDAttr(True)
        # A single physical link needs the same contact accuracy as LINKS
        # assemblies; otherwise mechanism-free rigid authoring silently uses
        # the coarse native iteration defaults for the same fitted SDF solid.
        rigid.CreateSolverPositionIterationCountAttr(32)
        rigid.CreateSolverVelocityIterationCountAttr(1)
        mass, why = kit_mass(kit, self.s.scene_graph.get("dynamic_objects", {}).get(path.split("/")[-1], {}),
                             spec.get("mass", 0.05))
        if spec.get("model"):
            from calibrated_scene_reconstruction.assets.material_mass import mass as solid_mass
            volume = sum(trimesh.Trimesh(p["V"], p["F"], process=True).volume for p in kit["parts"])
            mass, why = solid_mass(volume, self.s.scene_graph["dynamic_objects"][path.split("/")[-1]])
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        prim.SetCustomDataByKey("calibrated_scene_reconstruction:mass_from", why)
        # its colliders: the kit's own surface as a signed distance field -- the fine shape, a cup's mouth and a
        # post's hole open (the user, 2026-10-27: a task object's collider is never one convex hull or a coarse
        # decomposition; one hull of a cup was a solid cone and the other cup could not go in)
        collision_settings = sdf_settings(spec.get('sdf'))
        for i, part_ in enumerate(kit["parts"]):
            col = add_mesh(self.stage, f"{path}/collision/part_{i}", np.asarray(part_["V"], float),
                           np.asarray(part_["F"], int))
            UsdGeom.Imageable(col.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(col.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(collision_settings['approximation'])
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(col.GetPrim())
            sdf.CreateSdfResolutionAttr(collision_settings['resolution'])
            pc = PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim())
            pc.CreateContactOffsetAttr(collision_settings['contact_offset'])
            pc.CreateRestOffsetAttr(collision_settings['rest_offset'])
            UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(
                self._grip_material(), UsdShade.Tokens.weakerThanDescendants, "physics")
        return prim

    def add_object(self, spec, support_z):
        """Manipulable rigid object: twin mesh (`name`), mesh file (`mesh_path`: a YCB scan, an augmented twin) or
        `ph_model`; optional `scale`, `tint`, `roughness`, `mass`."""
        name = spec.get("as", spec.get("name", spec.get("ph_model")))
        path = f"/World/Objects/{name}"
        UsdGeom.Scope.Define(self.stage, "/World/Objects")
        if spec.get("ph_model"):
            prim, _ = add_library_model(self.stage, path, spec["ph_model"], spec["xy"], support_z, yaw=spec.get("yaw", 0.0),
                                        height=spec.get("height"), scale=spec.get("scale"), dynamic=True,
                                        mass=spec.get("mass", 0.3))
            self.objects[name] = prim
            return prim
        if spec.get("kit") or spec.get("model"):  # the agent's measured primitive in place of the reconstruction (assets.kits)
            prim = self._add_kit_object(path, spec, support_z)
            self.objects[name] = prim
            return prim
        if spec.get("mesh_path"):
            mesh = trimesh.load(repo_path(spec["mesh_path"]), force="mesh")
            V = np.asarray(mesh.vertices)
            texture = spec.get("texture") and repo_path(spec["texture"])
        else:
            mesh, V = self.object_mesh_world(spec["name"])
            texture = self.s / "objects" / spec["name"] / "albedo.png"
        label = object_label(self.s, spec)
        round_prop = rolls(label) and name != self.s.scene_graph["task"]["target"]
        if round_prop:  # round produce among the distractors: on its steadiest flat spot (assets.physprops)
            R, _ = rest_pose(V, mass_props(label, V)["com"])
            c = (V.min(0) + V.max(0)) / 2
            V = (V - c) @ R.T + c
        else:  # a reconstruction (or a mesh made from one) set down on its own base (grasping.targets.level)
            V = level(V)
        bottom = np.array([(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2, V[:, 2].min()])
        local = (V - bottom) * spec.get("scale", 1.0)
        if spec.get("size") and spec.get("size_yaw_deg") is not None:
            # a measured box along the object's own footprint axes (a peg lying diagonally): the mesh turned onto its
            # principal axes, scaled to (length, width, height), turned to the measured heading
            q = local[:, :2] - local[:, :2].mean(0)
            _, _, Vt = np.linalg.svd(q, full_matrices=False)
            own = np.c_[q @ Vt.T, local[:, 2]]
            own = own * (np.asarray(spec["size"], float) / np.maximum(np.ptp(own, axis=0), 1e-4))
            ya = np.radians(spec["size_yaw_deg"])
            Rz = np.array([[np.cos(ya), -np.sin(ya)], [np.sin(ya), np.cos(ya)]])
            local = np.c_[own[:, :2] @ Rz.T, own[:, 2]]
            local[:, :2] -= (local[:, :2].min(0) + local[:, :2].max(0)) / 2
        elif spec.get("size"):  # a measured box: the mesh scaled to it along each axis
            local = local * (np.asarray(spec["size"], float) / np.maximum(np.ptp(local, axis=0), 1e-4))
        pos = bottom.copy() if not spec.get("mesh_path") else np.zeros(3)
        pos[2] = support_z + 0.003
        if spec.get("xy"):
            pos[:2] = spec["xy"][:2]
        yaw = spec.get("yaw", 0.0)
        xform = UsdGeom.Xform.Define(self.stage, path)
        set_pose(xform.GetPrim(), pos, (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        look = self.s.scene_graph.get("dynamic_objects", {}).get(spec.get("name"), {}).get("look") or {}
        if look and not spec.get("mesh_path"):  # the agent's material for a twin object (sim.setpieces): glass
            from .receptacle import look_material
            from ..assets.kits import GLASS

            mat = look_material(self.stage, f"/World/Looks/obj_{name}", {**GLASS, **look})
        else:
            mat = textured_material(self.stage, f"/World/Looks/obj_{name}", texture,
                                    roughness=spec.get("roughness", 0.55), tint=spec.get("tint", (1, 1, 1)))
        geom = add_mesh(self.stage, f"{path}/geom", local, np.asarray(mesh.faces), mesh.visual.uv, mat)
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # semantic centre of mass (assets.physprops): uniform density would put a lamp's or a vase's too high
        mp = mass_props(label, local)
        if round_prop:  # and the rolling resistance of its soft contact
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateAngularDampingAttr(ROLLING_DAMPING)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(spec.get("mass", mp["mass"])))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*mp["com"]))
        parts = collision_parts(self.s, mesh_key(spec, local))
        if parts:  # the grasp synthesis' convex parts: the simulator collides with what the grasps were planned on
            for i, P in enumerate(parts):
                hull = trimesh.convex.convex_hull(P)
                part = add_mesh(self.stage, f"{path}/collision/part_{i}", np.asarray(hull.vertices),
                                np.asarray(hull.faces))
                UsdGeom.Imageable(part.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
                UsdPhysics.CollisionAPI.Apply(part.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr("convexHull")
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(part.GetPrim()).CreateHullVertexLimitAttr(64)
                UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(
                    self._grip_material(), UsdShade.Tokens.weakerThanDescendants, "physics")
        else:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexDecomposition")
            cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(geom.GetPrim())
            cd.CreateMaxConvexHullsAttr(24)
            cd.CreateHullVertexLimitAttr(48)
            cd.CreateVoxelResolutionAttr(200000)
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(self._grip_material(),
                                                                     UsdShade.Tokens.weakerThanDescendants, "physics")
        self.objects[name] = prim
        return prim

    def _grip_material(self):
        if not hasattr(self, "_grip"):
            self._grip = UsdShade.Material.Define(self.stage, "/World/Looks/grip_physics")
            pm = UsdPhysics.MaterialAPI.Apply(self._grip.GetPrim())
            pm.CreateStaticFrictionAttr(1.2)
            pm.CreateDynamicFrictionAttr(1.0)
            pm.CreateRestitutionAttr(0.0)
        return self._grip

    # --- lights -------------------------------------------------------------------------------------------------
    def build_lights(self, scale=1.0, temp=5200.0, sun=True):
        """Daylight rect lights in both windows, recessed ceiling disks (or `ceiling_panels` [[x, y, w, h]]: fluorescent
        panels, intensity `panel`), sky dome, a sun unless `sun` is false, and two large soft emitters that stand in for
        multi-bounce light (real-time RTX); intensities and `temp` (K) come from configs/<name>/lights.json."""
        cfg = self.light_cfg
        UsdGeom.Scope.Define(self.stage, "/World/Lights")
        sky = UsdLux.DomeLight.Define(self.stage, "/World/Lights/sky")
        if cfg.get("envmap"):  # the room as the splat holds it, round the task zone (apps/scene/splat_envmap.py)
            sky.CreateTextureFileAttr(str(self.s / cfg["envmap"]))
            sky.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
            sky.CreateIntensityAttr(cfg.get("envmap_intensity", 1000.0) * scale)
        elif cfg.get("hdri"):  # an outdoor HDRI (style themes): sky light and what windows look out on
            sky.CreateTextureFileAttr(str(repo_path(cfg["hdri"])))
            sky.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
            sky.CreateIntensityAttr(cfg.get("hdri_intensity", 900.0) * scale)
            # no rotation: RTX maps the latlong's pole to the stage's up axis (Z here); turning the dome +Y -> +Z
            # stood the sky on its side (the horizon vertical in every gap and reflection, sky light from the side)
        else:
            sky.CreateIntensityAttr(400.0 * scale)
            sky.CreateEnableColorTemperatureAttr(True)
            sky.CreateColorTemperatureAttr(7000.0)

        def rect(path, w, h, intensity, kelvin, pos, quat=(1, 0, 0, 0)):
            l = UsdLux.RectLight.Define(self.stage, path)
            l.CreateWidthAttr(float(w))
            l.CreateHeightAttr(float(h))
            l.CreateIntensityAttr(float(intensity))
            l.CreateEnableColorTemperatureAttr(True)
            l.CreateColorTemperatureAttr(float(kelvin))
            set_pose(l.GetPrim(), pos, quat)  # rect lights emit along local -Z

        parts = self.manifest["parts"]
        for name, part in parts.items():
            if part.get("kind") == "structure" and name in self.windows:
                V = np.load(self.s / "scene" / part["npz"])["V"]
                ext = V.max(0) - V.min(0)
                n = np.asarray(part["normal"], float)  # emit into the room: local -Z = n, local X horizontal
                xa = np.cross([0, 0, 1], n)
                xa /= np.linalg.norm(xa)
                R = np.column_stack([xa, np.cross(-n, xa), -n])
                rect(f"/World/Lights/{name}", max(ext[0], ext[1]), ext[2], cfg.get("window", 9000.0) * scale, 6500.0,
                     V.mean(0) + 0.03 * n, quat_from_matrix(R))
        sf = self.manifest["surfaces"]
        zc = sf["ceiling_z"] - 0.01
        (x0, x1), (y0, y1) = sf["room_x"], sf["room_y"]
        disks = cfg.get("ceiling_disks") or [[x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)]
                                              for fx, fy in [(0.45, 0.65), (0.72, 0.65), (0.59, 0.83), (0.22, 0.58), (0.59, 0.32)]]
        temp = cfg.get("temp", temp)  # the agent's measured colour temperature (lights.json; a cousin's theme wins)
        for i, (px, py, pw, ph) in enumerate(cfg.get("ceiling_panels") or []):  # office / lab fluorescent panels
            rect(f"/World/Lights/panel_{i}", pw, ph, cfg.get("panel", 9000.0) * scale, temp, [px, py, zc])
        if cfg.get("ceiling_panels") and not cfg.get("ceiling_disks"):
            disks = []  # panels instead of the default recessed disks
        for i, xy in enumerate(disks):
            d = UsdLux.DiskLight.Define(self.stage, f"/World/Lights/ceiling_{i}")
            d.CreateRadiusAttr(0.08)
            d.CreateIntensityAttr(cfg.get("disk", 30000.0) * scale)
            d.CreateEnableColorTemperatureAttr(True)
            d.CreateColorTemperatureAttr(temp)
            set_pose(d.GetPrim(), [xy[0], xy[1], zc])
        if cfg.get("fill", True):  # stand-ins for multi-bounce light; path tracing computes the bounces (fill: false)
            self._fill_lights(rect, cfg, scale, temp, x0, x1, y0, y1, zc)
        if sun and cfg.get("sun", True):  # lights.json "sun": false -- blinds down, no direct sunlight in the photo
            self._sun(scale)

    def _fill_lights(self, rect, cfg, scale, temp, x0, x1, y0, y1, zc):
        rect("/World/Lights/ceiling_bounce", 0.9 * (x1 - x0), 0.9 * (y1 - y0), cfg.get("bounce", 1200.0) * scale, temp,
             [(x0 + x1) / 2, (y0 + y1) / 2, zc - 0.02])
        rect("/World/Lights/room_fill", 0.9 * (x1 - x0), 2.0, cfg.get("front", 1500.0) * scale, temp,
             [(x0 + x1) / 2, y0 + 0.05, 1.3], (0.7071, 0.7071, 0.0, 0.0))

    def _sun(self, scale):
        sunl = UsdLux.DistantLight.Define(self.stage, "/World/Lights/sun")
        sunl.CreateIntensityAttr(2500.0 * scale)
        sunl.CreateAngleAttr(1.5)
        set_pose(sunl.GetPrim(), [0, 0, 0], quat_between([0, 0, 1], -np.array([-0.75, 0.25, -0.6])))

    # --- observed-coverage rendering ------------------------------------------------------------------------------
    def reviewed_parts(self):
        """Room parts whose look the photos ground (qa/surfaces.json: a surface with photo evidence, reviewed)."""
        import json

        f = self.s / "qa" / "surfaces.json"
        if not f.exists() or "_surface_review" not in self.s.scene_graph:
            return set()
        groups = json.loads(f.read_text()).get("groups", {})
        return {p for g in groups.values() if g.get("proposal") for p in g.get("parts", [])}

    def coverage_mode(self, on: bool):
        """Room shows its observed masks (white = seen in the photo, black = synthesized); objects/furniture count
        as observed (white). The robot keeps its own materials (unlit here) and is excluded by the caller via semantic
        labels: binding and unbinding a material on the referenced robot drops its instanced base / torso meshes from
        every later render (a placement sweep's overview images lost the Vega's body after the first candidate)."""
        if not hasattr(self, "_cov"):
            self._cov, self._orig = {}, {}
            self._white = color_material(self.stage, "/World/Looks/cov_white", (0, 0, 0), emissive=(1, 1, 1))
            reviewed = self.reviewed_parts()
            for name, part in self.manifest["parts"].items():
                prim = self.stage.GetPrimAtPath(f"/World/Room/{name}")
                if not prim.IsValid():
                    continue
                self._orig[name] = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
                # a reviewed surface (recon.surfaces: its material matched to the photo's clean pixels) counts as seen
                self._cov[name] = self._white if name in reviewed else textured_material(
                    self.stage, f"/World/Looks/cov_{name}", self.s / "scene" / part["observed"], tint=(0, 0, 0),
                    emissive=1.0, emissive_tint=(1, 1, 1))
        for name in self._cov:
            UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(f"/World/Room/{name}")).Bind(
                self._cov[name] if on else self._orig[name])
        lights = self.stage.GetPrimAtPath("/World/Lights")
        UsdGeom.Imageable(lights).GetVisibilityAttr().Set("invisible" if on else "inherited")
        roots = ["/World/Objects", "/World/Static", "/World/Support"]
        if self.s.config("completion.json", {}).get("source") == "captured":
            # nothing generated: the room's objects were reconstructed from the photo, its views and the walks'
            # frames, so they count as observed
            roots.append("/World/Synthetic")
        for root, mat in [(r, self._white) for r in roots]:
            p = self.stage.GetPrimAtPath(root)
            if p.IsValid():
                api = UsdShade.MaterialBindingAPI.Apply(p)
                api.Bind(mat, UsdShade.Tokens.strongerThanDescendants) if on else api.UnbindDirectBinding()


def build_scene(stage, session, variant):
    """Populate a stage from prepared scene inputs and a composed variant."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    ts = TwinScene(stage, session)
    ts.build_physics()
    # the agent's furniture looks (sim.setpieces) where the variant has no override of its own (a cousin's theme wins)
    ov = variant.get("room_overrides") or {}
    looks = {p: o for p, o in ts.inputs.furniture_overrides.items() if not any(p.startswith(k) for k in ov)}
    ts.build_room({**looks, **ov})
    ts.build_synthetic(variant.get("synthetic_furniture", []))  # first: a static object may stand on one (snap)
    ts.build_static_objects(variant.get("skip_static", ()), variant.get("static_swaps"))
    ts.build_set_pieces(ts.s.scene_graph.get("set_pieces"), variant.get("set_piece_overrides"))
    ts.build_stools(variant.get("stool_models", []))
    li = variant.get("lights", {})
    ts.light_cfg = li
    ts.build_lights(scale=li.get("scale", 1.0), temp=li.get("temp", 5200.0), sun=li.get("sun", True))
    zt = ts.manifest["surfaces"]["counter_top_z"]  # the task support; each twin object rests on its own slab
    zs = ts.manifest.get("support_z", {})
    # the agent's fix of a twin object: `size` (its box, m, along x, y, z: the reconstruction scaled to it axis by axis)
    # and `xy` (its centre), measured where many frames saw it (recon.object_measure: the splat)
    default = [{"name": n, "mass": spec.get("mass", 0.3),
                **{k: v for k, v in (ts.s.scene_graph.get("dynamic_objects", {}).get(n, {}).get("fix") or {}).items()
                   if k in ("size", "xy", "size_yaw_deg", "support_z", "support_rotation", "kit", "params", "yaw_deg", "look", "model", "mesh", "color", "opening", "sdf", "link_physics")}}
               for n, spec in ts.manifest["dynamic_objects"].items()]
    flat = ts.flat_pieces(zt)
    for spec in variant.get("objects", default):  # `support_z`: standing on something (calibrated_scene_reconstruction.dr: a board, a cooktop)
        z = spec.get("support_z", zs.get(spec.get("as", spec.get("name")), zt))
        if spec.get("xy"):  # over a flat set piece (a cooktop): on its top as built
            z = max([z] + [hi[2] for lo, hi in flat if np.all(np.asarray(spec["xy"][:2]) >= lo[:2])
                           and np.all(np.asarray(spec["xy"][:2]) <= hi[:2])])
        ts.add_object(spec, z)
    # no placed item may intersect the room or another item: check and push out horizontally
    ts.clearance = resolve(ts, first=[ts.s.scene_graph["task"]["target"]])
    print(summarize(ts.clearance), flush=True)
    # a randomized configuration's receptacles (calibrated_scene_reconstruction.dr: one under the target, a tray to put the goal on), static,
    # placed clear by the sampler (after the resolve: an object standing in a plate's well is no penetration)
    for k, rd in enumerate(variant.get("receptacles", [])):
        from ..assets import receptacles as rcp
        from .receptacle import add_receptacle

        prim = add_receptacle(stage, f"/World/Static/dr_{rd.get('role', 'item')}_{k}", rcp.rebuild(rd), rd["origin"],
                              rd["yaw"])
        prim.SetCustomDataByKey("calibrated_scene_reconstruction:role", rd.get("role", "item"))
    priors = getattr(ts.inputs, 'physical_priors', None)
    if priors:
        from .physical_priors import apply as apply_priors
        ts.physical_estimates = apply_priors(stage, priors, ts.inputs.mechanisms,
                                            getattr(ts.inputs, 'physics_case', 'nominal'))
        from .support_placement import apply as apply_sampled_support
        apply_sampled_support(stage,ts,ts.physical_estimates)
    apply_scene_physics(stage, ts, variant)
    from .simulation import apply_rigid_iterations
    apply_rigid_iterations(stage,getattr(ts.inputs,'simulation',None))
    return ts


def robot_semantics(stage, path="/World/Robot"):
    """Label every robot gprim so cameras can mask the robot out (coverage ratio)."""
    from isaacsim.core.utils.semantics import add_labels

    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.IsA(UsdGeom.Gprim):
            add_labels(prim, labels=["robot"], instance_name="class")
