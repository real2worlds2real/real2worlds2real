"""Use existing rigid kit surfaces as explicit starting geometry for editable models."""
from . import kits


# Session modules may extend or replace KITS. Only these built-in fitted
# builders implement the rigid, metric, -Y-front kit frame used below.
_RIGID_TEMPLATES = {f.__name__: f for f in (kits.cylinder, kits.cup, kits.box_frame, kits.plate_post_hole)}


def is_rigid_template(kind):
    return kind in _RIGID_TEMPLATES and kits.KITS.get(kind) is _RIGID_TEMPLATES[kind]


def build_workspace(kind, build_model, params):
    """Adapt the complete authored kit from its native -Y front to workspace +Y."""
    import bpy
    from mathutils import Matrix, Vector
    from real2sim_harness.asset_metadata import asset_value

    if not is_rigid_template(kind):
        raise ValueError("Provide build_scene() with the custom model's asset contract")
    scene = bpy.context.scene
    for key, value in {"real2sim_asset_kind": "rigid", "real2sim_front_axis": "+Y",
                       "real2sim_up_axis": "+Z", "real2sim_units": "m"}.items():
        if asset_value(scene, key, value) != value:
            raise ValueError(f"The generated rigid kit's workspace frame conflicts with {key}")
        scene[key] = value
    existing = set(scene.objects)
    build_model(params)
    bpy.context.view_layer.update()
    created = set(scene.objects) - existing
    half_turn = Matrix.Diagonal(Vector((-1, -1, 1, 1)))
    for obj in created:
        if obj.parent not in created:
            obj.matrix_world = half_turn @ obj.matrix_world
    bpy.context.view_layer.update()


def build(kind, params):
    import bpy

    for part in kits.build(kind, params)["parts"]:
        mesh = bpy.data.meshes.new(part["name"])
        mesh.from_pydata(part["V"].tolist(), [], part["F"].tolist())
        mesh.update()
        layer = mesh.uv_layers.new(name="source_uv")
        for loop in mesh.loops:
            layer.data[loop.index].uv = part["UV"][loop.vertex_index]
        obj = bpy.data.objects.new(part["name"], mesh)
        bpy.context.collection.objects.link(obj)
        surface = part["material"]
        if "tex" in surface:
            raise ValueError("Use the native textured kit or author the texture in the editable source")
        mat = bpy.data.materials.new(surface.get("name", part["name"]))
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        color = (*surface.get("color", (.8, .8, .8)), 1)
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = surface.get("roughness", .5)
        shader.inputs["Metallic"].default_value = surface.get("metallic", 0.)
        if surface.get("glass"):
            shader.inputs["Transmission Weight"].default_value = 1.
            shader.inputs["IOR"].default_value = surface.get("ior", 1.5)
        mat.diffuse_color = color
        mesh.materials.append(mat)
