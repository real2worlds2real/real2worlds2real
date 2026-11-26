"""Render an accepted Blender attempt at presentation quality."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_surface import is_asset_object  # noqa: E402


def axis_vector(value):
    return {
        "+X": Vector((1.0, 0.0, 0.0)),
        "-X": Vector((-1.0, 0.0, 0.0)),
        "+Y": Vector((0.0, 1.0, 0.0)),
        "-Y": Vector((0.0, -1.0, 0.0)),
    }[value]


def object_points(objects):
    return [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]


def bounds(points):
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return lo, hi


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def fit_camera(scene, camera, center, direction, radius, points, fill=0.82):
    distance = max(radius * 2.5, 0.15)
    for _ in range(7):
        camera.location = center + direction.normalized() * distance
        look_at(camera, center)
        bpy.context.view_layer.update()
        projected = [world_to_camera_view(scene, camera, point) for point in points]
        if any(point.z <= 0 for point in projected):
            distance *= 1.5
            continue
        width = max(point.x for point in projected) - min(point.x for point in projected)
        height = max(point.y for point in projected) - min(point.y for point in projected)
        scale = max(width, height) / fill
        if 0.99 <= scale <= 1.01:
            break
        distance = max(distance * min(max(scale, 0.55), 1.8), radius * 1.1)
    # Perspective depth can move the projected box away from the image center.
    # Extent-only fitting can otherwise clip one end even when width/height fit.
    projected = [world_to_camera_view(scene, camera, point) for point in points]
    middle_x = (min(p.x for p in projected) + max(p.x for p in projected)) / 2
    middle_y = (min(p.y for p in projected) + max(p.y for p in projected)) / 2
    original_x, original_y = camera.data.shift_x, camera.data.shift_y
    probe = projected[0]
    camera.data.shift_x = original_x + .01
    slope_x = (world_to_camera_view(scene, camera, points[0]).x - probe.x) / .01
    camera.data.shift_x = original_x
    camera.data.shift_y = original_y + .01
    slope_y = (world_to_camera_view(scene, camera, points[0]).y - probe.y) / .01
    camera.data.shift_y = original_y
    if abs(slope_x) < 1e-8 or abs(slope_y) < 1e-8:
        raise ValueError("Camera shift could not center the projected asset bounds")
    camera.data.shift_x += (.5 - middle_x) / slope_x
    camera.data.shift_y += (.5 - middle_y) / slope_y
    bpy.context.view_layer.update()


def configure_cycles(resolution, samples):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.3
    try:
        bpy.context.view_layer.cycles.use_denoising = True
    except AttributeError:
        pass

    enabled = []
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is not None:
        preferences = addon.preferences
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                preferences.compute_device_type = backend
                preferences.get_devices()
            except (TypeError, RuntimeError):
                continue
            devices = [device for device in preferences.devices if device.type != "CPU"]
            if not devices:
                continue
            for device in preferences.devices:
                device.use = device in devices
            enabled = [f"{device.type}:{device.name}" for device in devices]
            scene.cycles.device = "GPU"
            break
    return enabled


def render_view(scene, camera, output, name, points, direction, fill=0.82):
    lo, hi = bounds(points)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.05)
    fit_camera(scene, camera, center, direction, radius, points, fill=fill)
    scene.render.filepath = str(output / name)
    bpy.ops.render.render(write_still=True)
    return name


def main():
    separator = sys.argv.index("--")
    output = Path(sys.argv[separator + 1]).resolve()
    resolution = int(sys.argv[separator + 2])
    samples = int(sys.argv[separator + 3])
    report = {"ok": False}
    try:
        audit_path = output / "report.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            if not audit.get("contract", {}).get("ok"):
                raise RuntimeError("Fix the asset contract before final rendering")
            if audit.get("articulation", {}).get("required") and not audit["articulation"].get("ok"):
                raise RuntimeError(
                    "Fix the articulation gate before final rendering: "
                    + str(audit.get("error") or audit["articulation"])
                )
        scene = bpy.context.scene
        renderable = [
            obj
            for obj in scene.objects
            if is_asset_object(obj)
        ]
        if not renderable:
            raise RuntimeError("Scene contains no renderable asset objects")
        points = object_points(renderable)
        front = axis_vector(asset_value(scene, "real2sim_front_axis", "+Y"))
        right = Vector((front.y, -front.x, 0.0))
        up = Vector((0.0, 0.0, 1.0))
        camera = scene.camera
        if camera is None:
            data = bpy.data.cameras.new("Real2Sim_HeroCamera")
            camera = bpy.data.objects.new("Real2Sim_HeroCamera", data)
            bpy.context.collection.objects.link(camera)
            scene.camera = camera
        camera.data.lens = 58
        camera.data.sensor_width = 36
        camera.data.sensor_fit = "AUTO"
        camera.data.shift_x = camera.data.shift_y = 0
        devices = configure_cycles(resolution, samples)
        images = []
        views = [
            ("hero_front.png", front + up * 0.24),
            ("hero_front_left.png", front + right * 0.70 + up * 0.34),
            ("hero_front_right.png", front - right * 0.70 + up * 0.34),
            ("hero_back.png", -front + right * 0.52 + up * 0.38),
        ]
        lo, hi = bounds(points)
        horizontal = (hi.x - lo.x, hi.y - lo.y)
        if hi.z - lo.z < min(horizontal) * 0.4:
            views.append(("hero_top.png", up + front * 0.15))
        if max(horizontal) / max(min(horizontal), 1e-6) > 2.0:
            views.extend(
                [
                    ("hero_left.png", right + up * 0.18),
                    ("hero_right.png", -right + up * 0.18),
                ]
            )
        for name, direction in views:
            images.append(render_view(scene, camera, output, name, points, direction))

        detail = [obj for obj in renderable if asset_value(obj, "real2sim_detail")]
        if detail:
            detail_points = object_points(detail)
            detail_lo, detail_hi = bounds(detail_points)
            flat_detail = detail_hi.z - detail_lo.z < min(detail_hi.x - detail_lo.x, detail_hi.y - detail_lo.y) * 0.4
            images.append(
                render_view(
                    scene,
                    camera,
                    output,
                    "hero_detail.png",
                    detail_points,
                    front + up * (1.4 if flat_detail else 0.08),
                    fill=0.72,
                )
            )
        report = {
            "ok": True,
            "engine": "CYCLES",
            "resolution": [resolution, resolution],
            "samples": samples,
            "denoising": True,
            "gpu_devices": devices,
            "images": images,
        }
    except BaseException as exc:
        report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-12000:],
        }
        raise
    finally:
        (output / "final_render.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
