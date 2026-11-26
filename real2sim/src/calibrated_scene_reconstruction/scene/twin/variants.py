"""Compose scene and cousin configuration without importing the simulator."""
from calibrated_scene_reconstruction.scene.session import Session


def photo_goal(session, variant):
    """The variant with the scene's place target where the photo shows it (scene graph task.place.photo_goal {kind,
    xy, yaw}, built to task.place.match), for the views of a scene (fly-through, robot views): an episode places its
    own goal per plan, a configuration's receptacles are left as they are."""
    from calibrated_scene_reconstruction.assets import receptacles as rcp

    s = session if isinstance(session, Session) else Session(session)
    place = (s.scene_graph.get("task") or {}).get("place") or {}
    g = place.get("photo_goal")
    if not g:
        return variant
    rec = rcp.make(g["kind"], g.get("r_obj", 0.04), g.get("h_obj", 0.1), g.get("seed", 0),
                   (place.get("match") or {}).get(g["kind"]))
    if rec is None:
        return variant
    top = float(s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"][1][2])
    rd = {**rec.to_dict(), "origin": [float(g["xy"][0]), float(g["xy"][1]), top], "yaw": float(g.get("yaw", 0.0)),
          "role": "goal"}
    return {**variant, "receptacles": list(variant.get("receptacles") or []) + [rd]}


def prepare_variant(session, variant=None):
    """scene/twin.json defaults overlaid with a variant. Furniture in a robot's way is moved once the robot's motion
    is known (sim.sweep)."""
    s = session if isinstance(session, Session) else Session(session)
    base = s.read_json("scene/twin.json")
    variant = variant or {}
    v = {**base, **variant}
    v["lights"] = {**base.get("lights", {}), **variant.get("lights", {})}
    v["room_overrides"] = {**base.get("room_overrides", {}), **variant.get("room_overrides", {})}
    v["static_swaps"] = {**base.get("static_swaps", {}), **variant.get("static_swaps", {})}
    return v
