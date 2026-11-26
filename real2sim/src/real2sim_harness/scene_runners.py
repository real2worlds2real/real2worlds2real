"""Choose an operator-authorized runtime by the assembled task capabilities."""

from __future__ import annotations


def select_runner(configuration, required, *, record_video=False, configurable_timestep=False):
    if configuration.get("schema_version") != 1:
        raise ValueError("Runner configuration needs schema_version 1")
    routes = configuration.get("routes")
    if routes is None:
        routes = [configuration]
    if not isinstance(routes, list) or not 1 <= len(routes) <= 32:
        raise ValueError("Runner routes must be a bounded nonempty list")
    needed = set(required)
    if record_video:
        needed.add("scene_video")
    if configurable_timestep:
        needed.add("configurable_timestep")
    candidates = []
    ids = set()
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise ValueError("Runner routes must be objects")
        name = route.get("id", f"route-{index + 1}")
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Runner route IDs must be unique nonempty strings")
        ids.add(name)
        argv, caps = route.get("argv"), route.get("capabilities")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(v, str) and v for v in argv)
        ):
            raise ValueError("Runner configuration needs a nonempty argv list of strings")
        if not isinstance(caps, list) or not all(isinstance(v, str) and v for v in caps):
            raise ValueError("The runner must declare its actual capabilities")
        missing = needed - set(caps)
        if record_video and not any("{review_request}" in value for value in argv):
            missing.add("scene_video")
        if configurable_timestep and not any("{review_request}" in value for value in argv):
            missing.add("configurable_timestep")
        candidates.append(
            {
                "id": name,
                "argv": argv,
                "capabilities": caps,
                "missing_capabilities": sorted(missing),
            }
        )
    # Operator order determines preference between equally capable runtimes.
    chosen = next((route for route in candidates if not route["missing_capabilities"]), None)
    if chosen is not None:
        return {"ok": True, "route": chosen, "routes_considered": candidates}
    closest = min(candidates, key=lambda route: len(route["missing_capabilities"]))
    return {
        "ok": False,
        "missing_capabilities": closest["missing_capabilities"],
        "routes_considered": candidates,
    }
