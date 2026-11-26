"""Choose a task-level physics backend before materializing simulator assets."""
from __future__ import annotations

KINDS = {"rigid", "articulated", "deformable", "hybrid", "fluid", "strand", "fracture", "solid"}


def check_support_response(*, initial_bottom_m, final_bottom_m, support_top_m, late_speed_m_s,
                           footprint_inside, minimum_drop_m=0.005, separation_tolerance_m=0.01,
                           settled_speed_tolerance_m_s=0.03):
    """An observed drop-and-settle test, not a direct contact-force measurement."""
    import math
    finite = all(math.isfinite(value) for value in (initial_bottom_m, final_bottom_m, support_top_m, late_speed_m_s))
    failures = []
    if not finite:
        failures.append("non_finite_response")
    if initial_bottom_m - final_bottom_m < minimum_drop_m:
        failures.append("no_testable_free_drop")
    if abs(final_bottom_m - support_top_m) > separation_tolerance_m:
        failures.append("not_at_support_surface")
    if late_speed_m_s > settled_speed_tolerance_m_s or late_speed_m_s < 0:
        failures.append("not_settled")
    if not footprint_inside:
        failures.append("outside_support_footprint")
    return {"ok": not failures, "failures": failures,
            "method": "observed_free_drop_and_settle_on_declared_horizontal_support",
            "direct_contact_force_measured": False,
            "initial_bottom_m": initial_bottom_m, "final_bottom_m": final_bottom_m,
            "support_top_m": support_top_m, "late_speed_m_s": late_speed_m_s,
            "separation_tolerance_m": separation_tolerance_m,
            "minimum_drop_m": minimum_drop_m, "footprint_inside": bool(footprint_inside),
            "footprint_scope": "Body reference point inside declared support XY; not full footprint coverage"}


def plan_scene_physics(*, objects, interactions=None, preferred_backend="auto"):
    """A plan requests capabilities; backend selection does not verify an adapter."""
    if preferred_backend not in {"auto", "physx", "newton"}:
        raise ValueError("Choose auto, physx or newton")
    if not isinstance(objects, list) or not objects or len(objects) > 200:
        raise ValueError("Describe the physical participants in this task")
    participants = {}
    for item in objects:
        if not isinstance(item, dict) or set(item) != {"id", "kind"}:
            raise ValueError("Each physical participant needs id and kind")
        if not isinstance(item["id"], str) or not item["id"].strip() or item["id"] in participants or item["kind"] not in KINDS:
            raise ValueError("Physical participants need unique IDs and supported kinds")
        participants[item["id"]] = item["kind"]
    if interactions is None:
        interactions = []
        interaction_scope = "All listed physical participants share one task backend; unspecified contacts remain possible."
    else:
        if not isinstance(interactions, list) or any(not isinstance(pair, list) or len(pair) != 2
                or pair[0] == pair[1] or any(name not in participants for name in pair) for pair in interactions):
            raise ValueError("Interactions must name pairs of physical participants")
        interaction_scope = "Listed interactions describe expected contacts; one backend still covers the whole physical task."
    # Cloth and volumetric soft bodies share the PhysX default with rigid assets.
    # Fluid/strand tasks retain their existing default; an explicit choice wins.
    default_newton = any(kind in {"fluid", "strand"} for kind in participants.values())
    backend = ("newton" if default_newton else "physx") if preferred_backend == "auto" else preferred_backend
    capabilities = set(participants.values()) | {backend, "static_environment_collision"}
    rigid = any(kind in {"rigid", "articulated", "hybrid", "fracture"} for kind in participants.values())
    soft = any(kind in {"deformable", "hybrid", "strand", "solid"} for kind in participants.values())
    fluid = "fluid" in participants.values()
    if rigid and soft:
        capabilities.add("rigid_deformable_contact")
    if rigid and fluid:
        capabilities.add("rigid_fluid_contact")
    if soft and fluid:
        capabilities.add("deformable_fluid_contact")
    return {"schema_version": 1, "backend": backend, "participants": objects, "interactions": interactions,
            "solver_hint": ("physx" if backend == "physx" else "mpm_with_deformable_coupling" if fluid and soft else "mpm" if fluid else "vbd"),
            "required_capabilities": sorted(capabilities), "cross_engine_coupling": False,
            "scope": interaction_scope, "simulation_verified": False,
            "solver_selection": "The adapter must verify the selected backend and all required contact capabilities. A backend label or available constructor alone is insufficient; missing capabilities require an adapter, not a silent backend change.",
            "asset_policy": "Preserve editable geometry, topology, collision descriptions, parameters and bindings. Materialize or convert each participant for the selected task backend; a prior per-object backend is not a permanent assignment."}
