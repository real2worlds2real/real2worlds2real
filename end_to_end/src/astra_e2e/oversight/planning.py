"""Task-specific module obligations and the outer feedback graph."""
import copy

from .contracts import MODULES, intake


DEPENDENCIES = {
    "research": [], "capture": ["research"], "model": ["capture"],
    "physics": ["model"], "environment": ["model"],
    "twin": ["physics", "environment"], "data": ["twin"], "train": ["data"],
    "validation": ["train"], "holdout": ["validation"], "delivery": ["holdout"],
}
REQUIRED_REVIEWS = {
    "research": ["capability_assessment", "transfer_hypothesis"],
    "capture": ["source_identity", "legal_metric_scale", "camera_registration"],
    "model": ["source_geometry", "semantic_structure", "collision_fidelity"],
    "physics": ["native_task", "material_uncertainty", "no_hidden_assistance"],
    "environment": ["task_preservation", "visual_coherence", "deployment_coverage"],
    "twin": ["native_gates", "independent_freeze"],
    "data": ["label_alignment", "split_integrity", "coverage", "deployment_observations"],
    "train": ["io_parity", "training_provenance", "checkpoint_selection"],
    "validation": ["closed_loop", "failure_analysis", "deployment_gap"],
    "holdout": ["sealed_protocol", "closed_loop", "deployment_gap"],
    "delivery": ["reproducible_package", "controller_parity", "human_handoff", "no_real_success_claim"],
}
ROUTES = {"source": "capture", "geometry": "model", "physics": "physics",
          "appearance": "environment", "coverage": "data", "labels": "data",
          "policy": "train", "latency": "train", "novel": "research"}
REFERENCES = {
    "research": ["end_to_end/skills/research-transfer/SKILL.md"],
    "capture": ["real2sim/skills/twin-capture/SKILL.md"],
    "model": ["real2sim/skills/twin-model/SKILL.md"],
    "physics": ["real2sim/docs/physics/README.md", "real2sim/skills/twin-deliver/SKILL.md"],
    "environment": ["real2sim/docs/scene/blender_environment.md", "real2sim/skills/twin-deliver/SKILL.md"],
    "twin": ["real2sim/skills/twin-deliver/SKILL.md"],
    "data": ["end_to_end/skills/data-policy-loop/SKILL.md", "real2sim/docs/datagen/episodes.md"],
    "train": ["end_to_end/skills/data-policy-loop/SKILL.md", "policy_development/docs/station_harness.md"],
    "validation": ["end_to_end/skills/data-policy-loop/SKILL.md"],
    "holdout": ["end_to_end/skills/data-policy-loop/SKILL.md"],
    "delivery": ["end_to_end/docs/transfer.md"],
}


def review(name):
    return {"id": name, "kind": "review", "description": name.replace("_", " "),
            "operator": "eq", "target": True, "unit": "boolean"}


def metric(name, target, operator, unit):
    return {"id": name, "kind": "metric", "description": name.replace("_", " "),
            "operator": operator, "target": target, "unit": unit}


def build_plan(spec):
    assessed = intake(spec)
    if not assessed["ready"]:
        raise ValueError("Intake is incomplete: " + "; ".join(assessed["questions"]))
    requirements = {m: [review(k) for k in REQUIRED_REVIEWS[m]] for m in MODULES}
    simulation_verification = spec.get("validation_scope", "deployment") == "simulation_verification"
    if simulation_verification:
        requirements["data"] = [review("declared_observation_scope")
                                if check["id"] == "deployment_observations" else check
                                for check in requirements["data"]]
    fidelity = spec["fidelity"]
    requirements["model"] += [metric("geometry_error_m", fidelity["geometry_tolerance_m"], "le", "m")]
    requirements["model"] += [review("critical_feature:" + name) for name in fidelity["critical_features"]]
    requirements["physics"] += [metric("contact_error_m", fidelity["contact_tolerance_m"], "le", "m")]
    if fidelity["concave_contact"]:
        requirements["model"] += [review("concave_surface_and_collider")]
        requirements["physics"] += [review("native_concave_contact")]
    if fidelity["flexible_response"] == "quantitative":
        requirements["physics"] += [review("quantitative_material_response")]
    if spec["novelty"]:
        requirements["research"] += [review("novel_capability_validated")]
    requirements["data"] += [metric("usable_episodes", spec["data"]["minimum_episodes"], "ge", "episodes")]
    requirements["data"] += [review("coverage:" + name) for name in spec["data"]["coverage"]]
    requirements["environment"] += [metric("qualified_cousins", spec["variation"]["cousin_count"], "ge", "scenes")]
    for module in ("validation", "holdout"):
        requirements[module] += [metric("latency_ms", spec["policy"]["max_latency_ms"], "le", "ms")]
        requirements[module] += [review("deployment:" + name) for name in spec["evaluation"]["deployment_checks"]]
    for module, extra in spec["requirements"].items():
        existing = {item["id"] for item in requirements[module]}
        if existing.intersection(c["id"] for c in extra):
            raise ValueError(f"Custom requirements cannot replace mandatory checks: {module}")
        requirements[module] += copy.deepcopy(extra)
    return {"version": 1, "name": spec["name"], "objective": spec["objective"],
            "completion_claim": (
                "simulation_pipeline_verified; deployment I/O qualification and real validation pending"
                if simulation_verification else
                "simulation_qualified_deployment_candidate; real validation pending"),
            "modules": {m: {"depends_on": DEPENDENCIES[m], "checks": requirements[m],
                             "instructions": REFERENCES[m]} for m in MODULES},
            # Preserve the entire decision context instead of reducing task requirements to flags.
            "contract": copy.deepcopy(spec)}


def descendants(module):
    affected = {module}
    while True:
        updated = affected | {m for m, deps in DEPENDENCIES.items() if affected.intersection(deps)}
        if updated == affected:
            return affected
        affected = updated


def feedback_targets(failures):
    return sorted({ROUTES[f["category"]] for f in failures})
