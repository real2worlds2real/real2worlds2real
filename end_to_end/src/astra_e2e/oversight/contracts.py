"""Strict intake and work-packet contracts; no simulator imports."""
import math

from jsonschema import Draft202012Validator


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
COUNT = {"type": "integer", "minimum": 1}
BOOL = {"type": "boolean"}


def array(item, minimum=0):
    return {"type": "array", "items": item, "minItems": minimum}


def enum(*values):
    return {"enum": list(values)}


MODULES = ("research", "capture", "model", "physics", "environment", "twin",
           "data", "train", "validation", "holdout", "delivery")
CHECK = obj({"id": TEXT, "kind": enum("review", "metric"), "description": TEXT,
             "operator": enum("eq", "ge", "le"), "target": {"type": ["number", "boolean"]},
             "unit": TEXT})
CHANNEL = obj({"name": TEXT, "shape": array(COUNT, 1), "dtype": TEXT,
               "units": TEXT, "frame": TEXT, "available_on_robot": BOOL})
SPLIT = obj({"groups": array(TEXT, 1), "seeds": array({"type": "integer"}, 1)})
SUCCESS_PREDICATE = obj({"field": TEXT, "operator": enum("le", "ge", "abs_le", "eq"),
                         "target": {"type": ["number", "boolean"]}})

SPEC_SCHEMA = obj({
    "version": {"const": 1}, "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
    "validation_scope": enum("deployment", "simulation_verification"),
    "objective": TEXT,
    "source": obj({"files": array(TEXT, 1), "session": TEXT,
                   "calibration": TEXT, "independent_reconstruction": BOOL}),
    "task": obj({"description": TEXT, "success_definition": TEXT, "failure_modes": array(TEXT, 1),
                 "robot": TEXT, "deployment_environment": TEXT,
                 "success_predicates": array(SUCCESS_PREDICATE, 1)},
                required=["description", "success_definition", "failure_modes", "robot", "deployment_environment"]),
    "policy": obj({"family": enum("lerobot_dp", "official_dp", "openpi", "custom"),
                   "selection_reason": TEXT, "observations": array(CHANNEL, 1),
                   "actions": array(CHANNEL, 1), "control_hz": POSITIVE,
                   "max_latency_ms": POSITIVE, "action_alignment": TEXT,
                   "normalization": TEXT, "deployment_controller": TEXT}),
    "fidelity": obj({"profile": enum("task", "precision"), "basis": TEXT,
                     "critical_features": array(TEXT), "concave_contact": BOOL,
                     "geometry_tolerance_m": POSITIVE, "contact_tolerance_m": POSITIVE,
                     "flexible_response": enum("task_and_semantics", "quantitative"),
                     "uncertain_properties": array(TEXT)}),
    "variation": obj({"cousin_count": {"type": "integer", "minimum": 0},
                      "background": enum("zone", "mesh", "completion", "splat", "blender", "hybrid"),
                      "rationale": TEXT, "preserve": array(TEXT, 1),
                      "randomize": array(obj({"name": TEXT, "distribution": TEXT,
                                              "bounds": array(NUMBER, 2), "basis": TEXT}))}),
    "data": obj({"minimum_episodes": COUNT, "expert": TEXT,
                 "failure_policy": enum("success_only", "labelled_recovery"),
                 "coverage": array(TEXT, 1), "split_unit": enum("scene", "object", "source"),
                 "splits": obj({key: SPLIT for key in ("train", "validation", "holdout")})}),
    "training": obj({"recipe": TEXT, "initialization": TEXT,
                     "feedback_methods": array(enum("targeted_data", "dagger", "curriculum", "retrain", "research"), 1),
                     "dagger_expert": TEXT, "checkpoint_selection": TEXT}),
    "evaluation": obj({"success_rate_min": {"type": "number", "minimum": 0, "maximum": 1},
                       "minimum_trials": COUNT, "statistic": enum("rate", "wilson_lower_95"),
                       "strata": array(TEXT, 1), "deployment_checks": array(TEXT, 1)}),
    "budget": obj({"max_cycles": COUNT, "max_attempts_per_module": COUNT, "max_jobs": COUNT,
                   "wall_seconds": POSITIVE, "command_timeout_seconds": POSITIVE,
                   "gpu_ids": array(TEXT), "memory_gb": POSITIVE}),
    "authority": obj({"execute_local_jobs": BOOL, "network_research": BOOL,
                      "physical_robot": {"const": False}, "publish": {"const": False},
                      "allowed_workdirs": array(TEXT, 1)}),
    "novelty": array(obj({"topic": TEXT, "gap": TEXT, "validation": TEXT})),
    "requirements": obj({key: array(CHECK) for key in MODULES}),
    "unresolved": array(TEXT), "assumptions": array(TEXT),
})
# Existing contracts retain deployment semantics when no scope is supplied.
SPEC_SCHEMA["required"].remove("validation_scope")
SPEC_SCHEMA["allOf"] = [{
    "if": {"properties": {"validation_scope": {"const": "simulation_verification"}},
           "required": ["validation_scope"]},
    "else": {"properties": {"policy": {"properties": {
        key: {"items": {"properties": {"available_on_robot": {"const": True}}}}
        for key in ("observations", "actions")
    }}}},
}]

COMMAND_SCHEMA = obj({
    "argv": array(TEXT, 1), "cwd": TEXT,
    "env": {"type": "object", "additionalProperties": {"type": "string"}},
    "inputs": array(TEXT, 1), "timeout_seconds": POSITIVE,
})
PACKET_SCHEMA = obj({"module": enum(*MODULES), "purpose": TEXT,
                     "commands": array(COMMAND_SCHEMA, 1)})
REPORT_SCHEMA = obj({
    "module": enum(*MODULES), "attempt": COUNT, "plan_sha256": TEXT,
    "summary": TEXT, "artifacts": array(TEXT, 1),
    "checks": {"type": "object", "additionalProperties": obj({
        "value": {"type": ["boolean", "number"]}, "reason": TEXT, "evidence": array(TEXT, 1)})},
    "failures": array(obj({"category": enum("source", "geometry", "physics", "appearance", "coverage",
                                            "labels", "policy", "latency", "novel"),
                           "diagnosis": TEXT, "evidence": array(TEXT, 1)})),
    "native_session": {"type": ["string", "null"]},
    "policy_io": {"type": ["string", "null"]},
    "dataset_manifest": {"type": ["string", "null"]},
    "checkpoint": {"type": ["string", "null"]},
    "evaluation": {"type": ["object", "null"], "properties": {
        "split": enum("validation", "holdout"), "episodes": array(obj({
            "id": TEXT, "group": TEXT, "seed": {"type": "integer"}, "stratum": TEXT,
            "success": BOOL, "trace": TEXT}), 1)}, "required": ["split", "episodes"],
        "additionalProperties": False},
})


def validate(value, schema):
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError("; ".join(f"/{'/'.join(map(str, e.path))}: {e.message}" for e in errors))
    # JSON Schema considers nonfinite Python floats numbers; persisted contracts must not.
    def finite(x):
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("Nonfinite contract value")
        if isinstance(x, dict):
            for child in x.values():
                finite(child)
        if isinstance(x, list):
            for child in x:
                finite(child)
    finite(value)


def intake(spec):
    """Collect questions before execution. No defaults infer the user's intent."""
    issues = []
    try:
        validate(spec, SPEC_SCHEMA)
    except ValueError as exc:
        return {"ready": False, "questions": [str(exc)]}
    issues.extend(spec["unresolved"])
    predicates = spec["task"].get("success_predicates", [])
    if len({(p["field"], p["operator"]) for p in predicates}) != len(predicates):
        issues.append("Resolve duplicate success predicates.")
    for predicate in predicates:
        if predicate["operator"] != "eq" and type(predicate["target"]) is not int and type(predicate["target"]) is not float:
            issues.append("Ordered success predicates need numeric targets.")
    for names, label in ((spec["fidelity"]["critical_features"], "critical features"),
                         (spec["data"]["coverage"], "data coverage"),
                         (spec["evaluation"]["strata"], "evaluation strata"),
                         (spec["evaluation"]["deployment_checks"], "deployment checks")):
        if len(names) != len(set(names)):
            issues.append(f"Resolve duplicate {label}.")
    splits = spec["data"]["splits"]
    for key in ("groups", "seeds"):
        for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout")):
            if set(splits[left][key]) & set(splits[right][key]):
                issues.append(f"Choose disjoint {key} for {left} and {right}.")
    for name in ("observations", "actions"):
        names = [c["name"] for c in spec["policy"][name]]
        if len(names) != len(set(names)):
            issues.append(f"Resolve duplicate policy {name} names.")
    if spec["fidelity"]["profile"] == "precision" and not spec["fidelity"]["critical_features"]:
        issues.append("Name the critical physical features for precision reconstruction.")
    for item in spec["variation"]["randomize"]:
        if len(item["bounds"]) != 2 or item["bounds"][0] > item["bounds"][1]:
            issues.append(f"Specify ordered numeric bounds for {item['name']}.")
    for module, checks in spec["requirements"].items():
        if len({c["id"] for c in checks}) != len(checks):
            issues.append(f"Resolve duplicate requirements for {module}.")
        for check in checks:
            if check["kind"] == "review" and (check["operator"] != "eq" or check["target"] is not True):
                issues.append(f"Review {module}/{check['id']} must require true.")
            if check["kind"] == "metric" and type(check["target"]) not in (int, float):
                issues.append(f"Metric {module}/{check['id']} needs a numeric target.")
    return {"ready": not issues, "questions": issues}
