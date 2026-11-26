"""Retrieve methods across both reconstruction libraries and the policy station."""
from pathlib import Path

from astra_station.common import digest, file_hash, read
from .harness.service import Harness
from .layout import REPO


def _skill_text(entry, validations=()):
    name = entry["id"]
    def lines(value):
        if isinstance(value, list):
            return "\n".join("- " + str(item) for item in value)
        return str(value)
    description = " ".join(str(entry["problem"]).splitlines())
    paragraphs = [f"---\nname: {name}\ndescription: {description}\n---", "# " + description]
    for title, key in (("Procedure", "procedure"), ("Applicability", "applicability"),
                       ("Limitations", "limitations"), ("Negative results", "negative_results")):
        if entry[key]:
            paragraphs.extend(["## " + title, lines(entry[key])])
    if entry.get("required_capabilities"):
        paragraphs.extend(["## Required operations", lines(entry["required_capabilities"])])
        pairs = sorted({tuple(pair) for v in validations for pair in v.get("bindings", [])})
        paragraphs.extend(["## Observed native bindings", lines([f"{op} via {native}" for op, native in pairs]),
                           "Each binding names a cited tool execution. Validate the method on the current "
                           "candidate and requested simulator before reusing its conclusions."])
        runtimes = sorted({tuple(row) for v in validations for row in v.get("runtimes", [])})
        if runtimes:
            paragraphs.extend(["## Observed runtimes", lines([f"{op} on {name} {version}"
                                                             for op, name, version in runtimes])])
    return "\n\n".join(paragraphs) + "\n"


def _write_skill(result):
    """Only independently validated experience becomes a discoverable skill revision."""
    if result["status"] != "transfer_validated":
        return
    path = Path(result["path"])
    row = read(path)
    entry = row["entry"]
    skill = path.with_suffix("") / entry["id"] / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    with skill.open("x", encoding="utf-8") as f:
        f.write(_skill_text(entry, row["validations"]))
    result["skill"] = str(skill)


def memory(chain, operation, request, state=None):
    from real2sim_harness.skills import SkillRegistry

    harness = Harness(state or chain.run / "memory")
    if operation == "learn":
        result = harness.dispatch("experience_learn", {"entry": request})
        _write_skill(result)
    else:
        context = request.get("context", {})
        query = request.get("query", chain.request["task"])
        arguments = {"context": context}
        if "policy_query" in request:
            arguments["policy_query"] = request["policy_query"]
        result = harness.dispatch("experience_recall", arguments)
        registry = SkillRegistry()
        result["skills"] = registry.search(query)
        module = request.get("module")
        if module is not None:
            from .model_modules import DOMAINS, METHODS
            if module not in DOMAINS:
                raise ValueError("Choose object or environment for skill retrieval")
            result["module_skill"] = registry.read("reconstruct-" + module)
            result["methods"] = {op: options for op, options in METHODS.items() if op.startswith(module + ".")}
        result["validated_skills"] = []
        for item in result["reconstruction"]:
            if item["status"] != "transfer_validated":
                continue
            scopes = [scope for scope in item["validated_scopes"] if scope["module"] == module]
            if module is not None and not scopes:
                continue
            skill = Path(item["path"]).with_suffix("") / item["entry"]["id"] / "SKILL.md"
            if skill.is_file() and skill.read_text(encoding="utf-8") == _skill_text(item["entry"], read(item["path"])["validations"]):
                result["validated_skills"].append({"path": str(skill), "sha256": file_hash(skill),
                                                   **({"validated_scopes": scopes} if module is not None else {})})
        if "skills_root" in request:
            result["personal_skills"] = SkillRegistry(Path(request["skills_root"])).search(query)
        if "evolution_root" in request:
            from calibrated_scene_reconstruction.agent.evolution import recall
            result["evolution"] = recall(request["evolution_root"], context)
        if "policy_tags" in request:
            from astra_station.memory import Library
            root = Path(request.get("policy_library", REPO / "policy_development/library"))
            result["station_cards"] = Library(root).retrieve(request["policy_tags"])
    chain.ledger.append("experience_" + operation, store=str(harness.root), request_sha256=digest(request),
                        result_sha256=digest(result))
    return result
