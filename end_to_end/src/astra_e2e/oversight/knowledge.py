"""Versioned procedural experience; never substitute old fits for new evidence."""
from itertools import combinations
from pathlib import Path
import re

from .controller import Controller, read, write_new
from .evidence import digest, fingerprint, unchanged


def validation(run, evidence=(), *, capabilities=None):
    if (Path(run) / "chain.json").is_file():
        from ..chain import Chain
        from ..reconstruction import Workflow
        chain = Chain(run)
        accepted = chain.done("real2sim")
        if chain.current() is not None or not accepted or "workflow_sha256" not in accepted:
            raise ValueError("A module-plan validation run must have current complete evidence")
        workflow = Workflow(chain)
        path, identity = workflow.delivery()
        progress = workflow.status()
        trace = progress["receipts"]
        used = {p for pin in trace for p in read(pin["path"])["files"]}
        if not used.intersection(evidence):
            raise ValueError("Cite current evidence used in each validating module plan")
        if identity != accepted["workflow_sha256"] or str(path) != accepted["delivery"]:
            raise ValueError("Chain delivery lineage changed")
        sources = sorted(set(chain.done("observations")["files"].values()))
        facts = {"run": str(chain.run), "source_sha256": digest(sources), "plan_sha256": identity}
        if capabilities is not None:
            facts["observations"] = sources
            cited = []
            for head in progress.get("models", {}).values():
                needed = {op for op in capabilities if op.startswith(head["candidate"]["module"] + ".")}
                uses = [use for use in head["uses"] if use["operation"] in needed
                        and use.get("check_passed", True) and set(use["outputs"]).intersection(evidence)]
                if needed <= {use["operation"] for use in uses}:
                    cited.extend(uses)
            facts["bindings"] = [list(pair) for pair in sorted({(u["operation"], u["native"]) for u in cited})]
            facts["runtimes"] = [list(row) for row in sorted({
                (u["operation"], u["runtime"]["name"], u["runtime"]["version"])
                for u in cited if "runtime" in u})]
            # Keep the method and runtime from the same native execution.
            executions = {(u["operation"], u["method"], u["native"],
                           u.get("runtime", {}).get("name", ""), u.get("runtime", {}).get("version", ""))
                          for u in cited}
            facts["executions"] = [list(execution) for execution in sorted(executions)]
        return facts
    controller = Controller(run)
    if not controller.status()["complete"]:
        raise ValueError("A knowledge validation run must have current complete evidence")
    _, state = controller._load()
    return {"run": str(controller.root), "source_sha256": digest(state["sources"]),
            "plan_sha256": state["plan_sha256"]}


def record(root, entry):
    required = {"id", "problem", "context", "procedure", "applicability", "limitations",
                "negative_results", "transfer_rationale", "evidence", "validation_runs"}
    if set(entry) - {"required_capabilities"} != required or not all(entry[k] for k in required - {"negative_results", "validation_runs"}):
        raise ValueError("Knowledge needs a scoped problem, procedure, applicability, limits and evidence")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", entry["id"]):
        raise ValueError("Invalid knowledge id")
    if not isinstance(entry["context"], dict) or not isinstance(entry["validation_runs"], list):
        raise ValueError("Context must be a mapping and validation_runs a list")
    files = fingerprint(entry["evidence"])
    capabilities = entry.get("required_capabilities")
    if "required_capabilities" in entry:
        from ..model_modules import METHODS
        if (not isinstance(capabilities, list) or not capabilities
                or any(not isinstance(name, str) or name not in METHODS for name in capabilities)
                or len(set(capabilities)) != len(capabilities)):
            raise ValueError("Choose distinct semantic operations from the capability catalog")
    validations = [validation(run, files, capabilities=capabilities) for run in entry["validation_runs"]]
    # General reuse needs transfer across distinct sources, not two copies of one experiment.
    generalized = len({v["source_sha256"] for v in validations}) >= 2
    if capabilities:
        # Multiple views of the same acquisition can otherwise manufacture transfer.
        qualified = [v for v in validations if set(capabilities) <= {op for op, _ in v.get("bindings", [])}]
        generalized = any(set(a["observations"]).isdisjoint(b["observations"])
                          for i, a in enumerate(qualified) for b in qualified[i + 1:])
    row = {"version": 1, "entry": entry, "files": files, "validations": validations,
           "status": "transfer_validated" if generalized else "candidate",
           "scope": "procedural_only; source geometry, dimensions and fits are not reusable evidence"}
    directory = Path(root) / entry["id"]
    directory.mkdir(parents=True, exist_ok=True)
    revision = len(list(directory.glob("*.json"))) + 1
    path = directory / f"{revision:04d}.json"
    write_new(path, row)
    return {"path": str(path), "status": row["status"]}


def _validated_scopes(validations, capabilities):
    required = set(capabilities)
    if not required:
        return []
    scopes = []
    qualified = [v for v in validations if v.get("observations")
                 and required <= {row[0] for row in v.get("executions", [])}]
    for a, b in combinations(qualified, 2):
        if not set(a["observations"]).isdisjoint(b["observations"]):
            continue
        common = set(map(tuple, a["executions"])) & set(map(tuple, b["executions"]))
        for module in sorted({op.partition(".")[0] for op in required}):
            operations = {op for op in required if op.startswith(module + ".")}
            bindings = [{"operation": op, "method": method, "native": native,
                         **({"runtime": {"name": name, "version": version}} if name else {})}
                        for op, method, native, name, version in sorted(common) if op in operations]
            if operations <= {binding["operation"] for binding in bindings}:
                scopes.append({"module": module, "bindings": bindings, "validation_runs": [a["run"], b["run"]]})
    return scopes


def lookup(root, context):
    """Exact context filters are deliberate; novel contexts are research candidates."""
    results = []
    for directory in sorted(Path(root).glob("*")):
        versions = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not versions:
            continue
        row = read(versions[-1])
        if not unchanged(row["files"]):
            continue
        entry = row["entry"]
        if not all(entry["context"].get(k) == v for k, v in context.items()):
            continue
        try:
            validations = [validation(v["run"], row["files"], capabilities=entry.get("required_capabilities"))
                           for v in row["validations"]]
            # Older revisions pin the same workflow and evidence but predate the
            # joint binding field. Recover it only from those current receipts.
            current = all({k: value for k, value in fresh.items() if k != "executions" or k in stored} == stored
                          for fresh, stored in zip(validations, row["validations"]))
        except (OSError, KeyError, ValueError, RuntimeError):
            current = False
        results.append({"path": str(versions[-1]), "entry": entry,
                        "status": row["status"] if current else "candidate",
                        "validated_bindings": sorted({tuple(binding) for v in row["validations"]
                                                      for binding in v.get("bindings", [])}) if current else [],
                        "validated_runtimes": sorted({tuple(runtime) for v in row["validations"]
                                                      for runtime in v.get("runtimes", [])}) if current else [],
                        "validated_scopes": (_validated_scopes(validations, entry.get("required_capabilities", []))
                                             if current and row["status"] == "transfer_validated" else []),
                        "reuse": "Check applicability and validate on the current source before adoption."})
    return results
