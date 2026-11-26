"""Load pinned control-plane revisions without modifying a frozen worker checkout.

This is a trusted-code bootstrap, not a security sandbox. Run it in a fresh
Python process. Worker commands, environments and scene bindings remain owned
by the registered backend; only the three named orchestration modules change.
"""
import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import time


MODULES = ("continuation", "preflight", "agent")
PREFIX = "astra_e2e.harness."
# first-party packages and where a checkout keeps them (the same partition as astra_e2e.layout, which cannot be
# imported here: this bootstrap must run before any first-party import)
PACKAGES = ("calibrated_scene_reconstruction", "real2sim_harness", "astra_station", "astra_e2e")
# Previously frozen checkouts can contain these packages under their original names.
FROZEN_PACKAGE_ALIASES = ("r2s2r", "ucb_remo")
SOURCE_DIRS = ("real2sim/src", "policy_development/src", "end_to_end/src")


def first_party(name):
    return name.partition(".")[0] in PACKAGES + FROZEN_PACKAGE_ALIASES


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def inside(path, root):
    """`root` is one directory or a sequence of them (the source roots of a checkout)."""
    for one in ([root] if isinstance(root, Path) else root):
        try:
            path.relative_to(one)
            return True
        except ValueError:
            pass
    return False


class PinnedLoader(importlib.abc.SourceLoader):
    """Execute the exact verified bytes, never a coincidentally fresh pyc."""

    def __init__(self, path, data):
        self.path, self.data = path, data

    def get_filename(self, fullname):
        return str(self.path)

    def get_data(self, path):
        if Path(path) != self.path:
            raise ImportError("Pinned loader cannot load another file")
        return self.data

    def get_code(self, fullname):
        return self.source_to_code(self.data, str(self.path))


class FrozenImports(importlib.abc.MetaPathFinder):
    """Reject mixed first-party checkouts, including imports deferred until execution."""

    def __init__(self, source, replacements):
        self.source, self.replacements = source, replacements

    def find_spec(self, fullname, path=None, target=None):
        if not first_party(fullname):
            return None
        if fullname in self.replacements:
            file, data = self.replacements[fullname]
            return importlib.util.spec_from_file_location(
                fullname, file, loader=PinnedLoader(file, data))
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            raise ModuleNotFoundError("Frozen checkout has no module " + fullname, name=fullname)
        if not spec.origin or not inside(Path(spec.origin).resolve(), self.source):
            raise ImportError("Package import escaped frozen checkout: " + fullname)
        if any(not inside(Path(p).resolve(), self.source)
               for p in spec.submodule_search_locations or []):
            raise ImportError("Package search path escaped frozen checkout: " + fullname)
        return spec


def inputs(frozen_repository, control_plane, config_path, report_path):
    frozen = Path(frozen_repository).resolve(strict=True)
    source = tuple(frozen / d for d in SOURCE_DIRS if (frozen / d).is_dir())
    control = Path(control_plane).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    report = Path(report_path).resolve()
    if not (frozen / "end_to_end/src/astra_e2e/harness/__init__.py").is_file():
        raise ValueError("Frozen repository has no astra_e2e harness package")
    if inside(control, frozen) or inside(frozen, control):
        raise ValueError("Control-plane directory must be separate from the frozen repository")
    if inside(report, frozen) or report.exists():
        raise ValueError("Audit report must be new and outside the frozen repository")
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)
    files = {Path(__file__).resolve(): Path(__file__).read_bytes()}
    replacements = {}
    for name in MODULES:
        file = control / (name + ".py")
        if file.is_symlink() or file.resolve(strict=True).parent != control:
            raise ValueError("Control-plane modules must be ordinary files in their directory")
        files[file] = file.read_bytes()
        replacements[PREFIX + name] = (file, files[file])
    if report in files or report == config_file:
        raise ValueError("Audit report cannot replace an input")
    expected = {str(path): sha256(data) for path, data in files.items()}
    if config.get("control_plane_inputs") != expected:
        raise ValueError("control_plane_inputs must exactly pin the bootstrap and three module files")
    if inside(Path(__file__).resolve(), frozen):
        raise ValueError("Bootstrap must also live outside the frozen repository")
    return frozen, source, control, config_file, config_bytes, report, expected, replacements


def import_audit(source, replacements):
    """Record actual loaded files, including third-party dependency origins."""
    loaded = {}
    hash_cache = {}
    for name, module in sorted(list(sys.modules.items())):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if first_party(name):
            if name in replacements:
                if path != replacements[name][0]:
                    raise ImportError("Control-plane module resolved to an unexpected file: " + name)
                role = "control_plane"
            elif inside(path, source):
                role = "frozen_repository"
            else:
                raise ImportError("Loaded package escaped the frozen checkout: " + name)
        else:
            role = "python_dependency"
        if path.is_file():
            if path not in hash_cache:
                hash_cache[path] = sha256(path.read_bytes())
            loaded[name] = {"path": str(path), "sha256": hash_cache[path], "role": role}
    return loaded


def run(*, frozen_repository, control_plane, mode, config, report, timeout_seconds=None):
    if mode not in {"agent", "preflight"}:
        raise ValueError("Mode must be agent or preflight")
    if mode != "preflight" and timeout_seconds is not None:
        raise ValueError("Timeout applies only to preflight")
    if any(first_party(name) for name in sys.modules):
        raise ValueError("Control-plane bootstrap requires a fresh Python process")
    (frozen, source, control, config_file, config_bytes, report_file,
     pins, replacements) = inputs(frozen_repository, control_plane, config, report)
    finder = FrozenImports(source, replacements)
    original_path = list(sys.path)
    original_bytecode = sys.dont_write_bytecode
    sys.path[:0] = [str(root) for root in source]
    sys.meta_path.insert(0, finder)
    sys.dont_write_bytecode = True
    audit = {"schema": "astra_control_plane_bootstrap_v1", "mode": mode,
             "frozen_repository": str(frozen), "control_plane": str(control),
             "configuration_sha256": sha256(config_bytes), "inputs": pins,
             "scope": "Control-plane import provenance only; no scene or policy acceptance.",
             "worker_environment_modified": False, "started_at": time.time()}
    error = None
    result = None
    try:
        for name in MODULES:
            importlib.import_module(PREFIX + name)
        audit["loaded_before_entrypoint"] = import_audit(source, replacements)
        print(json.dumps({"control_plane_loaded": pins, "mode": mode,
                          "audit_report": str(report_file)}), file=sys.stderr)
        argv = ["--config", str(config_file)]
        if timeout_seconds is not None:
            argv += ["--timeout-seconds", str(timeout_seconds)]
        result = sys.modules[PREFIX + mode].main(argv)
        audit["entrypoint_exit_code"] = result
    except BaseException as exc:
        error = exc
        audit["error_type"] = type(exc).__name__
        audit["error"] = str(exc)
    finally:
        try:
            audit["loaded_after_entrypoint"] = import_audit(source, replacements)
            changed = [path for path, expected in pins.items()
                       if not Path(path).is_file() or sha256(Path(path).read_bytes()) != expected]
            if config_file.read_bytes() != config_bytes:
                changed.append(str(config_file))
            audit["inputs_unchanged"] = not changed
            if changed:
                raise ValueError("Control-plane input changed during execution: " + ", ".join(changed))
        except BaseException as exc:
            audit["integrity_error_type"] = type(exc).__name__
            audit["integrity_error"] = str(exc)
            error = exc
        finally:
            sys.meta_path.remove(finder)
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_bytecode
            audit["finished_at"] = time.time()
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with report_file.open("x", encoding="utf-8") as stream:
                json.dump(audit, stream, indent=2, sort_keys=True)
                stream.write("\n")
    if error is not None:
        raise error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-repository", "--frozen-repo", required=True)
    parser.add_argument("--control-plane", required=True)
    parser.add_argument("--mode", choices=("agent", "preflight"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timeout-seconds", type=int)
    return run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
