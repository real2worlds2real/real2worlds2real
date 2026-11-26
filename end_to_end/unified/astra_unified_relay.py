"""Relay bounded public decisions to a locally authenticated real Astra client.

The simulation worker retains source assets, execution authority, raw traces and
sealed evaluation data. SSH transfers only the unified runner's decision spool;
no authentication material is copied to the worker. This is an optional transport
for installations whose authenticated Codex client and GPU worker differ.
"""
import argparse
import base64
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

from astra_unified_decide import decide_request
from campaign_ledger import model_facts, strict_json
from astra_e2e.oversight.evidence import digest, file_hash


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if strict_json(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Immutable local relay record changed")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


class SSHDecisionRelay:
    def __init__(self, host, remote_state, local_state, *, max_model_calls=40, wall_seconds=14400,
                 transport_prefix=None):
        if (host is None) == (transport_prefix is None):
            raise ValueError("Select exactly one SSH host or pinned command transport")
        if host is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
            raise ValueError("Expected an SSH host or user@host, without command options")
        if transport_prefix is not None and (not isinstance(transport_prefix, list) or not transport_prefix
                or not all(isinstance(value, str) and value for value in transport_prefix)
                or transport_prefix[-1] != "--command"):
            raise ValueError("Command transport must be an argv list ending in --command")
        if (not PurePosixPath(remote_state).is_absolute() or ".." in PurePosixPath(remote_state).parts
                or not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_state)):
            raise ValueError("The worker state must be a concrete absolute POSIX directory")
        if not 1 <= max_model_calls <= 100 or not 1 <= wall_seconds <= 86400:
            raise ValueError("A finite relay call/time budget is required")
        self.host, self.remote = host, str(PurePosixPath(remote_state))
        self.transport_prefix = transport_prefix
        self.transport_inputs = {str(Path(value).resolve()): file_hash(value)
                                 for value in (transport_prefix or []) if Path(value).is_file()}
        self.root = Path(local_state).resolve()
        self.max_calls, self.wall_seconds = max_model_calls, wall_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        write_once(self.root / "config.json", {"host": host, "transport_prefix": transport_prefix,
            "transport_inputs": self.transport_inputs, "remote_state": self.remote,
            "max_model_calls": max_model_calls, "wall_seconds": wall_seconds,
            "relay_source_sha256": file_hash(__file__)})

    def _ssh(self, script, *args):
        if any(file_hash(path) != sha for path, sha in self.transport_inputs.items()):
            raise ValueError("Pinned command transport changed")
        command = "python3 -c " + shlex.quote(script) + " " + " ".join(shlex.quote(str(x)) for x in args)
        argv = self.transport_prefix + [command] if self.transport_prefix else [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, command]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError("SSH decision spool operation failed; inspect operator transport logs")
        return result.stdout

    def poll(self):
        script = """import json,sys
from pathlib import Path
root=Path(sys.argv[1]); spool=root/'decision_spool'
pending=[p.name for p in sorted(spool.glob('step-*.request.json')) if not p.with_name(p.name.replace('.request.','.response.')).exists()]
print(json.dumps({'pending':pending,'completed':(root/'report.json').exists()}))
"""
        return strict_json(self._ssh(script, self.remote))

    def process(self, name):
        if not re.fullmatch(r"step-\d{3}\.request\.json", name):
            raise ValueError("Invalid worker decision spool entry")
        local = self.root / "requests" / name
        local.parent.mkdir(exist_ok=True)
        source = self.remote + "/decision_spool/" + name
        request = strict_json(self._ssh("""import sys
from pathlib import Path
data=Path(sys.argv[1]).read_bytes()
if len(data)>65536: raise ValueError('Decision request too large')
sys.stdout.write(data.decode('utf-8'))
""", source))
        write_once(local, request)
        if request.get("operation") != "decide" or request["phase"] + ".request.json" != name:
            raise ValueError("Worker request identity mismatch")
        if type(request.get("max_model_calls")) is not int or not 1 <= request["max_model_calls"] <= self.max_calls:
            raise ValueError("Worker model budget exceeds relay authority")
        if len(list((self.root / "requests").glob("*.request.json"))) > self.max_calls:
            raise ValueError("Relay model request budget exhausted")
        model_facts(request["facts"])
        response_name = name.replace(".request.", ".response.")
        response_path = self.root / "responses" / response_name
        if not response_path.exists():
            envelope = {"schema": "astra_decision_relay_v1", "request_sha256": digest(request)}
            try:
                response = decide_request({**request, "ledger_root": str(self.root / "astra_ledger")})
                envelope["response"] = response
            except Exception as exc:
                # Keep detailed model/usage failures in the local campaign ledger.
                envelope["error"] = type(exc).__name__ + ": actual Astra decision failed; inspect local ledger"
            write_once(response_path, envelope)
        else:
            envelope = strict_json(response_path.read_text(encoding="utf-8"))
            if envelope["request_sha256"] != digest(request):
                raise ValueError("Local response belongs to another worker request")
        temporary = self.remote + "/decision_spool/." + response_name + ".upload"
        target = self.remote + "/decision_spool/" + response_name
        encoded = base64.b64encode(response_path.read_bytes()).decode("ascii")
        # Small chunks also work through command-only jump-host transports and
        # Windows command-line limits; only the public response is transferred.
        for index in range(0, len(encoded), 8000):
            self._ssh("""import base64,sys
from pathlib import Path
with Path(sys.argv[1]).open('wb' if sys.argv[2]=='0' else 'ab') as stream:
    stream.write(base64.b64decode(sys.argv[3],validate=True))
""", temporary, str(index), encoded[index:index+8000])
        self._ssh("""import os,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
if target.exists():
    if source.read_bytes()!=target.read_bytes(): raise ValueError('Published response changed')
else: os.link(source,target)
source.unlink()
""", temporary, target)
        return {"phase": request["phase"], "status": "failed" if "error" in envelope else "returned",
                "request_sha256": envelope["request_sha256"]}

    def run(self):
        deadline = time.monotonic() + self.wall_seconds
        while time.monotonic() < deadline:
            state = self.poll()
            for name in state["pending"]:
                print(json.dumps(self.process(name)), flush=True)
            if state["completed"]:
                return 0
            time.sleep(1)
        print(json.dumps({"status": "relay_wall_budget_exhausted"}), flush=True)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--host")
    transport.add_argument("--transport-prefix-json")
    parser.add_argument("--remote-state", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--max-model-calls", type=int, default=40)
    parser.add_argument("--wall-seconds", type=int, default=14400)
    args = parser.parse_args(argv)
    return SSHDecisionRelay(args.host, args.remote_state, args.state,
        max_model_calls=args.max_model_calls, wall_seconds=args.wall_seconds,
        transport_prefix=strict_json(args.transport_prefix_json) if args.transport_prefix_json else None).run()


if __name__ == "__main__":
    raise SystemExit(main())
