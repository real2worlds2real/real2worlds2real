"""Isolated stdio Codex transport using explicit per-user authorization."""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

PROVIDER = "real2sim"
TOKEN_ENV = "REAL2SIM_MODEL_TOKEN"
NATIVE_ITEMS = {"commandExecution", "fileChange", "imageView"}


class AppServerTransport:
    """One local stdio process; requests are never retried or physically replayed."""

    def __init__(self, argv, workspace, env, *, secret):
        self.pending = deque()
        self.incoming = queue.Queue()
        self.serial = 0
        self.secret = secret
        self.stderr = (workspace.parent / "app-server.stderr.log").open("a")
        self.process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self.readers:
            reader.start()

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("Expected an object")
                    self.incoming.put(message)
                except ValueError:
                    self.incoming.put(RuntimeError("Invalid app-server JSON message"))
        finally:
            self.incoming.put(EOFError("Codex app-server closed stdout"))

    def _read_stderr(self):
        for line in self.process.stderr:
            self.stderr.write(line.replace(self.secret, "[REDACTED]"))
            self.stderr.flush()

    def send(self, message):
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _receive(self, timeout):
        try:
            event = self.incoming.get(timeout=max(0.001, timeout))
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for Codex app-server") from exc
        if isinstance(event, BaseException):
            raise event
        return event

    def request(self, method, params, timeout=30):
        self.serial += 1
        identifier = self.serial
        self.send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            event = self._receive(deadline - time.monotonic())
            if event.get("id") == identifier and "method" not in event:
                if "error" in event:
                    path = Path(self.stderr.name).with_name("rpc-errors.jsonl")
                    with path.open("a") as stream:
                        stream.write(
                            json.dumps({"method": method, "error": event["error"]}).replace(
                                self.secret, "[REDACTED]"
                            )
                            + "\n"
                        )
                    # Do not expose arbitrary server/provider error bodies or credentials.
                    code = event["error"].get("code", "unknown")
                    raise RuntimeError(f"Codex RPC {method} rejected (code={code})")
                return event.get("result", {})
            self.pending.append(event)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Codex RPC {method}")

    def next_event(self, timeout=30):
        return self.pending.popleft() if self.pending else self._receive(timeout)

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr, self.stderr):
            stream.close()


def launch_config(workspace, *, auth_mode, compact_at, codex="codex"):
    """Isolated config: no inherited personal account, MCP servers, or project rules."""
    from .chatgpt_auth import RESOURCE

    config = {
        "model_provider": PROVIDER,
        f"model_providers.{PROVIDER}.name": "real2sim explicit authorization",
        f"model_providers.{PROVIDER}.base_url": RESOURCE,
        f"model_providers.{PROVIDER}.env_key": TOKEN_ENV,
        f"model_providers.{PROVIDER}.wire_api": "responses",
        f"model_providers.{PROVIDER}.requires_openai_auth": False,
        f"model_providers.{PROVIDER}.supports_websockets": False,
        "project_root_markers": [".real2sim-agent-root"],
        "project_doc_max_bytes": 0,
        "default_permissions": PROVIDER,
        f"permissions.{PROVIDER}.extends": ":read-only",
        f"permissions.{PROVIDER}.filesystem": {
            ":root": "deny",
            ":minimal": "read",
            # Start without implicit host temporary-directory writes. Explicit
            # nested grants also work when this task itself lives below /tmp.
            ":workspace_roots": {
                ".": "write",
                ".git": "read",
                ".agents": "read",
                ".codex": "read",
                ".aws": "read",
            },
            str(Path(sys.prefix).resolve()): "read",
            str(Path(shutil.which(codex) or codex).resolve().parent): "read",
            str(Path(sys.executable).resolve().parent.parent): "read",
        },
        f"permissions.{PROVIDER}.network.enabled": False,
        "shell_environment_policy.inherit": "none",
        "shell_environment_policy.set": {
            "PATH": str(Path(sys.executable).resolve().parent) + os.pathsep + os.defpath,
            "PYTHONPATH": str(
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            ),
        },
        "features.multi_agent": False,
        "features.shell_snapshot": False,
        "features.skill_mcp_dependency_install": False,
        "web_search": "disabled",
    }

    if compact_at is not None:
        config["model_auto_compact_token_limit"] = compact_at
    return config


def toml_literal(value):
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(json.dumps(key) + "=" + toml_literal(item) for key, item in value.items())
            + "}"
        )
    return json.dumps(value)


def server_launch(
    workspace,
    home,
    *,
    auth_mode,
    auth_dir="",
    compact_at=None,
    codex="codex",
    credential=None,
):
    """Shared explicit authorization and process setup for both agent roles."""
    if auth_mode not in {"chatgpt_plan", "api_key"}:
        raise ValueError("Choose explicit ChatGPT-plan or API-key authorization")
    config = launch_config(workspace, auth_mode=auth_mode, compact_at=compact_at, codex=codex)
    if auth_mode == "api_key" and os.environ.get("OPENAI_BASE_URL"):
        config[f"model_providers.{PROVIDER}.base_url"] = os.environ["OPENAI_BASE_URL"]
    argv = [codex, "app-server", "--stdio", "--strict-config"]
    for key, value in config.items():
        argv.extend(["-c", key + "=" + toml_literal(value)])
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")
        if key in os.environ
    }
    env["CODEX_HOME"] = str(home.resolve())
    if credential is None:
        if auth_mode == "chatgpt_plan":
            from .chatgpt_auth import ChatGPTConnection, CredentialStore

            connection = ChatGPTConnection(CredentialStore(auth_dir))
            try:
                credential = connection.access_token()
            finally:
                connection.close()
        else:
            credential = os.environ.get("OPENAI_API_KEY")
    if not credential:
        raise ValueError("Explicit authorization is missing")
    env[TOKEN_ENV] = credential
    return argv, env, config, credential
