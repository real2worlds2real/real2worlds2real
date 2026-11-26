"""Project-owned Sign in with ChatGPT credentials; never read Codex login files.

Official flow: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
Only the optional auth commands import httpx / PyJWT. All diagnostics are redacted.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
JWKS = ISSUER + "/.well-known/jwks.json"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
APP_NAME = "real2sim"


class LoginError(RuntimeError):
    """Only put safe diagnostic text in this exception, never response bodies."""


def private_write(path, value):
    temporary = path.with_name("." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CredentialStore:
    def __init__(self, directory):
        self.directory = Path(directory).absolute()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise LoginError("Use a private, owner-only auth directory (mode 0700), not a symlink")

    def read(self, name):
        path = self.directory / name
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise LoginError("Credential files must be owner-only regular files (mode 0600)")
            return json.load(stream)

    @contextlib.contextmanager
    def lock(self):
        import fcntl

        path = self.directory / "session.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode):
                raise LoginError("Unsafe credential lock permissions")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def host_id(self):
        with self.lock():
            value = self.read("host.json")
            if value is None:
                value = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
                private_write(self.directory / "host.json", value)
            return value["ext_agent_host_id"]

    def status(self):
        value = self.read("credentials.json")
        return {
            "connected": value is not None,
            "plan_enabled": value is not None and DIRECT_SCOPE in value.get("scopes", []),
            "access_token_expired": value is not None and value.get("expires_at", 0) <= time.time(),
            "credential_directory": str(self.directory),
            "usage_settings": "https://chatgpt.com/#settings/Usage",
        }


def checked_json(response, operation):
    if not response.is_success:
        # Do not include HTTP bodies or request URLs containing authorization codes.
        raise LoginError(f"{operation} failed (HTTP {response.status_code}); no credentials logged")
    try:
        return response.json()
    except ValueError:
        raise LoginError(f"{operation} returned invalid JSON") from None


def verify_identity(http, token, client_id, *, nonce=None):
    import jwt

    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") not in {"RS256", "ES256"}:
            raise LoginError("Unsupported identity signature algorithm")
        keys = checked_json(http.get(JWKS), "Identity signing keys")
        matching = [item for item in keys["keys"] if item.get("kid") == header.get("kid")]
        if len(matching) != 1:
            raise LoginError("Identity signing key was not uniquely identified")
        key = jwt.PyJWK.from_dict(matching[0])
        if key.algorithm_name != header["alg"]:
            raise LoginError("Identity signature algorithm mismatch")
        required = ["iss", "sub", "aud", "exp", "iat"] + (["nonce"] if nonce else [])
        claims = jwt.decode(
            token,
            key.key,
            algorithms=[header["alg"]],
            issuer=ISSUER,
            audience=client_id,
            options={"require": required},
        )
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise LoginError("Identity subject missing")
        audience = claims["aud"]
        if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != client_id:
            raise LoginError("Identity authorized party mismatch")
        if nonce is not None and not hmac.compare_digest(str(claims["nonce"]), nonce):
            raise LoginError("Identity nonce mismatch")
        return claims
    except LoginError:
        raise
    except (ValueError, KeyError, TypeError, jwt.PyJWTError):
        raise LoginError("Identity signature or claims could not be verified") from None


def token_record(tokens, claims, client_id, host_id, *, previous=None):
    if tokens.get("token_type", "").lower() != "bearer" or not isinstance(
        tokens.get("access_token"), str
    ):
        raise LoginError("Token response missing a bearer credential")
    if tokens.get("client_id", client_id) != client_id:
        raise LoginError("Token registration mismatch")
    if previous and (previous["subject"] != claims["sub"] or previous["client_id"] != client_id):
        raise LoginError(
            "Sign-in identity changed; use a separate auth directory for another account"
        )
    expires = tokens.get("expires_in")
    if (
        isinstance(expires, bool)
        or not isinstance(expires, (int, float))
        or not 0 < expires <= 86400 * 30
    ):
        raise LoginError("Token response missing a valid expiration")
    scopes = tokens.get("scope")
    if scopes is None and previous:
        scopes = " ".join(previous["scopes"])
    if not isinstance(scopes, str):
        raise LoginError("Token response did not specify granted permissions")
    return {
        "issuer": ISSUER,
        "subject": claims["sub"],
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "email": claims.get("email"),
        "access_token": tokens["access_token"],
        "token_type": "Bearer",
        "id_token": tokens.get("id_token", (previous or {}).get("id_token")),
        "refresh_token": tokens.get("refresh_token", (previous or {}).get("refresh_token")),
        "scopes": scopes.split(),
        "saved_at": time.time(),
        "expires_at": time.time() + expires,
    }


class ChatGPTConnection:
    def __init__(self, store, http=None):
        import httpx

        self.store = store
        self.http = http or httpx.Client(timeout=30, follow_redirects=False)

    def access_token(self):
        with self.store.lock():
            record = self.store.read("credentials.json")
            if record is None:
                raise LoginError("Run chatgpt-login first")
            if DIRECT_SCOPE not in record["scopes"]:
                raise LoginError("ChatGPT plan usage was not authorized; enable it through sign-in")
            if record["expires_at"] < time.time() + 60:
                if not record.get("refresh_token"):
                    raise LoginError("Sign-in expired; sign in again")
                tokens = checked_json(
                    self.http.post(
                        TOKEN,
                        data={
                            "grant_type": "refresh_token",
                            "client_id": record["client_id"],
                            "refresh_token": record["refresh_token"],
                            "resource": RESOURCE,
                        },
                    ),
                    "Token refresh",
                )
                claims = {"sub": record["subject"], "email": record.get("email")}
                if tokens.get("id_token"):
                    claims = verify_identity(self.http, tokens["id_token"], record["client_id"])
                record = token_record(
                    tokens,
                    claims,
                    record["client_id"],
                    record["ext_agent_host_id"],
                    previous=record,
                )
                private_write(self.store.directory / "credentials.json", record)
                if DIRECT_SCOPE not in record["scopes"]:
                    raise LoginError("ChatGPT plan permission was removed")
            return record["access_token"]

    def models(self):
        value = checked_json(
            self.http.get(
                RESOURCE + "/models",
                headers={
                    "Authorization": "Bearer " + self.access_token(),
                },
            ),
            "Account model catalog",
        )
        if not isinstance(value.get("models"), list):
            raise LoginError("Unexpected ChatGPT account model catalog")
        return [
            {"slug": item["slug"], "display_name": item.get("display_name", item["slug"])}
            for item in value["models"]
            if item.get("visibility") == "list"
        ]

    def close(self):
        self.http.close()


class LoginFlow:
    def __init__(self, store, redirect_uri, http):
        parsed = urlsplit(redirect_uri)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.path != "/auth/callback"
            or not parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise LoginError("Use an HTTP 127.0.0.1 loopback /auth/callback URI")
        self.store, self.redirect_uri, self.http = store, redirect_uri, http
        self.host_id = store.host_id()
        self.previous = store.read("credentials.json")
        self.transaction = None

    def start(self):
        verifier = secrets.token_urlsafe(48)
        self.transaction = {
            "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32),
            "verifier": verifier,
            "expires": time.monotonic() + 600,
        }
        client_id = self.previous["client_id"] if self.previous else "dynamic_agent_client"
        parameters = {
            "client_id": client_id,
            "ext_agent_host_id": self.host_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": self.transaction["state"],
            "nonce": self.transaction["nonce"],
            "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode(),
            "code_challenge_method": "S256",
        }
        if self.previous is None:
            parameters["agent_name_hint"] = APP_NAME
        # Deliberately omit id_token_hint from URLs so no credential appears in a URL.
        return AUTHORIZE + "?" + urlencode(parameters)

    def complete(self, query):
        parameters = parse_qs(query, keep_blank_values=True)

        def one(key, default=None):
            values = parameters.get(key, [default])
            if len(values) != 1:
                raise LoginError("Duplicate authorization callback parameter")
            return values[0]

        transaction = self.transaction
        if (
            not transaction
            or time.monotonic() >= transaction["expires"]
            or not hmac.compare_digest(one("state", ""), transaction["state"])
        ):
            raise LoginError("Authorization state is missing, expired or mismatched")
        self.transaction = None  # single-use; even a declined grant cannot be replayed
        if one("error") is not None:
            raise LoginError("Sign-in was declined or failed; no code exchanged")
        code = one("code")
        client_id = one("client_id", self.previous["client_id"] if self.previous else None)
        if (
            not code
            or not isinstance(client_id, str)
            or not re.fullmatch(r"oaiapp_[A-Za-z0-9_-]+", client_id)
        ):
            raise LoginError(
                "Registration did not return an issued client ID and authorization code"
            )
        if self.previous and client_id != self.previous["client_id"]:
            raise LoginError("Authorization registration changed")
        tokens = checked_json(
            self.http.post(
                TOKEN,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "code_verifier": transaction["verifier"],
                    "redirect_uri": self.redirect_uri,
                    "resource": RESOURCE,
                },
            ),
            "Authorization exchange",
        )
        claims = verify_identity(
            self.http, tokens.get("id_token", ""), client_id, nonce=transaction["nonce"]
        )
        record = token_record(tokens, claims, client_id, self.host_id, previous=self.previous)
        with self.store.lock():
            # Do not overwrite credentials another concurrent login just installed.
            if self.store.read("credentials.json") != self.previous:
                raise LoginError("Another login changed this connection; start a fresh sign-in")
            private_write(self.store.directory / "credentials.json", record)
        return self.store.status()


def serve_login(directory, *, port=1455, timeout=1800, ready=print):
    """Serve only loopback. Remote users connect through an SSH local forward."""
    import httpx

    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("Login port must be 1024..65535")
    store = CredentialStore(directory)
    browser_session = secrets.token_urlsafe(32)
    result = None
    with httpx.Client(timeout=30, follow_redirects=False) as http:
        flow = LoginFlow(store, f"http://127.0.0.1:{port}/auth/callback", http)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Callback URLs contain authorization codes.

            def send_page(self, status, message, *, cookie=False):
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
                )
                if cookie:
                    self.send_header(
                        "Set-Cookie", f"ad_login={browser_session}; HttpOnly; SameSite=Lax; Path=/"
                    )
                self.end_headers()
                self.wfile.write(message.encode())

            def do_GET(self):
                nonlocal result
                from http.cookies import SimpleCookie

                if self.headers.get("Host") != f"127.0.0.1:{port}":
                    self.send_page(400, "Use the exact 127.0.0.1 login URL.")
                    return
                parsed = urlsplit(self.path)
                if parsed.path == "/":
                    self.send_page(
                        200,
                        '<h1>real2sim</h1><p>Connect your ChatGPT plan for development experiments.</p><p>Model requests use your plan allowance. No password is collected by this app.</p><a href="/start">Continue with ChatGPT</a>',
                        cookie=True,
                    )
                    return
                cookies = SimpleCookie()
                cookies.load(self.headers.get("Cookie", ""))
                given = cookies.get("ad_login")
                if not given or not hmac.compare_digest(given.value, browser_session):
                    self.send_page(400, "Open the login start page in this browser first.")
                    return
                if parsed.path == "/start":
                    self.send_response(302)
                    self.send_header("Location", flow.start())
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                elif parsed.path == "/auth/callback":
                    try:
                        result = flow.complete(parsed.query)
                    except Exception:
                        # Avoid leaking code/token response details into browser or logs.
                        self.send_page(
                            400, 'Sign-in could not be verified. <a href="/">Return and retry</a>.'
                        )
                        return
                    self.send_page(
                        200,
                        "Connected. ChatGPT plan usage is enabled. You may close this page."
                        if result["plan_enabled"]
                        else "Identity connected, but plan usage was not granted. No model requests will be made.",
                    )
                else:
                    self.send_page(404, "Not found")

        with HTTPServer(("127.0.0.1", port), Handler) as server:
            server.timeout = 1
            ready(
                {
                    "login_url": f"http://127.0.0.1:{port}/",
                    "port": port,
                    "expires_in_seconds": timeout,
                    "status": "waiting_for_browser",
                }
            )
            deadline = time.monotonic() + timeout
            while result is None and time.monotonic() < deadline:
                server.handle_request()
    if result is None:
        raise LoginError("Login timed out; run chatgpt-login again")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="real2sim's separate ChatGPT plan connection")
    parser.add_argument("action", choices=["login", "status", "models"])
    parser.add_argument("--auth-dir", required=True)
    parser.add_argument("--port", type=int, default=1455)
    args = parser.parse_args()
    if args.action == "login":
        result = serve_login(
            args.auth_dir, port=args.port, ready=lambda value: print(json.dumps(value), flush=True)
        )
    else:
        store = CredentialStore(args.auth_dir)
        if args.action == "status":
            result = store.status()
        else:
            connection = ChatGPTConnection(store)
            try:
                result = {"models": connection.models()}
            finally:
                connection.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
