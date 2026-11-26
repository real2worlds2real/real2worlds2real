"""Streaming Responses adapter for explicitly authorized ChatGPT plan usage.

Uses public api.openai.com endpoints only. Never falls back to API-key billing.
"""

from __future__ import annotations

import re
import time

from .chatgpt_auth import RESOURCE, ChatGPTConnection, CredentialStore, LoginError


class ChatGPTPlanClient:
    auth_mode = "chatgpt_plan"

    def __init__(self, directory, *, connection=None, http_client=None, max_response_seconds=180):
        self.connection = connection or ChatGPTConnection(CredentialStore(directory))
        self.http_client = http_client
        self.max_response_seconds = max_response_seconds
        self.responses = self

    def create(self, **kwargs):
        from openai import APIError, OpenAI

        # Deliberate request contract, not a general proxy to arbitrary endpoints.
        allowed = {
            "model",
            "instructions",
            "input",
            "tools",
            "parallel_tool_calls",
            "store",
            "reasoning",
        }
        if set(kwargs) - allowed:
            raise ValueError("Unsupported ChatGPT-plan request parameter")
        if not isinstance(kwargs.get("input"), list):
            raise ValueError("ChatGPT-plan requests require complete history as an input array")
        request = dict(kwargs)
        functions = request.pop("tools", [])
        if functions:
            if any(item.get("type") != "function" for item in functions):
                raise ValueError("Only registered function tools are supported by this adapter")
            request["tools"] = [
                {
                    "type": "namespace",
                    "name": "remo",
                    "description": "real2sim asset generation and review tools.",
                    "tools": functions,
                }
            ]
        request.update(store=False, stream=True)
        # A token is supplied internally as a bearer credential, never in context/logs.
        client = OpenAI(
            api_key=self.connection.access_token(),
            base_url=RESOURCE,
            max_retries=0,
            timeout=self.max_response_seconds,
            http_client=self.http_client,
        )
        completed = None
        output_items = {}
        started = time.monotonic()
        try:
            with client.responses.create(**request) as events:
                for event in events:
                    if time.monotonic() - started > self.max_response_seconds:
                        raise LoginError("ChatGPT-plan response exceeded its time budget")
                    if event.type in {"response.failed", "response.incomplete", "error"}:
                        error = getattr(getattr(event, "response", None), "error", None)
                        code = getattr(error, "code", "unknown_error")
                        safe = (
                            code
                            if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", code)
                            else "unknown_error"
                        )
                        raise LoginError(
                            f"ChatGPT-plan stream did not complete: {event.type} ({safe})"
                        )
                    if event.type == "response.completed":
                        completed = event.response
                    elif event.type == "response.output_item.done":
                        output_items[event.output_index] = event.item
            if completed is None or completed.status != "completed":
                raise LoginError("ChatGPT-plan stream ended without response.completed")
            # Plan streams can leave terminal response.output empty; completed
            # items arrive separately. Never turn those calls into an empty reply.
            if not completed.output and output_items:
                if sorted(output_items) != list(range(len(output_items))):
                    raise LoginError("ChatGPT-plan stream has missing output items")
                completed = completed.model_copy(
                    update={"output": [output_items[i] for i in sorted(output_items)]}
                )
            if not completed.output:
                raise LoginError("ChatGPT-plan completed without any output items")
            return completed
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            raise LoginError(
                f"ChatGPT-plan request failed ({type(exc).__name__}, HTTP {status}); no billing fallback"
            ) from None
        finally:
            # The caller owns injected transports (used by offline SDK tests).
            if self.http_client is None:
                client.close()

    def close(self):
        self.connection.close()
