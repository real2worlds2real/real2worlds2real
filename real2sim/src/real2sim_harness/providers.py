from __future__ import annotations

from pathlib import Path
from typing import Any


def openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the API extra: python -m pip install -e '.[api]'") from exc
    return OpenAI()

def chatgpt_plan_client(auth_dir: Path) -> Any:
    from .chatgpt_plan import ChatGPTPlanClient
    return ChatGPTPlanClient(auth_dir)
