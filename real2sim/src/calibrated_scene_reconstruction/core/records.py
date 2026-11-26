"""JSON record writes and explicit contract assertions shared by domains and runtime."""
import json
import os
import uuid
from pathlib import Path

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.pending_' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)
