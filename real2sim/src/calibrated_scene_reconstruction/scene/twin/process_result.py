"""Consume the optional process result contract without executing physics/tasks."""
import hashlib
import json
from pathlib import Path


def state(session, task):
    receipt = session / 'qa/physics_process.json'
    required = (task.get('physics_process_required') or (session / 'qa/physics_process_required.json').exists()
                or receipt.exists() or any((session / 'physics_process').glob('round_*')))
    if not required:
        return {'applicable': False, 'passed': True, 'reason': 'No physical process requested'}
    try:
        value = json.loads(receipt.read_text())
        if value.get('version') != 1 or value.get('pass') is not True or not value.get('files'):
            raise ValueError('Accepted process evidence is required')
        for path, expected in value['files'].items():
            h = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError('Process input, model, review or execution evidence changed: ' + path)
        return {'applicable': True, 'passed': True, 'reason': 'Current accepted native process evidence'}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'applicable': True, 'passed': False, 'reason': str(exc)}
