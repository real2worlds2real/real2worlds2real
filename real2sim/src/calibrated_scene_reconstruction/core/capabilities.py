"""Declarative, lazily resolved optional capability contributions.

Descriptors live beside this module. Removing a descriptor removes its entries;
resolving an unavailable declaration always fails explicitly. Discovery does not
import providers or initialize simulation runtimes.
"""
from functools import lru_cache
from importlib import import_module
import json
from pathlib import Path


@lru_cache(maxsize=1)
def catalog():
    result = {}
    for path in source_files():
        for category, entries in json.loads(path.read_text()).items():
            target = result.setdefault(category, {})
            overlap = target.keys() & entries.keys()
            if overlap:
                raise ValueError('Duplicate capability registrations: ' + str(sorted(overlap)))
            target.update(entries)
    return result


def entries(category):
    return dict(catalog().get(category, {}))


def require(category, name):
    registered = entries(category)
    if name not in registered:
        raise ValueError('Unsupported ' + category.replace('_', ' ') + ': ' + str(name))
    return registered[name]


def invoke(category, name, *args, **kwargs):
    reference = require(category, name)
    module, function = reference.split(':', 1)
    return getattr(import_module(module), function)(*args, **kwargs)


def source_files():
    return tuple(sorted(Path(__file__).with_name('capability_units').glob('*.json')))
