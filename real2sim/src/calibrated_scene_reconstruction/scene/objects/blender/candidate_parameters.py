"""Use the actual build's complete parameters in candidate task diagnostics.

Fit records contain only optimized variables; build records also carry fixed
model choices such as sampling resolution and part count. This matches normal export
and prevents a task check from reading incomplete or unrelated parameters.
"""
from copy import deepcopy


def complete(report, fitted=None):
    actual = report.get('params')
    if not isinstance(actual, dict) or not actual:
        raise ValueError('Candidate needs the actual build parameter record')
    if fitted is not None:
        if not isinstance(fitted, dict):
            raise ValueError('Fitted parameters must be a mapping')
        for name, value in fitted.items():
            if name not in actual or actual[name] != value:
                raise ValueError('Fitted parameter differs from actual candidate build: ' + str(name))
    return deepcopy(actual)
