"""Exact Python/launcher readiness before model calls, never policy execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from .backends import load_backend, read, write_new
from ..oversight.evidence import digest, file_hash


PROBE = r'''
import importlib, json, sys
from pathlib import Path
try:
    module = importlib.import_module(sys.argv[1])
    if Path(module.__file__).resolve() != Path(sys.argv[2]).resolve():
        raise ValueError('Worker import resolved to a different source checkout')
    from astra_e2e.oversight.policy_bridge import verify_binding
    bound = verify_binding(sys.argv[3])
    if any(name in sys.modules for name in ('isaacsim.simulation_app', 'omni.kit.app')):
        raise ValueError('Import-only preflight unexpectedly loaded simulation application modules')
    print(json.dumps({'ok':True, 'binding_sha256':bound['binding_sha256'],
                      'python':sys.executable, 'python_version':sys.version.split()[0],
                      'worker_module':sys.argv[1]}))
except BaseException as exc:
    print(json.dumps({'ok':False, 'error_type':type(exc).__name__,
                      'missing_module':getattr(exc, 'name', None)}))
    raise SystemExit(1)
'''
PUBLIC_ENV = {'PYTHONPATH', 'CUDA_VISIBLE_DEVICES',
              'ASTRA_E2E_OVERSIGHT_RUN', 'ASTRA_E2E_OVERSIGHT_PLAN_SHA256',
              'R2S2R_OVERSIGHT_RUN', 'R2S2R_OVERSIGHT_PLAN_SHA256'}


def _contracts(config):
    """Derive commands from the registered worker, including launcher and env."""
    binding = read(config['binding'])
    plan = read(Path(binding['oversight_run'])/'plan.json')
    grouped, packages = {}, {}
    for entry in config['backends']:
        package = load_backend(entry['path'])
        packages[entry['path']] = package['sha256']
        for operation, command in package['manifest']['operations'].items():
            argv = command['argv']
            candidates = [index for index, value in enumerate(argv[1:], 1)
                          if value.endswith('.py') and Path(value).is_file()]
            if not candidates:
                raise ValueError('Runtime preflight requires an explicit importable Python worker entry point')
            index = candidates[0]
            worker = Path(argv[index]).resolve()
            if not worker.stem.isidentifier():
                raise ValueError('Worker must have an importable Python module name')
            environment = dict(command['env'])
            environment['CUDA_VISIBLE_DEVICES'] = command['env'].get(
                'CUDA_VISIBLE_DEVICES', ','.join(plan['contract']['budget']['gpu_ids']))
            source = Path(command['cwd'])/'src'
            if source.is_dir():
                environment['PYTHONPATH'] = str(source) + (os.pathsep + environment['PYTHONPATH']
                                                          if environment.get('PYTHONPATH') else '')
            for prefix in ('ASTRA_E2E', 'R2S2R'):
                environment[f'{prefix}_OVERSIGHT_RUN'] = binding['oversight_run']
                environment[f'{prefix}_OVERSIGHT_PLAN_SHA256'] = binding['plan_sha256']
            contract = {'launcher':argv[:index], 'worker':str(worker), 'module':worker.stem,
                        'cwd':command['cwd'], 'environment':environment}
            identity = digest(contract)
            grouped.setdefault(identity, {'contract':contract, 'operations':[]})['operations'].append(operation)
    identity = digest({'config':config, 'packages':packages, 'binding_sha256':file_hash(config['binding']),
                       'probe_source_sha256':file_hash(__file__), 'contracts':grouped})
    return grouped, identity


def run(config_path, *, timeout_seconds=60):
    """Run import/binding checks only and retain each attempt as a new report."""
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 180:
        raise ValueError('Preflight timeout must be an integer from 1 to 180 seconds per runtime')
    config = read(config_path)
    if not config.get('runtime_preflight', {}).get('directory'):
        raise ValueError('Runner configuration must declare its runtime-preflight evidence directory')
    contracts, identity = _contracts(config)
    expected_binding = read(config['binding'])['binding_sha256']
    checks = []
    from ..oversight.controller import Controller
    for key, item in sorted(contracts.items()):
        contract = item['contract']
        argv = [*contract['launcher'], '-c', PROBE, contract['module'], contract['worker'], config['binding']]
        environment = {**os.environ, **contract['environment']}
        started = time.monotonic()
        try:
            child = subprocess.Popen(argv, cwd=contract['cwd'], env=environment,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                stdout, stderr = child.communicate(timeout=timeout_seconds)
                code, failure = child.returncode, None
            except subprocess.TimeoutExpired:
                Controller._terminate(child)
                stdout, stderr = child.communicate()
                code, failure = child.returncode, 'timeout'
            finally:
                Controller._terminate(child)
        except OSError as exc:
            stdout, stderr, code, failure = b'', b'', None, type(exc).__name__
        details = {}
        for line in stdout.decode('utf-8', 'replace').splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and type(value.get('ok')) is bool:
                details = value
        public_environment = {name:(value if name in PUBLIC_ENV else {'value_sha256':digest(value)})
                              for name,value in contract['environment'].items()}
        checks.append({'id':key, 'operations':sorted(set(item['operations'])), 'argv':argv,
                       'cwd':contract['cwd'], 'environment':public_environment,
                       'exit_code':code, 'failure':failure, 'timeout_seconds':timeout_seconds,
                       'elapsed_seconds':time.monotonic()-started,
                       'stdout_sha256':hashlib.sha256(stdout).hexdigest(), 'stderr_sha256':hashlib.sha256(stderr).hexdigest(),
                       'stdout_bytes':len(stdout), 'stderr_bytes':len(stderr), 'details':details,
                       'ready':failure is None and code == 0 and details.get('ok') is True
                               and details.get('binding_sha256') == expected_binding
                               and details.get('worker_module') == contract['module']})
    report = {'schema':'astra_runtime_preflight_v1', 'configuration_sha256':digest(config),
              'contract_sha256':identity, 'checks':checks, 'ready':bool(checks) and all(row['ready'] for row in checks),
              'scope':'Import and binding readiness only; no simulator app, collection, training, model call or hardware.',
              'time':time.time()}
    directory = Path(config['runtime_preflight']['directory'])
    path = directory/f'{time.time_ns()}-{uuid.uuid4().hex}.json'
    write_new(path, report)
    return {'path':str(path), **report}


def verify(config):
    """Fail before model invocation if this exact preparation has no ready report."""
    descriptor = config.get('runtime_preflight', {})
    paths = sorted(Path(descriptor['directory']).glob('*.json')) if descriptor.get('directory') else []
    if not paths:
        raise ValueError('Runtime preflight is required before starting actual Astra')
    report = read(paths[-1])
    contracts, identity = _contracts(config)
    expected_binding = read(config['binding'])['binding_sha256']
    checks = report.get('checks', [])
    if (report.get('schema') != 'astra_runtime_preflight_v1' or report.get('ready') is not True
            or report.get('configuration_sha256') != digest(config) or report.get('contract_sha256') != identity
            or {row.get('id') for row in checks} != set(contracts) or len(checks) != len(contracts)
            or not all(row.get('ready') is True and row.get('exit_code') == 0
                       and row.get('failure') is None and row.get('details', {}).get('ok') is True
                       and row['details'].get('binding_sha256') == expected_binding
                       and row['details'].get('worker_module') == contracts[row['id']]['contract']['module']
                       for row in checks)):
        raise ValueError('Runtime preflight failed or is stale; model execution remains blocked')
    return {'report':str(paths[-1]), 'sha256':file_hash(paths[-1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=60)
    args = parser.parse_args(argv)
    result = run(args.config, timeout_seconds=args.timeout_seconds)
    print(json.dumps({'report':result['path'], 'ready':result['ready'],
                      'checks':[{'operations':r['operations'], 'ready':r['ready'], 'exit_code':r['exit_code'],
                                 'details':r['details']} for r in result['checks']]}))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
