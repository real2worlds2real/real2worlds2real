"""Bind input-acceptance proofs to physical geometry, inputs and saved evidence.

This module reads result contracts only. It never imports a task checker or
turns a diagnostic trial into a passing criterion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np










def read(session,criterion,expected_subject,case):
    """Missing/failed/stale proofs remain failures; the input cannot self-approve."""
    root=session.dir.resolve()
    try:
        plan=session.config('agent/acceptance_evidence.json',{})
        if plan.get('version')!=1:raise ValueError('No input acceptance evidence declaration')
        entry=plan['criteria'][criterion][case];path=(root/entry['path']).resolve()
        if not path.is_relative_to(root/'qa') or not path.is_file() or sha(path)!=entry['sha256']:
            raise ValueError('Acceptance result missing or changed')
        result=json.loads(path.read_text())
        if result.get('criterion')!=criterion or result.get('subject_sha256')!=expected_subject:
            raise ValueError('Acceptance proof belongs to different geometry, physics or inputs')
        if not result.get('evidence_sha256') or not result.get('input_sha256'):
            raise ValueError('Acceptance proof needs saved measurements and current implementation inputs')
        for name,expected in result['evidence_sha256'].items():
            file=(root/name).resolve()
            if not file.is_relative_to(root/'qa') or not file.is_file() or sha(file)!=expected:
                raise ValueError('Saved acceptance evidence changed')
        for name,expected in result['input_sha256'].items():
            file=Path(name)
            if not file.is_file() or sha(file)!=expected:raise ValueError('Acceptance implementation or source inputs changed')
        result['evidence_files']=list(dict.fromkeys([str(path.relative_to(root)),*result['evidence_sha256']]))
        from calibrated_scene_reconstruction.assets.acceptance import declaration,evaluate
        requests=declaration(session.config('task.json',{}))['criteria']
        request=next((row for row in requests if row['id']==criterion),None)
        if request is None:raise ValueError('Criterion is not specified in the current task input')
        if not evaluate({'acceptance':{'version':1,'criteria':[request],'relaxations':[]}}, {}, {criterion:result})['pass']:
            raise ValueError('Specified criterion or one of its named checks did not pass')
        return result
    except (ValueError,KeyError,TypeError,OSError) as error:
        return {'pass':False,'criterion':criterion,'why':str(error),'evidence_files':[]}


def completed_task_criteria(session,obj,mesh):
    """Reuse current complete native results without a circular proof rewrite.

    An independently qualified provisional asset may be needed to start a task
    check. Once all native cases pass, those current results can directly
    justify its explicit input criteria; no task implementation is imported.
    """
    if not hasattr(session,'dir') or not hasattr(session,'scene_graph'):return {}
    path=session.dir/'qa/task_check.json'
    if not path.exists():return {}
    try:
        result=json.loads(path.read_text())
        if result.get('pass') is not True:return {}
        from calibrated_scene_reconstruction.core.paths import resolve
        fix=session.scene_graph['dynamic_objects'][obj]['fix']
        if mesh_digest(mesh)!=mesh_digest(resolve(fix['mesh'])):return {}
        from calibrated_scene_reconstruction.scene.twin.reviews import task_result
        valid,_=task_result(session,True)
        if not valid:return {}
        return {name:{**record,'evidence_files':list(dict.fromkeys(['qa/task_check.json',*record.get('evidence_files',[])]))}
                for name,record in result.get('input_criteria',{}).items()}
    except (OSError,ValueError,KeyError,TypeError):
        return {}


from calibrated_scene_reconstruction.assets.acceptance_subject import digest, sha, mesh_digest, subject  # compatibility exports
