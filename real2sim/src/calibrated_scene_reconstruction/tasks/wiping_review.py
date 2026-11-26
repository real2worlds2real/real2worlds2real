"""File-mediated visual agent gate; no simulator or fluid state is accepted.

The running simulation freezes between observations until an external agent
inspects the raw RGB and localization overlay and writes a bound decision.
"""
import hashlib
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_decision(request, decision, proposed_mask):
    for key in ['request_id', 'rgb_sha256', 'proposal_sha256']:
        if decision.get(key) != request[key]:
            raise ValueError(f'Agent decision does not match this observation: {key}')
    if decision.get('reviewer') != 'codex_agent':
        raise ValueError('An external codex_agent visual review is required')
    if not all(decision.get(k) is True for k in ['inspected_raw_rgb', 'inspected_localization_overlay']):
        raise ValueError('Agent must inspect the raw image and localization overlay')
    if not isinstance(decision.get('reason'), str) or len(decision['reason'].strip()) < 12:
        raise ValueError('A concrete visual review reason is required')
    action = decision.get('action')
    mask = proposed_mask.copy().astype(np.uint8)
    if action == 'clear':
        mask[:] = 0
    elif action == 'replace':
        polygons = decision.get('water_polygons_uv')
        if not polygons:
            raise ValueError('A replacement requires agent-drawn image polygons')
        mask[:] = 0
        for polygon in polygons:
            uv = np.asarray(polygon, float)
            if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 3 or not np.isfinite(uv).all()
                    or (uv < 0).any() or (uv[:,0] >= mask.shape[1]).any() or (uv[:,1] >= mask.shape[0]).any()):
                raise ValueError('Invalid image-space water polygon')
            cv2.fillPoly(mask, [np.rint(uv).astype(np.int32)], 1)
    elif action != 'approve':
        raise ValueError('Agent must explicitly approve, replace, or clear')
    if request.get('review_controls_motion'):
        motion=decision.get('motion_decision')
        if motion not in ['wipe','stop']:
            raise ValueError('Agent must explicitly decide wipe or stop from the photo')
        if motion=='stop' and mask.any():
            raise ValueError('Stopping requires the agent to clear any proposed water region first')
        if motion=='wipe' and not mask.any():
            raise ValueError('An extra wipe requires an agent-localized visible/uncertain target; no automatic historical sweep')
    return mask


def wait_for_agent(output, index, camera, proposed_mask, proposal, *, timeout_s=3600,review_controls_motion=False):
    output = Path(output)
    directory = output/'agent_reviews'
    directory.mkdir(exist_ok=True)
    prefix = f'round_{index:02d}'
    request = dict(request_id=uuid.uuid4().hex, round_index=index,
                   created_unix_s=time.time(),
                   rgb_path=str((output/f'{prefix}_rgb.png').resolve()),
                   overlay_path=str((output/f'{prefix}_proposal_overlay.png').resolve()),
                   proposal_mask_path=str((output/f'{prefix}_proposal_mask.png').resolve()),
                   rgb_sha256=digest(output/f'{prefix}_rgb.png'),
                   proposal_sha256=digest(output/f'{prefix}_proposal_mask.png'),
                   proposal=proposal,
                   camera=dict(K=camera.K.tolist(),distortion=camera.distortion.tolist(),
                               size=list(camera.size),T_world_color=camera.T_world_color.tolist(),
                               table_z=camera.table_z,table_bounds=camera.table_bounds.tolist()),
                   available_inputs='Head RGB + known calibration/table + CAD/FK exclusion + prior approved image history',
                   physics_frozen_until_review=True,
                   review_controls_motion=review_controls_motion,
                   instructions='Compare raw RGB with detected footprint and its projected strokes. Approve, replace with image polygons, or clear. No particle state or dry-reference image is supplied.')
    if review_controls_motion:
        request['instructions']+=' Decide motion_decision=wipe or stop from this photo. Empty masks never trigger a mandatory historical sweep; any additional wipe needs an explicit image-space target.'
    if proposal.get('execution_scope')=='batch':
        request['instructions']+=' Wipe approval releases all coverage routes from this observation; the next photo is taken only after that whole batch completes.'
    request_path = directory/f'{prefix}_request.json'
    temporary = directory/f'{prefix}_request.tmp'
    temporary.write_text(json.dumps(request,indent=2)+'\n')
    temporary.replace(request_path)
    decision_path = directory/f'{prefix}_decision.json'
    print(f'[agent_review] Waiting for external visual review: {request_path}',flush=True)
    deadline = time.monotonic()+timeout_s
    while not decision_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'No agent decision; execution remains blocked: {request_path}')
        time.sleep(.25)
    decision = json.loads(decision_path.read_text())
    approved = validate_decision(request,decision,proposed_mask)
    y,x = np.nonzero(approved)
    if len(x):
        _,valid = camera.pixel_to_table(np.c_[x,y])
        if not valid.all():
            raise ValueError('Agent-approved water extends outside calibrated tabletop')
    receipt = dict(request_id=request['request_id'],round_index=index,
                   request_created_unix_s=request['created_unix_s'],accepted_unix_s=time.time(),
                   decision_sha256=digest(decision_path),action=decision['action'],
                   reason=decision['reason'],approved_pixels=int(approved.sum()),
                   motion_decision=decision.get('motion_decision'),
                   physics_steps_during_wait=0)
    (directory/f'{prefix}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return approved, receipt
