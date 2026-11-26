"""Functional-structure invariants shared by cousin generation and acceptance."""
from __future__ import annotations

from collections import Counter


def compare_structure(parent, child):
    """Preserve joint roles/types/connectivity, while allowing new metric geometry."""
    failures = []
    old = {row['name']: row for row in parent.get('joints', [])}
    new = {row['name']: row for row in child.get('joints', [])}
    if len(old) != len(parent.get('joints', [])) or len(new) != len(child.get('joints', [])):
        failures.append('duplicate_joint_identifiers')
    for name in sorted(old.keys()-new.keys()):
        failures.append(f'missing_joint:{name}')
    for name in sorted(new.keys()-old.keys()):
        failures.append(f'extra_joint:{name}')
    for name in sorted(old.keys() & new.keys()):
        for key in ('joint_type', 'parent_joint'):
            if old[name].get(key) != new[name].get(key):
                failures.append(f'changed_{key}:{name}')

    def loops(snapshot):
        return Counter((row['type'], *sorted((row['body_a'], row['body_b'])))
                       for row in snapshot.get('loop_constraints', []))

    if loops(parent) != loops(child):
        failures.append('changed_mechanical_loop_topology')

    def sewing(snapshot):
        return Counter((row['control_object'], row['group'], row['body'])
                       for row in snapshot.get('cloth_attachments', []))

    if sewing(parent) != sewing(child):
        failures.append('changed_cloth_attachment_ownership')
    if sorted(parent.get('controlled_joints', [])) != sorted(child.get('controlled_joints', [])):
        failures.append('changed_mechanism_control_roles')
    return {'ok': not failures, 'failures': failures, 'parent_joint_count': len(old),
            'child_joint_count': len(new),
            'scope': 'Joint identifiers/roles, types and connectivity; mechanical loops and cloth attachment ownership. Metric dimensions, joint positions/orientations, limits and mesh topology may change; function still needs agent and target-simulator review.'}


def validate_cousin(parent, child, *, allow_surface_only=False, sibling_geometry=()):
    result = compare_structure(parent, child)
    fingerprint = child.get('geometry_fingerprint')
    if not fingerprint:
        result['failures'].append('missing_geometry_fingerprint')
    changed = fingerprint != parent.get('geometry_fingerprint')
    result['geometry_changed_from_parent'] = changed
    result['duplicate_sibling_geometry'] = fingerprint in sibling_geometry
    if not allow_surface_only:
        if not changed:
            result['failures'].append('geometry_unchanged_from_parent')
        if result['duplicate_sibling_geometry']:
            result['failures'].append('geometry_duplicates_accepted_sibling')
    result['ok'] = not result['failures']
    result['novelty_scope'] = 'Canonical visible vertex-position fingerprint detects exact geometric reuse; it is not a semantic diversity score. Agent review must assess meaningful prompt-directed redesign.'
    return result
