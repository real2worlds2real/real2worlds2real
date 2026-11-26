"""A declared GS-preserving owner needs actual usable same-owner source views.

Zero-view masks produce complete native backing. They must not be accepted as
successful hybrid preservation merely because their source-owner labels match.
"""


def audit(ownership, masks):
    rows = []
    for entity, owner in ownership.items():
        declaration = owner.get('mesh_coverage', {})
        if (owner.get('role') != 'gap_fill' or owner.get('preserve_gs') is not True
                or declaration.get('compositing') != 'captured_projection'):
            continue
        measured = masks.get(entity, {}).get('capture_coverage', {})
        count = measured.get('views', 0)
        passed = (type(count) is int and count >= 2 and measured.get('coverage_entity') == entity
                  and declaration.get('coverage_entity', entity) == entity)
        mode = declaration.get('evidence_mode')
        if mode is not None or measured.get('evidence_mode') is not None:
            passed &= (mode == measured.get('evidence_mode') == 'opaque_source_rays'
                       and measured.get('minimum_separated_views') == 2
                       and measured.get('minimum_ray_angle_deg', 0) >= 2
                       and measured.get('ray_mode') == declaration.get('ray_mode') == 'opaque_layers')
        rows.append({'entity': entity, 'usable_source_views': count, 'pass': passed,
                     'action': None if passed else 'Review and acquire same-owner source masks, then remeasure actual opaque GS coverage; keep appearance unapproved.'})
    return {'pass': all(row['pass'] for row in rows), 'checks': rows, 'scope': __doc__.strip()}


def validate(ownership, masks):
    result = audit(ownership, masks)
    if not result['pass']:
        raise ValueError('GS-preserving owners lack two usable same-owner coverage views: '+
                         ', '.join(row['entity'] for row in result['checks'] if not row['pass']))
    return result
