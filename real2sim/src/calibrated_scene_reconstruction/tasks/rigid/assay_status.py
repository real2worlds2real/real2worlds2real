"""Separate a candidate assay's measured checks from native semantic approval."""
import copy


def summarize(result):
    """Do not turn a relaxed secondary numeric pass into object acceptance.

    This diagnostic adapter cannot approve a semantic review. The ordinary
    object-understanding/structural review path verifies that evidence later.
    Existing raw measurements and flexible approximation policy are preserved.
    """
    if not isinstance(result, dict) or type(result.get('pass')) is not bool:
        raise ValueError('A measured boolean raw assay result is required')

    def pending(value):
        if isinstance(value, dict):
            return value.get('native_semantic_review_required') is True or any(pending(v) for v in value.values())
        return isinstance(value, list) and any(pending(v) for v in value)

    required = bool(pending(result))
    raw = result['pass']
    status = 'failed' if not raw else 'pending_native_semantic_review' if required else 'measured_checks_passed'
    report = copy.deepcopy(result)
    report['raw_assay_pass'] = raw
    report['pass'] = None if raw and required else raw
    report['assay_status'] = {
        'status': status, 'measured_checks_pass': raw,
        'native_semantic_review_required': required,
        'native_semantic_review': 'pending' if required else 'not_requested_by_assay',
        'object_approved': False, 'task_approved': False,
        'scope': 'Candidate measurements only. A secondary numeric allowance cannot approve '
                 'loss of object shape or material plausibility; current native semantic evidence '
                 'and all ordinary source/task/object gates remain mandatory.'}
    return report
